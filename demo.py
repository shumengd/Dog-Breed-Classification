"""Serve the local PyTorch image classifier and experiment viewer."""

import argparse
import base64
import binascii
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
import mimetypes
from pathlib import Path
from threading import Lock
from urllib.parse import urlparse

import pandas as pd
from PIL import Image, UnidentifiedImageError
import torch

from inference import DogBreedPredictor, LABELS, PROJECT_ROOT, model_specs
from utils import config


MAX_REQUEST_BYTES = 12 * 1024 * 1024


class DemoApplication:
    def __init__(self, bundle_dir: Path | None = None) -> None:
        self.specs = model_specs()
        self.bundle_dir = bundle_dir or PROJECT_ROOT / "checkpoints/demo"
        self.predictors: dict[str, DogBreedPredictor] = {}
        self.lock = Lock()
        self.samples = self._samples()

    def _samples(self) -> dict:
        metadata_path = PROJECT_ROOT / config("csv_file")
        image_dir = (PROJECT_ROOT / config("image_path")).resolve()
        if not metadata_path.is_file():
            return {}
        metadata = pd.read_csv(metadata_path)
        rows = metadata[(metadata.task == "target") & (metadata.partition == "test")]
        samples = {}
        for _, row in rows.groupby("numeric_label", sort=True).head(2).iterrows():
            path = (image_dir / row.filename).resolve()
            if not path.is_relative_to(image_dir) or not path.is_file():
                continue
            sample_id = f"sample-{len(samples)}"
            samples[sample_id] = {"path": path, "label": LABELS[int(row.numeric_label)]}
        return samples

    def status(self) -> dict:
        metadata_exists = (PROJECT_ROOT / config("csv_file")).is_file()
        models = []
        for model_id, spec in self.specs.items():
            bundle = self.bundle_dir / f"{model_id}.bundle.pt"
            legacy_exists = any(spec.checkpoint_dir.glob("*.checkpoint.pth.tar"))
            models.append(
                {
                    "id": model_id,
                    "name": spec.name,
                    "available": bundle.is_file()
                    or (legacy_exists and metadata_exists),
                }
            )
        available = [model["id"] for model in models if model["available"]]
        return {
            "models": models,
            "classes": LABELS,
            "image_size": 64,
            "default_model": available[0] if available else None,
            "samples": [
                {
                    "id": key,
                    "url": f"/api/samples/{key}",
                    "label": value["label"],
                    "partition": "test",
                }
                for key, value in self.samples.items()
            ],
            "ready": bool(available),
        }

    def predictor(self, model_id: str) -> DogBreedPredictor:
        if model_id not in self.specs:
            raise ValueError("Unknown model selection")
        with self.lock:
            if model_id not in self.predictors:
                bundle = self.bundle_dir / f"{model_id}.bundle.pt"
                if bundle.is_file():
                    self.predictors[model_id] = DogBreedPredictor.from_bundle(bundle)
                else:
                    self.predictors[model_id] = DogBreedPredictor.from_checkpoint(
                        model_id
                    )
            return self.predictors[model_id]

    def predict(self, payload: dict) -> dict:
        if not isinstance(payload, dict):
            raise ValueError("Expected a JSON object")
        model_id = payload.get("model", "transfer")
        if not isinstance(model_id, str):
            raise ValueError("Select a valid model")
        if payload.get("sample"):
            if not isinstance(payload["sample"], str):
                raise ValueError("Select a valid sample")
            sample = self.samples.get(payload["sample"])
            if sample is None:
                raise ValueError("Unknown sample")
            with Image.open(sample["path"]) as image:
                result = self.predictor(model_id).predict(image, sample["path"].name)
            result["actual_label"] = sample["label"]
            return result
        encoded = payload.get("image")
        if not isinstance(encoded, str) or not encoded:
            raise ValueError("Select an image first")
        try:
            content = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("Invalid image encoding") from exc
        if len(content) > 8 * 1024 * 1024:
            raise ValueError("Use an image smaller than 8 MB")
        with Image.open(BytesIO(content)) as image:
            if image.format not in {"JPEG", "PNG", "WEBP"}:
                raise ValueError("Use a JPEG, PNG, or WebP image")
            return self.predictor(model_id).predict(
                image, str(payload.get("filename", "image"))
            )


class DemoHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, application: DemoApplication, **kwargs) -> None:
        self.application = application
        super().__init__(*args, **kwargs)

    def respond(self, status: int, content: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(content)

    def json_response(self, status: int, payload: dict) -> None:
        self.respond(
            status, json.dumps(payload).encode(), "application/json; charset=utf-8"
        )

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/status":
            self.json_response(200, self.application.status())
        elif path == "/api/benchmarks":
            historical = json.loads((PROJECT_ROOT / "docs/benchmark.json").read_text())
            reproduced = PROJECT_ROOT / "docs/reproduced-benchmark.json"
            self.json_response(
                200,
                {
                    "historical": historical,
                    "reproduced": json.loads(reproduced.read_text())
                    if reproduced.is_file()
                    else None,
                },
            )
        elif path.startswith("/api/samples/"):
            sample = self.application.samples.get(path.removeprefix("/api/samples/"))
            if sample is None:
                self.json_response(404, {"error": "Sample not found"})
                return
            self.respond(200, sample["path"].read_bytes(), "image/png")
        elif path == "/" or path.startswith("/static/") or path.startswith("/figures/"):
            base = PROJECT_ROOT / (
                "docs/figures" if path.startswith("/figures/") else "web"
            )
            relative = "index.html" if path == "/" else path.split("/", 2)[2]
            asset = (base / relative).resolve()
            if not asset.is_relative_to(base.resolve()) or not asset.is_file():
                self.json_response(404, {"error": "File not found"})
                return
            mime = mimetypes.guess_type(asset.name)[0] or "application/octet-stream"
            self.respond(200, asset.read_bytes(), mime)
        else:
            self.json_response(404, {"error": "Not found"})

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/predict":
            self.json_response(404, {"error": "Not found"})
            return
        if self.headers.get_content_type() != "application/json":
            self.json_response(415, {"error": "Expected application/json"})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size <= 0 or size > MAX_REQUEST_BYTES:
                self.json_response(413, {"error": "Use an image smaller than 8 MB"})
                return
            payload = json.loads(self.rfile.read(size))
            self.json_response(200, self.application.predict(payload))
        except (
            ValueError,
            TypeError,
            UnidentifiedImageError,
            Image.DecompressionBombError,
        ) as exc:
            self.json_response(400, {"error": str(exc)})
        except FileNotFoundError:
            self.json_response(
                503,
                {
                    "error": "Model files unavailable. Train a model or export a bundle first."
                },
            )
        except OSError:
            self.json_response(
                400,
                {
                    "error": "This image could not be decoded. Select another JPEG, PNG, or WebP image."
                },
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--bundle-dir", type=Path, help="directory containing exported model bundles"
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    handler = partial(DemoHandler, application=DemoApplication(args.bundle_dir))
    with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
        print(f"Dog Breed Classification: http://127.0.0.1:{args.port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
