"""Exercise the demo's actual HTTP routes with an isolated local server."""

import base64
from functools import partial
from http.server import ThreadingHTTPServer
from io import BytesIO
import json
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image
import pytest

from dataset import ImageStandardizer
from demo import DemoApplication, DemoHandler
from inference import DogBreedPredictor
from model.target import Target


@pytest.fixture
def demo_server(tmp_path, monkeypatch):
    monkeypatch.setattr("demo.PROJECT_ROOT", tmp_path)
    web = tmp_path / "web"
    web.mkdir()
    (web / "index.html").write_text("<title>Dog Breed Classification</title>")
    (web / "styles.css").write_text("body { color: black; }")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/benchmark.json").write_text(json.dumps({"results": []}))
    (tmp_path / "private.txt").write_text("not public")
    standardizer = ImageStandardizer()
    standardizer.fit(np.arange(48).reshape(1, 4, 4, 3))
    predictor = DogBreedPredictor(
        Target(), standardizer, "transfer", "Frozen CNN", "target", 5
    )
    predictor.export(tmp_path / "bundles/transfer.bundle.pt")
    application = DemoApplication(tmp_path / "bundles")
    with ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(DemoHandler, application=application)
    ) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}"
        finally:
            server.shutdown()
            thread.join(timeout=5)


def request(base, path, payload=None, content_type="application/json"):
    data = json.dumps(payload).encode() if payload is not None else None
    req = Request(base + path, data=data, headers={"Content-Type": content_type})
    try:
        with urlopen(req, timeout=10) as response:
            return response.status, response.headers.get_content_type(), response.read()
    except HTTPError as response:
        return response.code, response.headers.get_content_type(), response.read()


def test_demo_serves_interface_and_available_model(demo_server):
    status, mime, content = request(demo_server, "/")
    assert status == 200 and mime == "text/html"
    assert b"Dog Breed Classification" in content
    status, mime, _ = request(demo_server, "/static/styles.css")
    assert status == 200 and mime == "text/css"
    status, _, body = request(demo_server, "/api/status")
    assert status == 200 and json.loads(body)["ready"]
    status, _, body = request(demo_server, "/api/benchmarks")
    assert status == 200 and json.loads(body)["historical"]["results"] == []


def test_demo_upload_returns_real_normalized_class_probabilities(demo_server):
    image = BytesIO()
    Image.new("RGB", (96, 48), (150, 90, 50)).save(image, format="PNG")
    status, _, body = request(
        demo_server,
        "/api/predict",
        {
            "model": "transfer",
            "filename": "dog.png",
            "image": base64.b64encode(image.getvalue()).decode(),
        },
    )
    result = json.loads(body)
    assert status == 200
    assert result["filename"] == "dog.png"
    assert result["epoch"] == 5
    assert [item["label"] for item in result["probabilities"]] == [
        "Golden Retriever",
        "Collie",
    ]
    assert sum(
        item["probability"] for item in result["probabilities"]
    ) == pytest.approx(1)


@pytest.mark.parametrize(
    "path", ["/data/dogs.csv", "/static/../private.txt", "/api/samples/missing"]
)
def test_demo_does_not_serve_dataset_or_outside_assets(demo_server, path):
    assert request(demo_server, path)[0] == 404


@pytest.mark.parametrize(
    "payload", [{}, {"image": "invalid!"}, {"sample": ["invalid"]}, {"image": "AA=="}]
)
def test_demo_returns_json_errors_for_invalid_uploads(demo_server, payload):
    status, mime, body = request(demo_server, "/api/predict", payload)
    assert status == 400 and mime == "application/json"
    assert json.loads(body)["error"]


def test_demo_rejects_non_json_requests(demo_server):
    assert request(demo_server, "/api/predict", {}, content_type="text/plain")[0] == 415
