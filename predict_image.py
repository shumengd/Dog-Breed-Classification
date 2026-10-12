"""Predict one image using a local checkpoint or a self-contained model bundle."""

import argparse
import json
from pathlib import Path

from PIL import Image

from inference import DogBreedPredictor, model_specs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", choices=model_specs(), default="transfer")
    parser.add_argument(
        "--bundle", type=Path, help="use an exported bundle without local training data"
    )
    parser.add_argument(
        "--output", type=Path, help="optional prediction JSON destination"
    )
    args = parser.parse_args()
    predictor = (
        DogBreedPredictor.from_bundle(args.bundle)
        if args.bundle
        else DogBreedPredictor.from_checkpoint(args.model)
    )
    with Image.open(args.image) as image:
        result = predictor.predict(image, args.image.name)
    serialized = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")
    print(serialized)


if __name__ == "__main__":
    main()
