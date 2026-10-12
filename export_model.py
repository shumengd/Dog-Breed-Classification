"""Export the selected local checkpoint with its training normalization."""

import argparse

from inference import DogBreedPredictor, model_specs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=model_specs(), default="transfer")
    parser.add_argument("--output", help="model bundle destination")
    parser.add_argument(
        "--checkpoint-dir", help="optional experiment checkpoint directory"
    )
    args = parser.parse_args()
    output = args.output or f"checkpoints/demo/{args.model}.bundle.pt"
    predictor = DogBreedPredictor.from_checkpoint(
        args.model, checkpoint_dir=args.checkpoint_dir
    )
    print(
        f"Exported {predictor.model_name}, epoch {predictor.epoch}: {predictor.export(output)}"
    )


if __name__ == "__main__":
    main()
