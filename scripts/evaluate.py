"""Evaluate a saved model on the regenerated test split.

Example:
    python scripts/evaluate.py --data-dir /path/to/blood-cell-images \\
        --model runs/default/model.keras
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import keras

from wbc_multistream.data import SPLIT_SEED, load_images, make_split
from wbc_multistream.evaluation import evaluate_model, input_ablation


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--ablation", action="store_true")
    args = parser.parse_args()

    images, labels = load_images(args.data_dir)
    split = make_split(images, labels, seed=args.seed)
    model = keras.models.load_model(args.model)

    result = evaluate_model(model, split.x_test, split.y_test)
    print(json.dumps({k: result[k] for k in ("accuracy", "loss", "num_samples")}, indent=2))
    print(json.dumps(result["confusion_matrix"]))
    if args.ablation:
        print(json.dumps(input_ablation(model, split.x_test, split.y_test), indent=2))


if __name__ == "__main__":
    main()
