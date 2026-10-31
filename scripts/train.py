"""Train the three-stream model on the blood-cell image folders and evaluate it.

Example:
    python scripts/train.py --data-dir /path/to/blood-cell-images --protocol holdout
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import keras

from wbc_multistream.data import CLASS_NAMES, SPLIT_SEED, load_images, make_split
from wbc_multistream.evaluation import input_ablation
from wbc_multistream.model import build_model, compile_model
from wbc_multistream.training import TrainConfig, train_and_evaluate

logger = logging.getLogger("train")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Folder with one sub-folder per class (basophil, erythroblast, ...).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/default"))
    parser.add_argument("--protocol", choices=("notebook", "holdout"), default="notebook")
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--val-fraction", type=float, default=TrainConfig.val_fraction)
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Also report accuracy with inputs zeroed at inference time.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    keras.utils.set_random_seed(args.seed)

    images, labels = load_images(args.data_dir)
    logger.info("Loaded %d images; building optical-density and diffusion views", len(images))
    split = make_split(images, labels, seed=args.seed)

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        protocol=args.protocol,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    model = compile_model(build_model(num_classes=len(CLASS_NAMES)), args.learning_rate)
    result = train_and_evaluate(model, split, cfg, CLASS_NAMES, args.output_dir)
    logger.info("Test accuracy (%s protocol): %.4f", args.protocol, result["accuracy"])

    if args.ablation:
        logger.info(
            "Input ablation (zeroed at inference): %s",
            input_ablation(model, split.x_test, split.y_test),
        )


if __name__ == "__main__":
    main()
