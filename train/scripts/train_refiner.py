from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, random_split

from prep import preprocess_dataset

from ..src.data.query_index import load_query_samples
from ..src.data.refiner_dataset import RefinerDataset
from ..src.training.refiner_trainer import save_refiner, train_refiner
from ..src.utils.logging import setup_logging
from ..src.utils.torch_utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the bounding-box refiner.")
    parser.add_argument("--config", type=Path, default=Path("train/configs/refiner.yaml"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit query samples for lightweight training. Leave unset to use all available samples.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--prep-limit", type=int, default=None, help="Limit JSONs per split/group during preprocessing.")
    parser.add_argument("--skip-prep", action="store_true", help="Assume preprocessing already ran.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    output_path = Path(config.get("output_model", "outputs/refiner/refiner.pt"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logging("refiner", output_path.parent / "train.log")
    set_seed(args.seed)

    prep_dir = Path(config.get("preprocessed_dir", "train/preprocessed"))
    if args.skip_prep:
        logger.info("Skipping preprocessing; reusing CSVs in %s", prep_dir)
    else:
        preprocess_dataset(
            data_root=config["data_root"],
            output_dir=prep_dir,
            splits=config.get("prep_splits", ["train"]),
            groups=config.get("prep_groups", ["press", "report"]),
            limit=args.prep_limit,
        )

    train_csvs = [prep_dir / "train_press.csv", prep_dir / "train_report.csv"]
    samples = load_query_samples(train_csvs, config["data_root"])
    if args.max_samples:
        samples = samples[: args.max_samples]

    dataset = RefinerDataset(
        query_samples=samples,
        categories=set(config.get("categories") or []),
        noise_std=config.get("noise_std", 0.05),
        samples_per_annotation=config.get("samples_per_annotation", 2),
    )
    valid_size = int(0.1 * len(dataset))
    if valid_size > 0:
        train_size = len(dataset) - valid_size
        train_subset, valid_subset = random_split(dataset, [train_size, valid_size])
    else:
        train_subset, valid_subset = dataset, None

    train_loader = DataLoader(train_subset, batch_size=config.get("batch_size", 256), shuffle=True)
    valid_loader = (
        DataLoader(valid_subset, batch_size=config.get("batch_size", 256))
        if valid_subset is not None
        else None
    )

    device = torch.device(args.device)
    model, history = train_refiner(
        train_loader=train_loader,
        valid_loader=valid_loader,
        epochs=config.get("epochs", 5),
        lr=config.get("learning_rate", 1e-3),
        device=device,
    )
    save_refiner(model, output_path)


if __name__ == "__main__":
    main()
