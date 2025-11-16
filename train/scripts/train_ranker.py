from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from prep import preprocess_dataset

from ..src.data.query_index import load_query_samples
from ..src.data.ranker_dataset import RankerDataset, build_ranker_dataset
from ..src.training.ranker_trainer import save_ranker, train_ranker
from ..src.utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the LightGBM ranker.")
    parser.add_argument("--config", type=Path, default=Path("train/configs/ranker.yaml"))
    parser.add_argument("--max-train", type=int, default=None, help="Limit number of query samples for training.")
    parser.add_argument("--max-valid", type=int, default=None, help="Limit number of query samples for validation.")
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--prep-limit", type=int, default=None, help="Limit JSON files per split/group during preprocessing.")
    parser.add_argument("--skip-prep", action="store_true", help="Assume preprocessed CSVs already exist.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    output_path = Path(config.get("output_model", "outputs/ranker/ranker.txt"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logging("ranker", output_path.parent / "train.log")

    prep_dir = Path(config.get("preprocessed_dir", "train/preprocessed"))
    if args.skip_prep:
        logger.info("Skipping preprocessing; using existing CSVs in %s", prep_dir)
    else:
        preprocess_dataset(
            data_root=config["data_root"],
            output_dir=prep_dir,
            splits=config.get("prep_splits", ["train", "valid"]),
            groups=config.get("prep_groups", ["press", "report"]),
            limit=args.prep_limit,
        )

    train_csvs = [prep_dir / "train_press.csv", prep_dir / "train_report.csv"]
    valid_csvs = [prep_dir / "valid_press.csv", prep_dir / "valid_report.csv"]

    train_samples = load_query_samples(train_csvs, config["data_root"])
    valid_samples = load_query_samples(valid_csvs, config["data_root"])

    if args.max_train:
        train_samples = train_samples[: args.max_train]
    if args.max_valid:
        valid_samples = valid_samples[: args.max_valid]

    train_dataset = build_ranker_dataset(
        train_samples,
        max_queries_per_page=args.max_queries or config.get("max_queries_per_page"),
    )
    valid_dataset = build_ranker_dataset(
        valid_samples,
        max_queries_per_page=args.max_queries or config.get("max_queries_per_page"),
    )

    model = train_ranker(
        train_data=train_dataset,
        valid_data=valid_dataset,
        params=config.get("lightgbm"),
    )
    save_ranker(model, output_path)

    metadata = {
        "train_samples": len(train_dataset.frame),
        "valid_samples": len(valid_dataset.frame),
        "groups_train": len(train_dataset.group_counts),
        "groups_valid": len(valid_dataset.group_counts),
    }
    (output_path.parent / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
