from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml

from ..src.training.yolo_detector import train_yolo_detector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the DocLayout YOLO detector.")
    parser.add_argument("--config", type=Path, default=Path("train/configs/detector.yaml"))
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs defined in the config.")
    parser.add_argument("--imgsz", type=int, default=None, help="Override image size (pixels).")
    parser.add_argument("--batch", type=int, default=None, help="Override batch size.")
    parser.add_argument("--device", type=str, default=None, help="Device string passed to Ultralytics (e.g., '0' or 'cpu').")
    parser.add_argument("--fraction", type=float, default=None, help="Subset fraction for YOLO training.")
    parser.add_argument("--resume", action="store_true", help="Resume the most recent YOLO run.")
    parser.add_argument("--name", type=str, default=None, help="Override YOLO run name.")
    parser.add_argument("--project", type=str, default=None, help="Override YOLO project directory.")
    parser.add_argument("--model", type=str, default=None, help="Override model checkpoint path.")
    parser.add_argument("--data-yaml", type=str, default=None, help="Override dataset YAML path.")
    return parser.parse_args()


def _resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _collect_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "fraction": args.fraction,
        "resume": args.resume,
        "name": args.name,
        "project": args.project,
        "model": args.model,
        "data": args.data_yaml,
    }


def main() -> None:
    args = parse_args()
    repo_root = _resolve_repo_root()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text())

    yolo_cfg = config.get("yolo", {})
    output_dir = repo_root / config.get("output_dir", "outputs/detector")
    overrides = {k: v for k, v in _collect_overrides(args).items() if v is not None}

    train_yolo_detector(
        yolo_cfg=yolo_cfg,
        repo_root=repo_root,
        output_dir=output_dir,
        cli_overrides=overrides,
    )


if __name__ == "__main__":
    main()
