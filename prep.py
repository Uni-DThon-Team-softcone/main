"""
Utility script for preprocessing the Query-Conditioned Table/Chart dataset.

It scans `~/data` (or any provided data root), collects query-bearing annotations,
and writes normalized CSVs. Each row contains enough metadata to recover the
originating JSON (for downstream candidate generation) while also pointing to the
image path so lightweight trainers can operate directly on preprocessed inputs.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


CSV_FIELDS = [
    "split",
    "group",
    "json_path",
    "image_path",
    "page_id",
    "instance_id",
    "class_id",
    "class_name",
    "visual_instruction",
    "visual_answer",
    "x",
    "y",
    "w",
    "h",
    "width",
    "height",
]

DOC_TYPE_TO_GROUP = {
    "보도자료": "press",
    "보고서": "report",
}


def infer_group(doc_type: str | None) -> str:
    doc_type = (doc_type or "").strip()
    return DOC_TYPE_TO_GROUP.get(doc_type, "press")


def collect_annotations(
    data: Dict[str, Any],
    json_path: Path,
    image_path: Path,
    split: str,
    group: str,
) -> Iterable[Dict[str, str | float]]:
    source_info = data.get("source_data_info", {})
    learning_info = data.get("learning_data_info", {})

    document_resolution = source_info.get("document_resolution", [0, 0])
    width, height = document_resolution if len(document_resolution) == 2 else (0, 0)

    rows: List[Dict[str, str | float]] = []
    for annotation in learning_info.get("annotation", []):
        if "visual_instruction" not in annotation:
            continue
        bbox = annotation.get("bounding_box")
        if not bbox or len(bbox) != 4:
            continue
        row = {
            "split": split,
            "group": group,
            "json_path": str(json_path),
            "image_path": str(image_path),
            "page_id": json_path.stem,
            "instance_id": annotation.get("instance_id", ""),
            "class_id": annotation.get("class_id", ""),
            "class_name": annotation.get("class_name", ""),
            "visual_instruction": annotation.get("visual_instruction", ""),
            "visual_answer": annotation.get("visual_answer", ""),
            "x": bbox[0],
            "y": bbox[1],
            "w": bbox[2],
            "h": bbox[3],
            "width": width,
            "height": height,
        }
        rows.append(row)
    return rows


def preprocess_split(
    data_root: Path,
    split: str,
    groups: Sequence[str],
    output_dir: Path,
    limit: int | None = None,
) -> List[Path]:
    json_dir = data_root / split / "json"
    image_dir = data_root / split / "jpg"
    csv_paths = []
    requested_groups = list(groups)
    group_rows: Dict[str, List[Dict[str, str | float]]] = {group: [] for group in requested_groups}
    group_counts = defaultdict(int)

    json_files = sorted(json_dir.glob("*.json"))

    for json_file in json_files:
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"[prep] Skipping malformed JSON {json_file}: {exc}")
            continue
        doc_type = data.get("raw_data_info", {}).get("doc_type")
        group = infer_group(doc_type)
        if requested_groups and group not in requested_groups:
            continue
        if limit is not None and group_counts[group] >= limit:
            continue

        image_name = data.get("source_data_info", {}).get("source_data_name_jpg")
        if not image_name:
            continue
        image_path = image_dir / image_name
        rows = collect_annotations(
            data=data,
            json_path=json_file,
            image_path=image_path,
            split=split,
            group=group,
        )
        if rows:
            group_rows.setdefault(group, []).extend(rows)
        group_counts[group] += 1

    for group in requested_groups:
        csv_path = output_dir / f"{split}_{group}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for row in group_rows.get(group, []):
                writer.writerow(row)
        csv_paths.append(csv_path)
    return csv_paths


def preprocess_dataset(
    data_root: str | Path,
    output_dir: str | Path,
    splits: Iterable[str] = ("train", "valid"),
    groups: Iterable[str] = ("press", "report"),
    limit: int | None = None,
) -> List[Path]:
    root = Path(data_root).expanduser()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_paths: List[Path] = []
    for split in splits:
        csv_paths.extend(preprocess_split(root, split, list(groups), out_dir, limit=limit))
    return csv_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess query-bearing annotations into CSV files.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/workspace/data"),
        help="Root directory containing split folders with 'json' and 'jpg' children.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("train/preprocessed"), help="Destination directory for CSVs.")
    parser.add_argument("--splits", nargs="+", default=["train", "valid"], help="Dataset splits to process.")
    parser.add_argument("--groups", nargs="+", default=["press", "report"], help="Groups (press/report).")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of JSON files per split/group for fast smoke tests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_paths = preprocess_dataset(
        data_root=args.data_root,
        output_dir=args.output_dir,
        splits=args.splits,
        groups=args.groups,
        limit=args.limit,
    )
    print("Wrote CSVs:")
    for path in csv_paths:
        print(path)


if __name__ == "__main__":
    main()
