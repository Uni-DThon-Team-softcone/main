from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from ultralytics import YOLO

from ..src.data.refiner_dataset import _features as refiner_anchor_features
from ..src.data.schema import AnnotationRecord, PageRecord
from ..src.features.ranker_features import extract_candidate_features
from ..src.pipeline.candidate_selector import class_category, gate_candidates
from ..src.pipeline.query_parser import parse_intent
from ..src.training.refiner_trainer import BoxRefiner

DOC_TYPE_TO_GROUP = {"보도자료": "press", "보고서": "report"}
READABLE_CLASS_NAMES = {
    "L01": "머리말",
    "L02": "꼬리말",
    "L03": "페이지번호",
    "T01": "제목",
    "T02": "소제목",
    "T03": "시각요소 제목",
    "V01": "표",
    "V02": "차트",
    "V03": "다이어그램",
}
TARGET_CATEGORIES = {"chart", "table", "diagram", "body"}


def normalize_class_id(class_id: str | None) -> str | None:
    if not class_id:
        return None
    if class_id.startswith("V02"):
        return "V02"
    return class_id


def resolve_torch_device(device: str | torch.device | None) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if not device:
        return torch.device("cpu")
    device = device.strip()
    if "," in device:
        device = device.split(",")[0].strip()
    if device.isdigit():
        return torch.device(f"cuda:{device}")
    if device.lower().startswith("cuda") or device.lower().startswith("cpu"):
        return torch.device(device)
    return torch.device(device)


@dataclass
class QueryEntry:
    query_id: str
    query_text: str
    page_id: str
    target_class_id: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end inference pipeline (Detector → Ranker → Refiner).")
    parser.add_argument("--data-root", type=Path, default=Path("/workspace/data"))
    parser.add_argument("--query-dir", type=Path, default=None, help="Directory containing query JSON files.")
    parser.add_argument("--image-dir", type=Path, default=None, help="Directory containing page JPGs.")
    parser.add_argument("--detector-weights", type=Path, default=Path("outputs/yolo_models/doclayout_yolo/weights/best.pt"))
    parser.add_argument("--ranker-model", type=Path, default=Path("outputs/ranker/lightgbm.txt"))
    parser.add_argument("--refiner-model", type=Path, default=Path("outputs/refiner/refiner.pt"))
    parser.add_argument("--submission-template", type=Path, default=Path("data/sample_submission.csv"))
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", type=str, default=None, help="Device string for YOLO/torch (e.g., '0' or 'cpu').")
    parser.add_argument("--chunk-size", type=int, default=256, help="Number of pages to send per YOLO predict() call.")
    return parser.parse_args()


def load_query_index(query_dir: Path, image_dir: Path) -> tuple[Dict[str, PageRecord], Dict[str, QueryEntry]]:
    page_records: Dict[str, PageRecord] = {}
    query_index: Dict[str, QueryEntry] = {}

    json_paths = sorted(query_dir.glob("*.json"))
    for json_path in tqdm(json_paths, desc="[inference] load queries", leave=False):
        data = json.loads(json_path.read_text(encoding="utf-8"))
        page_id = Path(json_path).stem
        source_info = data.get("source_data_info", {})
        learning_info = data.get("learning_data_info", {})
        width, height = source_info.get("document_resolution", [2480, 3508])
        image_name = source_info.get("source_data_name_jpg")
        if not image_name:
            continue
        image_path = image_dir / image_name
        doc_type = data.get("raw_data_info", {}).get("doc_type", "")
        annotations = learning_info.get("annotation", [])
        if not annotations:
            continue

        page_record = PageRecord(
            json_path=json_path,
            image_path=image_path,
            width=int(width),
            height=int(height),
            annotations=[],  # populated after detector runs
            visual_context=learning_info.get("visual_context", ""),
            type_id=learning_info.get("type_id", ""),
            type_name=learning_info.get("type_name", ""),
            split="test",
            group=DOC_TYPE_TO_GROUP.get(doc_type, "press"),
        )
        page_records[page_id] = page_record

        for ann in annotations:
            query_id = ann.get("instance_id")
            if not query_id:
                continue
            query_text = ann.get("visual_instruction", "")
            query_class = normalize_class_id(ann.get("class_id"))
            query_index[query_id] = QueryEntry(
                query_id=query_id,
                query_text=query_text,
                page_id=page_id,
                target_class_id=query_class,
            )
    return page_records, query_index


def run_detector(
    model: YOLO,
    page_records: Dict[str, PageRecord],
    imgsz: int,
    batch: int,
    conf: float,
    device: str | None,
    chunk_size: int,
) -> None:
    if not page_records:
        return
    items = sorted(page_records.items(), key=lambda kv: kv[0])
    chunk_size = max(1, chunk_size)
    for start in tqdm(range(0, len(items), chunk_size), desc="[inference] detect", leave=False):
        chunk = items[start : start + chunk_size]
        sources = [str(record.image_path) for _, record in chunk]
        results = model.predict(
            source=sources,
            imgsz=imgsz,
            conf=conf,
            batch=batch,
            verbose=False,
            device=device,
        )
        for (page_id, record), result in zip(chunk, results):
            annotations = convert_detections_to_annotations(result, page_id, record.width, record.height)
            record.annotations = annotations


def convert_detections_to_annotations(result, page_id: str, width: int, height: int) -> List[AnnotationRecord]:
    annotations: List[AnnotationRecord] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.xywhn is None:
        return annotations

    xywhn = boxes.xywhn.cpu().numpy()
    cls_ids = boxes.cls.cpu().numpy().astype(int)
    confidences = boxes.conf.cpu().numpy()
    for idx, (coords, cls_id, conf) in enumerate(zip(xywhn, cls_ids, confidences)):
        cls_label = result.names.get(int(cls_id), f"class_{cls_id}")
        readable_name = READABLE_CLASS_NAMES.get(cls_label, cls_label)
        x_c, y_c, w_n, h_n = coords
        w = w_n * width
        h = h_n * height
        x = (x_c - w_n / 2) * width
        y = (y_c - h_n / 2) * height
        x = max(0.0, min(x, width - 1))
        y = max(0.0, min(y, height - 1))
        w = max(1.0, min(w, width - x))
        h = max(1.0, min(h, height - y))
        annotations.append(
            AnnotationRecord(
                class_id=cls_label,
                class_name=readable_name,
                bbox=[float(x), float(y), float(w), float(h)],
                instance_id=f"{page_id}_det_{idx}",
                visual_instruction=None,
                visual_answer=f"conf={conf:.3f}",
            )
        )
    return annotations


def load_ranker(path: Path) -> tuple[lgb.Booster, Sequence[str]]:
    booster = lgb.Booster(model_file=str(path))
    feature_names = booster.feature_name()
    return booster, feature_names


def load_refiner(path: Path | None, device: str | None) -> BoxRefiner | None:
    if path is None:
        return None
    path = path.expanduser()
    if not path.exists() or not path.is_file():
        return None
    model = BoxRefiner()
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    torch_device = resolve_torch_device(device)
    if torch_device.type != "cpu":
        model.to(torch_device)
    model.eval()
    return model


def select_candidate(
    record: PageRecord,
    query_entry: QueryEntry,
    ranker: lgb.Booster,
    feature_names: Sequence[str],
) -> AnnotationRecord | None:
    query_ann = AnnotationRecord(
        class_id="query",
        class_name="query",
        bbox=[0.0, 0.0, 0.0, 0.0],
        instance_id=query_entry.query_id,
        visual_instruction=query_entry.query_text,
        visual_answer=None,
    )
    candidates = [ann for ann in record.annotations if class_category(ann) in TARGET_CATEGORIES]
    if query_entry.target_class_id:
        filtered = [
            ann for ann in candidates if normalize_class_id(ann.class_id) == query_entry.target_class_id
        ]
        if filtered:
            candidates = filtered
    if not candidates:
        return None
    gated = gate_candidates(record, query_entry.query_text or "", candidates)
    intent = parse_intent(query_entry.query_text or "")

    best_candidate = None
    best_score = float("-inf")
    for cand in gated:
        features = extract_candidate_features(record, query_ann, cand, intent=intent)
        prefixed = {f"feat_{k}": v for k, v in features.items()}
        vector = np.array([[prefixed.get(name, 0.0) for name in feature_names]], dtype=np.float32)
        if ranker.best_iteration and ranker.best_iteration > 0:
            score = float(ranker.predict(vector, num_iteration=ranker.best_iteration)[0])
        else:
            score = float(ranker.predict(vector)[0])
        if score > best_score:
            best_score = score
            best_candidate = cand
    return best_candidate


def refine_bbox(
    bbox: List[float],
    candidate: AnnotationRecord,
    record: PageRecord,
    refiner: BoxRefiner | None,
    torch_device: torch.device,
) -> List[float]:
    if refiner is None or class_category(candidate) not in {"chart", "table", "diagram"}:
        return bbox
    feats = refiner_anchor_features(candidate.bbox, record.width, record.height, class_category(candidate))
    tensor = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)
    if torch_device.type != "cpu":
        tensor = tensor.to(torch_device)
    with torch.no_grad():
        deltas = refiner(tensor).squeeze(0).cpu().numpy()
    ax, ay, aw, ah = candidate.bbox
    dx, dy, dw, dh = deltas
    aw = max(aw, 1e-3)
    ah = max(ah, 1e-3)
    x = ax + dx * aw
    y = ay + dy * ah
    w = aw * math.exp(dw)
    h = ah * math.exp(dh)
    x = max(0.0, min(x, record.width - 1))
    y = max(0.0, min(y, record.height - 1))
    w = max(1.0, min(w, record.width - x))
    h = max(1.0, min(h, record.height - y))
    return [float(x), float(y), float(w), float(h)]


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser()
    query_dir = args.query_dir or (data_root / "test" / "query")
    image_dir = args.image_dir or (data_root / "test" / "images")

    page_records, query_index = load_query_index(query_dir, image_dir)
    if not query_index:
        raise RuntimeError(f"No queries discovered under {query_dir}")

    detector = YOLO(str(args.detector_weights))
    run_detector(
        detector,
        page_records,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        device=args.device,
        chunk_size=args.chunk_size,
    )

    ranker, feature_names = load_ranker(args.ranker_model)
    torch_device = resolve_torch_device(args.device)
    refiner = load_refiner(args.refiner_model, torch_device)

    if args.submission_template and args.submission_template.exists():
        template_df = pd.read_csv(args.submission_template)
        target_queries = [
            QueryEntry(
                query_id=row["query_id"],
                query_text=row.get(
                    "query_text",
                    (query_index.get(row["query_id"]).query_text if row["query_id"] in query_index else ""),
                ),
                page_id=query_index.get(row["query_id"]).page_id if row["query_id"] in query_index else "",
                target_class_id=(
                    query_index.get(row["query_id"]).target_class_id if row["query_id"] in query_index else None
                ),
            )
            for _, row in template_df.iterrows()
        ]
    else:
        target_queries = [query_index[qid] for qid in sorted(query_index.keys())]

    predictions: Dict[str, List[float]] = {}
    for entry in tqdm(target_queries, desc="[inference] rank+refine", leave=False):
        page_record = page_records.get(entry.page_id)
        if page_record is None:
            predictions[entry.query_id] = [0.0, 0.0, 0.0, 0.0]
            continue
        candidate = select_candidate(page_record, entry, ranker, feature_names)
        if candidate is None:
            predictions[entry.query_id] = [0.0, 0.0, 0.0, 0.0]
            continue
        refined = refine_bbox(candidate.bbox, candidate, page_record, refiner, torch_device)
        predictions[entry.query_id] = refined

    rows = []
    for entry in target_queries:
        bbox = predictions.get(entry.query_id, [0.0, 0.0, 0.0, 0.0])
        rows.append(
            {
                "query_id": entry.query_id,
                "query_text": entry.query_text,
                "pred_x": float(bbox[0]),
                "pred_y": float(bbox[1]),
                "pred_w": float(bbox[2]),
                "pred_h": float(bbox[3]),
            }
        )
    output_df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output, index=False, encoding="utf-8")
    print(f"[inference] wrote submission to {args.output} ({len(output_df)} rows)")


if __name__ == "__main__":
    main()


import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from ultralytics import YOLO

from ..src.data.refiner_dataset import _features as refiner_anchor_features
from ..src.data.schema import AnnotationRecord, PageRecord
from ..src.features.ranker_features import extract_candidate_features
from ..src.pipeline.candidate_selector import class_category, gate_candidates
from ..src.pipeline.query_parser import parse_intent
from ..src.training.refiner_trainer import BoxRefiner

DOC_TYPE_TO_GROUP = {"보도자료": "press", "보고서": "report"}
READABLE_CLASS_NAMES = {
    "L01": "머리말",
    "L02": "꼬리말",
    "L03": "페이지번호",
    "T01": "제목",
    "T02": "소제목",
    "T03": "시각요소 제목",
    "V01": "표",
    "V02": "차트",
    "V03": "다이어그램",
}
TARGET_CATEGORIES = {"chart", "table", "diagram", "body"}


def normalize_class_id(class_id: str | None) -> str | None:
    if not class_id:
        return None
    if class_id.startswith("V02"):
        return "V02"
    return class_id


@dataclass
class QueryEntry:
    query_id: str
    query_text: str
    page_id: str
    target_class_id: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end inference pipeline (Detector → Ranker → Refiner).")
    parser.add_argument("--data-root", type=Path, default=Path("/workspace/data"))
    parser.add_argument("--query-dir", type=Path, default=None, help="Directory containing query JSON files.")
    parser.add_argument("--image-dir", type=Path, default=None, help="Directory containing page JPGs.")
    parser.add_argument("--detector-weights", type=Path, default=Path("outputs/yolo_models/doclayout_yolo/weights/best.pt"))
    parser.add_argument("--ranker-model", type=Path, default=Path("outputs/ranker/lightgbm.txt"))
    parser.add_argument("--refiner-model", type=Path, default=Path("outputs/refiner/refiner.pt"))
    parser.add_argument("--submission-template", type=Path, default=Path("data/sample_submission.csv"))
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", type=str, default=None, help="Device string for YOLO/torch (e.g., '0' or 'cpu').")
    parser.add_argument("--chunk-size", type=int, default=256, help="Number of pages to send per YOLO predict() call.")
    return parser.parse_args()


def load_query_index(query_dir: Path, image_dir: Path) -> tuple[Dict[str, PageRecord], Dict[str, QueryEntry]]:
    page_records: Dict[str, PageRecord] = {}
    query_index: Dict[str, QueryEntry] = {}

    json_paths = sorted(query_dir.glob("*.json"))
    for json_path in tqdm(json_paths, desc="[inference] load queries", leave=False):
        data = json.loads(json_path.read_text(encoding="utf-8"))
        page_id = Path(json_path).stem
        source_info = data.get("source_data_info", {})
        learning_info = data.get("learning_data_info", {})
        width, height = source_info.get("document_resolution", [2480, 3508])
        image_name = source_info.get("source_data_name_jpg")
        if not image_name:
            continue
        image_path = image_dir / image_name
        doc_type = data.get("raw_data_info", {}).get("doc_type", "")
        annotations = learning_info.get("annotation", [])
        if not annotations:
            continue

        page_record = PageRecord(
            json_path=json_path,
            image_path=image_path,
            width=int(width),
            height=int(height),
            annotations=[],  # populated after detector runs
            visual_context=learning_info.get("visual_context", ""),
            type_id=learning_info.get("type_id", ""),
            type_name=learning_info.get("type_name", ""),
            split="test",
            group=DOC_TYPE_TO_GROUP.get(doc_type, "press"),
        )
        page_records[page_id] = page_record

        for ann in annotations:
            query_id = ann.get("instance_id")
            if not query_id:
                continue
            query_text = ann.get("visual_instruction", "")
            query_class = normalize_class_id(ann.get("class_id"))
            query_index[query_id] = QueryEntry(
                query_id=query_id,
                query_text=query_text,
                page_id=page_id,
                target_class_id=query_class,
            )
    return page_records, query_index


def run_detector(
    model: YOLO,
    page_records: Dict[str, PageRecord],
    imgsz: int,
    batch: int,
    conf: float,
    device: str | None,
    chunk_size: int,
) -> None:
    if not page_records:
        return
    items = sorted(page_records.items(), key=lambda kv: kv[0])
    chunk_size = max(1, chunk_size)
    for start in tqdm(range(0, len(items), chunk_size), desc="[inference] detect", leave=False):
        chunk = items[start : start + chunk_size]
        sources = [str(record.image_path) for _, record in chunk]
        results = model.predict(
            source=sources,
            imgsz=imgsz,
            conf=conf,
            batch=batch,
            verbose=False,
            device=device,
        )
        for (page_id, record), result in zip(chunk, results):
            annotations = convert_detections_to_annotations(result, page_id, record.width, record.height)
            record.annotations = annotations


def convert_detections_to_annotations(result, page_id: str, width: int, height: int) -> List[AnnotationRecord]:
    annotations: List[AnnotationRecord] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.xywhn is None:
        return annotations

    xywhn = boxes.xywhn.cpu().numpy()
    cls_ids = boxes.cls.cpu().numpy().astype(int)
    confidences = boxes.conf.cpu().numpy()
    for idx, (coords, cls_id, conf) in enumerate(zip(xywhn, cls_ids, confidences)):
        cls_label = result.names.get(int(cls_id), f"class_{cls_id}")
        readable_name = READABLE_CLASS_NAMES.get(cls_label, cls_label)
        x_c, y_c, w_n, h_n = coords
        w = w_n * width
        h = h_n * height
        x = (x_c - w_n / 2) * width
        y = (y_c - h_n / 2) * height
        x = max(0.0, min(x, width - 1))
        y = max(0.0, min(y, height - 1))
        w = max(1.0, min(w, width - x))
        h = max(1.0, min(h, height - y))
        annotations.append(
            AnnotationRecord(
                class_id=cls_label,
                class_name=readable_name,
                bbox=[float(x), float(y), float(w), float(h)],
                instance_id=f"{page_id}_det_{idx}",
                visual_instruction=None,
                visual_answer=f"conf={conf:.3f}",
            )
        )
    return annotations


def load_ranker(path: Path) -> tuple[lgb.Booster, Sequence[str]]:
    booster = lgb.Booster(model_file=str(path))
    feature_names = booster.feature_name()
    return booster, feature_names


def load_refiner(path: Path | None, device: str | None) -> BoxRefiner | None:
    if path is None or not path.exists():
        return None
    model = BoxRefiner()
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    if device and device != "cpu":
        model.to(device)
    model.eval()
    return model


def select_candidate(
    record: PageRecord,
    query_entry: QueryEntry,
    ranker: lgb.Booster,
    feature_names: Sequence[str],
) -> AnnotationRecord | None:
    query_ann = AnnotationRecord(
        class_id="query",
        class_name="query",
        bbox=[0.0, 0.0, 0.0, 0.0],
        instance_id=query_entry.query_id,
        visual_instruction=query_entry.query_text,
        visual_answer=None,
    )
    candidates = [ann for ann in record.annotations if class_category(ann) in TARGET_CATEGORIES]
    expected_class = query_entry.target_class_id
    if expected_class:
        strict = [ann for ann in candidates if normalize_class_id(ann.class_id) == expected_class]
        if strict:
            candidates = strict
    if not candidates:
        return None
    gated = gate_candidates(record, query_entry.query_text or "", candidates)
    intent = parse_intent(query_entry.query_text or "")

    best_candidate = None
    best_score = float("-inf")
    for cand in gated:
        features = extract_candidate_features(record, query_ann, cand, intent=intent)
        prefixed = {f"feat_{k}": v for k, v in features.items()}
        vector = np.array([[prefixed.get(name, 0.0) for name in feature_names]], dtype=np.float32)
        if ranker.best_iteration and ranker.best_iteration > 0:
            score = float(ranker.predict(vector, num_iteration=ranker.best_iteration)[0])
        else:
            score = float(ranker.predict(vector)[0])
        if score > best_score:
            best_score = score
            best_candidate = cand
    return best_candidate


def refine_bbox(
    bbox: List[float],
    candidate: AnnotationRecord,
    record: PageRecord,
    refiner: BoxRefiner | None,
    device: str | None,
) -> List[float]:
    if refiner is None or class_category(candidate) not in {"chart", "table", "diagram"}:
        return bbox
    feats = refiner_anchor_features(candidate.bbox, record.width, record.height, class_category(candidate))
    tensor = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)
    if device and device != "cpu":
        tensor = tensor.to(device)
    with torch.no_grad():
        deltas = refiner(tensor).squeeze(0).cpu().numpy()
    ax, ay, aw, ah = candidate.bbox
    dx, dy, dw, dh = deltas
    aw = max(aw, 1e-3)
    ah = max(ah, 1e-3)
    x = ax + dx * aw
    y = ay + dy * ah
    w = aw * math.exp(dw)
    h = ah * math.exp(dh)
    x = max(0.0, min(x, record.width - 1))
    y = max(0.0, min(y, record.height - 1))
    w = max(1.0, min(w, record.width - x))
    h = max(1.0, min(h, record.height - y))
    return [float(x), float(y), float(w), float(h)]


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser()
    query_dir = args.query_dir or (data_root / "test" / "query")
    image_dir = args.image_dir or (data_root / "test" / "images")

    page_records, query_index = load_query_index(query_dir, image_dir)
    if not query_index:
        raise RuntimeError(f"No queries discovered under {query_dir}")

    detector = YOLO(str(args.detector_weights))
    run_detector(
        detector,
        page_records,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        device=args.device,
        chunk_size=args.chunk_size,
    )

    ranker, feature_names = load_ranker(args.ranker_model)
    refiner = load_refiner(args.refiner_model, args.device)

    if args.submission_template and args.submission_template.exists():
        template_df = pd.read_csv(args.submission_template)
        target_queries = []
        for _, row in template_df.iterrows():
            qid = row["query_id"]
            base_entry = query_index.get(qid)
            if base_entry is None:
                target_queries.append(
                    QueryEntry(
                        query_id=qid,
                        query_text=row.get("query_text", ""),
                        page_id="",
                        target_class_id=None,
                    )
                )
                continue
            query_text = row.get("query_text", base_entry.query_text)
            target_queries.append(
                QueryEntry(
                    query_id=qid,
                    query_text=query_text,
                    page_id=base_entry.page_id,
                    target_class_id=base_entry.target_class_id,
                )
            )
    else:
        target_queries = [query_index[qid] for qid in sorted(query_index.keys())]

    predictions: Dict[str, List[float]] = {}
    for entry in tqdm(target_queries, desc="[inference] rank+refine", leave=False):
        page_record = page_records.get(entry.page_id)
        if page_record is None:
            predictions[entry.query_id] = [0.0, 0.0, 0.0, 0.0]
            continue
        candidate = select_candidate(page_record, entry, ranker, feature_names)
        if candidate is None:
            predictions[entry.query_id] = [0.0, 0.0, 0.0, 0.0]
            continue
        refined = refine_bbox(candidate.bbox, candidate, page_record, refiner, args.device)
        predictions[entry.query_id] = refined

    rows = []
    for entry in target_queries:
        bbox = predictions.get(entry.query_id, [0.0, 0.0, 0.0, 0.0])
        rows.append(
            {
                "query_id": entry.query_id,
                "query_text": entry.query_text,
                "pred_x": int(round(bbox[0])),
                "pred_y": int(round(bbox[1])),
                "pred_w": int(round(bbox[2])),
                "pred_h": int(round(bbox[3])),
            }
        )
    output_df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output, index=False, encoding="utf-8")
    print(f"[inference] wrote submission to {args.output} ({len(output_df)} rows)")


if __name__ == "__main__":
    main()

