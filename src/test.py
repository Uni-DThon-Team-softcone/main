"""
Stage 3: YOLO + CLIP Inference Pipeline

YOLO로 test data 이미지에서 bbox 후보 검출 후,
CLIP으로 query와 유사도 계산하여 최종 bbox 예측
"""

import os
import json
import argparse
import csv
from glob import glob
from typing import List, Dict, Any, Optional
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

# YOLO
from ultralytics import YOLO

# CLIP (HuggingFace transformers 사용)
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModel

# utils 모듈에서 공통 함수 및 상수 import
from utils import (
    YOLO_CLASS_NAMES, YOLO_NAME2ID,
    normalize_class_id, clean_query_text,
    find_jsons, read_json, get_image_path
)

# model.py에서 CLIPMatchingModel import
from model import CLIPMatchingModel


# ==================== YOLO Detection ====================


# [width,height]
min_stat={
    "T01": [11.41,3.63],
    "C01": [3.35,1.36],
    "T03":[4.06,1.36],
    "V02-4":[3.99,2.70],
    "V01":[7.84,1.96],
    "L03":[5.89,0.84],
    "T02":[4.06,3.55],
    "L01":[9.79,4.06],
    "V02-1":[8.38,3.77],
    "L02":[16.09,2.93],
    "V02-8":[4.45,2.93],
    "C02":[4.45,3.07],
    "V02-3":[56.28,4.46],
    "V02-5":[81.76,65.98],
    "V02-2":[290.03,8.94],
    "V03":[19.28,3.21],
    "I01":[41.68,3.34],
    "V02-6":[473.74,384.38],
    "V02-7":[57.28,47.95],
    "V02":[3.99,2.70],
}

def yolo_detect(image: Image.Image, yolo_model, device: str):
    """YOLO로 bbox 후보 검출 및 logit 기반 가장 확률 높은 클래스 선택, min_stat로 필터링"""
    results = yolo_model(image, device=device, verbose=False, conf = 0.1, iou=0.6)
    detections = []

    for result in results:
        if result.boxes is None:
            continue
        # Make sure result.boxes contains tensor with .cls and .cls_logits
        # For each detected box, check all logit/probabilities and find the eligible class with highest probability within valid classes and size
        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy()    # (N, 4)
        classes = boxes.cls.cpu().numpy()  # (N,)
        confs = boxes.conf.cpu().numpy()   # (N,)

        # The YOLOv8 result provides result.probs (per box) only if you run with return_prob=True.
        # There may be result.boxes.conf (objectness), boxes.cls (class index), but not per-class probs by default.
        # To access class logits, use boxes.cls_logits or result.probs if possible.
        # For now, assume result.probs is available and is (N, num_classes)
        if hasattr(result, "probs") and result.probs is not None:
            probs = result.probs.cpu().numpy()  # (N, num_classes)
        else:
            # Fall back to single-class confidence (for each box's predicted class), use zeros for unavailable
            n_boxes = xyxy.shape[0]
            n_classes = len(getattr(result, "names", YOLO_CLASS_NAMES))
            probs = np.zeros((n_boxes, n_classes), dtype=np.float32)
            for i, c in enumerate(classes):
                probs[i, int(c)] = confs[i]

        for i in range(xyxy.shape[0]):
            x1, y1, x2, y2 = xyxy[i]
            width = x2 - x1
            height = y2 - y1

            # Prepare class-candidates: classes in YOLO_NAME2ID (plus V02-xx variants) and over min_stat
            best_cls_name = None
            best_cls_idx = None
            best_prob = -np.inf

            for cname, cidx in YOLO_NAME2ID.items():
                # V02-xx handled below
                min_wh = min_stat.get(cname)
                if min_wh is not None:
                    min_w, min_h = min_wh
                    if width < min_w or height < min_h:
                        continue  # skip this class if box too small

                if probs[i, cidx] > best_prob:
                    best_prob = probs[i, cidx]
                    best_cls_idx = cidx
                    best_cls_name = cname

            # Also check if any detected class_name startswith V02- and min_stat for V02 applies
            # Allow for box.cls to indicate V02-xx and use it as V02 if needed
            pred_cls = int(classes[i])
            pred_cls_name = result.names[pred_cls]
            # Treat V02-xx variants
            if pred_cls_name.startswith("V02-"):
                min_wh = min_stat.get("V02")
                if min_wh is not None:
                    min_w, min_h = min_wh
                    if width >= min_w and height >= min_h:
                        v02idx = YOLO_NAME2ID.get("V02", None)
                        if v02idx is not None and probs[i, v02idx] > best_prob:
                            best_prob = probs[i, v02idx]
                            best_cls_idx = v02idx
                            best_cls_name = "V02"

            # Only keep if a valid class with sufficient prob & over min_stat was found
            if best_cls_name is not None and best_prob > 0.0:
                detections.append({
                    "bbox": [float(x1), float(y1), float(width), float(height)],  # [x, y, w, h]
                    "cls": best_cls_idx,
                    "conf": float(best_prob),
                    "cls_name": best_cls_name,
                })

    return detections


# ==================== CLIP Similarity ====================

@torch.no_grad()
def compute_clip_similarity(query: str, crops: List[Image.Image],
                          clip_model: CLIPMatchingModel, processor, device: str):
    """CLIP으로 query와 image crops 간 유사도 계산 (HuggingFace 방식)"""
    if len(crops) == 0:
        return []

    # Text encoding
    text_inputs = processor(
        text=[query],
        padding="max_length",
        truncation=True,
        max_length=77,
        return_tensors="pt"
    )
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    txt_emb = clip_model.encode_text_tensor(
        text_inputs["input_ids"], 
        text_inputs["attention_mask"]
    )  # (1, D)

    # Image encoding
    image_inputs = processor(images=crops, return_tensors="pt")
    pixel_values = image_inputs["pixel_values"].to(device)
    img_emb = clip_model.encode_image_tensor(pixel_values)  # (N, D)

    # Similarity 계산
    logit_scale = clip_model.logit_scale.exp()
    sims = (logit_scale * (img_emb @ txt_emb.T).squeeze(1)).cpu().numpy()

    return sims.tolist()


# ==================== Inference Pipeline ====================

def load_clip_from_ckpt(ckpt_path: str, device: str = "cuda"):
    """CLIP 체크포인트 로드"""
    print(f"[CLIP] Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    clip_model_name = ckpt.get("clip_model", "koclip/koclip-base-pt")
    
    model = CLIPMatchingModel(model_name=clip_model_name, device=device)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    
    processor = AutoProcessor.from_pretrained(clip_model_name)
    print(f"[CLIP] Checkpoint loaded successfully")
    return model, processor


def process_single_file(json_path: str, jpg_dir: str,
                        yolo_model, clip_model: CLIPMatchingModel, processor, device: str):
    """단일 파일 처리"""
    data = read_json(json_path)
    img_path = get_image_path(json_path, data, jpg_dir)

    # 이미지 로드
    image = Image.open(img_path).convert("RGB")

    # Step 1: YOLO로 bbox 후보 검출
    detections = yolo_detect(image, yolo_model, device)

    if len(detections) == 0:
        print(f"[WARN] No detections found in {json_path}")
        return []

    # Step 2: Query 처리
    annotations = data.get("learning_data_info", {}).get("annotation", [])
    results = []

    for ann in annotations:
        query = ann.get("visual_instruction", "").strip()
        if not query:
            continue

        instance_id = ann.get("instance_id", "")
        class_id_raw = ann.get("class_id", "")
        # visual_answer 가져오기 (test 데이터에도 있을 수 있음)
        visual_answer = ann.get("visual_answer", "").strip() if ann.get("visual_answer") else None

        # class_id 정규화
        class_id_norm = normalize_class_id(class_id_raw)
        if class_id_norm is None:
            continue  # C01/C02/I01 등은 제외

        # 같은 class의 bbox만 필터링
        candidate_boxes = [
            det for det in detections
            if det["cls_name"] == class_id_norm
        ]

        if len(candidate_boxes) == 0:
            # 같은 class가 없으면 모든 후보 사용
            candidate_boxes = detections
            print(f"[WARN] No {class_id_norm} boxes found for {instance_id}, using all candidates")

        # Step 3: 각 bbox crop
        crops = []
        for det in candidate_boxes:
            x, y, w, h = det["bbox"]
            x1 = int(max(0, x))
            y1 = int(max(0, y))
            x2 = int(min(image.width, x + w))
            y2 = int(min(image.height, y + h))
            crop = image.crop((x1, y1, x2, y2))
            crops.append(crop)

        # Step 4: Query text 정리 (visual_answer 포함)
        query_clean = clean_query_text(query, class_id_norm, visual_answer)

        # Step 5: CLIP similarity 계산
        sims = compute_clip_similarity(query_clean, crops, clip_model, processor, device)
        sims = np.array(sims, dtype=np.float32)
        confs = np.array([det["conf"] for det in candidate_boxes], dtype=np.float32)

        
        


        # 최종 선택
        best_idx = int(sims.argmax())
        best_score = float(sims[best_idx])
        best_bbox = candidate_boxes[best_idx]["bbox"]

        results.append({
            "query_id": instance_id,  # query_id로 저장 (baseline.py와 동일)
            "query_text": query,  # 원본 query text 저장
            "bbox": best_bbox,  # [x, y, w, h]
            "score": best_score,
            "class_id": class_id_norm,
            "query_clean": query_clean  # CLIP에 사용된 정리된 query (참고용)
        })

    return results


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(
        description="Stage 3: YOLO + CLIP Inference Pipeline"
    )
    parser.add_argument("--test_json_dir", type=str, default="data/test/query",
                       help="Path to test JSON directory")
    parser.add_argument("--test_jpg_dir", type=str, default="data/test/images",
                       help="Path to test JPG directory")

    parser.add_argument("--yolo_model_path", type=str, required=True,
                       help="Path to YOLO model checkpoint (best.pt)")
    parser.add_argument("--clip_ckpt", type=str, default="clip_matching/best_clip_cached.pth",
                       help="Path to CLIP checkpoint (best_clip.pth)")
                       
    parser.add_argument("--out_csv", type=str, required=True,
                       help="Output CSV file path")
    parser.add_argument("--sample_submission", type=str, default="sample_submission.csv",
                       help="Path to sample_submission.csv template")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use: 'cuda', 'cpu'")

    args = parser.parse_args()

    # 디렉토리 생성
    os.makedirs(os.path.dirname(args.out_csv) if os.path.dirname(args.out_csv) else ".", exist_ok=True)

    # 모델 로드
    print(f"[LOAD] YOLO model from: {args.yolo_model_path}")
    yolo_model = YOLO(args.yolo_model_path)

    print(f"[LOAD] CLIP model...")
    clip_model, processor = load_clip_from_ckpt(args.clip_ckpt, args.device)

    # JSON 파일 찾기
    json_files = find_jsons(args.test_json_dir)
    print(f"[INFO] Found {len(json_files)} JSON files")

    # 모든 결과 수집
    all_results = []

    for json_path in tqdm(json_files, desc="Processing"):
        try:
            results = process_single_file(
                json_path, args.test_jpg_dir,
                yolo_model, clip_model, processor, args.device
            )
            all_results.extend(results)
        except Exception as e:
            print(f"[ERROR] Failed to process {json_path}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # sample_submission.csv를 템플릿으로 사용하여 결과 생성
    print(f"[LOAD] Loading sample_submission.csv template from {args.sample_submission}")
    if not os.path.exists(args.sample_submission):
        print(f"[WARN] sample_submission.csv not found at {args.sample_submission}, using generated results directly")
        # Fallback: 기존 방식으로 저장
        with open(args.out_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["query_id", "query_text", "pred_x", "pred_y", "pred_w", "pred_h"])
            for result in all_results:
                query_id = result["query_id"]
                query_text = result["query_text"]
                x, y, w, h = result["bbox"]
                writer.writerow([query_id, query_text, x, y, w, h])
        print(f"[DONE] Saved {len(all_results)} predictions to {args.out_csv}")
        return

    # 현재 결과를 딕셔너리로 변환 (query_id를 키로)
    results_dict = {}
    for result in all_results:
        query_id = result["query_id"]
        results_dict[query_id] = result

    # sample_submission.csv 읽기 및 매칭
    print(f"[MATCH] Matching results with sample_submission.csv...")
    matched_count = 0
    unmatched_count = 0
    
    with open(args.sample_submission, "r", encoding="utf-8-sig") as template_f:
        reader = csv.DictReader(template_f)
        template_rows = list(reader)
    
    print(f"[INFO] Template has {len(template_rows)} rows")
    print(f"[INFO] Generated {len(all_results)} results")

    # query_id 기준으로 정렬: TY1이 먼저, TY2가 나중에 오도록
    def sort_key(row):
        query_id = row["query_id"]
        # TY1이면 0, TY2이면 1로 매핑하여 TY1이 먼저 오도록
        if "TY1" in query_id:
            ty_priority = 0
        elif "TY2" in query_id:
            ty_priority = 1
        else:
            ty_priority = 2  # 다른 경우는 마지막에
        # (TY 우선순위, query_id 전체)로 정렬
        return (ty_priority, query_id)
    
    template_rows = sorted(template_rows, key=sort_key)
    print(f"[SORT] Sorted template rows: TY1 first, then TY2")
    
    # 검증: template_rows의 개수가 sample_submission.csv와 일치하는지 확인
    print(f"[VERIFY] Template rows after sorting: {len(template_rows)} (should match sample_submission.csv)")

    # CSV 저장 (정렬된 순서대로)
    print(f"[SAVE] Saving results to {args.out_csv}")
    with open(args.out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "query_text", "pred_x", "pred_y", "pred_w", "pred_h"])

        for template_row in template_rows:
            query_id = template_row["query_id"]
            query_text = template_row["query_text"]
            
            # 현재 결과에서 매칭되는 것 찾기
            if query_id in results_dict:
                result = results_dict[query_id]
                x, y, w, h = result["bbox"]
                matched_count += 1
            else:
                # 매칭되지 않으면 (0,0,0,0)으로 채움
                x, y, w, h = 0, 0, 0, 0
                unmatched_count += 1
                print(f"[WARN] No result found for query_id: {query_id}")
            
            writer.writerow([query_id, query_text, x, y, w, h])

    print(f"[DONE] Saved {len(template_rows)} predictions to {args.out_csv}")
    print(f"[STATS] Matched: {matched_count}, Unmatched: {unmatched_count}")


if __name__ == "__main__":
    main()

