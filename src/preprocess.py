import os
import json
import argparse
import shutil
import yaml
from typing import List, Optional

from utils import find_jsons, read_json, get_image_path

# ==================== YOLO 학습용 데이터 변환 ====================

# 클래스 매핑 (class_id -> YOLO class index)
# 원본 데이터용 (전처리 전)
CLASS_MAPPING_ORIGINAL = {
    "C01": 0,   # 본문
    "C02": 1,   # 목록
    "I01": 2,   # 발행정보
    "L01": 3,   # 머리말
    "L02": 4,   # 꼬리말
    "L03": 5,   # 페이지번호
    "T01": 6,   # 제목
    "T02": 7,   # 소제목
    "T03": 8,   # 시각요소 제목
    "V01": 9,   # 표
    "V02-1": 10,  # 차트(세로 막대형)
    "V02-2": 11,  # 차트(가로 막대형)
    "V02-3": 12,  # 차트(원형)
    "V02-4": 13,  # 차트(꺾은선형)
    "V02-5": 14,  # 차트(영역형)
    "V02-6": 15,  # 차트(분산형)
    "V02-7": 16,  # 차트(방사형)
    "V02-8": 17,  # 차트(혼합형)
    "V03": 18,  # 다이어그램
}

CLASS_NAMES_ORIGINAL = [
    "C01", "C02", "I01", "L01", "L02", "L03",
    "T01", "T02", "T03",
    "V01", "V02-1", "V02-2", "V02-3", "V02-4", "V02-5", "V02-6", "V02-7", "V02-8", "V03"
]

# 전처리된 데이터용 (C01, C02, I01 제거, V02-n -> V02 통합)
CLASS_MAPPING_PREPROCESSED = {
    "L01": 0,   # 머리말
    "L02": 1,   # 꼬리말
    "L03": 2,   # 페이지번호
    "T01": 3,   # 제목
    "T02": 4,   # 소제목
    "T03": 5,   # 시각요소 제목
    "V01": 6,   # 표
    "V02": 7,   # 차트 (V02-1~V02-8 통합)
    "V03": 8,   # 다이어그램
}

CLASS_NAMES_PREPROCESSED = [
    "L01", "L02", "L03",
    "T01", "T02", "T03",
    "V01", "V02", "V03"
]


def convert_bbox_to_yolo(bbox: List[float], img_width: int, img_height: int) -> Optional[List[float]]:
    """
    COCO 형식 (x, y, w, h) 좌상단 기준을 YOLO 형식 (x_center, y_center, w, h) normalized로 변환
    Returns None if bbox is invalid (out of bounds, negative, etc.)
    """
    x, y, w, h = bbox
    
    # 유효성 검사: 음수 또는 잘못된 값 체크
    if w <= 0 or h <= 0 or img_width <= 0 or img_height <= 0:
        return None
    
    # 좌표를 이미지 범위 내로 제한
    x = max(0, min(x, img_width))
    y = max(0, min(y, img_height))
    w = min(w, img_width - x)
    h = min(h, img_height - y)
    
    # 다시 유효성 검사
    if w <= 0 or h <= 0:
        return None
    
    # 중심점 계산
    x_center = (x + w / 2.0) / img_width
    y_center = (y + h / 2.0) / img_height
    
    # 너비, 높이 정규화
    w_norm = w / img_width
    h_norm = h / img_height
    
    # YOLO 형식: 0~1 범위를 벗어나면 None 반환 (corrupt 방지)
    if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0 and 
            0.0 < w_norm <= 1.0 and 0.0 < h_norm <= 1.0):
        return None
    
    return [x_center, y_center, w_norm, h_norm]


def convert_json_to_yolo_format(
    json_files: List[str],
    jpg_dir: str,
    output_dir: str,
    split: str = "train"
):
    """
    전처리된 JSON annotation을 YOLO 형식으로 변환
    (전처리된 데이터만 사용: C01/C02/I01 제거, V02-n -> V02)
    """
    # 전처리된 데이터용 클래스 매핑 사용
    class_mapping = CLASS_MAPPING_PREPROCESSED
    class_names = CLASS_NAMES_PREPROCESSED
    images_dir = os.path.join(output_dir, "images", split)
    labels_dir = os.path.join(output_dir, "labels", split)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    stats = {cls_id: 0 for cls_id in class_names}
    total_images = 0
    total_annotations = 0
    
    # 이미 복사된 이미지 추적 (중복 복사 방지)
    copied_images = set()
    
    print(f"[YOLO Dataset] Converting {split} data...")
    print(f"  Total files: {len(json_files)}")
    
    for json_idx, json_path in enumerate(json_files):
        try:
            # JSON 파일 읽기 (빈 파일 스킵)
            try:
                data = read_json(json_path)
            except (ValueError, json.JSONDecodeError) as e:
                # 빈 파일이나 잘못된 JSON은 스킵 (로그는 너무 많을 수 있으므로 조용히 스킵)
                continue
            
            # 이미지 경로 찾기
            try:
                img_path = get_image_path(json_path, data, jpg_dir=jpg_dir)
            except FileNotFoundError:
                continue
            
            if not os.path.exists(img_path):
                continue
            
            # 이미지 정보
            source_info = data.get("source_data_info", {})
            # 전처리된 JSON에는 document_resolution이 없을 수 있으므로 기본값 사용
            resolution = source_info.get("document_resolution", [2480, 3508])
            img_width, img_height = resolution[0], resolution[1]
            
            # 라벨 파일 먼저 생성 (annotation이 없으면 이미지도 복사하지 않음)
            annotations = data.get("learning_data_info", {}).get("annotation", [])
            yolo_labels = []
            
            for ann in annotations:
                class_id = ann.get("class_id", "")
                bbox = ann.get("bounding_box")
                
                if not class_id or bbox is None or len(bbox) != 4:
                    continue
                
                # 클래스 매핑
                if class_id not in class_mapping:
                    continue
                
                # YOLO 형식으로 변환 (유효성 검사 포함)
                yolo_bbox = convert_bbox_to_yolo(bbox, img_width, img_height)
                if yolo_bbox is None:
                    # 유효하지 않은 bbox는 스킵 (corrupt 방지)
                    continue
                
                yolo_class_idx = class_mapping[class_id]
                stats[class_id] += 1
                
                # YOLO 형식: class_id x_center y_center width height
                yolo_labels.append(f"{yolo_class_idx} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n")
            
            # annotation이 없거나 유효한 라벨이 없으면 스킵 (이미지도 복사하지 않음)
            if not yolo_labels:
                continue
            
            # 이미지 복사 (중복 체크 최적화)
            img_name = os.path.basename(img_path)
            dest_img_path = os.path.join(images_dir, img_name)
            label_name = os.path.splitext(img_name)[0] + ".txt"
            label_path = os.path.join(labels_dir, label_name)
            
            # 이미 복사된 이미지는 스킵
            if img_name not in copied_images:
                if not os.path.exists(dest_img_path):
                    try:
                        # 같은 파일시스템이면 하드링크 시도 (더 빠름)
                        try:
                            os.link(img_path, dest_img_path)
                        except (OSError, AttributeError):
                            # 하드링크 실패시 일반 복사
                            shutil.copy2(img_path, dest_img_path)
                    except Exception as e:
                        print(f"[WARN] Failed to copy {img_path}: {e}")
                        continue
                copied_images.add(img_name)
            
            # 라벨 파일 저장 (유효한 라벨이 있는 경우만)
            with open(label_path, "w") as f:
                f.writelines(yolo_labels)
            total_images += 1
            total_annotations += len(yolo_labels)
        
        except Exception as e:
            print(f"[WARN] Error processing {json_path}: {e}")
            continue
        
        # 더 자주 진행 상황 출력 (100개마다 또는 1%마다)
        if (json_idx + 1) % max(100, len(json_files) // 100) == 0:
            progress = (json_idx + 1) / len(json_files) * 100
            print(f"  Processed {json_idx + 1}/{len(json_files)} files ({progress:.1f}%)...")
    
    print(f"[YOLO Dataset] {split}: {total_images} images, {total_annotations} annotations")
    print(f"[YOLO Dataset] Class distribution:")
    for cls_id, count in sorted(stats.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"  {cls_id}: {count}")
    
    return total_images, total_annotations


def create_yolo_dataset_yaml(output_dir: str):
    """YOLO dataset.yaml 파일 생성 (전처리된 데이터용)"""
    # 전처리된 데이터용 클래스 매핑 사용
    class_names = CLASS_NAMES_PREPROCESSED
    num_classes = len(CLASS_NAMES_PREPROCESSED)
    
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    
    dataset_config = {
        "path": os.path.abspath(output_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": num_classes,
        "names": {str(i): name for i, name in enumerate(class_names)}
    }
    
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(dataset_config, f, allow_unicode=True, default_flow_style=False)
    
    print(f"[YOLO Dataset] Created dataset.yaml at {yaml_path}")
    print(f"[YOLO Dataset] Number of classes: {num_classes}")
    print(f"[YOLO Dataset] Class names: {class_names}")
    return yaml_path


def prepare_yolo_dataset(args):
    """YOLO 학습용 데이터셋 준비 (전처리된 JSON 사용)"""
    output_dir = args.yolo_dataset_dir
    
    print("[YOLO Dataset] Preparing dataset from preprocessed JSON files...")
    print("[YOLO Dataset] Using preprocessed JSON files (C01/C02/I01 removed, V02-n -> V02)")
    
    # Train 데이터 변환
    train_json_files = find_jsons(args.train_json_dir)
    if len(train_json_files) == 0:
        print(f"[WARN] No JSON files found in {args.train_json_dir}")
        train_images, train_anns = 0, 0
    else:
        train_images, train_anns = convert_json_to_yolo_format(
            train_json_files, args.train_jpg_dir, output_dir, split="train"
        )
    
    # Valid 데이터 변환
    valid_json_files = find_jsons(args.valid_json_dir)
    if len(valid_json_files) == 0:
        print(f"[WARN] No JSON files found in {args.valid_json_dir}")
        valid_images, valid_anns = 0, 0
    else:
        valid_images, valid_anns = convert_json_to_yolo_format(
            valid_json_files, args.valid_jpg_dir, output_dir, split="val"
        )
    
    # dataset.yaml 생성
    yaml_path = create_yolo_dataset_yaml(output_dir)
    
    print(f"\n[YOLO Dataset] Dataset preparation completed!")
    print(f"  Train: {train_images} images, {train_anns} annotations")
    print(f"  Valid: {valid_images} images, {valid_anns} annotations")
    print(f"  YAML: {yaml_path}")
    
    return yaml_path

def preprocess_json_files(json_dir: str, output_dir: str = "pre_json"):
    """
    JSON 파일 전처리:
    - source_data_name_jpg와 annotation만 남기고 나머지 제거
    - C01, C02, I01 클래스의 instance는 drop
    - V02-n은 모두 V02로 변경
    """
    os.makedirs(output_dir, exist_ok=True)
    
    json_files = find_jsons(json_dir)
    print(f"[JSON Preprocessing] Processing {len(json_files)} files from {json_dir}...")
    
    processed_count = 0
    skipped_count = 0
    
    for json_path in json_files:
        try:
            # JSON 파일 읽기
            data = read_json(json_path)
            
            # source_data_name_jpg 추출
            source_data_name_jpg = data.get("source_data_info", {}).get("source_data_name_jpg", None)
            if source_data_name_jpg is None:
                print(f"[WARN] No source_data_name_jpg found in {json_path}, skipping...")
                skipped_count += 1
                continue
            
            # annotation 추출 및 전처리
            annotations = data.get("learning_data_info", {}).get("annotation", [])
            
            # 전처리된 annotation 리스트
            processed_annotations = []
            
            for ann in annotations:
                class_id = ann.get("class_id", "")
                
                # C01, C02, I01 클래스는 drop
                if class_id in ["C01", "C02", "I01"]:
                    continue
                
                # V02-n을 V02로 변경
                if class_id.startswith("V02-"):
                    ann = ann.copy()  # 원본 수정 방지
                    ann["class_id"] = "V02"
                    # class_name도 업데이트 (있는 경우)
                    if "class_name" in ann:
                        ann["class_name"] = "차트"
                
                processed_annotations.append(ann)
            
            # 새로운 JSON 구조 생성
            processed_data = {
                "source_data_info": {
                    "source_data_name_jpg": source_data_name_jpg
                },
                "learning_data_info": {
                    "annotation": processed_annotations
                }
            }
            
            # 출력 파일 경로 생성
            json_filename = os.path.basename(json_path)
            output_path = os.path.join(output_dir, json_filename)
            
            # JSON 파일 저장
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=4)
            
            processed_count += 1
            
        except Exception as e:
            print(f"[WARN] Error processing {json_path}: {e}")
            skipped_count += 1
            continue
    
    print(f"[JSON Preprocessing] Completed!")
    print(f"  Processed: {processed_count} files")
    print(f"  Skipped: {skipped_count} files")
    print(f"  Output directory: {output_dir}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_json_dir", type=str, default="data/train/json")
    parser.add_argument("--train_jpg_dir", type=str, default="data/train/jpg")
    parser.add_argument("--valid_json_dir", type=str, default="data/valid/json")
    parser.add_argument("--valid_jpg_dir", type=str, default="data/valid/jpg")
    parser.add_argument("--yolo_dataset_dir", type=str, default="./outputs/yolo_datasets")
    parser.add_argument("--preprocess_json", action="store_true", help="Run JSON preprocessing for both train and valid")
    return parser.parse_args()

if __name__ == '__main__':
   args = get_args()
   
   # pre_json 폴더 경로 확인
   train_pre_json_dir = os.path.join(os.path.dirname(args.train_json_dir), "pre_json")
   valid_pre_json_dir = os.path.join(os.path.dirname(args.valid_json_dir), "pre_json")
   
   train_pre_json_exists = os.path.exists(train_pre_json_dir) and os.path.isdir(train_pre_json_dir)
   valid_pre_json_exists = os.path.exists(valid_pre_json_dir) and os.path.isdir(valid_pre_json_dir)
   
   # pre_json이 이미 존재하는지 확인
   if train_pre_json_exists and valid_pre_json_exists:
       # pre_json 폴더에 JSON 파일이 있는지 확인
       train_pre_json_files = find_jsons(train_pre_json_dir) if train_pre_json_exists else []
       valid_pre_json_files = find_jsons(valid_pre_json_dir) if valid_pre_json_exists else []
       
       if len(train_pre_json_files) > 0 and len(valid_pre_json_files) > 0:
           print(f"[INFO] Preprocessed JSON files found in:")
           print(f"  Train: {train_pre_json_dir} ({len(train_pre_json_files)} files)")
           print(f"  Valid: {valid_pre_json_dir} ({len(valid_pre_json_files)} files)")
           print(f"[INFO] Skipping preprocessing, proceeding with YOLO dataset construction...")
           
           # 전처리된 JSON 사용하여 YOLO 데이터셋 생성
           args.train_json_dir = train_pre_json_dir
           args.valid_json_dir = valid_pre_json_dir
           prepare_yolo_dataset(args)
       else:
           # pre_json 폴더는 있지만 파일이 없으면 전처리 진행
           if args.preprocess_json:
               print(f"[INFO] Pre_json directories exist but are empty. Running preprocessing...")
               # 전처리 진행
               train_output_dir = train_pre_json_dir
               valid_output_dir = valid_pre_json_dir
               preprocess_json_files(args.train_json_dir, train_output_dir)
               preprocess_json_files(args.valid_json_dir, valid_output_dir)
               print(f"\n[JSON Preprocessing] All preprocessing completed!")
               
               # 전처리 후 YOLO 데이터셋 생성
               args.train_json_dir = train_output_dir
               args.valid_json_dir = valid_output_dir
               prepare_yolo_dataset(args)
           else:
               print(f"[WARN] Pre_json directories exist but are empty.")
               print(f"[WARN] Run with --preprocess_json to preprocess JSON files.")
   else:
       # pre_json이 없으면 전처리 필요
       if args.preprocess_json:
           # train과 valid 모두 전처리
           # train: data/train/json -> data/train/pre_json
           train_output_dir = os.path.join(os.path.dirname(args.train_json_dir), "pre_json")
           print(f"[JSON Preprocessing] Processing train data...")
           print(f"  Input: {args.train_json_dir}")
           print(f"  Output: {train_output_dir}")
           preprocess_json_files(args.train_json_dir, train_output_dir)
           
           # valid: data/valid/json -> data/valid/pre_json
           valid_output_dir = os.path.join(os.path.dirname(args.valid_json_dir), "pre_json")
           print(f"\n[JSON Preprocessing] Processing valid data...")
           print(f"  Input: {args.valid_json_dir}")
           print(f"  Output: {valid_output_dir}")
           preprocess_json_files(args.valid_json_dir, valid_output_dir)
           
           print(f"\n[JSON Preprocessing] All preprocessing completed!")
           
           # 전처리 후 YOLO 데이터셋 생성
           args.train_json_dir = train_output_dir
           args.valid_json_dir = valid_output_dir
           prepare_yolo_dataset(args)
       else:
           # pre_json이 없고 --preprocess_json도 없으면 에러
           print(f"[ERROR] Preprocessed JSON files not found.")
           print(f"[ERROR] Please run with --preprocess_json flag to preprocess JSON files first.")
           print(f"[ERROR] Expected directories:")
           print(f"  Train: {train_pre_json_dir}")
           print(f"  Valid: {valid_pre_json_dir}") 