"""
Stage 1: YOLO 문서 객체 탐지 모델 학습

YOLO를 사용한 문서 객체 탐지 및 분류 모델 학습
- YOLO detection 모델 학습
"""

import os
import argparse
import yaml
from typing import List

# YOLO (ultralytics)
try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False
    print("[WARN] ultralytics not available, please install: pip install ultralytics")


# ==================== 클래스 매핑 ====================

# 클래스 매핑 (class_id -> YOLO class index)
CLASS_MAPPING = {
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

CLASS_NAMES = [
    "C01", "C02", "I01", "L01", "L02", "L03",
    "T01", "T02", "T03",
    "V01", "V02-1", "V02-2", "V02-3", "V02-4", "V02-5", "V02-6", "V02-7", "V02-8", "V03"
]

NUM_CLASSES = len(CLASS_NAMES)

# 기본 설정
DEFAULT_SEED = 42
DEFAULT_YOLO_MODEL_DIR = "./outputs/yolo_models"


def train_yolo_segmentation(args):
    """YOLO 문서 분할 모델 학습"""
    if not _YOLO_AVAILABLE:
        raise RuntimeError("ultralytics not available. Please install: pip install ultralytics")
    
    # YOLO 로깅 레벨 조정
    # 학습 진행 상황은 verbose로 제어하고, 불필요한 경고만 필터링
    import logging
    from ultralytics.utils import LOGGER
    # INFO 레벨로 설정하여 학습 진행 상황은 볼 수 있도록 함
    LOGGER.setLevel(logging.INFO)
    
    # dataset.yaml 검증 및 클래스 수 확인
    if not os.path.exists(args.yolo_dataset_yaml):
        raise FileNotFoundError(f"Dataset YAML not found: {args.yolo_dataset_yaml}")
    
    with open(args.yolo_dataset_yaml, 'r', encoding='utf-8') as f:
        dataset_config = yaml.safe_load(f)
    
    
    print("[YOLO Train] Initializing YOLO model...")
    
    # 모델 로드 (verbose=False로 로드 시 출력 최소화)
    base_model = args.yolo_base_model  # 예: "yolo11m.pt", "yolo11l.pt"
    model = YOLO(base_model, verbose=False)
    
    # 핵심 수정: YOLO 헤드의 클래스 수를 19로 설정
    # YOLO v11은 학습 시 dataset.yaml의 nc 값을 읽어서 자동으로 헤드를 재구성합니다.
    # 모델 객체의 속성도 명시적으로 설정하여 일관성을 보장합니다.
    print(f"[YOLO Train] Configuring model for {NUM_CLASSES} classes...")
    model.model.nc = NUM_CLASSES
    model.model.names = CLASS_NAMES
    # 주의: model.train() 호출 시 dataset.yaml의 nc 값을 읽어서 헤드가 자동으로 재구성됩니다.
    
    print(f"[YOLO Train] Using base model: {base_model}")
    print(f"[YOLO Train] Number of classes: {NUM_CLASSES}")
    print(f"[YOLO Train] Dataset YAML: {args.yolo_dataset_yaml} (nc={dataset_config['nc']})")
    
    # 학습 파라미터
    train_params = {
        "data": args.yolo_dataset_yaml,
        "epochs": args.yolo_epochs,
        "imgsz": args.yolo_imgsz,
        "batch": args.yolo_batch,
        "device": args.yolo_device,
        "project": args.yolo_project,
        "name": args.yolo_name,
        "seed": args.seed if hasattr(args, 'seed') else DEFAULT_SEED,
        "optimizer": "AdamW",
        "lr0": 5e-5,
        "cos_lr": True,
        "weight_decay": 1e-4,
        "compile": True,
        # dropout은 YOLO에서 공식적으로 지원하지 않으므로 제거
        "verbose": not args.yolo_quiet,  # 학습 진행 상황 출력 (epoch, loss, metrics, ETA 등)
        "plots": False,  # 그래프 생성 비활성화 (학습 속도 향상),
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.1,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
        "mosaic": 0.7,      # 0~1, 필요하면 0.5 나중에 실험
        "mixup": 0.0,
        "copy_paste": 0.0,  
    }
    
    print(f"[YOLO Train] Training parameters:")
    for key, value in train_params.items():
        print(f"  {key}: {value}")
    
    # 학습 시작
    print(f"\n[YOLO Train] Starting training...")
    results = model.train(**train_params)
    
    # Best model 경로
    best_model_path = os.path.join(args.yolo_project, args.yolo_name, "weights", "best.pt")
    print(f"\n[YOLO Train] Training completed!")
    print(f"[YOLO Train] Best model saved at: {best_model_path}")
    
    return best_model_path


# ==================== Main ====================

def get_args():
    parser = argparse.ArgumentParser(description="Stage 1: YOLO Document Object Detection")
    
    parser.add_argument("--yolo_dataset_yaml", type=str, required=True,
                       help="Path to dataset.yaml file")
    parser.add_argument("--yolo_base_model", type=str, default="yolo11n.pt",
                       help="Base YOLO model (yolo11n.pt=lightest, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt)")
    parser.add_argument("--yolo_epochs", type=int, default=40)
    parser.add_argument("--yolo_imgsz", type=int, default=1024)
    parser.add_argument("--yolo_batch", type=int, default=16)
    parser.add_argument("--yolo_device", type=str, default="cuda",
                       help="Device to use: 'cuda', 'cpu', 'mps', or device index like '0'")
    parser.add_argument("--yolo_project", type=str, default=DEFAULT_YOLO_MODEL_DIR)
    parser.add_argument("--yolo_name", type=str, default="doc_detection")
    parser.add_argument("--yolo_quiet", action='store_true', default=False,
                       help="Disable training progress output (sets verbose=False). Default: verbose=True (shows progress)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    
    return parser.parse_args()


def main():
    args = get_args()
    train_yolo_segmentation(args)


if __name__ == "__main__":
    main()

