# 1. Preprocess
python src/preprocess.py --preprocess_json

# 2. Train YOLO
python src/yolo_train.py --yolo_dataset_yaml outputs/yolo_datasets/dataset.yaml --yolo_base_model yolo11n.pt --yolo_epochs 40 --yolo_imgsz 1024 --yolo_batch 16 --yolo_device cuda --yolo_project outputs/yolo_models --yolo_name doc_detection --yolo_quiet --seed 42

# 3. Prepare CLIP
python prepare_clip_cache.py \
  --json_dir data/train/pre_json \
  --jpg_dir  data/train/jpg \
  --split    train \
  --cache_dir ./clip_cache \
  --clip_model koclip/koclip-base-pt

python prepare_clip_cache.py \
  --json_dir data/valid/pre_json \
  --jpg_dir  data/valid/jpg \
  --split    val \
  --cache_dir ./clip_cache \
  --clip_model koclip/koclip-base-pt

# 4. Train CLIP
python clip_train_cached.py \
  --cache_root ./clip_cache \
  --train_split train \
  --val_split val \
  --batch_size 256 \
  --num_workers 8 \
  --device cuda \
  --use_amp \
  --clip_model koclip/koclip-base-pt \
  --save_ckpt outputs/clip_matching/best_clip_cached.pth

# 5. Predict
python predict.py --yolo_model_path outputs/yolo_models/doc_detection/weights/best.pt --out_csv result_final.csv --clip_ckpt outputs/clip_matching/best_clip_cached_answer.pth --sample_submission data/sample_submission.csv