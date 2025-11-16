# how to run

FRACTION=1 bash train/run_subset.sh

# how to inference

python -m train.scripts.run_inference \
  --data-root /workspace/data \
  --detector-weights outputs/yolo_models/doclayout_yolo/weights/best.pt \
  --ranker-model outputs/ranker/lightgbm.txt \
  --refiner-model ""
  --submission-template /workspace/data/sample_submission.csv \
  --output submissions/doclayout_yolo_no_refiner.csv \
  --imgsz 1024 \
  --batch 16 \
  --conf 0.25 \
  --device 0 \
  --chunk-size 256
