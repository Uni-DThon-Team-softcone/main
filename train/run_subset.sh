#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   FRACTION=0.2 bash train/run_subset.sh
# Environment overrides:
#   DATA_ROOT   - defaults to ~/data
#   PREP_DIR    - defaults to <repo>/train/preprocessed
#   FRACTION    - fraction of JSON pages per split/group to use (e.g., 0.2). If unset, falls back to DEFAULT_LIMIT.
#   DEFAULT_LIMIT - number of JSON pages per group when FRACTION is unset (default 200).
#   DEVICE      - defaults to cuda if available.

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"

DATA_ROOT="${DATA_ROOT:-/workspace/data}"
PREP_DIR="${PREP_DIR:-${REPO_ROOT}/train/preprocessed}"
LOG_DIR="${REPO_ROOT}/outputs/logs"
mkdir -p "${LOG_DIR}"

if [ ! -d "${DATA_ROOT}" ]; then
  echo "DATA_ROOT='${DATA_ROOT}' is missing." >&2
  exit 1
fi

source "/workspace/.venv/bin/activate"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

calc_group_limit() {
  local split=$1
  local group=$2
  python - "$DATA_ROOT" "$split" "$group" "$FRACTION" <<'PY'
import sys, math, json
from pathlib import Path
DOC_TYPE_TO_GROUP = {"보도자료": "press", "보고서": "report"}
data_root, split, group, frac = sys.argv[1:]
frac = float(frac)
json_dir = Path(data_root) / split / "json"
total = 0
for json_path in json_dir.glob("*.json"):
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        continue
    doc_type = (data.get("raw_data_info", {}).get("doc_type") or "").strip()
    mapped = DOC_TYPE_TO_GROUP.get(doc_type, "press")
    if mapped != group:
        continue
    total += 1
limit = max(1, math.ceil(total * frac))
print(limit)
PY
}

if [ -n "${FRACTION:-}" ]; then
  echo "[prep] fraction mode enabled (FRACTION=${FRACTION})"
  declare -A LIMITS
  for split in train valid; do
    for group in press report; do
      LIMITS["${split}_${group}"]=$(calc_group_limit "${split}" "${group}")
      echo "  -> ${split}/${group}: ${LIMITS["${split}_${group}"]} pages"
      python "${REPO_ROOT}/prep.py" \
        --data-root "${DATA_ROOT}" \
        --output-dir "${PREP_DIR}" \
        --splits "${split}" \
        --groups "${group}" \
        --limit "${LIMITS["${split}_${group}"]}" \
        > "${LOG_DIR}/prep_${split}_${group}.log" 2>&1
    done
  done
  TRAIN_RECORD_LIMIT=$((LIMITS["train_press"] + LIMITS["train_report"]))
  VALID_RECORD_LIMIT=$((LIMITS["valid_press"] + LIMITS["valid_report"]))
  RANKER_PREP_FLAG=(--skip-prep)
  REFINER_PREP_FLAG=(--skip-prep)
  PREP_LIMIT_VALUE=0
else
  DEFAULT_LIMIT="${DEFAULT_LIMIT:-200}"
  echo "[prep] limiting to ~${DEFAULT_LIMIT} JSONs per split/group" | tee "${LOG_DIR}/prep.log"
  python "${REPO_ROOT}/prep.py" \
    --data-root "${DATA_ROOT}" \
    --output-dir "${PREP_DIR}" \
    --limit "${DEFAULT_LIMIT}" \
    2>&1 | tee -a "${LOG_DIR}/prep.log"
  TRAIN_RECORD_LIMIT=$((DEFAULT_LIMIT * 2))
  VALID_RECORD_LIMIT=$((DEFAULT_LIMIT * 2))
  RANKER_PREP_FLAG=(--prep-limit "${DEFAULT_LIMIT}")
  REFINER_PREP_FLAG=(--prep-limit "${DEFAULT_LIMIT}")
fi

DEVICE="${DEVICE:-cuda}"
if ! python - <<'PY' >/dev/null 2>&1; then
import torch, os
device = os.environ.get("DEVICE", "cuda")
if device == "cuda" and not torch.cuda.is_available():
    raise SystemExit(1)
PY
  echo "CUDA not available, falling back to cpu."
  DEVICE="cpu"
fi

YOLO_DATASET_DIR="${REPO_ROOT}/outputs/yolo_datasets"
echo "[detector] preparing YOLO dataset @ ${YOLO_DATASET_DIR}" | tee "${LOG_DIR}/detector.log"
python "${REPO_ROOT}/../preprocess.py" \
  --train_json_dir "${DATA_ROOT}/train/json" \
  --train_jpg_dir "${DATA_ROOT}/train/jpg" \
  --valid_json_dir "${DATA_ROOT}/valid/json" \
  --valid_jpg_dir "${DATA_ROOT}/valid/jpg" \
  --yolo_dataset_dir "${YOLO_DATASET_DIR}" \
  2>&1 | tee -a "${LOG_DIR}/detector.log"

DETECTOR_ARGS=(
  --config "${REPO_ROOT}/train/configs/detector.yaml"
  --epochs 40
  --data-yaml "${YOLO_DATASET_DIR}/dataset.yaml"
)
if [ -n "${FRACTION:-}" ]; then
  DETECTOR_ARGS+=(--fraction "${FRACTION}")
fi
python -m train.scripts.train_detector \
  "${DETECTOR_ARGS[@]}" \
  2>&1 | tee -a "${LOG_DIR}/detector.log"

echo "[ranker] LightGBM LambdaMART on preprocessed samples" | tee "${LOG_DIR}/ranker.log"
python -m train.scripts.train_ranker \
  --config "${REPO_ROOT}/train/configs/ranker.yaml" \
  --max-train "${TRAIN_RECORD_LIMIT}" \
  --max-valid "${VALID_RECORD_LIMIT}" \
  --max-queries 4 \
  "${RANKER_PREP_FLAG[@]}" \
  2>&1 | tee -a "${LOG_DIR}/ranker.log"

echo "[refiner] box-regression head from query samples" | tee "${LOG_DIR}/refiner.log"
python -m train.scripts.train_refiner \
  --config "${REPO_ROOT}/train/configs/refiner.yaml" \
  --max-samples 200 \
  --device "${DEVICE}" \
  "${REFINER_PREP_FLAG[@]}" \
  2>&1 | tee -a "${LOG_DIR}/refiner.log"

echo "All stages completed. Logs stored in ${LOG_DIR}"
