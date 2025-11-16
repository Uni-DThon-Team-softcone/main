# Training Pipeline

This folder hosts the full detector → ranker → refiner training stack that implements the policy described in `Agents.md`. Every script is runnable with `python -m ...` **after** activating the project environment (`source ~/.venv/bin/activate`). The scripts assume the competition dataset lives in `~/data` with the `{press,report}_{jpg,json}` splits.

## Components

| Stage | Goal | Key files |
|-------|------|-----------|
| Detector | DocLayout-YOLO finetuned over 19 layout classes (표/차트/T03 우선). | `configs/detector.yaml`, `scripts/train_detector.py`, `src/training/yolo_detector.py` |
| Ranker | LightGBM LambdaMART ranker that scores gated candidates per query. Implements the 4-step policy (explicit class → title reference → figure fallback → 본문). | `configs/ranker.yaml`, `scripts/train_ranker.py`, `src/pipeline/*`, `src/features/ranker_features.py`, `src/training/ranker_trainer.py` |
| Refiner | Lightweight box-adjuster that learns Δ(x, y, w, h) from noisy anchors (CIoU-style offsets). | `configs/refiner.yaml`, `scripts/train_refiner.py`, `src/data/refiner_dataset.py`, `src/training/refiner_trainer.py` |

Supporting utilities live under `src/utils` (logging, deterministic seeds, collate fns) and `src/data` (JSON ingestion, dataset builders). `train/scripts` exposes CLI entrypoints so each stage can be trained independently.

## Running the stages

### 0. Preprocess queries with `prep.py`

All query-bearing annotations are normalized into CSVs via `prep.py`. Every training script either invokes the helper internally (with `--prep-limit` for smoke tests) or you can run it ahead of time:

```bash
source ~/.venv/bin/activate
python prep.py --data-root ~/data --output-dir train/preprocessed
```

The script emits `train/valid × press/report` CSVs with columns such as `image_path`, `instance_id`, `class_name`, and `(x,y,w,h)`. These files become the canonical “input feed” for downstream query-aware trainers (ranker + refiner).

> Always activate the provided Python before running commands:
>
> ```bash
> source ~/.venv/bin/activate
> ```

1. **Detector**

    ```bash
    python -m train.scripts.train_detector \
      --config train/configs/detector.yaml \
      --epochs 40 \
      --data-yaml outputs/yolo_datasets/dataset.yaml
    ```

    *Runs Ultralytics DocLayout-YOLO: the script expects `outputs/yolo_datasets/dataset.yaml` (generated via `preprocess.py`) and finetunes the specified backbone (`yolo.model`). The best/last weights are copied to `outputs/detector/detector_{best,last}.pt`, and the raw YOLO run artifacts stay under `outputs/yolo_models/<run_name>/`. Pass `--fraction 0.2` for subset experiments or `--resume` to continue the latest run.*

2. **Ranker**

    ```bash
    python -m train.scripts.train_ranker \
      --config train/configs/ranker.yaml \
      --max-train 400 \
      --max-valid 80 \
      --prep-limit 5
    ```

    *Automatically calls `prep.py`, ingests the resulting CSVs through `load_query_samples`, rehydrates the full JSON for candidate generation, and builds the ranking dataframe:
    - Parses intent with `src/pipeline/query_parser.py`.
    - Applies the 4-step gating via `candidate_selector.gate_candidates`.
    - Extracts lexical/layout cues (`token_overlap`, `title_gap`, `context_mentions_instance`, centers/area ratios, intent flags) for each candidate.
    - Trains a LightGBM `LGBMRanker` with LambdaMART (NDCG@1/5 objectives).*

3. **Refiner**

    ```bash
    python -m train.scripts.train_refiner \
      --config train/configs/refiner.yaml \
      --max-samples 1000 \
      --prep-limit 5
    ```

    *Starts from the same preprocessed CSV rows (query-positive boxes only), jitters anchors, and learns SmoothL1 deltas with a small MLP (`BoxRefiner`). Outputs `outputs/refiner/refiner.pt`.*

Ranker/Refiner scripts accept `--max-*` arguments for quick smoke tests; omit them for full training. The detector uses Ultralytics’ built-in `--fraction` argument instead.

## Configuration highlights

* `configs/detector.yaml` — adjust DocLayout-YOLO settings (backbone checkpoint, dataset yaml, epochs/batch/imgsz, project/name).
* `configs/ranker.yaml` — control max queries processed, LightGBM hyperparameters, and where the preprocessed CSVs live.
* `configs/refiner.yaml` — tune jitter amount, samples per annotation, targeted categories, and the preprocessing options shared with the ranker.

Each script produces a log file in its output directory (`train.log`) plus metadata (history curves, dataset stats). Downstream inference code can load:

* Detector checkpoints via PyTorch (`outputs/detector/detector_best.pt`).
* Ranker boosters via `lightgbm.Booster`.
* Refiner weights via `BoxRefiner.load_state_dict`.

This structure keeps the data contracts clear: `prep.py` produces the canonical query-positive feed, scripts rehydrate JSONs only when needed, intent/gating logic is centralized, and outputs are written in reproducible folders for later integration with the submission runner.
