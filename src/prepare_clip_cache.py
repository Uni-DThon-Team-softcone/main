# prepare_clip_cache.py

import os
import json
from glob import glob
from typing import Dict, Any, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from transformers import AutoProcessor

# utils 모듈에서 공통 함수 import
from utils import (
    normalize_class_id, clean_query_text,
    find_jsons, read_json, get_image_path
)


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Prepare cached CLIP dataset (cropped images + tokenized texts)")
    ap.add_argument("--json_dir", type=str, required=True)
    ap.add_argument("--jpg_dir", type=str, required=True)
    ap.add_argument("--split", type=str, required=True, help="e.g., train or val")
    ap.add_argument("--cache_dir", type=str, default="./clip_cache")
    ap.add_argument("--clip_model", type=str, default="koclip/koclip-base-pt")
    ap.add_argument("--max_samples", type=int, default=-1, help="-1 for all")
    args = ap.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    split_root = os.path.join(args.cache_dir, args.split)
    img_out_dir = os.path.join(split_root, "images")
    txt_out_dir = os.path.join(split_root, "texts")
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(txt_out_dir, exist_ok=True)

    print(f"[CachePrep] Loading processor: {args.clip_model}")
    processor = AutoProcessor.from_pretrained(args.clip_model)

    json_files = find_jsons(args.json_dir)
    print(f"[CachePrep] Found {len(json_files)} json files in {args.json_dir}")

    global_idx = 0
    skipped_files = 0
    skipped_anns = 0

    for jf in tqdm(json_files, desc=f"[CachePrep] {args.split} JSON", unit="file"):
        try:
            data = read_json(jf)
        except Exception:
            skipped_files += 1
            continue

        try:
            img_path = get_image_path(jf, data, jpg_dir=args.jpg_dir)
        except FileNotFoundError:
            skipped_files += 1
            continue

        try:
            page_img = Image.open(img_path).convert("RGB")
        except Exception:
            skipped_files += 1
            continue

        anns = data.get("learning_data_info", {}).get("annotation", [])
        H, W = page_img.size[1], page_img.size[0]

        for ann in anns:
            if args.max_samples > 0 and global_idx >= args.max_samples:
                break

            qtxt = str(ann.get("visual_instruction", "")).strip()
            visual_answer = str(ann.get("visual_answer", "")).strip() if ann.get("visual_answer") else None
            bbox = ann.get("bounding_box", None)
            cid_raw = ann.get("class_id", "")

            if not qtxt or bbox is None or len(bbox) != 4:
                skipped_anns += 1
                continue

            cid_norm = normalize_class_id(cid_raw)
            if cid_norm is None:
                skipped_anns += 1
                continue

            qtxt_clean = clean_query_text(qtxt, cid_norm, visual_answer)

            x, y, w, h = bbox
            x1 = int(max(0, x))
            y1 = int(max(0, y))
            x2 = int(min(W, x + w))
            y2 = int(min(H, y + h))
            if x2 <= x1 or y2 <= y1:
                skipped_anns += 1
                continue

            crop = page_img.crop((x1, y1, x2, y2))

            img_save_path = os.path.join(img_out_dir, f"{global_idx}.jpg")
            crop.save(img_save_path, quality=95)

            # 텍스트 토크나이즈 (한 번만)
            tokens = processor(
                text=[qtxt_clean],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=77,
            )
            # batch=1 이므로 0번 index만 사용
            input_ids = tokens["input_ids"][0]
            attn_mask = tokens["attention_mask"][0]

            txt_save_path = os.path.join(txt_out_dir, f"{global_idx}.pt")
            torch.save(
                {
                    "input_ids": input_ids,
                    "attention_mask": attn_mask,
                },
                txt_save_path,
            )

            global_idx += 1

        if args.max_samples > 0 and global_idx >= args.max_samples:
            break

    meta = {
        "num_samples": global_idx,
        "skipped_files": skipped_files,
        "skipped_annotations": skipped_anns,
    }
    with open(os.path.join(split_root, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[CachePrep] Done for split={args.split}")
    print(f"  - num_samples       : {global_idx}")
    print(f"  - skipped_files     : {skipped_files}")
    print(f"  - skipped_annotations: {skipped_anns}")


if __name__ == "__main__":
    main()
