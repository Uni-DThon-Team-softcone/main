# clip_train_cached.py

import os
import json
from typing import List

# 토크나이저 병렬 경고 끄기
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
from transformers import AutoModel


# ===================== Dataset =====================

class CachedCLIPDataset(Dataset):
    """
    prepare_clip_cache.py 로 만들어진 캐시를 사용하는 Dataset
    - images:   cache_root/{split}/images/{i}.jpg
    - texts:    cache_root/{split}/texts/{i}.pt  (input_ids, attention_mask)
    - meta:     cache_root/{split}/meta.json     (num_samples)
    """

    def __init__(self, cache_root: str, split: str = "train", image_size: int = 224):
        super().__init__()
        self.split_root = os.path.join(cache_root, split)
        self.img_dir = os.path.join(self.split_root, "images")
        self.txt_dir = os.path.join(self.split_root, "texts")

        meta_path = os.path.join(self.split_root, "meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"meta.json not found: {meta_path}")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.num_samples = int(meta["num_samples"])

        # OpenAI CLIP / KoCLIP 기본 mean/std
        self.transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])

        print(f"[CachedDataset] split={split}, num_samples={self.num_samples}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        img_path = os.path.join(self.img_dir, f"{idx}.jpg")
        txt_path = os.path.join(self.txt_dir, f"{idx}.pt")

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), color=(0, 0, 0))

        img_tensor = self.transform(img)

        tok = torch.load(txt_path)
        input_ids = tok["input_ids"]
        attn_mask = tok["attention_mask"]

        return {
            "image": img_tensor,              # (3, H, W)
            "input_ids": input_ids,          # (L,)
            "attention_mask": attn_mask,     # (L,)
        }


# ===================== Model =====================
from model import CLIPMatchingModel

# ===================== Training Loop =====================

def train_clip_matching(args):
    device = args.device
    torch.manual_seed(args.seed)

    # Dataset
    train_ds = CachedCLIPDataset(args.cache_root, split=args.train_split, image_size=args.image_size)
    val_ds = CachedCLIPDataset(args.cache_root, split=args.val_split, image_size=args.image_size)

    def collate_fn(batch):
        # batch: list of dicts
        images = torch.stack([b["image"] for b in batch], dim=0)  # (B, 3, H, W)
        input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)  # (B, L)
        attn_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)  # (B, L)
        return images, input_ids, attn_mask

    num_workers = args.num_workers
    prefetch_factor = 4 if num_workers > 0 else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(device != "cpu"),
        prefetch_factor=prefetch_factor,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(device != "cpu"),
        prefetch_factor=prefetch_factor,
        persistent_workers=(num_workers > 0),
    )

    # Model
    model = CLIPMatchingModel(
        model_name=args.clip_model,
        freeze_backbone=not args.unfreeze_backbone,
        device=device,
    ).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"[Training] Trainable parameter tensors: {len(trainable_params)}")

    optim = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=1e-4,
    )

    scaler = torch.cuda.amp.GradScaler() if args.use_amp and device != "cpu" else None

    def clip_loss(logits_per_text, logits_per_image):
        batch_size = logits_per_text.size(0)
        target = torch.arange(batch_size, dtype=torch.long, device=device)
        loss_t = F.cross_entropy(logits_per_text, target)
        loss_i = F.cross_entropy(logits_per_image, target)
        return (loss_t + loss_i) / 2.0

    best_val_loss = float("inf")
    os.makedirs(os.path.dirname(args.save_ckpt), exist_ok=True)

    if args.use_amp:
        print("[Training] Using AMP")

    for epoch in range(1, args.epochs + 1):
        # ---- train ----
        model.train()
        total_loss = 0.0
        total_cnt = 0

        for images, input_ids, attn_mask in tqdm(train_loader, desc=f"[Train] Epoch {epoch}", unit="batch"):
            optim.zero_grad()

            if scaler is not None:
                with torch.cuda.amp.autocast():
                    logits_t, logits_i = model.forward_batch(images, input_ids, attn_mask)
                    loss = clip_loss(logits_t, logits_i)
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                logits_t, logits_i = model.forward_batch(images, input_ids, attn_mask)
                loss = clip_loss(logits_t, logits_i)
                loss.backward()
                optim.step()

            bs = images.size(0)
            total_loss += float(loss.item()) * bs
            total_cnt += bs

        avg_train = total_loss / max(1, total_cnt)

        # ---- valid ----
        model.eval()
        total_vloss = 0.0
        total_vcnt = 0

        with torch.no_grad():
            for images, input_ids, attn_mask in tqdm(val_loader, desc=f"[Valid] Epoch {epoch}", unit="batch"):
                logits_t, logits_i = model.forward_batch(images, input_ids, attn_mask)
                loss = clip_loss(logits_t, logits_i)
                bs = images.size(0)
                total_vloss += float(loss.item()) * bs
                total_vcnt += bs

        avg_val = total_vloss / max(1, total_vcnt)
        print(f"[Epoch {epoch}] train_loss={avg_train:.4f}  val_loss={avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "clip_model": args.clip_model,
                },
                args.save_ckpt,
            )
            print(f"[Saved] best model to {args.save_ckpt} (val_loss={best_val_loss:.4f})")


def get_args():
    import argparse
    ap = argparse.ArgumentParser(description="Stage 3: CLIP matching training (cached dataset)")
    ap.add_argument("--cache_root", type=str, required=True, help="Root of clip_cache created by prepare_clip_cache.py")
    ap.add_argument("--train_split", type=str, default="train")
    ap.add_argument("--val_split", type=str, default="val")
    ap.add_argument("--save_ckpt", type=str, default="./outputs/clip_matching/best_clip_cached.pth")

    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--use_amp", action="store_true")

    ap.add_argument("--clip_model", type=str, default="koclip/koclip-base-pt")
    ap.add_argument("--unfreeze_backbone", action="store_true")

    return ap.parse_args()


def main():
    args = get_args()
    train_clip_matching(args)


if __name__ == "__main__":
    main()
