import argparse
import contextlib
import csv
import hashlib
import json
import os
import random
import zipfile
from glob import glob
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModel,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from transformers.modeling_outputs import SequenceClassifierOutput


ImageFile.LOAD_TRUNCATED_IMAGES = True


class CFG:
    SEED: int = 42
    DOC_WIDTH: int = 1024
    IMG_SIZE: int = 256
    MAX_TEXT_LEN: int = 96
    DIM: int = 512
    NUM_FEATURE_LEVELS: int = 3
    DECODER_LAYERS: int = 3
    DECODER_HEADS: int = 8
    DECODER_DROPOUT: float = 0.1
    FREEZE_TEXT: bool = False
    FREEZE_VISION: bool = False
    USE_CACHE: bool = False
    CACHE_DIR: str = "./cache"
    EPOCHS: int = 8
    LEARNING_RATE: float = 1e-4
    BATCH_SIZE: int = 8
    EVAL_BATCH_SIZE: int = 8
    NUM_WORKERS: int = 4
    PREFETCH_FACTOR: int = 4
    IMAGE_BACKBONE: str = "microsoft/swinv2-base-patch4-window8-256"
    TEXT_BACKBONE: str = "intfloat/multilingual-e5-large-instruct"
    JSON_DIR: str = "./data/json"
    JPG_DIR: Optional[str] = None
    OUTPUT_DIR: str = "./outputs/ckpt/swinv2_small"
    EVAL_CSV: str = "./outputs/preds/eval_pred.csv"
    PRED_CSV: str = "./outputs/preds/test_pred.csv"
    SUBMISSION_ZIP: str = "./outputs/submission.zip"

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def find_jsons(json_dir: str) -> List[str]:
    if os.path.isdir(json_dir):
        return sorted(glob(os.path.join(json_dir, "*.json")))
    raise FileNotFoundError(f"json_dir not found: {json_dir}")


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_image_path(json_path: str, data: Dict[str, Any], jpg_dir: Optional[str] = None) -> str:
    src = data.get("source_data_info", {})
    jpg_name = src.get("source_data_name_jpg")
    if jpg_dir and jpg_name:
        candidate = os.path.join(jpg_dir, jpg_name)
        if os.path.exists(candidate):
            return candidate
    if jpg_name:
        maybe = json_path.replace(os.sep + "json" + os.sep, os.sep + "jpg" + os.sep)
        maybe = os.path.join(os.path.dirname(maybe), jpg_name) if os.path.isdir(os.path.dirname(maybe)) else maybe
        if os.path.exists(maybe):
            return maybe
    base = os.path.splitext(os.path.basename(json_path))[0]
    sibling = os.path.join(os.path.dirname(json_path), base.replace("MI3", "MI2") + ".jpg")
    if os.path.exists(sibling):
        return sibling
    raise FileNotFoundError(f"Could not resolve JPG for {json_path} (jpg_dir={jpg_dir})")


def is_visual_ann(ann: Dict[str, Any]) -> bool:
    cid = str(ann.get("class_id", "") or "")
    cname = str(ann.get("class_name", "") or "")
    has_q = bool(str(ann.get("visual_instruction", "") or "").strip())
    looks_visual = cid.startswith("V") or any(tok in cname for tok in ["표", "차트", "그래프", "chart", "table"])
    return has_q and looks_visual


def iou_xywh_pixel(pred_xywh, gt_xywh):
    px, py, pw, ph = pred_xywh
    gx, gy, gw, gh = gt_xywh
    px2, py2 = px + pw, py + ph
    gx2, gy2 = gx + gw, gy + gh
    ix1, iy1 = max(px, gx), max(py, gy)
    ix2, iy2 = min(px2, gx2), min(py2, gy2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    denom = pw * ph + gw * gh - inter
    if denom <= 0:
        return 0.0
    return inter / denom


def _xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return torch.stack([x1, y1, x2, y2], dim=-1)


def generalized_iou(pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
    pred = _xywh_to_xyxy(pred_boxes)
    target = _xywh_to_xyxy(target_boxes)
    x1 = torch.max(pred[..., 0], target[..., 0])
    y1 = torch.max(pred[..., 1], target[..., 1])
    x2 = torch.min(pred[..., 2], target[..., 2])
    y2 = torch.min(pred[..., 3], target[..., 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    pred_area = (pred[..., 2] - pred[..., 0]).clamp(min=0) * (pred[..., 3] - pred[..., 1]).clamp(min=0)
    target_area = (target[..., 2] - target[..., 0]).clamp(min=0) * (target[..., 3] - target[..., 1]).clamp(min=0)
    union = pred_area + target_area - inter
    iou = inter / union.clamp(min=1e-6)
    c_x1 = torch.min(pred[..., 0], target[..., 0])
    c_y1 = torch.min(pred[..., 1], target[..., 1])
    c_x2 = torch.max(pred[..., 2], target[..., 2])
    c_y2 = torch.max(pred[..., 3], target[..., 3])
    area_c = (c_x2 - c_x1).clamp(min=0) * (c_y2 - c_y1).clamp(min=0)
    giou = iou - (area_c - union) / area_c.clamp(min=1e-6)
    return giou


# def _sdp_context():
#     if not torch.cuda.is_available():
#         return contextlib.nullcontext()
#     return torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION)


def _get_image_size(path: str) -> Tuple[int, int]:
    with Image.open(path) as img:
        return img.size


def _target_size(orig: Tuple[int, int], target_width: int) -> Tuple[int, int]:
    ow, oh = orig
    if target_width is None or target_width <= 0 or ow == 0:
        return ow, oh
    if ow == target_width:
        return ow, oh
    scale = target_width / ow
    nh = max(8, int(round(oh * scale)))
    return target_width, nh


def _scale_bbox(bbox: List[float], orig: Tuple[int, int], resized: Tuple[int, int]) -> List[float]:
    if bbox is None:
        return None
    x, y, w, h = bbox
    ow, oh = orig
    rw, rh = resized
    if ow == 0 or oh == 0:
        return None
    sx = rw / ow
    sy = rh / oh
    px = x * sx
    py = y * sy
    pw = w * sx
    ph = h * sy
    if rw == 0 or rh == 0:
        return None
    cx = (px + pw / 2.0) / rw
    cy = (py + ph / 2.0) / rh
    nw = pw / rw
    nh = ph / rh
    return [cx, cy, nw, nh]


class DocLayoutDataset(Dataset):
    def __init__(
        self,
        json_dir: Optional[str],
        jpg_dir: Optional[str],
        image_processor,
        tokenizer,
        max_text_len: int,
        doc_width: int,
        include_labels: bool = True,
        supervised_only: bool = False,
        csv_path: Optional[str] = None,
        csv_root: Optional[str] = None,
        use_cache: bool = False,
        freeze_text: bool = False,
        freeze_vision: bool = False,
    ):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.doc_width = doc_width
        self.include_labels = include_labels
        self.supervised_only = supervised_only
        self.use_cache = use_cache
        self.freeze_text = freeze_text
        self.freeze_vision = freeze_vision
        self.cache_encoded_prompt = use_cache and freeze_text
        self.cache_encoded_image = use_cache and freeze_vision
        self.cache_resized_image = use_cache and not freeze_vision
        self.cache_dir = None
        self.cache_manifest: Dict[str, str] = {}
        self.items: List[Dict[str, Any]] = []

        if csv_path:
            self._load_from_csv(csv_path, csv_root)
        elif json_dir:
            self._load_from_json(json_dir, jpg_dir)
        else:
            raise ValueError("Either json_dir or csv_path must be provided.")

        if not self.items:
            source = csv_path or json_dir or "dataset"
            raise RuntimeError(f"No samples found in {source}")

        if include_labels and supervised_only:
            self.items = [it for it in self.items if it["bbox"] is not None]
            if not self.items:
                source = csv_path or json_dir or "dataset"
                raise RuntimeError(f"No supervised samples (bbox missing) in {source}")
        self._assign_resized_sizes()
        print("Use cache:", self.use_cache)
        if self.use_cache:
            self._init_cache_dir(json_dir or csv_path)
            self._build_cache()
    def _init_cache_dir(self, source_path: str):
        base_name = os.path.splitext(os.path.basename(source_path))[0]
        cache_root = os.path.join(CFG.CACHE_DIR, base_name)
        os.makedirs(cache_root, exist_ok=True)
        self.cache_dir = cache_root

    def _load_from_json(self, json_dir: str, jpg_dir: Optional[str]):
        for jf in find_jsons(json_dir):
            data = read_json(jf)
            ann_list = data.get("learning_data_info", {}).get("annotation", [])
            try:
                img_path = get_image_path(jf, data, jpg_dir=jpg_dir)
            except FileNotFoundError:
                continue
            orig_size = _get_image_size(img_path)
            for ann in ann_list:
                if not is_visual_ann(ann):
                    continue
                bbox = ann.get("bounding_box")
                self.items.append(
                    {
                        "json": jf,
                        "img": img_path,
                        "query_id": ann.get("instance_id", ""),
                        "query": str(ann.get("visual_instruction", "")).strip(),
                        "bbox": bbox if bbox and len(bbox) == 4 else None,
                        "class_name": ann.get("class_name", ""),
                        "orig_size": orig_size,
                    }
                )

    def _load_from_csv(self, csv_path: str, csv_root: Optional[str]):
        csv_path = os.path.abspath(csv_path)
        base_dir = os.path.abspath(csv_root) if csv_root else os.path.dirname(csv_path)
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                rel = (row.get("image_path") or "").strip()
                if not rel:
                    continue
                img_path = os.path.normpath(os.path.join(base_dir, rel))
                if not os.path.isabs(img_path):
                    img_path = os.path.abspath(img_path)
                if not os.path.exists(img_path):
                    continue
                try:
                    coords = [float(row.get(col, "")) for col in ("x", "y", "w", "h")]
                except (TypeError, ValueError):
                    coords = []
                bbox = coords if len(coords) == 4 and np.isfinite(coords).all() else None
                query = str(row.get("visual_instruction", "")).strip()
                query_id = str(row.get("query_id") or f"{os.path.basename(img_path)}#{idx}")
                # orig_size = _get_image_size(img_path)
                self.items.append(
                    {
                        "json": csv_path,
                        "img": img_path,
                        "query_id": query_id,
                        "query": query,
                        "bbox": bbox,
                        "class_name": "",
                        "orig_size": [float(row.get("img_w", "")), float(row.get("img_h", ""))],
                    }
                )

    def _assign_resized_sizes(self):
        for rec in self.items:
            rec["resized_size"] = _target_size(rec["orig_size"], self.doc_width)

    def _build_cache(self):
        if self.cache_dir is None:
            self.cache_dir = CFG.CACHE_DIR
            os.makedirs(self.cache_dir, exist_ok=True)

        for rec in tqdm(self.items, desc="Building cache", disable=len(self.items) < 128):
            if self.cache_encoded_prompt:
                rec["cached_prompt"] = self._prepare_text(rec["query"])
            resized_size = rec["resized_size"]
            if self.cache_encoded_image:
                rec["cached_pixel"] = self._prepare_image(rec["img"], resized_size)
            elif self.cache_resized_image:
                cache_path = self._cached_image_path(rec)
                if not os.path.exists(cache_path):
                    arr = self._resize_to_array(rec["img"], resized_size)
                    np.save(cache_path, arr)
                rec["cached_resized_path"] = cache_path
                rec_id = rec["query_id"]
                self.cache_manifest[rec_id] = cache_path

    def _cached_image_path(self, rec: Dict[str, Any]) -> str:
        if self.cache_dir is None:
            self.cache_dir = CFG.CACHE_DIR
        name = rec["query_id"].replace("/", "_")
        fname = f"{name}_{rec['resized_size'][0]}x{rec['resized_size'][1]}.npy"
        return os.path.join(self.cache_dir, fname)

    @staticmethod
    def _clone_prompt(cache: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in cache.items()}

    @staticmethod
    def _array_to_image(arr: np.ndarray) -> Image.Image:
        return Image.fromarray(arr.astype(np.uint8), mode="RGB")

    def __len__(self):
        return len(self.items)

    def _prepare_text(self, text: str) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            text or "",
            max_length=self.max_text_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in encoded.items()}

    def _prepare_image(self, img_path: str, target_size: Tuple[int, int]) -> torch.Tensor:
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            if target_size != img.size:
                img = img.resize(target_size, Image.BILINEAR)
            proc = self.image_processor(images=img, return_tensors="pt")
            return proc["pixel_values"].squeeze(0)

    def _resize_to_array(self, img_path: str, target_size: Tuple[int, int]) -> np.ndarray:
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            if target_size != img.size:
                img = img.resize(target_size, Image.BILINEAR)
            return np.array(img, dtype=np.uint8)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.items[idx]
        orig_size = record["orig_size"]
        resized_size = record.get("resized_size")
        if resized_size is None:
            resized_size = _target_size(orig_size, self.doc_width)
            record["resized_size"] = resized_size

        if self.use_cache and "cached_prompt" in record:
            text_inputs = self._clone_prompt(record["cached_prompt"])
        else:
            text_inputs = self._prepare_text(record["query"])

        if "cached_pixel" in record:
            pixel_values = record["cached_pixel"]
        elif "cached_resized_path" in record:
            arr = np.load(record["cached_resized_path"])
            img = self._array_to_image(arr)
            proc = self.image_processor(images=img, return_tensors="pt")
            pixel_values = proc["pixel_values"].squeeze(0)
        else:
            pixel_values = self._prepare_image(record["img"], resized_size)
        item: Dict[str, Any] = {
            "pixel_values": pixel_values,
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
        }
        if self.include_labels:
            label = torch.zeros(9, dtype=torch.float32)
            scaled = _scale_bbox(record["bbox"], orig_size, resized_size)
            if scaled is not None:
                label[:4] = torch.tensor(scaled, dtype=torch.float32)
                label[4] = 1.0
            ow, oh = orig_size
            rw, rh = resized_size
            label[5:] = torch.tensor([float(ow), float(oh), float(rw), float(rh)], dtype=torch.float32)
            item["labels"] = label
        return item

    def get_item_meta(self, idx: int) -> Dict[str, Any]:
        rec = self.items[idx]
        return {
            "query_id": rec["query_id"],
            "query": rec["query"],
            "img": rec["img"],
            "orig_size": rec["orig_size"],
        }


class DocBatchCollator:
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        pixel_values = torch.stack([b["pixel_values"] for b in batch])
        input_ids = torch.stack([b["input_ids"] for b in batch])
        attention_mask = torch.stack([b["attention_mask"] for b in batch])
        out = {"pixel_values": pixel_values, "input_ids": input_ids, "attention_mask": attention_mask}
        if "labels" in batch[0]:
            labels = torch.stack([b["labels"] for b in batch])
            out["labels"] = labels
        return out


class FlashMHA(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def _shape(self, tensor: torch.Tensor) -> torch.Tensor:
        B, L, _ = tensor.shape
        return tensor.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        context = hidden_states if key_value_states is None else key_value_states
        q = self._shape(self.q_proj(hidden_states))
        k = self._shape(self.k_proj(context))
        v = self._shape(self.v_proj(context))
        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = key_padding_mask[:, None, None, :].to(torch.bool)
        # with _sdp_context():
        attn = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0, is_causal=False
        )
        attn = attn.transpose(1, 2).contiguous().view(hidden_states.size(0), hidden_states.size(1), self.dim)
        return self.out_proj(attn)


class MultiScaleDecoderLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn1 = FlashMHA(dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = FlashMHA(dim, num_heads, dropout=dropout)
        self.norm3 = nn.LayerNorm(dim)
        self.self_attn2 = FlashMHA(dim, num_heads, dropout=dropout)
        self.norm4 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor, context_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = residual + self.self_attn1(x)

        residual = x
        q = self.norm2(x)
        key_padding = context_mask
        x = residual + self.cross_attn(q, key_value_states=context, key_padding_mask=key_padding)

        residual = x
        q = self.norm3(x)
        x = residual + self.self_attn2(q)

        residual = x
        x = residual + self.mlp(self.norm4(x))
        return x


class MultiscaleDecoder(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_layers: int, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList(
            [MultiScaleDecoderLayer(dim, num_heads, mlp_ratio=4.0, dropout=dropout) for _ in range(num_layers)]
        )
        self.context_norm = nn.LayerNorm(dim)
        self.out_norm = nn.LayerNorm(dim)

    def forward(
        self,
        vision_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        key_padding = None
        if text_mask is not None:
            key_padding = ~text_mask.bool()
        context = self.context_norm(text_tokens)
        x = vision_tokens
        for layer in self.layers:
            x = layer(x, context, key_padding)
        return self.out_norm(x)


class CrossModalBBoxPredictor(nn.Module):
    def __init__(
        self,
        image_model_name: str,
        text_model_name: str,
        hidden_dim: int,
        num_feature_levels: int,
        decoder_layers: int,
        decoder_heads: int,
        decoder_dropout: float,
        freeze_text: bool = False,
        freeze_vision: bool = False,
    ):
        super().__init__()
        self.freeze_text = freeze_text
        self.freeze_vision = freeze_vision
        self.image_config = AutoConfig.from_pretrained(image_model_name, output_hidden_states=True)
        self.text_config = AutoConfig.from_pretrained(text_model_name, output_hidden_states=True)
        self.image_encoder = AutoModel.from_pretrained(image_model_name, config=self.image_config)
        self.text_encoder = AutoModel.from_pretrained(text_model_name, config=self.text_config)
        
        for p in self.image_encoder.parameters():
            p.requires_grad = True
        for p in self.text_encoder.parameters():
            p.requires_grad = True
            
        if freeze_vision:
            for p in self.image_encoder.parameters():
                p.requires_grad = False
        if freeze_text:
            for p in self.text_encoder.parameters():
                p.requires_grad = False
        inferred_dims = self._infer_feature_dims(num_feature_levels)
        self.num_feature_levels = len(inferred_dims)
        self.image_projs = nn.ModuleList([nn.Linear(dim, hidden_dim) for dim in inferred_dims])
        self.level_embed = nn.Parameter(torch.zeros(self.num_feature_levels, 1, 1, hidden_dim))
        self.text_proj = nn.Linear(self.text_config.hidden_size, hidden_dim)
        self.decoder = MultiscaleDecoder(
            dim=hidden_dim,
            num_heads=decoder_heads,
            num_layers=decoder_layers,
            dropout=decoder_dropout,
        )
        self.bbox_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4),
        )

    def _infer_feature_dims(self, desired_levels: int) -> List[int]:
        img_size = getattr(self.image_config, "image_size", CFG.IMG_SIZE)
        if isinstance(img_size, dict):
            height = img_size.get("height") or img_size.get("shortest_edge") or CFG.IMG_SIZE
            width = img_size.get("width") or height
        elif isinstance(img_size, (tuple, list)):
            height, width = img_size[:2]
        else:
            height = width = int(img_size)
        channels = getattr(self.image_config, "num_channels", 3)
        device = next(self.image_encoder.parameters()).device
        dummy = torch.zeros(1, channels, height, width, device=device)
        training = self.image_encoder.training
        self.image_encoder.eval()
        with torch.no_grad():
            outputs = self.image_encoder(pixel_values=dummy, output_hidden_states=True, return_dict=True)
        self.image_encoder.train(training)
        hs = outputs.hidden_states[-desired_levels:]
        dims = [tensor.shape[-1] for tensor in hs]
        if not dims:
            dims = [self.image_config.hidden_size]
        return dims

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        img_out = self.image_encoder(pixel_values=pixel_values, output_hidden_states=True, return_dict=True)
        hs = img_out.hidden_states[-self.num_feature_levels :]
        vision_tokens = []
        for idx, (tensor, proj) in enumerate(zip(hs, self.image_projs)):
            tok = proj(tensor)
            tok = tok + self.level_embed[idx]
            vision_tokens.append(tok)
        vision_tokens = torch.cat(vision_tokens, dim=1)

        txt_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        text_tokens = self.text_proj(txt_out.last_hidden_state)
        decoded = self.decoder(vision_tokens, text_tokens, attention_mask)
        pooled = decoded.mean(dim=1)
        logits = torch.sigmoid(self.bbox_head(pooled))

        loss = None
        if labels is not None:
            target = labels[:, :4]
            mask = labels[:, 4:5]
            valid = mask.squeeze(-1) > 0
            if valid.any():
                l1 = F.smooth_l1_loss(logits, target, reduction="none")
                l1 = (l1 * mask).sum() / (mask.sum() * logits.size(-1))
                giou = generalized_iou(logits[valid], target[valid])
                giou_loss = (1.0 - giou).mean()
                loss = l1 + giou_loss
            else:
                loss = logits.sum() * 0
        return SequenceClassifierOutput(loss=loss, logits=logits)


class FastTrainer(Trainer):
    def __init__(self, *args, dataloader_kwargs=None, **kwargs):
        super().__init__(*args, **kwargs)
        default = {"num_workers": CFG.NUM_WORKERS, "prefetch_factor": CFG.PREFETCH_FACTOR}
        self.fast_loader_kwargs = {**default, **(dataloader_kwargs or {})}

    def _build_dataloader(self, dataset, train: bool):
        if dataset is None:
            return None
        batch_size = self.args.train_batch_size if train else self.args.eval_batch_size
        num_workers = self.fast_loader_kwargs["num_workers"]
        loader_kwargs = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": train,
            "collate_fn": self.data_collator,
            "num_workers": num_workers,
            "pin_memory": True,
            "persistent_workers": num_workers > 0,
        }
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = self.fast_loader_kwargs["prefetch_factor"]
        return DataLoader(**loader_kwargs)

    def get_train_dataloader(self):
        return self._build_dataloader(self.train_dataset, train=True)

    def get_eval_dataloader(self, eval_dataset=None):
        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        return self._build_dataloader(dataset, train=False)

    def get_test_dataloader(self, test_dataset):
        return self._build_dataloader(test_dataset, train=False)


def denorm_boxes(norm_boxes: np.ndarray, label_info: np.ndarray) -> np.ndarray:
    proc_w = label_info[:, 7]
    proc_h = label_info[:, 8]
    orig_w = label_info[:, 5]
    orig_h = label_info[:, 6]
    cx = norm_boxes[:, 0]
    cy = norm_boxes[:, 1]
    w = norm_boxes[:, 2]
    h = norm_boxes[:, 3]
    px = (cx - w / 2.0) * proc_w
    py = (cy - h / 2.0) * proc_h
    pw = w * proc_w
    ph = h * proc_h
    scale_x = orig_w / np.maximum(proc_w, 1e-6)
    scale_y = orig_h / np.maximum(proc_h, 1e-6)
    px *= scale_x
    py *= scale_y
    pw *= scale_x
    ph *= scale_y
    return np.stack([px, py, pw, ph], axis=1)


def build_compute_metrics():
    def compute_metrics_fn(eval_pred):
        preds = eval_pred.predictions
        labels = eval_pred.label_ids
        mask = labels[:, 4]
        valid = mask > 0.5
        if valid.sum() == 0:
            return {"mIoU": 0.0}
        pred_pix = denorm_boxes(preds[valid], labels[valid])
        gt_pix = denorm_boxes(labels[valid][:, :4], labels[valid])
        ious = [iou_xywh_pixel(pred_pix[i], gt_pix[i]) for i in range(pred_pix.shape[0])]
        return {"mIoU": float(np.mean(ious))}

    return compute_metrics_fn


def save_artifacts(
    output_dir: str,
    model: CrossModalBBoxPredictor,
    image_processor,
    tokenizer,
    bundle: Dict[str, Any],
):
    os.makedirs(output_dir, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "bundle": bundle}, os.path.join(output_dir, "model.pt"))
    image_processor.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    with open(os.path.join(output_dir, "bundle.json"), "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2, ensure_ascii=False)


def load_artifacts(ckpt_dir: str, device: torch.device):
    bundle_path = os.path.join(ckpt_dir, "bundle.json")
    if not os.path.exists(bundle_path):
        raise FileNotFoundError(f"Missing bundle.json in {ckpt_dir}")
    with open(bundle_path, "r", encoding="utf-8") as f:
        bundle = json.load(f)
    image_processor = AutoImageProcessor.from_pretrained(ckpt_dir)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    model = CrossModalBBoxPredictor(
        image_model_name=bundle["image_model"],
        text_model_name=bundle["text_model"],
        hidden_dim=bundle["dim"],
        num_feature_levels=bundle["num_feature_levels"],
        decoder_layers=bundle.get("decoder_layers", CFG.DECODER_LAYERS),
        decoder_heads=bundle.get("decoder_heads", CFG.DECODER_HEADS),
        decoder_dropout=bundle.get("decoder_dropout", CFG.DECODER_DROPOUT),
    )
    state = torch.load(os.path.join(ckpt_dir, "model.pt"), map_location=device)
    model.load_state_dict(state["model_state"])
    return model, image_processor, tokenizer, bundle


def predictions_to_rows(
    preds: np.ndarray,
    dataset: DocLayoutDataset,
    doc_width: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for pred, idx in zip(preds, range(len(dataset))):
        meta = dataset.get_item_meta(idx)
        orig_w, orig_h = meta["orig_size"]
        resized = _target_size(meta["orig_size"], doc_width)
        label_stub = np.zeros(9, dtype=np.float32)
        label_stub[5:] = np.array([orig_w, orig_h, resized[0], resized[1]], dtype=np.float32)
        pixel_box = denorm_boxes(pred[None, :], label_stub[None, :])[0]
        rows.append(
            {
                "query_id": meta["query_id"],
                "query_text": meta["query"],
                "pred_x": float(pixel_box[0]),
                "pred_y": float(pixel_box[1]),
                "pred_w": float(pixel_box[2]),
                "pred_h": float(pixel_box[3]),
            }
        )
    return rows


def write_rows_to_csv(rows: List[Dict[str, Any]], out_csv: str):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df = pd.DataFrame(rows, columns=["query_id", "query_text", "pred_x", "pred_y", "pred_w", "pred_h"])
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[Saved] {out_csv}")


def train_loop(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_processor = AutoImageProcessor.from_pretrained(args.image_model,use_fast=True)
    if args.img_size:
        image_processor.size = {"height": args.img_size, "width": args.img_size}
    tokenizer = AutoTokenizer.from_pretrained(args.text_model,use_fast=True)
    train_ds = DocLayoutDataset(
        json_dir=None if args.train_csv else args.json_dir,
        jpg_dir=args.jpg_dir,
        image_processor=image_processor,
        tokenizer=tokenizer,
        max_text_len=args.max_text_len,
        doc_width=args.doc_width,
        include_labels=True,
        supervised_only=True,
        csv_path=args.train_csv,
        csv_root=args.csv_root,
        use_cache=args.use_cache,
        freeze_text=args.freeze_text,
        freeze_vision=args.freeze_vision,
    )
    eval_ds = None
    eval_strategy = "no"
    if args.valid_csv or args.valid_json_dir:
        eval_ds = DocLayoutDataset(
            json_dir=None if args.valid_csv else args.valid_json_dir,
            jpg_dir=args.jpg_dir,
            image_processor=image_processor,
            tokenizer=tokenizer,
            max_text_len=args.max_text_len,
            doc_width=args.doc_width,
            include_labels=True,
            supervised_only=False,
            csv_path=args.valid_csv,
            csv_root=args.csv_root,
            use_cache=args.use_cache,
            freeze_text=args.freeze_text,
            freeze_vision=args.freeze_vision,
        )
        eval_strategy = args.eval_strategy

    model = CrossModalBBoxPredictor(
        image_model_name=args.image_model,
        text_model_name=args.text_model,
        hidden_dim=args.dim,
        num_feature_levels=args.num_feature_levels,
        decoder_layers=args.decoder_layers,
        decoder_heads=args.decoder_heads,
        decoder_dropout=args.decoder_dropout,
        freeze_text=args.freeze_text,
        freeze_vision=args.freeze_vision,
    )
    model.to(device)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        eval_strategy=eval_strategy,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.lr,
        load_best_model_at_end=True,
        metric_for_best_model="mIoU",
        greater_is_better=False,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        save_strategy=args.save_strategy,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        gradient_accumulation_steps=args.grad_accum,
        bf16=torch.cuda.is_available(),
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,
        report_to=["wandb"],
    )

    trainer = FastTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DocBatchCollator(),
        tokenizer=None,
        compute_metrics=build_compute_metrics() if eval_ds is not None else None,
        dataloader_kwargs={"num_workers": args.num_workers, "prefetch_factor": args.prefetch_factor},
    )

    trainer.train(resume_from_checkpoint=args.resume_from)

    bundle = {
        "image_model": args.image_model,
        "text_model": args.text_model,
        "dim": args.dim,
        "num_feature_levels": args.num_feature_levels,
        "doc_width": args.doc_width,
        "max_text_len": args.max_text_len,
        "img_size": args.img_size,
        "decoder_layers": args.decoder_layers,
        "decoder_heads": args.decoder_heads,
        "decoder_dropout": args.decoder_dropout,
        "freeze_text": args.freeze_text,
        "freeze_vision": args.freeze_vision,
    }
    save_artifacts(args.output_dir, model, image_processor, tokenizer, bundle)


def evaluate_loop(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, image_processor, tokenizer, bundle = load_artifacts(args.ckpt_dir, device)
    img_size = bundle.get("img_size", args.img_size)
    if img_size:
        image_processor.size = {"height": img_size, "width": img_size}
    eval_ds = DocLayoutDataset(
        json_dir=None if args.csv_path else args.json_dir,
        jpg_dir=args.jpg_dir,
        image_processor=image_processor,
        tokenizer=tokenizer,
        max_text_len=bundle.get("max_text_len", args.max_text_len),
        doc_width=bundle.get("doc_width", args.doc_width),
        include_labels=True,
        supervised_only=False,
        csv_path=args.csv_path,
        csv_root=args.csv_root,
        use_cache=args.use_cache,
        freeze_text=bundle.get("freeze_text", False),
        freeze_vision=bundle.get("freeze_vision", False),
    )
    training_args = TrainingArguments(
        output_dir=os.path.join(args.ckpt_dir, "eval"),
        per_device_eval_batch_size=args.eval_batch_size,
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,
    )
    trainer = FastTrainer(
        model=model.to(device),
        args=training_args,
        data_collator=DocBatchCollator(),
        train_dataset=None,
        eval_dataset=eval_ds,
        tokenizer=None,
        compute_metrics=build_compute_metrics(),
        dataloader_kwargs={"num_workers": args.num_workers, "prefetch_factor": args.prefetch_factor},
    )
    metrics = trainer.evaluate()
    print(json.dumps(metrics, indent=2))
    preds = trainer.predict(eval_ds)
    rows = predictions_to_rows(preds.predictions, eval_ds, bundle.get("doc_width", args.doc_width))
    write_rows_to_csv(rows, args.out_csv)


def predict_loop(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, image_processor, tokenizer, bundle = load_artifacts(args.ckpt_dir, device)
    img_size = bundle.get("img_size", args.img_size)
    if img_size:
        image_processor.size = {"height": img_size, "width": img_size}
    test_ds = DocLayoutDataset(
        json_dir=None if args.csv_path else args.json_dir,
        jpg_dir=args.jpg_dir,
        image_processor=image_processor,
        tokenizer=tokenizer,
        max_text_len=bundle.get("max_text_len", args.max_text_len),
        doc_width=bundle.get("doc_width", args.doc_width),
        include_labels=False,
        supervised_only=False,
        csv_path=args.csv_path,
        csv_root=args.csv_root,
        use_cache=args.use_cache,
        freeze_text=bundle.get("freeze_text", False),
        freeze_vision=bundle.get("freeze_vision", False),
    )
    training_args = TrainingArguments(
        output_dir=os.path.join(args.ckpt_dir, "predict"),
        per_device_eval_batch_size=args.eval_batch_size,
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,
        report_to=[],
    )
    trainer = FastTrainer(
        model=model.to(device),
        args=training_args,
        data_collator=DocBatchCollator(),
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        dataloader_kwargs={"num_workers": args.num_workers, "prefetch_factor": args.prefetch_factor},
    )
    preds = trainer.predict(test_ds)
    rows = predictions_to_rows(preds.predictions, test_ds, bundle.get("doc_width", args.doc_width))
    write_rows_to_csv(rows, args.out_csv)


def zip_submission(csv_path: str, zip_path: str):
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname=os.path.basename(csv_path))
    print(f"[Submission] Zipped {csv_path} → {zip_path}")


def get_args():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common(p):
        p.add_argument("--json_dir", type=str, default=CFG.JSON_DIR)
        p.add_argument("--jpg_dir", type=str, default=CFG.JPG_DIR)
        p.add_argument("--batch_size", type=int, default=CFG.BATCH_SIZE)
        p.add_argument("--eval_batch_size", type=int, default=CFG.EVAL_BATCH_SIZE)
        p.add_argument("--num_workers", type=int, default=CFG.NUM_WORKERS)
        p.add_argument("--prefetch_factor", type=int, default=CFG.PREFETCH_FACTOR)
        p.add_argument("--doc_width", type=int, default=CFG.DOC_WIDTH)
        p.add_argument("--img_size", type=int, default=CFG.IMG_SIZE)
        p.add_argument("--max_text_len", type=int, default=CFG.MAX_TEXT_LEN)
        p.add_argument("--dim", type=int, default=CFG.DIM)
        p.add_argument("--num_feature_levels", type=int, default=CFG.NUM_FEATURE_LEVELS)
        p.add_argument("--decoder_layers", type=int, default=CFG.DECODER_LAYERS)
        p.add_argument("--decoder_heads", type=int, default=CFG.DECODER_HEADS)
        p.add_argument("--decoder_dropout", type=float, default=CFG.DECODER_DROPOUT)
        p.add_argument("--freeze_text", action="store_true", default=CFG.FREEZE_TEXT)
        p.add_argument("--freeze_vision", action="store_true", default=CFG.FREEZE_VISION)
        p.add_argument("--use_cache", action="store_true", default=CFG.USE_CACHE)
        p.add_argument("--image_model", type=str, default=CFG.IMAGE_BACKBONE)
        p.add_argument("--text_model", type=str, default=CFG.TEXT_BACKBONE)

    train_p = sub.add_parser("train")
    add_common(train_p)
    train_p.add_argument("--valid_json_dir", type=str, default=None)
    train_p.add_argument("--train_csv", type=str, default=None, help="CSV produced by prep.py for training")
    train_p.add_argument("--valid_csv", type=str, default=None, help="CSV produced by prep.py for validation")
    train_p.add_argument("--csv_root", type=str, default=None, help="Root directory that contains CSV image paths")
    train_p.add_argument("--epochs", type=int, default=CFG.EPOCHS)
    train_p.add_argument("--lr", type=float, default=CFG.LEARNING_RATE)
    train_p.add_argument("--weight_decay", type=float, default=0.01)
    train_p.add_argument("--warmup_ratio", type=float, default=0.05)
    train_p.add_argument("--grad_accum", type=int, default=1)
    train_p.add_argument("--max_steps", type=int, default=-1)
    train_p.add_argument("--output_dir", type=str, default=CFG.OUTPUT_DIR)
    train_p.add_argument("--logging_steps", type=int, default=50)
    train_p.add_argument("--save_total_limit", type=int, default=3)
    train_p.add_argument("--save_strategy", type=str, default="epoch", choices=["epoch", "steps"])
    train_p.add_argument("--eval_strategy", type=str, default="epoch", choices=["no", "epoch", "steps"])
    train_p.add_argument("--resume_from", type=str, default=None)

    eval_p = sub.add_parser("eval")
    add_common(eval_p)
    eval_p.add_argument("--csv_path", type=str, default=None, help="CSV file to evaluate (prep.py output)")
    eval_p.add_argument("--csv_root", type=str, default=None, help="Root directory for CSV image paths")
    eval_p.add_argument("--ckpt_dir", type=str, required=True)
    eval_p.add_argument("--out_csv", type=str, default=CFG.EVAL_CSV)

    pred_p = sub.add_parser("predict")
    add_common(pred_p)
    pred_p.add_argument("--csv_path", type=str, default=None, help="CSV file to predict over (prep.py output)")
    pred_p.add_argument("--csv_root", type=str, default=None, help="Root directory for CSV image paths")
    pred_p.add_argument("--ckpt_dir", type=str, required=True)
    pred_p.add_argument("--out_csv", type=str, default=CFG.PRED_CSV)

    zip_p = sub.add_parser("zip")
    zip_p.add_argument("--csv", type=str, required=True)
    zip_p.add_argument("--out_zip", type=str, default=CFG.SUBMISSION_ZIP)

    return parser.parse_args()


def main():
    seed_everything(CFG.SEED)
    args = get_args()
    if args.cmd == "train":
        train_loop(args)
    elif args.cmd == "eval":
        evaluate_loop(args)
    elif args.cmd == "predict":
        predict_loop(args)
    elif args.cmd == "zip":
        zip_submission(args.csv, args.out_zip)
    else:
        raise ValueError(args.cmd)


if __name__ == "__main__":
    main()

