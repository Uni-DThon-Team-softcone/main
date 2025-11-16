"""
CLIP Matching Model 정의

KoCLIP backbone + projection head를 사용한 CLIP 모델
- 학습 시: stage2_koclip.py에서 사용 (encode_text, encode_images)
- 추론 시: final.py에서 사용 (encode_text_tensor, encode_image_tensor)
"""

import os
from typing import List, Optional
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

# transformers (KoCLIP)
try:
    from transformers import AutoModel, AutoProcessor
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False
    AutoModel = None
    AutoProcessor = None


class CLIPMatchingModel(nn.Module):
    """
    KoCLIP backbone + projection head
    - backbone: Hugging Face KoCLIP model (frozen by default)
    - head: small linear layers + logit_scale
    
    두 가지 인터페이스 제공:
    1. 학습용: encode_text(texts: List[str]), encode_images(images: List[Image.Image])
    2. 추론용: encode_text_tensor(input_ids, attention_mask), encode_image_tensor(pixel_values)
    """

    def __init__(self, model_name: str = "koclip/koclip-base-pt", freeze_backbone: bool = True, device: str = "cuda"):
        super().__init__()
        if not _TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers is not available. Install: pip install transformers")

        self.device = torch.device(device)
        print(f"[KoCLIP] Loading model={model_name}")
        self.clip = AutoModel.from_pretrained(model_name).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)

        if freeze_backbone:
            for p in self.clip.parameters():
                p.requires_grad = False

        # KoCLIP의 임베딩 차원 확인
        # 임시로 텍스트와 이미지를 인코딩해서 차원 확인
        with torch.no_grad():
            test_text_inputs = self.processor(
                text=["test"],
                return_tensors="pt",
                padding=True
            ).to(self.device)
            test_image_inputs = self.processor(
                images=Image.new("RGB", (224, 224)),
                return_tensors="pt"
            ).to(self.device)
            
            # 텍스트 임베딩 차원 확인
            if hasattr(self.clip, 'get_text_features'):
                test_txt = self.clip.get_text_features(**test_text_inputs)
            else:
                test_outputs = self.clip(**test_text_inputs)
                if hasattr(test_outputs, 'text_embeds'):
                    test_txt = test_outputs.text_embeds
                else:
                    test_txt = test_outputs[0] if isinstance(test_outputs, tuple) else test_outputs.last_hidden_state[:, 0]
            
            # 이미지 임베딩 차원 확인
            if hasattr(self.clip, 'get_image_features'):
                test_img = self.clip.get_image_features(**test_image_inputs)
            else:
                test_outputs = self.clip(**test_image_inputs)
                if hasattr(test_outputs, 'image_embeds'):
                    test_img = test_outputs.image_embeds
                else:
                    test_img = test_outputs[0] if isinstance(test_outputs, tuple) else test_outputs.last_hidden_state[:, 0]
            
            # 두 차원이 같아야 함
            embed_dim = test_txt.shape[-1]
            assert test_img.shape[-1] == embed_dim, f"Text dim ({test_txt.shape[-1]}) != Image dim ({test_img.shape[-1]})"

        self.txt_proj = nn.Linear(embed_dim, embed_dim).to(self.device)
        self.img_proj = nn.Linear(embed_dim, embed_dim).to(self.device)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).to(self.device)

        # 학습 가능한 파라미터 확인
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"[KoCLIP] Total params: {total_params:,}, Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        print(f"[KoCLIP] Embedding dim: {embed_dim}")

    # ==================== 학습용 인터페이스 (stage2_koclip.py에서 사용) ====================

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """
        텍스트 문자열 리스트를 인코딩 (학습용)
        
        Args:
            texts: 텍스트 문자열 리스트
        
        Returns:
            정규화된 텍스트 임베딩 텐서
        """
        # 텍스트만 처리 (최적화: 배치 처리)
        text_inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77  # CLIP 일반적인 max length
        )
        # GPU로 이동
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        
        with torch.no_grad():
            # KoCLIP 모델의 텍스트 인코딩
            if hasattr(self.clip, 'get_text_features'):
                txt_feat = self.clip.get_text_features(**text_inputs)
            elif hasattr(self.clip, 'text_model'):
                text_outputs = self.clip.text_model(**text_inputs)
                txt_feat = text_outputs.pooler_output if hasattr(text_outputs, 'pooler_output') else text_outputs.last_hidden_state[:, 0]
                if hasattr(self.clip, 'text_projection'):
                    txt_feat = self.clip.text_projection(txt_feat)
            else:
                outputs = self.clip(**text_inputs)
                if hasattr(outputs, 'text_embeds'):
                    txt_feat = outputs.text_embeds
                elif hasattr(outputs, 'pooler_output'):
                    txt_feat = outputs.pooler_output
                else:
                    txt_feat = outputs[0][:, 0] if isinstance(outputs, tuple) else outputs.last_hidden_state[:, 0]
        
        txt_feat = F.normalize(txt_feat, dim=-1)
        txt_proj = self.txt_proj(txt_feat)
        txt_proj = F.normalize(txt_proj, dim=-1)
        return txt_proj

    def encode_images(self, images: List[Image.Image]) -> torch.Tensor:
        """
        PIL Image 리스트를 인코딩 (학습용)
        
        Args:
            images: PIL Image 리스트
        
        Returns:
            정규화된 이미지 임베딩 텐서
        """
        # 이미지만 처리 (최적화: 배치 처리)
        image_inputs = self.processor(
            images=images,
            return_tensors="pt",
            padding=True
        )
        # GPU로 이동
        image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}
        
        with torch.no_grad():
            # KoCLIP 모델의 이미지 인코딩
            if hasattr(self.clip, 'get_image_features'):
                img_feat = self.clip.get_image_features(**image_inputs)
            elif hasattr(self.clip, 'vision_model'):
                vision_outputs = self.clip.vision_model(**image_inputs)
                img_feat = vision_outputs.pooler_output if hasattr(vision_outputs, 'pooler_output') else vision_outputs.last_hidden_state[:, 0]
                if hasattr(self.clip, 'vision_projection'):
                    img_feat = self.clip.vision_projection(img_feat)
            else:
                outputs = self.clip(**image_inputs)
                if hasattr(outputs, 'image_embeds'):
                    img_feat = outputs.image_embeds
                elif hasattr(outputs, 'pooler_output'):
                    img_feat = outputs.pooler_output
                else:
                    img_feat = outputs[0][:, 0] if isinstance(outputs, tuple) else outputs.last_hidden_state[:, 0]
        
        img_feat = F.normalize(img_feat, dim=-1)
        img_proj = self.img_proj(img_feat)
        img_proj = F.normalize(img_proj, dim=-1)
        return img_proj

    def forward_batch(self, images: List[Image.Image], texts: List[str]):
        """
        배치 단위로 이미지와 텍스트를 인코딩하여 유사도 계산 (학습용)
        
        Args:
            images: PIL Image 리스트 (길이 B)
            texts: 텍스트 문자열 리스트 (길이 B)
        
        Returns:
            logits_per_text: (B, B), logits_per_image: (B, B)
        """
        img_emb = self.encode_images(images)  # (B, D)
        txt_emb = self.encode_text(texts)     # (B, D)

        # similarity matrix
        logit_scale = self.logit_scale.exp()
        logits_per_text = logit_scale * txt_emb @ img_emb.t()
        logits_per_image = logits_per_text.t()
        return logits_per_text, logits_per_image

    # ==================== 추론용 인터페이스 (final.py에서 사용) ====================

    def encode_text_tensor(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        이미 토크나이즈된 텍스트 텐서를 인코딩 (추론용)
        
        Args:
            input_ids: 토크나이즈된 input_ids 텐서
            attention_mask: attention_mask 텐서
        
        Returns:
            정규화된 텍스트 임베딩 텐서
        """
        with torch.no_grad():
            feat = self.clip.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        feat = F.normalize(self.txt_proj(feat), dim=-1)
        return feat

    def encode_image_tensor(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        이미 전처리된 이미지 텐서를 인코딩 (추론용)
        
        Args:
            pixel_values: 전처리된 이미지 픽셀 값 텐서
        
        Returns:
            정규화된 이미지 임베딩 텐서
        """
        with torch.no_grad():
            feat = self.clip.get_image_features(pixel_values=pixel_values)
        feat = F.normalize(self.img_proj(feat), dim=-1)
        return feat

