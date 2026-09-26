"""
简单融合器模块 - 用于消融实验B6组。

将跨模态注意力机制替换为简单的拼接/平均融合，
验证跨模态注意力对系统性能的独立贡献。

融合策略：
    1. 各模态分别编码
    2. 将所有模态特征沿特征维度拼接
    3. 通过平均池化得到统一表示
    4. 输出投影+LayerNorm

对照：MultimodalFuser 使用跨模态注意力（Query=文本，KV=全模态）
"""

import logging
from typing import Optional, List, Tuple, Union

import torch
import torch.nn as nn

from models.encoders.audio_encoder import AudioEncoder
from models.encoders.gesture_encoder import GestureEncoder
from models.encoders.annotation_encoder import AnnotationEncoder
from models.encoders.text_encoder import TextEncoder
from utils.offline_config import get_weights_dir

import os

logger = logging.getLogger("experiment")


class SimpleConcatFuser(nn.Module):
    """
    简单拼接融合器（消融实验B6专用）。

    将跨模态注意力替换为简单的平均池化融合，
    保留相同的编码器和输出接口，仅改变融合策略。
    """

    def __init__(
        self,
        fusion_dim: int = 256,
        audio_model: str = "facebook/wav2vec2-base-960h",
        gesture_backbone: str = "resnet18",
        text_model: str = "all-MiniLM-L6-v2",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.fusion_dim = fusion_dim

        weights_base = get_weights_dir(server_mode=True)

        # 文本模型路径重定向
        text_folder_name = text_model.split('/')[-1]
        local_text_path = os.path.join(weights_base, text_folder_name)
        actual_text_model = local_text_path if os.path.exists(local_text_path) else text_model

        # 音频模型路径重定向
        audio_folder_name = audio_model.split('/')[-1]
        local_audio_path = os.path.join(weights_base, audio_folder_name)
        actual_audio_model = local_audio_path if os.path.exists(local_audio_path) else audio_model

        # 初始化编码器（与 MultimodalFuser 相同）
        self.audio_encoder = AudioEncoder(model_name=actual_audio_model, proj_dim=fusion_dim)
        self.gesture_encoder = GestureEncoder(backbone=gesture_backbone, proj_dim=fusion_dim)
        self.annotation_encoder = AnnotationEncoder(proj_dim=fusion_dim)
        self.text_encoder = TextEncoder(model_name=actual_text_model, proj_dim=fusion_dim)

        # 简单融合：不使用跨模态注意力，仅使用线性投影+LayerNorm
        self.output_proj = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
        )

        # 模态置信度预测头（与 MultimodalFuser 保持接口一致）
        self.modality_bias_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 4),
            nn.ReLU(),
            nn.Linear(fusion_dim // 4, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        text_emb: Optional[torch.Tensor] = None,
        audio_values: Optional[torch.Tensor] = None,
        gesture_images: Optional[torch.Tensor] = None,
        annotation_images: Optional[torch.Tensor] = None,
        return_bias: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        简单拼接+平均池化融合。

        关键区别：不使用跨模态注意力，直接对编码特征求平均。
        """
        features: List[torch.Tensor] = []

        if text_emb is not None:
            text_feat = self.text_encoder(text_emb)  # (B, D)
            features.append(text_feat)

        if audio_values is not None:
            audio_feat = self.audio_encoder(audio_values)  # (B, D)
            features.append(audio_feat)

        if gesture_images is not None:
            gest_feat = self.gesture_encoder(gesture_images)  # (B, D)
            features.append(gest_feat)

        if annotation_images is not None:
            annot_feat = self.annotation_encoder(annotation_images)  # (B, D)
            features.append(annot_feat)

        if not features:
            raise ValueError("At least one modality must be provided.")

        # 简单平均融合（替代跨模态注意力）
        stacked = torch.stack(features, dim=0)  # (num_modalities, B, D)
        fused = stacked.mean(dim=0)  # (B, D)

        fused = self.output_proj(fused)

        if return_bias:
            bias = self.modality_bias_head(fused)
            return fused, bias
        else:
            return fused

    def get_modality_bias(self, fused: torch.Tensor) -> torch.Tensor:
        """从融合向量预测模态置信度（接口兼容）。"""
        return self.modality_bias_head(fused)
