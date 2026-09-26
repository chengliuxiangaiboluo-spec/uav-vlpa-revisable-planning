"""
多模态融合基线对比模块。

本模块实现4种与MultimodalFuser（Cross-Modal Attention + MGHA）对照的多模态融合方法，
均基于"文献参考"文件夹中2024-2026年最新UAV多模态论文设计。

基线方法（均引用用户文献参考中的最新论文）：
    1. AFFNetFuser   — 自适应细粒度融合 (Tang et al., 2026, IEEE TIP)
    2. MAFTNetFuser  — Transformer自适应融合 (Liu et al., 2026, IEEE Sensors)
    3. SCALFuser     — 语义一致自适应对齐 (Chen et al., 2026, IEEE JSTARS)
    4. LPANetFuser   — LLM引导渐进式对齐 (Wu et al., 2026, IEEE TIP)

参考文献（均为用户"文献参考"文件夹中的论文）：
    [1] Tang et al., 2026. "Adaptive Fine-Grained Fusion Network for
        Multimodal UAV Object Detection" — IEEE TIP
        文件: 文献参考/UAV多模态导航文献/Adaptive_Fine-Grained_Fusion_Network_for_Multimodal_UAV_Object_Detection.pdf
    [2] Liu et al., 2026. "MAFTNet: Multimodal Adaptive Fusion-Based
        Transformer Network" — IEEE Sensors Journal
        文件: 文献参考/UAV多模态导航文献/MAFTNet_Multimodal_Adaptive_Fusion-Based_Transformer_Network_for_Infrared_and_Visible_Image_UAV_Object_Detection.pdf
    [3] Chen et al., 2026. "SCAL: A Semantic-Consistent Adaptive
        Alignment Learning Framework" — IEEE JSTARS
        文件: 文献参考/架构/SCAL_A_Semantic-Consistent_Adaptive_Alignment_Learning_Framework_for_UAV_Remote_Sensing_Cross-Modal_Retrieval.pdf
    [4] Wu et al., 2026. "Large Language Model Guided Progressive Feature
        Alignment for Multimodal UAV Object Detection" — IEEE TIP
        文件: 文献参考/UAV多模态导航文献/Large_Language_Model_Guided_Progressive_Feature_Alignment_for_Multimodal_UAV_Object_Detection.pdf
"""

import logging
import os
from typing import Dict, Optional, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoders.audio_encoder import AudioEncoder
from models.encoders.gesture_encoder import GestureEncoder
from models.encoders.annotation_encoder import AnnotationEncoder
from models.encoders.text_encoder import TextEncoder
from utils.offline_config import get_weights_dir

logger = logging.getLogger("experiment")


def _build_encoders(fusion_dim, audio_model, gesture_backbone, text_model):
    """构建四种模态编码器（与MultimodalFuser完全一致），保证公平对比。"""
    weights_base = get_weights_dir(server_mode=True)

    text_folder_name = text_model.split('/')[-1]
    local_text_path = os.path.join(weights_base, text_folder_name)
    actual_text_model = local_text_path if os.path.exists(local_text_path) else text_model

    audio_folder_name = audio_model.split('/')[-1]
    local_audio_path = os.path.join(weights_base, audio_folder_name)
    actual_audio_model = local_audio_path if os.path.exists(local_audio_path) else audio_model

    audio_encoder = AudioEncoder(model_name=actual_audio_model, proj_dim=fusion_dim)
    gesture_encoder = GestureEncoder(backbone=gesture_backbone, proj_dim=fusion_dim)
    annotation_encoder = AnnotationEncoder(proj_dim=fusion_dim)
    text_encoder = TextEncoder(model_name=actual_text_model, proj_dim=fusion_dim)

    return text_encoder, audio_encoder, gesture_encoder, annotation_encoder


def _build_bias_head(fusion_dim):
    return nn.Sequential(
        nn.Linear(fusion_dim, fusion_dim // 4),
        nn.ReLU(),
        nn.Linear(fusion_dim // 4, 1),
        nn.Sigmoid(),
    )


# ==================== 基线1: AFFNetFuser ====================

class AFFNetFuser(nn.Module):
    """
    自适应细粒度融合基线 (Tang et al., 2026, IEEE TIP)。

    策略（简化实现AFFNet的LFCMF模块）：
        1. 各模态分别编码
        2. 局部特征一致性门控：计算各模态与文本的相似度权重
        3. 加权融合而非简单拼接
        4. 输出投影

    文件: 文献参考/UAV多模态导航文献/Adaptive_Fine-Grained_Fusion_Network_for_Multimodal_UAV_Object_Detection.pdf
    """

    def __init__(self, fusion_dim=256, audio_model="facebook/wav2vec2-base-960h",
                 gesture_backbone="resnet18", text_model="all-MiniLM-L6-v2",
                 dropout=0.1):
        super().__init__()
        self.fusion_dim = fusion_dim
        (text_encoder, audio_encoder, gesture_encoder,
         annotation_encoder) = _build_encoders(fusion_dim, audio_model, gesture_backbone, text_model)
        self.text_encoder = text_encoder
        self.audio_encoder = audio_encoder
        self.gesture_encoder = gesture_encoder
        self.annotation_encoder = annotation_encoder

        self.consistency_gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim // 4),
            nn.ReLU(),
            nn.Linear(fusion_dim // 4, 1),
            nn.Sigmoid(),
        )
        self.fuse_proj = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
        )
        self.output_proj = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
        )
        self.modality_bias_head = _build_bias_head(fusion_dim)

    def forward(self, text_emb=None, audio_values=None, gesture_images=None,
                annotation_images=None, return_bias=False):
        batch_size = 1
        if text_emb is not None:
            batch_size = text_emb.size(0)
        elif audio_values is not None:
            batch_size = audio_values.size(0)
        device = self.fuse_proj[0].weight.device
        zero = torch.zeros(batch_size, self.fusion_dim, device=device)

        text_feat = self.text_encoder(text_emb) if text_emb is not None else zero
        audio_feat = self.audio_encoder(audio_values) if audio_values is not None else zero
        gest_feat = self.gesture_encoder(gesture_images) if gesture_images is not None else zero
        annot_feat = self.annotation_encoder(annotation_images) if annotation_images is not None else zero

        modal_feats = [audio_feat, gest_feat, annot_feat]
        weighted_sum = text_feat.clone()
        n_active = 1
        for m_feat in modal_feats:
            gate_input = torch.cat([text_feat, m_feat], dim=-1)
            weight = self.consistency_gate(gate_input)
            weighted_sum = weighted_sum + weight * m_feat
            n_active += 1

        fused = weighted_sum / max(n_active, 1)
        fused = self.fuse_proj(fused)
        fused = self.output_proj(fused)

        if return_bias:
            bias = self.modality_bias_head(fused)
            return fused, bias
        return fused

    def get_modality_bias(self, fused):
        return self.modality_bias_head(fused)


# ==================== 基线2: MAFTNetFuser ====================

class MAFTNetFuser(nn.Module):
    """
    Transformer自适应融合基线 (Liu et al., 2026, IEEE Sensors)。

    策略（简化实现MAFTNet的DB-MST+CDCR+HFFE）：
        1. 双分支模态协同：text与各模态做cross-attention
        2. 条件驱动通道重构：通道加权
        3. 混合特征融合编码器

    文件: 文献参考/UAV多模态导航文献/MAFTNet_Multimodal_Adaptive_Fusion-Based_Transformer_Network_for_Infrared_and_Visible_Image_UAV_Object_Detection.pdf
    """

    def __init__(self, fusion_dim=256, audio_model="facebook/wav2vec2-base-960h",
                 gesture_backbone="resnet18", text_model="all-MiniLM-L6-v2",
                 dropout=0.1, n_heads=4):
        super().__init__()
        self.fusion_dim = fusion_dim
        (text_encoder, audio_encoder, gesture_encoder,
         annotation_encoder) = _build_encoders(fusion_dim, audio_model, gesture_backbone, text_model)
        self.text_encoder = text_encoder
        self.audio_encoder = audio_encoder
        self.gesture_encoder = gesture_encoder
        self.annotation_encoder = annotation_encoder

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(fusion_dim)
        self.channel_gate = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 4),
            nn.GELU(),
            nn.Linear(fusion_dim // 4, fusion_dim),
            nn.Sigmoid(),
        )
        self.hffe = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
        )
        self.output_proj = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
        )
        self.modality_bias_head = _build_bias_head(fusion_dim)

    def forward(self, text_emb=None, audio_values=None, gesture_images=None,
                annotation_images=None, return_bias=False):
        if text_emb is None:
            raise ValueError("MAFTNetFuser requires text as anchor modality.")
        batch_size = text_emb.size(0)
        device = self.output_proj[0].weight.device
        zero = torch.zeros(batch_size, self.fusion_dim, device=device)

        text_feat = self.text_encoder(text_emb)
        other_feats = []
        if audio_values is not None:
            other_feats.append(self.audio_encoder(audio_values))
        if gesture_images is not None:
            other_feats.append(self.gesture_encoder(gesture_images))
        if annotation_images is not None:
            other_feats.append(self.annotation_encoder(annotation_images))

        if not other_feats:
            fused = self.hffe(text_feat)
        else:
            other_stack = torch.stack(other_feats, dim=1)
            text_q = text_feat.unsqueeze(1)
            attn_out, _ = self.cross_attn(text_q, other_stack, other_stack)
            attn_out = attn_out.squeeze(1)
            attn_out = self.norm1(text_feat + attn_out)
            gate = self.channel_gate(attn_out)
            channel_recon = attn_out * gate
            fused = self.hffe(channel_recon)

        fused = self.output_proj(fused)
        if return_bias:
            bias = self.modality_bias_head(fused)
            return fused, bias
        return fused

    def get_modality_bias(self, fused):
        return self.modality_bias_head(fused)


# ==================== 基线3: SCALFuser ====================

class SCALFuser(nn.Module):
    """
    语义一致自适应对齐融合基线 (Chen et al., 2026, IEEE JSTARS)。

    策略（简化实现SCAL的CSTLA+CSABG）：
        1. 置信度缩放的文本条件局部对齐
        2. 背景抑制的上下文语义对齐（注意力池化）

    文件: 文献参考/架构/SCAL_A_Semantic-Consistent_Adaptive_Alignment_Learning_Framework_for_UAV_Remote_Sensing_Cross-Modal_Retrieval.pdf
    """

    def __init__(self, fusion_dim=256, audio_model="facebook/wav2vec2-base-960h",
                 gesture_backbone="resnet18", text_model="all-MiniLM-L6-v2",
                 dropout=0.1):
        super().__init__()
        self.fusion_dim = fusion_dim
        (text_encoder, audio_encoder, gesture_encoder,
         annotation_encoder) = _build_encoders(fusion_dim, audio_model, gesture_backbone, text_model)
        self.text_encoder = text_encoder
        self.audio_encoder = audio_encoder
        self.gesture_encoder = gesture_encoder
        self.annotation_encoder = annotation_encoder

        self.confidence_net = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim // 4),
            nn.ReLU(),
            nn.Linear(fusion_dim // 4, 1),
            nn.Sigmoid(),
        )
        self.align_proj = nn.ModuleList([
            nn.Linear(fusion_dim, fusion_dim) for _ in range(3)
        ])
        self.context_attn = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 4),
            nn.Tanh(),
            nn.Linear(fusion_dim // 4, 1),
            nn.Softmax(dim=-1),
        )
        self.output_proj = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
        )
        self.modality_bias_head = _build_bias_head(fusion_dim)

    def forward(self, text_emb=None, audio_values=None, gesture_images=None,
                annotation_images=None, return_bias=False):
        if text_emb is None:
            raise ValueError("SCALFuser requires text as anchor modality.")
        batch_size = text_emb.size(0)
        device = self.output_proj[0].weight.device
        zero = torch.zeros(batch_size, self.fusion_dim, device=device)

        text_feat = self.text_encoder(text_emb)
        aligned_feats = []
        idx = 0
        for available, encoder, input_tensor in [
            (audio_values is not None, self.audio_encoder, audio_values),
            (gesture_images is not None, self.gesture_encoder, gesture_images),
            (annotation_images is not None, self.annotation_encoder, annotation_images),
        ]:
            if available:
                feat = encoder(input_tensor)
                aligned = self.align_proj[idx](feat)
                conf_input = torch.cat([text_feat, aligned], dim=-1)
                confidence = self.confidence_net(conf_input)
                aligned = confidence * aligned
                aligned_feats.append(aligned)
            idx += 1

        if not aligned_feats:
            fused = self.output_proj(text_feat)
        else:
            feat_stack = torch.stack(aligned_feats, dim=1)
            feat_with_text = torch.cat([text_feat.unsqueeze(1), feat_stack], dim=1)
            attn_weights = self.context_attn(feat_with_text)
            weighted = (feat_with_text * attn_weights).sum(dim=1)
            fused = self.output_proj(weighted)

        if return_bias:
            bias = self.modality_bias_head(fused)
            return fused, bias
        return fused

    def get_modality_bias(self, fused):
        return self.modality_bias_head(fused)


# ==================== 基线4: LPANetFuser ====================

class LPANetFuser(nn.Module):
    """
    LLM引导渐进式对齐融合基线 (Wu et al., 2026, IEEE TIP)。

    策略（简化实现LPANet的渐进式语义-空间对齐）：
        1. 第一阶段: 语义对齐（模态特征投影到文本语义空间）
        2. 第二阶段: 空间对齐（cross-attention空间重组）

    文件: 文献参考/UAV多模态导航文献/Large_Language_Model_Guided_Progressive_Feature_Alignment_for_Multimodal_UAV_Object_Detection.pdf
    """

    def __init__(self, fusion_dim=256, audio_model="facebook/wav2vec2-base-960h",
                 gesture_backbone="resnet18", text_model="all-MiniLM-L6-v2",
                 dropout=0.1):
        super().__init__()
        self.fusion_dim = fusion_dim
        (text_encoder, audio_encoder, gesture_encoder,
         annotation_encoder) = _build_encoders(fusion_dim, audio_model, gesture_backbone, text_model)
        self.text_encoder = text_encoder
        self.audio_encoder = audio_encoder
        self.gesture_encoder = gesture_encoder
        self.annotation_encoder = annotation_encoder

        self.semantic_align = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_dim, fusion_dim),
            )
            for _ in range(3)
        ])
        self.spatial_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=4,
            dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(fusion_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
        )
        self.modality_bias_head = _build_bias_head(fusion_dim)

    def forward(self, text_emb=None, audio_values=None, gesture_images=None,
                annotation_images=None, return_bias=False):
        if text_emb is None:
            raise ValueError("LPANetFuser requires text as anchor modality.")
        batch_size = text_emb.size(0)
        device = self.output_proj[0].weight.device
        zero = torch.zeros(batch_size, self.fusion_dim, device=device)

        text_feat = self.text_encoder(text_emb)
        aligned_feats = []
        idx = 0
        for available, encoder, input_tensor in [
            (audio_values is not None, self.audio_encoder, audio_values),
            (gesture_images is not None, self.gesture_encoder, gesture_images),
            (annotation_images is not None, self.annotation_encoder, annotation_images),
        ]:
            if available:
                feat = encoder(input_tensor)
                aligned = self.semantic_align[idx](feat)
                aligned = aligned + text_feat  # 残差对齐
                aligned_feats.append(aligned)
            idx += 1

        if not aligned_feats:
            fused = self.output_proj(text_feat)
        else:
            feat_stack = torch.stack(aligned_feats, dim=1)
            text_q = text_feat.unsqueeze(1)
            attn_out, _ = self.spatial_attn(text_q, feat_stack, feat_stack)
            attn_out = attn_out.squeeze(1)
            fused = self.norm(text_feat + attn_out)
            fused = self.output_proj(fused)

        if return_bias:
            bias = self.modality_bias_head(fused)
            return fused, bias
        return fused

    def get_modality_bias(self, fused):
        return self.modality_bias_head(fused)


# ==================== 工厂函数 ====================

def get_baseline_fuser(name, fusion_dim=256, **kwargs):
    """根据名称获取基线融合器实例。"""
    name_lower = name.lower().replace("_", "").replace("-", "")
    if name_lower == "affnet":
        return AFFNetFuser(fusion_dim=fusion_dim, **kwargs)
    elif name_lower == "maftnet":
        return MAFTNetFuser(fusion_dim=fusion_dim, **kwargs)
    elif name_lower == "scal":
        return SCALFuser(fusion_dim=fusion_dim, **kwargs)
    elif name_lower == "lpanet":
        return LPANetFuser(fusion_dim=fusion_dim, **kwargs)
    else:
        raise ValueError(
            f"Unknown baseline fuser: {name}. Supported: affnet, maftnet, scal, lpanet"
        )


def list_baseline_fusers():
    """返回所有可用基线融合器的(name, description)列表。"""
    return [
        ("affnet", "自适应细粒度融合 AFFNet (Tang et al., 2026, IEEE TIP)"),
        ("maftnet", "Transformer自适应融合 MAFTNet (Liu et al., 2026, IEEE Sensors)"),
        ("scal", "语义一致自适应对齐 SCAL (Chen et al., 2026, IEEE JSTARS)"),
        ("lpanet", "LLM引导渐进式对齐 LPANet (Wu et al., 2026, IEEE TIP)"),
    ]
