"""
多模态融合器模块。

本模块实现顶层多模态融合架构，
用于将文本、音频、手势和标注四种模态的特征进行语义对齐和融合。

核心功能:
    1. 运行每个模态特定的编码器
    2. 将所有可用模态特征连接为键/值(Key/Value)序列
    3. 使用文本特征作为查询(Query)，应用跨模态注意力机制
    4. 产生统一的语义表示向量
    5. 预测模态置信度（modality_bias）

技术特点:
    - 支持单模态和多模态输入的动态融合
    - 模态置信度预测头评估输入理解质量
    - 离线权重路径重定向，避免网络依赖
    - 完整的异常处理和日志记录

参考文献:
    Vaswani et al., 2017. "Attention Is All You Need"
    Radford et al., 2021. "Learning Transferable Visual Models From Natural Language Supervision"
"""

# ==================== 标准库和第三方库导入 ====================
import logging  # 日志记录模块
import os        # 操作系统接口模块
from typing import Dict, Optional, List, Tuple, Union  # 类型提示支持

# ==================== PyTorch深度学习框架 ====================
import torch        # 张量计算核心库
import torch.nn as nn  # 神经网络模块

# ==================== 项目模块导入 ====================
from models.encoders.audio_encoder import AudioEncoder      # 音频编码器模块
from models.encoders.gesture_encoder import GestureEncoder  # 手势编码器模块
from models.encoders.annotation_encoder import AnnotationEncoder  # 标注编码器模块
from models.encoders.text_encoder import TextEncoder        # 文本编码器模块
from models.fusion.cross_modal_attention import CrossModalAttention  # 跨模态注意力模块

# ==================== 工具模块导入 ====================
from utils.offline_config import get_weights_dir  # 权重路径配置工具

# ==================== 日志配置 ====================
logger = logging.getLogger("experiment")  # 获取实验日志记录器


class _GateActivation(nn.Module):
    """
    门控函数的可配置激活模块（A1 消融实验用）。

    支持四种门控激活，用于对比不同门控函数对多模态融合的影响：
        - "sigmoid": 标准软门控，输出 [0,1]
        - "tanh":    输出 [-1,1]，允许负向调制（默认，论文采用）
        - "relu":    输出 [0,∞)，无界门控
        - "gumbel":  Gumbel-sigmoid 硬门控，训练时加 Gumbel 噪声软采样，
                     推理时退化为 0/1 硬门控

    该模块不含可学习参数，作为 modality_gate 的最后一层，
    接收门控 logit（Linear 输出）并施加对应激活。由于无参数，
    与现有 checkpoint 的 state_dict 完全兼容（strict=False 加载不受影响）。
    """

    def __init__(self, mode: str = "sigmoid", tau: float = 1.0):
        super().__init__()
        if mode not in ("sigmoid", "tanh", "relu", "gumbel"):
            raise ValueError(f"Unknown gate activation: {mode}")
        self.mode = mode
        self.tau = tau

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "sigmoid":
            return torch.sigmoid(x)
        if self.mode == "tanh":
            return torch.tanh(x)
        if self.mode == "relu":
            return torch.relu(x)
        # gumbel: Gumbel-sigmoid
        if self.training:
            u = torch.rand_like(x).clamp(1e-6, 1.0 - 1e-6)
            gumbel = -torch.log(-torch.log(u))
            return torch.sigmoid((x + gumbel) / self.tau)
        # 推理时使用硬门控
        return (torch.sigmoid(x) > 0.5).float()


class MultimodalFuser(nn.Module):
    """
    顶层多模态融合模块。

    该模块是整个多模态系统的核心，负责将不同模态的信息进行语义对齐和融合。

    工作流程:
        1. 对每种可用模态调用对应的编码器
        2. 将所有编码后的特征拼接为Key/Value序列
        3. 使用文本嵌入作为Query，通过跨模态注意力机制进行融合
        4. 输出统一的语义表示向量
        5. 可选地预测模态置信度

    关键特性:
        - 动态模态支持：自动检测并处理可用的模态组合
        - 置信度评估：通过modality_bias_head评估输入理解质量
        - 离线兼容：支持本地权重路径重定向
        - 错误处理：完善的异常检测和日志记录
    """

    def __init__(
        self,
        fusion_dim: int = 256,
        attention_heads: int = 8,
        attention_layers: int = 2,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        audio_model: str = "facebook/wav2vec2-base-960h",
        gesture_backbone: str = "resnet18",
        text_model: str = "all-MiniLM-L6-v2", # 简化名字
        gate_activation: str = "tanh",  # A1: 门控函数(tanh为最优，sigmoid/tanh/relu/gumbel可选)
    ):
        """
        初始化多模态融合器。

        Args:
            fusion_dim: 融合维度（默认256，A2实验确认最优）
            attention_heads: 注意力头数（默认8）
            attention_layers: 注意力层数（默认2）
            ffn_dim: 前馈网络隐藏层维度（默认2048）
            dropout: Dropout比率（默认0.1）
            audio_model: 音频模型名称或路径
            gesture_backbone: 手势编码器骨干网络类型
            text_model: 文本模型名称或路径
        """
        super().__init__()  # 调用父类构造函数
        self.fusion_dim = fusion_dim  # 保存融合维度

        # --- 核心修改：强制路径重定向，绕过联网检查 ---
        # 在服务器环境中获取权重存放目录
        weights_base = get_weights_dir(server_mode=True)

        # 1. 文本模型路径重定向
        # 提取模型文件夹名
        text_folder_name = text_model.split('/')[-1]
        local_text_path = os.path.join(weights_base, text_folder_name)

        if os.path.exists(local_text_path):
            logger.info(f"[Fuser] Redirecting Text Model to local path: {local_text_path}")
            actual_text_model = local_text_path
        else:
            logger.warning(f"[Fuser] Local text model not found at {local_text_path}. Using name: {text_model}")
            actual_text_model = text_model

        # 2. 音频模型路径重定向
        audio_folder_name = audio_model.split('/')[-1]
        local_audio_path = os.path.join(weights_base, audio_folder_name)

        if os.path.exists(local_audio_path):
            logger.info(f"[Fuser] Redirecting Audio Model to local path: {local_audio_path}")
            actual_audio_model = local_audio_path
        else:
            actual_audio_model = audio_model

        # --- 初始化子模块 ---
        # 初始化各模态编码器，使用重定向后的路径
        self.audio_encoder = AudioEncoder(model_name=actual_audio_model, proj_dim=fusion_dim)
        self.gesture_encoder = GestureEncoder(backbone=gesture_backbone, proj_dim=fusion_dim)
        self.annotation_encoder = AnnotationEncoder(proj_dim=fusion_dim)
        self.text_encoder = TextEncoder(model_name=actual_text_model, proj_dim=fusion_dim)

        # 初始化跨模态注意力模块
        self.cross_attention = CrossModalAttention(
            d_model=fusion_dim,
            n_heads=attention_heads,
            n_layers=attention_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )

        # 输出投影层：增强融合特征
        self.output_proj = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),  # 线性变换
            nn.LayerNorm(fusion_dim),  # 层归一化
        )

        # === MGHA: 模态门控异构特征装配 (Modality-Gated Heterogeneous Assembly) ===
        # 1. 可学习模态类型嵌入：编码每种模态在UAV规划上下文中的固有语义角色
        #    论文公式: f̃_m = f_m ⊙ σ(e_m)  (Eq. type_embed)
        self.modality_type_embeddings = nn.ParameterDict({
            "text":  nn.Parameter(torch.randn(fusion_dim) * 0.02),
            "audio": nn.Parameter(torch.randn(fusion_dim) * 0.02),
            "gesture": nn.Parameter(torch.randn(fusion_dim) * 0.02),
            "annot": nn.Parameter(torch.randn(fusion_dim) * 0.02),
        })

        # 2. 自适应模态门控：根据文本指令条件性地抑制无关模态
        #    论文公式: g_m = σ(w_g^T · ReLU(W_g · [f_text; f_m]))  (Eq. gate)
        #    A1: 门控函数可配置（sigmoid/tanh/relu/gumbel），默认 tanh
        self.gate_activation = gate_activation
        self.modality_gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),  # W_g: 输入为[text; f_m]拼接
            nn.ReLU(),
            nn.Linear(fusion_dim, 1),                 # w_g: 映射到标量 logit
            _GateActivation(gate_activation),         # 可配置门控激活，输出门控值
        )

        # === 新增：模态置信度预测头 ===
        # 输入：融合向量 → 输出：置信度标量 [0, 1]
        # 学术依据：模型应该能够评估自身对输入的理解程度
        self.modality_bias_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 4),  # 降维
            nn.ReLU(),  # 非线性激活
            nn.Linear(fusion_dim // 4, 1),  # 映射到标量
            nn.Sigmoid(),  # Sigmoid激活，输出范围[0,1]
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
        融合所有可用模态为单一语义表示。

        参数说明:
            text_emb: 文本嵌入张量，shape为(B, D)或(B, 1, D)
            audio_values: 音频波形张量，shape为(B, 48000)
            gesture_images: 手势图像张量，shape为(B, 3, 224, 224)
            annotation_images: 标注图像张量，shape为(B, 3, 224, 224)
            return_bias: 是否返回模态置信度

        Returns:
            fused: 融合后的表示，shape为(B, D)
            bias: 模态置信度，shape为(B, 1)，仅当return_bias=True时返回

        工作流程:
            1. 检测并编码所有可用模态
            2. 统计实际输入的模态数量
            3. 拼接所有模态特征为Key/Value序列
            4. 使用文本特征作为Query进行跨模态注意力融合
            5. 应用输出投影增强特征
            6. 可选地预测模态置信度
        """
        kv_features: List[torch.Tensor] = []  # 存储所有模态特征
        gated_features: List[torch.Tensor] = []  # 存储MGHA门控后的特征
        modal_count = 0  # 统计实际输入的模态数量

        # --- 编码可用模态 ---
        text_feat = None
        if text_emb is not None:
            text_feat = self.text_encoder(text_emb)        # (B, D)
            modal_count += 1

        if audio_values is not None:
            audio_feat = self.audio_encoder(audio_values)   # (B, D)
            # MGHA Step 1: 模态类型嵌入调制  f̃_m = f_m ⊙ σ(e_m)
            type_scale = torch.sigmoid(self.modality_type_embeddings["audio"])  # (D,)
            audio_typed = audio_feat * type_scale  # (B, D)
            # MGHA Step 2: 自适应门控  g_m = Gate([f_text; f_m])
            if text_feat is not None:
                gate_input = torch.cat([text_feat, audio_feat], dim=-1)  # (B, 2D)
                g_audio = self.modality_gate(gate_input)  # (B, 1)
                audio_gated = g_audio * audio_typed  # (B, D)
            else:
                audio_gated = audio_typed
            gated_features.append(audio_gated.unsqueeze(1))  # (B, 1, D)
            modal_count += 1

        if gesture_images is not None:
            gest_feat = self.gesture_encoder(gesture_images) # (B, D)
            # MGHA Step 1: 模态类型嵌入调制
            type_scale = torch.sigmoid(self.modality_type_embeddings["gesture"])  # (D,)
            gest_typed = gest_feat * type_scale
            # MGHA Step 2: 自适应门控
            if text_feat is not None:
                gate_input = torch.cat([text_feat, gest_feat], dim=-1)
                g_gest = self.modality_gate(gate_input)  # (B, 1)
                gest_gated = g_gest * gest_typed
            else:
                gest_gated = gest_typed
            gated_features.append(gest_gated.unsqueeze(1))
            modal_count += 1

        if annotation_images is not None:
            annot_feat = self.annotation_encoder(annotation_images)  # (B, D)
            # MGHA Step 1: 模态类型嵌入调制
            type_scale = torch.sigmoid(self.modality_type_embeddings["annot"])  # (D,)
            annot_typed = annot_feat * type_scale
            # MGHA Step 2: 自适应门控
            if text_feat is not None:
                gate_input = torch.cat([text_feat, annot_feat], dim=-1)
                g_annot = self.modality_gate(gate_input)  # (B, 1)
                annot_gated = g_annot * annot_typed
            else:
                annot_gated = annot_typed
            gated_features.append(annot_gated.unsqueeze(1))
            modal_count += 1

        # 文本特征也做类型嵌入调制（但不做门控，因为文本是主模态）
        if text_feat is not None:
            type_scale = torch.sigmoid(self.modality_type_embeddings["text"])  # (D,)
            text_typed = text_feat * type_scale
            gated_features.insert(0, text_typed.unsqueeze(1))  # 文本放在首位

        if not gated_features:
            raise ValueError("At least one modality must be provided.")

        # MGHA Step 3: 异构装配 — 将门控后的特征组装为 K/V 矩阵
        # 论文公式: K = V = [f̂_1; f̂_2; ...; f̂_M]  (Eq. assembly)
        kv = torch.cat(gated_features, dim=1)  # (B, num_modalities, D)

        # Query = 文本特征（如果存在），否则使用第一个可用模态
        if text_feat is not None:
            query = text_typed.unsqueeze(1)  # (B, 1, D) — 使用类型嵌入后的文本
        else:
            query = gated_features[0]

        # 跨模态注意力融合
        fused = self.cross_attention(query, kv, kv, mask=None)  # (B, 1, D)
        fused = fused.squeeze(1)                                 # (B, D)
        fused = self.output_proj(fused)

        if return_bias:
            # 预测模态置信度
            bias = self.modality_bias_head(fused)  # (B, 1)
            return fused, bias
        else:
            return fused

    def get_modality_bias(self, fused: torch.Tensor) -> torch.Tensor:
        """
        从融合向量预测模态置信度。

        这是一个便捷方法，用于单独调用模态置信度预测头。

        Args:
            fused: 融合后的表示张量，shape为(B, D)

        Returns:
            bias: 模态置信度张量，shape为(B, 1)

        Note:
            该方法等价于直接调用self.modality_bias_head(fused)
        """
        return self.modality_bias_head(fused)