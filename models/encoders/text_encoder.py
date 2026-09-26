"""
文本编码器模块 - 使用Sentence-BERT (all-MiniLM-L6-v2)。

本模块实现基于预训练Sentence-BERT模型的文本编码器，
用于将自然语言指令编码为固定维度的特征向量。

主要功能:
    1. 加载预训练的Sentence-BERT模型
    2. 将文本指令编码为384维嵌入向量
    3. 通过线性投影层将特征映射到融合维度

使用模型:
    sentence-transformers/all-MiniLM-L6-v2
    - 输出维度: 384
    - 特点: 轻量级、高性能的句子嵌入模型

参考文献:
    Reimers & Gurevych, 2019. "Sentence-BERT: Sentence Embeddings
    using Siamese BERT-Networks"
"""

# 标准库导入
import logging  # 日志记录模块
from typing import List  # 类型提示支持
from pathlib import Path

# PyTorch深度学习框架
import torch  # 张量计算核心库
import torch.nn as nn  # 神经网络模块

# ==================== 离线环境配置 ====================
# 首先设置离线环境，避免网络请求
from utils.offline_config import get_weights_dir, setup_offline_environment
setup_offline_environment()

# ==================== HuggingFace补丁 ====================
# 应用HuggingFace客户端补丁，修复网络问题
from utils.hf_client_patch import patch_huggingface_client, safe_model_load
patch_huggingface_client()

# 获取日志记录器
logger = logging.getLogger("experiment")


class TextEncoder(nn.Module):
    """
    基于Sentence-BERT的文本编码器。

    该编码器使用预训练的Sentence-BERT模型将文本指令编码为
    固定维度的特征向量，并通过线性投影层映射到融合维度。

    属性:
        model_name (str): Sentence-BERT模型名称
        proj_dim (int): 投影后的特征维度
        _sbert: Sentence-BERT模型实例（延迟加载）
        projection (nn.Sequential): 线性投影层（384 → proj_dim）

    工作流程:
        1. 延迟加载Sentence-BERT模型
        2. 使用模型编码文本为384维向量
        3. 通过投影层映射到融合维度
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        proj_dim: int = 512,
    ):
        """
        初始化文本编码器。

        Args:
            model_name: Sentence-BERT模型名称或路径
            proj_dim: 投影后的特征维度（用于多模态融合）
        """
        super().__init__()  # 调用父类构造函数
        self.model_name = model_name  # 保存模型名称
        self.proj_dim = proj_dim  # 保存投影维度
        self._sbert = None  # 延迟加载标志，初始为None

        # MiniLM-L6-v2模型输出维度为384
        # 投影层：384维 → proj_dim维
        self.projection = nn.Sequential(
            nn.Linear(384, proj_dim),  # 线性变换
            nn.LayerNorm(proj_dim),  # 层归一化，稳定特征分布
        )
        logger.info("TextEncoder initialised (proj=%d).", proj_dim)

    def _load_model(self):
        """
        延迟加载Sentence-BERT模型。

        采用延迟加载策略，仅在首次使用时加载模型，
        避免不必要的内存占用和加载时间。

        加载流程:
            1. 检查模型是否已加载
            2. 使用安全加载器加载模型（带重试机制）
            3. 加载失败时使用随机嵌入作为后备方案
        """
        if self._sbert is not None:
            return  # 模型已加载，直接返回

        try:
            from sentence_transformers import SentenceTransformer

            # 定义加载函数
            def _load_sbert():
                # The server runs in strict offline mode.  Resolve the
                # repository's downloaded MiniLM directory explicitly rather
                # than asking SentenceTransformers to interpret the public
                # Hub identifier and silently falling back to random vectors.
                model_path = self.model_name
                if self.model_name == "sentence-transformers/all-MiniLM-L6-v2":
                    local_minilm = Path(get_weights_dir()) / "all-MiniLM-L6-v2"
                    if not (local_minilm / "config.json").is_file():
                        raise FileNotFoundError(
                            f"Offline MiniLM is missing: {local_minilm / 'config.json'}"
                        )
                    model_path = str(local_minilm)
                return SentenceTransformer(model_path)

            # 使用安全加载器，最多重试3次
            self._sbert = safe_model_load(_load_sbert, max_retries=3)

            if self._sbert is None:
                raise RuntimeError("Failed to load Sentence-BERT after retries")

            logger.info("Sentence-BERT loaded: %s", self.model_name)
        except Exception as exc:
            # 加载失败，使用后备方案
            logger.warning("Sentence-BERT unavailable (%s); using random embeddings.", exc)
            self._sbert = "fallback"  # 标记为后备模式

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """
        将文本字符串列表编码为嵌入向量。

        这是Sentence-BERT模型的直接接口，返回384维的句子嵌入。

        Args:
            texts: 文本指令列表，每个元素是一个字符串

        Returns:
            shape为(len(texts), 384)的张量，每行是一个句子的嵌入向量

        Note:
            - 如果模型加载失败，将返回随机嵌入
            - 输出维度固定为384（MiniLM-L6-v2的输出维度）
        """
        self._load_model()  # 确保模型已加载

        if self._sbert and self._sbert != "fallback":
            # 正常模式：使用Sentence-BERT编码
            embeddings = self._sbert.encode(texts, convert_to_tensor=True)
            return embeddings.float()  # 确保返回float类型
        else:
            # 后备模式：使用随机嵌入
            return torch.randn(len(texts), 384)

    def forward(self, text_embeddings: torch.Tensor) -> torch.Tensor:
        """
        将预计算的文本嵌入投影到融合维度。

        这是前向传播函数，将384维的Sentence-BERT嵌入
        投影到统一的多模态融合维度。

        Args:
            text_embeddings: 预计算的文本嵌入，shape为(batch, 384)

        Returns:
            投影后的特征张量，shape为(batch, proj_dim)

        Note:
            输入应先通过encode_texts方法获取，再传入本方法
        """
        return self.projection(text_embeddings)
