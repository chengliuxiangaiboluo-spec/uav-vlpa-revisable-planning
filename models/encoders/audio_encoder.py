"""
音频编码器模块 - 基于HuggingFace Wav2Vec2。

本模块实现基于预训练Wav2Vec2模型的音频编码器，
用于将原始音频波形编码为固定维度的特征向量。

主要功能:
    1. 加载预训练的Wav2Vec2模型
    2. 将16kHz音频波形编码为768维特征向量
    3. 通过线性投影层映射到融合维度
    4. 可选的ASR（自动语音识别）转录功能

使用模型:
    facebook/wav2vec2-base-960h
    - 隐藏层大小: 768
    - 预训练数据: 960小时LibriSpeech

参考文献:
    Baevski et al., 2020. "wav2vec 2.0: A Framework for
    Self-Supervised Learning of Speech Representations"
"""

# PyTorch深度学习框架
import torch  # 张量计算核心库
import torch.nn as nn  # 神经网络模块
import numpy as np  # 数值计算库
import logging  # 日志记录模块
from typing import Optional  # 可选类型提示

# ==================== 离线环境配置 ====================
# 首先设置离线环境，避免网络请求
from utils.offline_config import setup_offline_environment
setup_offline_environment()

# 获取日志记录器
logger = logging.getLogger("experiment")


class AudioEncoder(nn.Module):
    """
    基于Wav2Vec2的音频编码器。

    该编码器使用预训练的Wav2Vec2模型将原始音频波形编码为
    固定维度的特征向量，并通过投影层映射到融合维度。

    属性:
        model_name (str): Wav2Vec2模型名称或路径
        proj_dim (int): 投影后的特征维度
        _wav2vec2: Wav2Vec2模型实例（延迟加载）
        _processor: Wav2Vec2处理器（延迟加载）
        _freeze_base (bool): 是否冻结基础模型参数
        projection (nn.Linear): 线性投影层
        layer_norm (nn.LayerNorm): 层归一化

    工作流程:
        1. 延迟加载Wav2Vec2模型和处理器
        2. 对输入音频进行预处理（归一化、重采样）
        3. 通过Wav2Vec2提取声学特征
        4. 全局平均池化得到固定维度向量
        5. 投影到融合维度
    """

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base-960h",
        proj_dim: int = 512,
        freeze_base: bool = True,
    ):
        """
        初始化音频编码器。

        Args:
            model_name: Wav2Vec2模型名称或路径
            proj_dim: 投影后的特征维度（用于多模态融合）
            freeze_base: 是否冻结基础模型参数
                        True: 仅训练投影层（推荐）
                        False: 微调整个模型
        """
        super().__init__()  # 调用父类构造函数
        self.model_name = model_name  # 保存模型名称
        self.proj_dim = proj_dim  # 保存投影维度

        # 延迟加载，避免transformers库缺失时的导入错误
        self._wav2vec2 = None  # Wav2Vec2模型实例
        self._processor = None  # 音频处理器实例
        self._freeze_base = freeze_base  # 是否冻结基础参数

        # Wav2Vec2-base的隐藏层大小为768
        # 投影层：768维 → proj_dim维
        self.projection = nn.Linear(768, proj_dim)  # 线性投影
        self.layer_norm = nn.LayerNorm(proj_dim)  # 层归一化

    def _load_model(self):
        """
        延迟加载Wav2Vec2模型和处理器。

        采用延迟加载策略，仅在首次使用时加载模型。
        加载完成后，如果设置了freeze_base，将冻结基础模型参数。
        """
        if self._wav2vec2 is not None:
            return  # 模型已加载，直接返回

        from transformers import Wav2Vec2Model, Wav2Vec2Processor

        # 加载处理器：用于音频预处理
        self._processor = Wav2Vec2Processor.from_pretrained(self.model_name)
        # 加载模型：用于特征提取
        self._wav2vec2 = Wav2Vec2Model.from_pretrained(self.model_name)

        # 冻结基础模型参数（如果需要）
        if self._freeze_base:
            for param in self._wav2vec2.parameters():
                param.requires_grad = False  # 禁止梯度更新

        logger.info("Wav2Vec2 model loaded: %s", self.model_name)

    def forward(self, audio_values: torch.Tensor) -> torch.Tensor:
        """
        将原始音频波形编码为投影特征向量。

        前向传播流程:
            1. 加载模型并移至正确设备
            2. 通过Wav2Vec2提取声学特征
            3. 对时间维度进行全局平均池化
            4. 投影到融合维度并归一化

        Args:
            audio_values: 原始音频波形张量，shape为(batch, samples)
                         采样率应为16kHz

        Returns:
            投影后的特征张量，shape为(batch, proj_dim)

        Note:
            - 如果freeze_base=True，Wav2Vec2部分不计算梯度
            - 输入音频应在送入前通过preprocess方法预处理
        """
        self._load_model()  # 确保模型已加载
        device = audio_values.device  # 获取输入设备
        self._wav2vec2 = self._wav2vec2.to(device)  # 将模型移至正确设备

        # 根据是否冻结基础模型选择梯度计算模式
        with torch.no_grad() if self._freeze_base else torch.enable_grad():
            outputs = self._wav2vec2(audio_values)

        # 全局平均池化：对时间维度求平均
        hidden = outputs.last_hidden_state  # (batch, time, 768)
        pooled = hidden.mean(dim=1)          # (batch, 768)

        # 投影到融合维度并归一化
        projected = self.layer_norm(self.projection(pooled))  # (batch, proj_dim)
        return projected

    def preprocess(self, waveform: np.ndarray, sample_rate: int = 16000) -> torch.Tensor:
        """
        预处理音频波形用于前向传播。

        将numpy格式的音频波形转换为模型可接受的张量格式。

        Args:
            waveform: numpy音频波形数组，shape为(samples,)
            sample_rate: 音频采样率，默认为16000Hz

        Returns:
            预处理后的输入张量，shape为(1, samples)

        Note:
            Wav2Vec2要求输入采样率为16kHz
        """
        self._load_model()  # 确保处理器已加载
        inputs = self._processor(
            waveform,  # 音频波形
            sampling_rate=sample_rate,  # 采样率
            return_tensors="pt",  # 返回PyTorch张量
            padding=True  # 填充到相同长度
        )
        return inputs.input_values  # (1, samples)

    def transcribe(self, waveform: np.ndarray, sample_rate: int = 16000) -> str:
        """
        对音频波形执行ASR（自动语音识别）转录。

        使用Wav2Vec2ForCTC模型将语音转换为文本。
        这是一个独立的辅助功能，不参与特征提取。

        Args:
            waveform: numpy音频波形数组，shape为(samples,)
            sample_rate: 音频采样率，默认为16000Hz

        Returns:
            转录的文本字符串，失败时返回空字符串

        Note:
            此方法加载额外的CTC模型，可能消耗较多内存
        """
        try:
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
            import torch

            # 加载CTC模型和处理器
            processor = Wav2Vec2Processor.from_pretrained(self.model_name)
            asr_model = Wav2Vec2ForCTC.from_pretrained(self.model_name)

            # 预处理音频
            inputs = processor(
                waveform,  # 音频波形
                sampling_rate=sample_rate,  # 采样率
                return_tensors="pt",  # 返回PyTorch张量
                padding=True  # 填充
            )

            # 执行推理
            with torch.no_grad():
                logits = asr_model(inputs.input_values).logits  # 获取logits

            # 解码：取argmax得到预测的token ID
            predicted_ids = torch.argmax(logits, dim=-1)

            # 将token ID转换为文本
            return processor.batch_decode(predicted_ids)[0]

        except Exception as exc:
            logger.warning("ASR transcription failed: %s", exc)
            return ""  # 失败时返回空字符串
