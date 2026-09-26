"""
跨模态注意力模块。

本模块实现多头跨模态注意力机制，
用于在文本、音频、手势和标注四种模态之间建立语义关联。

主要功能:
    1. 实现多层跨模态注意力网络
    2. 支持缺失模态的注意力掩码
    3. 包含层归一化和前馈网络残差块
    4. 文本模态作为查询(Query)，其他模态作为键值(Key/Value)

核心原理:
    - Query: 通常来自文本编码器（指令理解）
    - Key/Value: 来自所有可用模态特征的拼接（多模态感知）
    - 注意力机制：让文本指令关注相关的多模态信息

参考文献:
    Vaswani et al., 2017. "Attention Is All You Need"
"""

# PyTorch深度学习框架
import torch  # 张量计算核心库
import torch.nn as nn  # 神经网络模块
import math  # 数学函数库
from typing import Optional  # 类型提示（兼容Python 3.8）


class CrossModalAttention(nn.Module):
    """
    多层跨模态注意力网络。

    该模块通过堆叠多个跨模态注意力块，实现文本指令与多模态感知信息
    之间的深层语义对齐。文本模态作为查询，其他模态作为键值，
    使模型能够根据指令关注相关的视觉、听觉和标注信息。

    属性:
        layers (nn.ModuleList): 跨模态注意力块列表

    工作流程:
        1. 文本Query向量作为输入
        2. 逐层通过跨模态注意力块
        3. 每层融合文本Query与多模态Key/Value
        4. 输出增强后的文本表示
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 2,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            _CrossModalBlock(d_model, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播：执行多层跨模态注意力计算。

        参数说明:
            query: 查询张量，shape为(batch, Lq, d_model)
                   通常来自文本编码器，Lq为文本序列长度
            key: 键张量，shape为(batch, Lk, d_model)
                 通常来自多模态特征拼接，Lk为多模态序列长度
            value: 值张量，shape为(batch, Lk, d_model)
                   与key相同维度，提供实际特征值
            mask: 注意力掩码，shape为(batch, 1, Lk)布尔类型
                  True表示忽略对应位置，用于处理缺失模态

        Returns:
            输出张量，shape为(batch, Lq, d_model)
            表示经过跨模态注意力增强的文本表示

        Note:
            掩码格式: True=忽略, False=保留
        """
        out = query  # 初始输出为查询向量
        # 逐层应用跨模态注意力块
        for layer in self.layers:
            out = layer(out, key, value, mask)
        return out


class _CrossModalBlock(nn.Module):
    """
    单个跨模态注意力块。

    每个块包含一个跨模态注意力子层和一个前馈网络子层，
    每个子层后都跟随层归一化和残差连接。

    组件:
        attn: 多头注意力层
        norm1: 第一层归一化
        ffn: 前馈网络（GELU激活）
        norm2: 第二层归一化
    """

    def __init__(self, d_model, n_heads, ffn_dim, dropout):
        """
        初始化单个跨模态注意力块。

        Args:
            d_model: 特征维度
            n_heads: 注意力头数
            ffn_dim: 前馈网络隐藏层维度
            dropout: Dropout比率
        """
        super().__init__()  # 调用父类构造函数
        # 多头注意力层：支持batch_first格式
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        # 第一层归一化
        self.norm1 = nn.LayerNorm(d_model)
        # 前馈网络：线性变换 → GELU激活 → Dropout → 线性变换 → Dropout
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),  # d_model → ffn_dim
            nn.GELU(),  # 高斯误差线性单元激活函数
            nn.Dropout(dropout),  # Dropout正则化
            nn.Linear(ffn_dim, d_model),  # ffn_dim → d_model
            nn.Dropout(dropout),  # Dropout正则化
        )
        # 第二层归一化
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, query, key, value, mask):
        """
        执行单个跨模态注意力块的前向传播。

        计算流程:
            1. 跨模态注意力：query与key/value交互
            2. 残差连接 + 层归一化
            3. 前馈网络：非线性特征变换
            4. 残差连接 + 层归一化

        Args:
            query: 查询张量
            key: 键张量
            value: 值张量
            mask: 注意力掩码

        Returns:
            经过注意力和前馈网络处理后的特征
        """
        # 跨模态注意力计算
        # 注意力掩码处理：如果mask存在，需要squeeze掉中间维度
        attn_out, _ = self.attn(
            query, key, value,
            key_padding_mask=mask.squeeze(1) if mask is not None else None,
        )
        # 残差连接 + 层归一化
        x = self.norm1(query + attn_out)

        # 前馈网络计算
        ffn_out = self.ffn(x)
        # 残差连接 + 层归一化
        x = self.norm2(x + ffn_out)
        return x
