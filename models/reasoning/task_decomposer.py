"""
任务分解器模块。

本模块实现基于Transformer架构的任务分解器，
用于将融合的多模态表示解码为原子任务序列。

核心功能:
    1. 将融合的多模态表示解码为原子任务序列
    2. 支持教师强制(teacher-forcing)训练
    3. 支持自回归推理
    4. 为强化学习优化提供对数概率输出

技术特点:
    - 基于小型Transformer解码器架构
    - 使用预定义的任务类型词汇表
    - 支持动态最大序列长度配置
    - 提供完整的训练和推理接口

参考文献:
    Vaswani et al., 2017. "Attention Is All You Need"
    Brown et al., 2020. "Language Models are Few-Shot Learners"
"""

# ==================== 标准库和第三方库导入 ====================
import logging  # 日志记录模块
from typing import List, Optional  # 类型提示支持

# PyTorch深度学习框架
import torch        # 张量计算核心库
import torch.nn as nn  # 神经网络模块
import torch.nn.functional as F  # 函数式接口

# ==================== 项目模块导入 ====================
from data.scenario_schema import AtomicTask  # 原子任务数据类

# 获取实验日志记录器
logger = logging.getLogger("experiment")

# ==================== 任务类型词汇表 ====================
# 固定的任务类型词汇表，用于将任务名称与ID互相转换
TASK_TYPES = ["fly_to", "circle", "avoid", "inspect", "return", "hover", "photograph", "EOS"]
# 任务类型到ID的映射
task_type_to_id = {t: i for i, t in enumerate(TASK_TYPES)}
# ID到任务类型的映射
id_to_task_type = {i: t for t, i in task_type_to_id.items()}
# 任务类型数量（包括结束标记EOS）
NUM_TASK_TYPES = len(TASK_TYPES)

# ==================== 语义转移约束矩阵 (HSATD) ====================
# 定义操作有效的任务转移规则：T[i][j] = 1 表示允许从任务i转移到任务j
# 设计原则:
#   - Navigation(fly_to, return) 后可跟 Navigation, Interaction, Constraint
#   - Interaction(circle, inspect, hover, photograph) 后可跟 Navigation, Interaction, Constraint
#   - Constraint(avoid) 后可跟 Navigation, Interaction
#   - return 后只可跟 EOS (返航后只能结束)
#   - EOS 是终止状态，后不可跟任何任务
#   - fly_to 是起始状态，任何任务都可转移到其他任务
def _build_transition_matrix():
    """构建语义转移约束矩阵 T ∈ {0,1}^{8×8}"""
    T = torch.ones(NUM_TASK_TYPES, NUM_TASK_TYPES, dtype=torch.float32)
    eos_id = task_type_to_id["EOS"]
    return_id = task_type_to_id["return"]

    # EOS 后不可跟任何任务（终止状态）
    T[eos_id, :] = 0.0
    # return 后只能跟 EOS
    T[return_id, :] = 0.0
    T[return_id, eos_id] = 1.0
    # avoid 后不可直接跟 return 或 avoid（需先有导航动作）
    avoid_id = task_type_to_id["avoid"]
    T[avoid_id, return_id] = 0.0
    T[avoid_id, avoid_id] = 0.0

    return T

TRANSITION_MATRIX = _build_transition_matrix()


class TaskDecomposer(nn.Module):
    """
    任务分解器类。

    该类基于Transformer解码器架构，将融合的多模态表示
    自回归地解码为原子任务序列。

    主要组件:
        - token_embedding: 任务类型嵌入层
        - pos_embedding: 位置编码层
        - decoder: Transformer解码器
        - output_head: 输出头（分类层）
    """

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 2,
        n_heads: int = 8,
        max_subtasks: int = 10,
        max_seq_len: Optional[int] = None,   # 新增：最大序列长度（包括起始 token）
        dropout: float = 0.1,
    ):
        """
        初始化任务分解器。

        Args:
            d_model: 特征维度（默认256）
            n_layers: 解码器层数（默认2）
            n_heads: 注意力头数（默认8）
            max_subtasks: 最大子任务数（默认10）
            max_seq_len: 最大序列长度（可选）
            dropout: Dropout比率（默认0.1）
        """
        super().__init__()  # 调用父类构造函数
        self.d_model = d_model  # 保存特征维度
        self.max_subtasks = max_subtasks  # 保存最大子任务数

        # HSATD 语义转移约束开关（A3 消融：置为 False 可禁用约束矩阵 T 的 logits 掩码）
        self.use_transition_constraint = True

        # 确定最大序列长度：若未指定，则使用 max_subtasks+1（与原行为一致）
        if max_seq_len is None:
            max_seq_len = max_subtasks + 1
        self.max_seq_len = max_seq_len  # 保存最大序列长度

        # 任务类型嵌入层：将任务ID映射到特征向量
        self.token_embedding = nn.Embedding(NUM_TASK_TYPES, d_model)
        # 位置编码层：为序列中的每个位置添加位置信息
        # 位置编码表大小由 max_seq_len 决定，不再硬编码
        self.pos_embedding = nn.Embedding(self.max_seq_len, d_model)

        # Transformer解码器层
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # 输出头：将解码后的特征映射到任务类型概率分布
        self.output_head = nn.Linear(d_model, NUM_TASK_TYPES)

    def forward(
        self,
        fused_repr: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        教师强制前向传播（训练模式）。

        该方法在训练时使用真实标签进行教师强制训练，
        让模型学习预测正确的任务序列。

        参数说明:
            fused_repr: 融合的多模态表示，shape为(batch, d_model)
            target_ids: 真实任务ID序列，shape为(batch, seq_len)

        Returns:
            logits: 预测logits，shape为(batch, seq_len, NUM_TASK_TYPES)

        Note:
            推理时请使用decompose()方法，而非此方法
        """
        if target_ids is None:
            # 在推理时，应使用decompose()方法
            raise ValueError("Use decompose() for inference.")

        batch_size, seq_len = target_ids.shape
        device = fused_repr.device

        # 检查序列长度是否超出位置编码表范围
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")

        # Memory = 融合表示扩展为序列长度1
        memory = fused_repr.unsqueeze(1)  # (B, 1, D)

        # 任务类型嵌入 + 位置编码
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        tgt = self.token_embedding(target_ids) + self.pos_embedding(positions)

        # 因果掩码（确保每个位置只能看到前面的位置）
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)

        # Transformer解码器
        decoded = self.decoder(tgt, memory, tgt_mask=causal_mask)
        # 输出头：映射到任务类型概率分布
        logits = self.output_head(decoded)  # (B, seq_len, NUM_TASK_TYPES)

        # HSATD: 应用语义转移约束掩码
        # 论文公式: P̃(a_t=j|a_{t-1}=i) = T_ij · P(...) / Σ_k[T_ik · P(...)]
        if self.use_transition_constraint and target_ids.size(1) > 1:
            prev_tokens = target_ids[:, :-1]  # (B, seq_len-1) — 前一步的任务ID
            transition_mask = TRANSITION_MATRIX.to(logits.device)  # (V, V)
            # 获取每个位置的允许转移掩码
            allowed = transition_mask[prev_tokens]  # (B, seq_len-1, V)
            # 对无效转移赋予 -inf logits（softmax后概率为0）
            padding = torch.ones(logits.size(0), 1, NUM_TASK_TYPES, device=logits.device)
            # 第一个位置不做约束（起始token）
            full_mask = torch.cat([padding, allowed], dim=1)  # (B, seq_len, V)
            logits = logits.masked_fill(full_mask == 0, float('-inf'))

        return logits

    @torch.no_grad()
    def decompose(self, fused_repr: torch.Tensor) -> List[List[AtomicTask]]:
        """
        自回归任务分解（推理模式）。

        该方法在推理时使用自回归方式生成任务序列，
        逐个预测下一个任务类型，直到遇到EOS标记。

        Args:
            fused_repr: 融合的多模态表示，shape为(batch, d_model)

        Returns:
            results: 每个批次元素对应的原子任务列表

        Note:
            该方法在无梯度模式下运行（@torch.no_grad）
        """
        self.eval()  # 设置为评估模式
        batch_size = fused_repr.size(0)
        device = fused_repr.device
        memory = fused_repr.unsqueeze(1)  # (B, 1, D)

        eos_id = task_type_to_id["EOS"]
        # 以"fly_to"作为起始token
        generated = torch.full((batch_size, 1), task_type_to_id["fly_to"], dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # 最大生成步数：不超过 max_seq_len - 1（因为已有起始 token）
        max_steps = self.max_seq_len - 1
        for step in range(max_steps):
            seq_len = generated.size(1)
            positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            tgt = self.token_embedding(generated) + self.pos_embedding(positions)

            causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)
            decoded = self.decoder(tgt, memory, tgt_mask=causal_mask)
            logits = self.output_head(decoded[:, -1, :])  # (B, NUM_TASK_TYPES)

            # HSATD: 应用语义转移约束，屏蔽无效的下一个任务类型
            if self.use_transition_constraint:
                prev_token = generated[:, -1]  # (B,) — 最后一个生成的token
                transition_mask = TRANSITION_MATRIX.to(logits.device)  # (V, V)
                allowed = transition_mask[prev_token]  # (B, V)
                logits = logits.masked_fill(allowed == 0, float('-inf'))

            next_token = logits.argmax(dim=-1)            # (B,)

            finished = finished | (next_token == eos_id)
            next_token[finished] = eos_id

            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)

            if finished.all():
                break

        # 转换为AtomicTask列表
        results = []
        for b in range(batch_size):
            tasks = []
            for tok in generated[b].tolist():  # 保留起始 fly_to — 与 RL 训练一致
                if tok == eos_id:
                    break
                tasks.append(AtomicTask(
                    task_type=id_to_task_type.get(tok, "fly_to"),
                    priority=len(tasks) + 1,
                ))
            results.append(tasks)

        return results

    def get_log_probs(
        self,
        fused_repr: torch.Tensor,
        action_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算给定动作序列的对数概率。

        该方法用于强化学习优化器，计算策略网络输出的对数概率，
        用于PPO等强化学习算法的损失计算。

        Args:
            fused_repr: 融合的多模态表示，shape为(batch, d_model)
            action_ids: 选择的动作ID序列，shape为(batch, seq_len)

        Returns:
            selected: 对数概率张量，shape为(batch, seq_len)
        """
        logits = self.forward(fused_repr, action_ids)  # (B, S, V)
        log_probs = F.log_softmax(logits, dim=-1)
        selected = log_probs.gather(2, action_ids.unsqueeze(-1)).squeeze(-1)
        return selected