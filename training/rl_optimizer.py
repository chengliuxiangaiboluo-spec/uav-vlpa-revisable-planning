"""
强化学习优化器模块 - 使用近端策略优化(PPO)训练任务分解器。

本模块实现基于PPO算法的强化学习优化器，
用于训练任务分解器以生成更优的任务序列。

核心训练流程:
    1. 从训练集采样场景
    2. 通过融合器编码得到fused_repr
    3. 自回归采样任务序列
    4. 使用增强规划器评估路线
    5. 通过奖励函数计算奖励
    6. 使用PPO更新策略网络

PPO算法特点:
    1. 稳定的策略更新（通过裁剪目标函数）
    2. 支持离散动作空间（任务类型选择）
    3. 样本效率高
    4. 易于实现和调试

参考文献:
    Schulman et al., 2017. "Proximal Policy Optimization Algorithms"
    Sutton & Barto, 2018. "Reinforcement Learning: An Introduction"
"""

# ==================== 标准库和第三方库导入 ====================
import logging  # 日志记录模块
from pathlib import Path
import time     # 墙钟时间，用于 Slurm 进度输出
import sys      # 系统功能模块
from typing import List, Dict  # 类型提示支持

# PyTorch深度学习框架
import torch        # 张量计算核心库
import torch.nn as nn  # 神经网络模块
import torch.nn.functional as F  # 函数式接口
import numpy as np  # NumPy数值计算库

# ==================== 项目模块导入 ====================
from data.scenario_schema import ScenarioSample, ExpertPath, ModalityType, AtomicTask, WaypointTarget, PlanResult  # 场景数据结构
from models.reasoning.task_decomposer import (
    TaskDecomposer,       # 任务分解器
    task_type_to_id,      # 任务类型映射
    id_to_task_type,      # ID到任务类型反向映射
    NUM_TASK_TYPES,       # 任务类型数量
    TRANSITION_MATRIX,    # HSATD语义转移约束矩阵
)
from models.fusion.multimodal_fuser import MultimodalFuser  # 多模态融合器
from models.encoders.text_encoder import TextEncoder        # 文本编码器
from models.planner.enhanced_planner import EnhancedPlanner  # 增强规划器
from training.reward_function import RewardFunction          # 奖励函数
from training.training_utils import CheckpointManager        # 检查点管理
from configs.experiment_config import TrainingConfig          # 训练配置

# 获取实验日志记录器
logger = logging.getLogger("experiment")


class RLTaskDecomposerOptimizer:
    """
    基于PPO的强化学习训练器类。

    该类使用近端策略优化算法训练任务分解器，
    通过组合奖励信号指导模型学习更优的任务分解策略。

    训练流程:
        1. 采样场景
        2. 编码融合表示
        3. 自回归采样任务序列
        4. 规划路径并评估
        5. 计算奖励
        6. PPO策略更新

    主要组件:
        - decomposer: 任务分解器（策略网络）
        - fuser: 多模态融合器
        - planner: 增强规划器
        - reward_fn: 奖励函数
        - value_head: 价值头（用于优势估计）
    """

    def __init__(
        self,
        config: TrainingConfig,
        decomposer: TaskDecomposer,
        fuser: MultimodalFuser,
        planner: EnhancedPlanner,
        reward_fn: RewardFunction,
        device: str = "cpu",
    ):
        """
        初始化强化学习优化器。

        Args:
            config: 训练配置对象
            decomposer: 任务分解器实例
            fuser: 多模态融合器实例
            planner: 增强规划器实例
            reward_fn: 奖励函数实例
            device: 计算设备（cpu/cuda）
        """
        # 设置默认浮点类型为 FP32，避免混合精度问题
        torch.set_default_dtype(torch.float32)
        # 设置 cuDNN 确定性模式，可能降低性能但增加稳定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Accept both the configuration's string ("cuda"/"cpu") and an
        # already-normalised torch.device.  Later RL code needs `.type` for
        # CUDA synchronisation, so retaining a raw string here crashes after
        # fusion training has completed.
        self.device = torch.device(device)
        self.config = config
        self.decomposer = decomposer.to(self.device)
        self.fuser = fuser.to(self.device)
        self.planner = planner
        self.reward_fn = reward_fn
        self._text_encoder = TextEncoder()

        # RL训练时禁用神经分解：避免 plan() 内部调用 decomposer 形成循环依赖
        # 训练时 plan() 用启发式提供独立的路径质量评估，
        # decomposer 只在 _sample_trajectory() 中用于动作采样
        if hasattr(self.planner, '_use_neural_decompose'):
            self.planner._use_neural_decompose = False

        # 价值头：用于优势估计
        self.value_head = nn.Linear(decomposer.d_model, 1).to(self.device)

        # 优化器：联合优化分解器和价值头参数
        self.optimizer = torch.optim.Adam(
            list(self.decomposer.parameters()) + list(self.value_head.parameters()),
            lr=config.rl_lr,
        )
        self.ckpt_mgr = CheckpointManager(config.checkpoint_dir)

    def _map_actions_to_tasks(self, action_ids, scenario):
        """将采样的动作ID序列映射为绑定场景目标/障碍物的任务列表。

        与 EnhancedPlanner._neural_task_decompose 的映射逻辑一致：
        - fly_to 消耗一个目标（导航到新目标）
        - 观察类任务（circle/inspect/hover/photograph）在当前目标处执行，
          不消耗新目标；若没有当前目标则跳过
        - avoid 消耗一个障碍物
        - return 不绑定目标

        此设计迫使策略必须显式输出 fly_to 才能访问新目标，
        避免策略通过输出观察类任务"免费"获得目标访问。

        Args:
            action_ids: (1, S) 的动作 ID 张量
            scenario: 场景样本

        Returns:
            mapped_tasks: AtomicTask 列表
            ordered_targets: 推导的目标访问顺序
        """
        OBSERVATION_TASK_TYPES = {"circle", "inspect", "hover", "photograph"}
        eos_id = task_type_to_id["EOS"]

        mapped_tasks = []
        target_idx = 0
        obstacle_idx = 0
        visited_target_names = []
        current_target = None  # 由 fly_to 设置的当前目标

        for tok in action_ids[0].tolist():
            if tok == eos_id:
                break
            task_type = id_to_task_type.get(tok, "fly_to")

            if task_type == "fly_to":
                # fly_to 消耗一个目标
                if target_idx < len(scenario.targets):
                    t = scenario.targets[target_idx]
                    mapped_tasks.append(AtomicTask(
                        task_type="fly_to", target=t,
                        priority=len(mapped_tasks) + 1))
                    if t.name not in visited_target_names:
                        visited_target_names.append(t.name)
                    target_idx += 1
                    current_target = t
            elif task_type in OBSERVATION_TASK_TYPES:
                # 观察类任务在当前目标处执行，不消耗新目标
                if current_target is not None:
                    mapped_tasks.append(AtomicTask(
                        task_type=task_type, target=current_target,
                        priority=len(mapped_tasks) + 1))
                # else: 没有当前目标，跳过此观察任务
            elif task_type == "avoid":
                if obstacle_idx < len(scenario.obstacles):
                    mapped_tasks.append(AtomicTask(
                        task_type="avoid",
                        target=scenario.obstacles[obstacle_idx],
                        priority=len(mapped_tasks) + 1))
                    obstacle_idx += 1
            elif task_type == "return":
                mapped_tasks.append(AtomicTask(
                    task_type="return", priority=len(mapped_tasks) + 1))
                current_target = None  # 返航后重置当前目标

        # 不再补全未覆盖的目标 — 奖励必须反映策略的真实输出
        # 如果策略只输出 ['return']，则 mapped_tasks 只有 return，
        # 规划器不访问任何目标，completion reward = 0，策略受到惩罚。

        # 确保序列以 return 结尾（无人机必须返航）
        has_return = any(t.task_type == "return" for t in mapped_tasks)
        if not has_return:
            mapped_tasks.append(AtomicTask(
                task_type="return", priority=len(mapped_tasks) + 1))

        for i, t in enumerate(mapped_tasks):
            t.priority = i + 1

        # ordered_targets 只包含策略实际选择的目标
        ordered_targets = []
        seen = set()
        for t in mapped_tasks:
            if t.task_type == "fly_to" and t.target is not None:
                if t.target.name not in seen:
                    ordered_targets.append(t.target)
                    seen.add(t.target.name)

        return mapped_tasks, ordered_targets

    def _plan_with_sampled_tasks(self, scenario, action_ids):
        """使用 RL 采样的任务序列执行规划，返回 PlanResult。

        将采样的动作序列映射为任务，然后复用规划器的 A* 路径搜索，
        使奖励信号真正依赖于策略输出的动作。

        Args:
            scenario: 场景样本
            action_ids: (1, S) 的动作 ID 张量

        Returns:
            PlanResult: 规划结果
        """
        import os
        mapped_tasks, ordered_targets = self._map_actions_to_tasks(action_ids, scenario)
        # 不再回退到全部目标 — 策略输出什么就评估什么

        image_id = scenario.image_id
        image_path = os.path.join(self.planner.images_dir, f"{image_id}.jpg")
        if not os.path.exists(image_path):
            return PlanResult()

        targets_pct = {
            t.name: {"type": t.target_type, "coordinates": list(t.coordinates_percent)}
            for t in ordered_targets
        }
        obstacles_pct = {
            o.name: {"type": o.target_type, "coordinates": list(o.coordinates_percent)}
            for o in scenario.obstacles
        }

        # Keep the planning geometry independent of the input modality count.
        # Otherwise the RL reward would leak the desired ablation outcome.
        enhanced_min_rad = 25

        traj, successful_segs, total_segs = self.planner._run_astar_enhanced(
            image_id, image_path, targets_pct, obstacles_pct,
            min_rad=enhanced_min_rad,
        )
        if not traj:
            traj, successful_segs, total_segs = self.planner._run_astar_enhanced(
                image_id, image_path, targets_pct, obstacles_pct,
                min_rad=25,
            )

        completed = self.planner._determine_completed_targets(
            ordered_targets, traj, successful_segs, total_segs,
        )

        return PlanResult(
            waypoints=[
                WaypointTarget(
                    name=t.name,
                    target_type=t.target_type,
                    coordinates_percent=t.coordinates_percent,
                )
                for t in ordered_targets
            ],
            trajectory_latlon=traj,
            execution_time_ms=0.0,
            completed_targets=completed,
        )

    def train(
        self,
        scenarios: List[ScenarioSample],
    ) -> Dict[str, List[float]]:
        """
        运行PPO训练循环。

        该方法执行指定轮数的PPO训练，
        每轮包含场景采样、任务序列生成、路径评估和策略更新。

        Args:
            scenarios: 场景样本列表

        Returns:
            history: 包含每轮奖励和策略损失的历史字典
        """
        started_at = time.monotonic()
        print(
            "[RL] start | episodes=%d | ppo_epochs=%d | max_steps=%d | scenarios=%d | device=%s"
            % (self.config.rl_episodes, getattr(self.config, "rl_ppo_epochs", 1),
               self.config.rl_max_steps_per_episode, len(scenarios), self.device),
            flush=True,
        )
        history: Dict[str, List[float]] = {"episode_rewards": [], "policy_losses": []}
        eos_id = task_type_to_id["EOS"]

        for ep in range(1, self.config.rl_episodes + 1):
            if ep == 1 or ep % self.config.log_interval == 0:
                print(
                    "[RL] episode %d/%d started | elapsed=%.1f min"
                    % (ep, self.config.rl_episodes, (time.monotonic() - started_at) / 60.0),
                    flush=True,
                )
            scenario = scenarios[np.random.randint(len(scenarios))]

            # 编码场景 — 加载所有可用模态，确保训练和推理一致
            text_emb = self._text_encoder.encode_texts(
                [scenario.text_instruction]
            ).to(self.device)

            # 加载多模态数据（与 enhanced_planner.plan() 一致）
            audio_values = None
            gesture_images = None
            annotation_images = None
            if hasattr(self.planner, '_load_audio_tensor'):
                if ModalityType.VOICE in scenario.modalities and scenario.audio_path:
                    audio_values = self.planner._load_audio_tensor(scenario.audio_path)
                if ModalityType.GESTURE in scenario.modalities and scenario.gesture_image_path:
                    gesture_images = self.planner._load_image_tensor(scenario.gesture_image_path)
                if ModalityType.ANNOTATION in scenario.modalities and scenario.annotation_image_path:
                    annotation_images = self.planner._load_image_tensor(scenario.annotation_image_path)

            with torch.no_grad():
                fused = self.fuser(
                    text_emb=text_emb,
                    audio_values=audio_values,
                    gesture_images=gesture_images,
                    annotation_images=annotation_images,
                )  # (1, D)

            # 采样动作序列
            action_ids, log_probs_old, values = self._sample_trajectory(fused)

            # 评估计划 — 使用 RL 采样的任务序列，而非启发式分解
            plan_result = self._plan_with_sampled_tasks(scenario, action_ids)

            # 计算奖励
            gt = (
                scenario.ground_truth_paths[0]
                if scenario.ground_truth_paths
                else ExpertPath()
            )
            reward = self.reward_fn.compute(plan_result, gt, scenario)

            # 序列长度激励：鼓励模型输出更长的任务序列
            # 排除起始 fly_to（固定）和 EOS（终止符），只计算有效动作
            eos_id_val = task_type_to_id["EOS"]
            valid_tokens = [
                t for t in action_ids[0].tolist()
                if t != eos_id_val
            ]
            # 起始 fly_to 是固定的，不计入长度激励
            extra_tokens = max(len(valid_tokens) - 1, 0)
            # 每个额外 token +0.02，上限 0.15（约 7-8 个额外 token）
            length_bonus = min(0.02 * extra_tokens, 0.15)
            reward += length_bonus

            # PPO更新 — 对同一条轨迹做多次更新，使 clip_eps 生效
            # 多次更新使策略逐渐偏离采样策略，ratio 不再≈1，裁剪才起作用
            ppo_epochs = getattr(self.config, 'rl_ppo_epochs', 1)
            total_policy_loss = 0.0
            for ppo_ep in range(ppo_epochs):
                policy_loss = self._ppo_update(fused, action_ids, log_probs_old, values, reward)
                total_policy_loss += policy_loss
            policy_loss = total_policy_loss / ppo_epochs

            history["episode_rewards"].append(reward)
            history["policy_losses"].append(policy_loss)

            if ep % self.config.log_interval == 0 or ep == 1:
                # 采样任务序列诊断 — 直接观察策略多样性，及时发现坍塌
                task_names = [
                    id_to_task_type.get(t, "?")
                    for t in action_ids[0].tolist()
                    if t != eos_id
                ]
                avg_r = np.mean(history["episode_rewards"][-self.config.log_interval:])
                logger.info(
                    "RL episode %d/%d  tasks=%s  len=%d  reward=%.4f(+%.3f)  avg_reward=%.4f  loss=%.4f",
                    ep, self.config.rl_episodes, task_names,
                    len(task_names), reward, length_bonus, avg_r, policy_loss,
                )
                print(
                    "[RL] episode %d/%d complete | tasks=%s | reward=%.4f | avg_reward=%.4f | loss=%.4f | elapsed=%.1f min"
                    % (ep, self.config.rl_episodes, task_names, reward, avg_r, policy_loss,
                       (time.monotonic() - started_at) / 60.0),
                    flush=True,
                )

            # 周期性保存中间 checkpoint（防止长训练中途失败丢失全部进度）
            if ep % 500 == 0 and ep != self.config.rl_episodes:
                self.ckpt_mgr.save(
                    self.decomposer, self.optimizer, ep,
                    {"avg_reward": float(np.mean(history["episode_rewards"][-100:]))},
                    name="decomposer_rl",
                )
                logger.info("RL intermediate checkpoint saved at episode %d", ep)
                print("[RL] checkpoint saved | episode=%d" % ep, flush=True)

        # Evaluation reloads only model weights.  The final optimizer state is
        # unnecessary for a completed submission run and is costly on shared
        # storage, so retain one final decomposer checkpoint without it.
        final_checkpoint = Path(self.ckpt_mgr.save(
            self.decomposer, None, self.config.rl_episodes,
            {"avg_reward": float(np.mean(history["episode_rewards"]))},
            name="decomposer_rl",
        ))
        # Intermediate checkpoints protect a job while it is running, but a
        # completed seed must retain only its final model to prevent checkpoint
        # accumulation across the 20 Exp5 runs.
        for stale_checkpoint in final_checkpoint.parent.glob("decomposer_rl_epoch*.pt"):
            if stale_checkpoint != final_checkpoint:
                stale_checkpoint.unlink()
        print(
            "[RL] complete | episodes=%d | elapsed=%.1f min"
            % (self.config.rl_episodes, (time.monotonic() - started_at) / 60.0),
            flush=True,
        )
        return history

    def _sample_trajectory(self, fused: torch.Tensor):
        """
        从当前策略中采样任务序列。

        该方法使用自回归方式生成任务序列，
        直到遇到EOS标记或达到最大步数限制。

        Args:
            fused: 融合表示张量

        Returns:
            action_ids: 动作ID序列
            log_probs: 对数概率
            values: 价值估计
        """
        self.decomposer.eval()
        batch_size = fused.size(0)
        eos_id = task_type_to_id["EOS"]
        start_id = task_type_to_id["fly_to"]

        generated = torch.full((batch_size, 1), start_id, dtype=torch.long, device=self.device)
        log_probs = []
        values = []

        memory = fused.unsqueeze(1)  # (1, 1, D)

        # 获取位置编码表的最大长度，并限制循环步数
        max_seq_len = getattr(self.decomposer, 'max_seq_len', None)
        if max_seq_len is None:
            # 兼容旧版本 decomposer 没有 max_seq_len 属性的情况
            max_seq_len = self.config.rl_max_steps_per_episode + 1
        max_steps = min(self.config.rl_max_steps_per_episode, max_seq_len - 1)

        for step in range(max_steps):
            seq_len = generated.size(1)
            positions = torch.arange(seq_len, device=self.device).unsqueeze(0)
            tgt = self.decomposer.token_embedding(generated) + self.decomposer.pos_embedding(positions)
            causal = nn.Transformer.generate_square_subsequent_mask(seq_len, device=self.device)

            # 确保张量连续
            tgt = tgt.contiguous()
            memory = memory.contiguous()
            if causal is not None:
                causal = causal.contiguous()

            # 可选同步（仅当 CUDA 可用）
            if self.device.type == 'cuda':
                torch.cuda.synchronize()

            decoded = self.decomposer.decoder(tgt, memory, tgt_mask=causal)

            logits = self.decomposer.output_head(decoded[:, -1, :])  # (B, V)

            # HSATD: 应用语义转移约束掩码（与 decompose() 保持一致）
            # 防止RL采样产生论文中定义为无效的任务转移
            prev_token = generated[:, -1]  # (B,) — 最后一个已生成的token
            transition_mask = TRANSITION_MATRIX.to(logits.device)  # (V, V)
            allowed = transition_mask[prev_token]  # (B, V)
            logits = logits.masked_fill(allowed == 0, float('-inf'))

            probs = F.softmax(logits, dim=-1)

            # 稳定概率分布
            if torch.isnan(probs).any() or torch.isinf(probs).any():
                print("Warning: NaN/inf in probs, replacing with uniform", flush=True)
                probs = torch.ones_like(probs) / probs.shape[-1]
            probs = probs.clamp(min=1e-8)
            probs = probs / probs.sum(dim=-1, keepdim=True)

            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))

            # 价值估计
            v = self.value_head(decoded[:, -1, :]).squeeze(-1)
            values.append(v)

            generated = torch.cat([generated, action.unsqueeze(1)], dim=1)
            if (action == eos_id).all():
                break

        action_ids = generated  # 保留起始 fly_to — 模型应获得该 token 的奖励信号
        # 为起始 fly_to 补零占位（它不是采样的，但需要维度对齐）
        zero_pad = torch.zeros(batch_size, 1, device=self.device)
        log_probs_t = torch.cat([zero_pad, torch.stack(log_probs, dim=1)], dim=1)  # (B, 1+S)
        values_t = torch.cat([zero_pad, torch.stack(values, dim=1)], dim=1)  # (B, 1+S)
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        return action_ids, log_probs_t, values_t

    def _ppo_update(self, fused, action_ids, old_log_probs, old_values, reward):
        """
        执行单次PPO更新步骤。

        该方法实现PPO算法的核心更新逻辑，包括：
        - 重新计算对数概率和价值估计
        - 计算优势函数
        - 策略比率和裁剪
        - 策略损失、价值损失和熵正则化

        Args:
            fused: 融合表示
            action_ids: 动作ID序列
            old_log_probs: 旧对数概率
            old_values: 旧价值估计
            reward: 奖励值

        Returns:
            total_loss: 总损失值
        """
        self.decomposer.train()

        # 重新计算对数概率和价值 — 必须与采样时的位置编码一致
        # action_ids 现在包含起始 fly_to，无需再拼接
        full_seq = action_ids  # (B, 1+S)
        full_len = full_seq.size(1)

        memory = fused.unsqueeze(1)
        positions = torch.arange(full_len, device=self.device).unsqueeze(0)
        tgt = self.decomposer.token_embedding(full_seq) + self.decomposer.pos_embedding(positions)
        causal = nn.Transformer.generate_square_subsequent_mask(full_len, device=self.device)

        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        decoded = self.decomposer.decoder(tgt, memory, tgt_mask=causal)

        # 从 decoded 中提取 log_probs
        # full_seq = [fly_to, a0, a1, ..., aS] 长度 1+S
        # decoded[:, i, :] 预测 full_seq[:, i+1]
        # 需要预测位置 0..S-1（共 S 个）来对应 action_ids 中位置 1..S 的 token
        # 起始 fly_to（位置 0）的 log_prob 为 0（占位），不影响梯度
        pred_logits = self.decomposer.output_head(decoded[:, :full_len-1, :])  # (B, S, V)
        # 应用转移约束（与采样时一致）
        prev_tokens = full_seq[:, :full_len-1]  # (B, S)
        transition_mask = TRANSITION_MATRIX.to(pred_logits.device)
        allowed = transition_mask[prev_tokens]  # (B, S, V)
        pred_logits = pred_logits.masked_fill(allowed == 0, float('-inf'))
        log_probs_all = F.log_softmax(pred_logits, dim=-1)  # (B, S, V)
        # action_ids 位置 1..S 对应 pred_logits 位置 0..S-1
        new_log_probs_sampled = log_probs_all.gather(2, action_ids[:, 1:].unsqueeze(-1)).squeeze(-1)  # (B, S)
        # 拼接起始 fly_to 的占位 log_prob=0
        zero_pad = torch.zeros(action_ids.size(0), 1, device=self.device)
        new_log_probs = torch.cat([zero_pad, new_log_probs_sampled], dim=1)  # (B, 1+S)

        # 价值估计（对应 action_ids 的每个位置）
        new_values_sampled = self.value_head(decoded[:, 1:, :]).squeeze(-1)  # (B, S)
        new_values = torch.cat([zero_pad, new_values_sampled], dim=1)  # (B, 1+S)

        # 优势函数（奖励 - 价值基线）
        # 注意：不能在单条轨迹内归一化 — 归一化会使优势均值恒为0，
        # 好轨迹和坏轨迹得到同等强度的更新，丢失轨迹好坏的信号
        reward_t = torch.full_like(old_values, reward)
        advantages = reward_t - old_values.detach()

        # 策略比率
        ratio = torch.exp(new_log_probs - old_log_probs.detach())
        clipped_ratio = torch.clamp(ratio, 1.0 - self.config.rl_clip_eps, 1.0 + self.config.rl_clip_eps)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        # 价值损失
        value_loss = F.mse_loss(new_values, reward_t)

        # 熵奖励（鼓励探索）— 必须使用与策略分布相同的带转移约束掩码的 logits，
        # 否则梯度会把概率质量推向被禁止的转移，与 HSATD 约束对抗
        # 用 clamp 避免 -inf × 0 产生 NaN
        ent_probs = F.softmax(pred_logits, dim=-1)
        ent_log_probs = torch.log(ent_probs.clamp(min=1e-8))
        entropy = -(ent_probs * ent_log_probs).sum(-1).mean()

        total_loss = (
            policy_loss
            + self.config.rl_value_coeff * value_loss
            - self.config.rl_entropy_coeff * entropy
        )

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.decomposer.parameters(), 0.5)
        self.optimizer.step()

        return total_loss.item()
