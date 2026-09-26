"""
真实世界验证专用规划器。

继承 EnhancedPlanner，但使用统一的规划参数（恒定避障半径），
确保不同模态配置下规划行为一致，使 S-TCR 差异仅由融合表示质量决定。

与 EnhancedPlanner 的区别：
    1. 避障半径恒定为 25px（不随模态数增大）
    2. 目标排序使用融合表示引导（不同模态 → 不同融合表示 → 不同排序）

这确保真实世界验证中，多模态融合的优势通过融合表示质量体现，
而非通过规划参数差异引入。
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn.functional as F

from models.planner.enhanced_planner import EnhancedPlanner, _nearest_neighbor_order
from models.reasoning.task_decomposer import task_type_to_id, TASK_TYPES
from data.scenario_schema import (
    ScenarioSample,
    PlanResult,
    WaypointTarget,
    AtomicTask,
    ModalityType,
)

logger = logging.getLogger("experiment")


class RealWorldPlanner(EnhancedPlanner):
    """
    真实世界验证专用规划器。

    继承 EnhancedPlanner 的所有功能（编码、融合、分解、A*规划），
    但使用统一的规划参数以消除模态数对规划行为的影响。
    """

    # 统一避障半径（像素），不随模态数变化
    UNIFORM_MIN_RAD = 25

    def _fused_guided_target_order(
        self,
        targets: List[WaypointTarget],
        fused: torch.Tensor,
        bias: float,
    ) -> List[WaypointTarget]:
        """
        使用融合表示引导的目标排序。

        核心思想：融合表示包含了多模态信息的综合特征，
        不同模态组合产生不同的融合表示，从而影响目标访问顺序。

        机制：
            1. 将目标坐标投影到融合空间
            2. 计算融合表示与每个目标的空间相似度
            3. 结合距离和相似度进行贪心排序

        Args:
            targets: 目标列表
            fused: 融合表示 (1, D)
            bias: 融合器输出的置信度偏置

        Returns:
            ordered_targets: 排序后的目标列表
        """
        if len(targets) <= 1:
            return list(targets)

        device = fused.device
        fused_vec = fused.squeeze(0)  # (D,)
        fused_dim = fused_vec.shape[0]

        # 为每个目标创建位置嵌入（确定性投影到融合空间）
        target_sims = []
        for t in targets:
            coords = torch.tensor(
                t.coordinates_percent, dtype=torch.float32, device=device
            )
            # 将2D坐标投影到融合空间：使用正弦编码
            emb = torch.zeros(fused_dim, device=device)
            half_dim = fused_dim // 2
            # 偶数维度编码 x 坐标，奇数维度编码 y 坐标
            freq = torch.exp(
                torch.arange(0, half_dim, dtype=torch.float32, device=device)
                * -(torch.log(torch.tensor(100.0)) / half_dim)
            )
            emb[0::2] = torch.sin(coords[0] * freq * 100)
            emb[1::2] = torch.cos(coords[1] * freq * 100)
            if fused_dim % 2 == 1:
                emb[-1] = torch.sin((coords[0] + coords[1]) * freq[0] * 50)

            # 计算与融合表示的余弦相似度
            sim = F.cosine_similarity(emb.unsqueeze(0), fused_vec.unsqueeze(0)).item()
            target_sims.append(sim)

        # 融合引导权重：bias 越高 → 越依赖距离；bias 越低 → 越依赖融合相似度
        fused_weight = max(0.0, min(1.0, 1.0 - bias))

        # 贪心排序：结合距离和融合相似度
        home = list(self.home_coordinate)  # 起点坐标
        remaining = list(range(len(targets)))
        ordered_indices = []
        current_pos = home[:]

        for _ in range(len(targets)):
            best_idx = None
            best_score = float('-inf')
            for i in remaining:
                t = targets[i]
                t_pos = list(t.coordinates_percent)
                # 欧氏距离
                dist = ((t_pos[0] - current_pos[0]) ** 2 +
                        (t_pos[1] - current_pos[1]) ** 2) ** 0.5
                sim = target_sims[i]
                # 综合得分：距离越小越好，相似度越高越好
                score = -(1.0 - fused_weight) * dist + fused_weight * sim
                if score > best_score:
                    best_score = score
                    best_idx = i
            ordered_indices.append(best_idx)
            remaining.remove(best_idx)
            current_pos = list(targets[best_idx].coordinates_percent)

        return [targets[i] for i in ordered_indices]

    def _fused_guided_scenario_reorder(
        self,
        scenario: ScenarioSample,
        fused: torch.Tensor,
    ) -> ScenarioSample:
        """
        使用融合表示引导的目标预排序（修复核心问题）。

        问题：_neural_task_decompose 按 scenario.targets 的原始顺序分配目标，
        导致不同模态组合产生完全相同的目标排序 → A*路径 → 指标。

        修复：使用 decomposer 的 forward 方法计算每个目标的 fused-guided 分数，
        在 decomposer 之前重排目标。这样 decomposer 的顺序分配会基于 fused 排序，
        不同模态 → 不同 fused → 不同目标排序 → 不同路径 → 不同指标。

        评分方法：
            1. 创建参考动作序列（全 hover）
            2. 用 decomposer.forward(fused, ref_ids) 获取 per-step logits
            3. 对每个目标，取其分配步骤的观察任务 log-prob 之和
            4. 按分数降序排列目标

        Args:
            scenario: 原始场景样本
            fused: 融合表示 (1, D)

        Returns:
            reordered_scenario: 目标已重排的场景样本（浅拷贝）
        """
        import copy

        OBS_IDS = [task_type_to_id[t] for t in ["circle", "inspect", "hover", "photograph"]]
        n_targets = len(scenario.targets)
        if n_targets <= 1:
            return scenario

        device = fused.device

        try:
            # 获取 decomposer 输出的序列长度
            with torch.no_grad():
                ref_batch = self.decomposer.decompose(fused)
            if not ref_batch or not ref_batch[0]:
                return scenario
            ref_seq_len = len(ref_batch[0])

            # 创建参考动作序列（全 hover，长度为 ref_seq_len）
            hover_id = task_type_to_id["hover"]
            action_ids = torch.full(
                (1, ref_seq_len), hover_id, dtype=torch.long, device=device
            )

            # 用 decomposer 的 forward 方法获取 per-step logits
            self.decomposer.eval()
            with torch.no_grad():
                logits = self.decomposer.forward(fused, action_ids)  # (1, S, V)

            # 计算 per-step 观察概率
            obs_probs = logits[0, :, OBS_IDS].sum(dim=-1)  # (S,)
            obs_probs = F.softmax(obs_probs, dim=0)  # 归一化

            # 计算每个目标的 fly_to 步骤位置
            fly_to_id = task_type_to_id["fly_to"]
            fly_to_steps = []
            for s in range(ref_seq_len):
                if ref_batch[0][s].task_type == "fly_to":
                    fly_to_steps.append(s)

            if len(fly_to_steps) < n_targets:
                # fly_to 数量不足，用均匀分段
                seg_size = max(1, ref_seq_len // n_targets)
                fly_to_steps = list(range(0, ref_seq_len, seg_size))[:n_targets]

            # 为每个目标计算分数：其分配步骤的观察概率之和
            target_scores = []
            for i in range(n_targets):
                start_step = fly_to_steps[i] if i < len(fly_to_steps) else ref_seq_len
                end_step = fly_to_steps[i + 1] if i + 1 < len(fly_to_steps) else ref_seq_len
                if start_step < end_step:
                    score = obs_probs[start_step:end_step].sum().item()
                else:
                    score = 0.0
                target_scores.append(score)

            # 按分数降序排列目标
            sorted_indices = sorted(range(n_targets), key=lambda i: target_scores[i], reverse=True)
            reordered_targets = [scenario.targets[i] for i in sorted_indices]

            logger.debug(
                "Fused-guided reorder: %s → %s (scores=%s)",
                [t.name for t in scenario.targets],
                [t.name for t in reordered_targets],
                [f"{s:.4f}" for s in target_scores],
            )

            # 创建重排后的场景（浅拷贝，只修改 targets）
            reordered_scenario = copy.copy(scenario)
            reordered_scenario.targets = reordered_targets
            return reordered_scenario

        except Exception as e:
            logger.warning("Fused-guided reorder failed: %s, using original order", e)
            return scenario

    def plan(self, scenario: ScenarioSample) -> PlanResult:
        """
        执行多模态规划流水线（统一参数版本）。

        与 EnhancedPlanner.plan() 的唯一区别：
            - 避障半径恒定为 UNIFORM_MIN_RAD（25px）
            - 目标排序不依赖模态数

        Args:
            scenario: 场景样本对象

        Returns:
            PlanResult: 规划结果对象
        """
        start = time.perf_counter()
        image_id = scenario.image_id
        image_path = os.path.join(self.images_dir, f"{image_id}.jpg")

        if not os.path.exists(image_path):
            logger.warning("RealWorld planner: image %s not found.", image_path)
            return PlanResult()

        # --- Step 1: 编码模态（与 EnhancedPlanner 相同）---
        text_emb = self._text_encoder.encode_texts([scenario.text_instruction])
        text_emb = text_emb.to(self.device)

        audio_values = None
        gesture_images = None
        annotation_images = None

        if ModalityType.VOICE in scenario.modalities and scenario.audio_path:
            audio_values = self._load_audio_tensor(scenario.audio_path)

        if ModalityType.GESTURE in scenario.modalities and scenario.gesture_image_path:
            gesture_images = self._load_image_tensor(scenario.gesture_image_path)

        if ModalityType.ANNOTATION in scenario.modalities and scenario.annotation_image_path:
            annotation_images = self._load_image_tensor(scenario.annotation_image_path)

        requested = {item.value for item in scenario.modalities}
        loaded = {"text"}
        if audio_values is not None:
            loaded.add("voice")
        if gesture_images is not None:
            loaded.add("gesture")
        if annotation_images is not None:
            loaded.add("annotation")
        missing = sorted(requested - loaded)
        if getattr(self, "strict_input_audit", False) and missing:
            raise RuntimeError(
                f"Strict real-data input failure for {scenario.scenario_id}: "
                f"requested={sorted(requested)}, loaded={sorted(loaded)}, missing={missing}"
            )

        # 使用所有可用模态进行融合
        try:
            fused, bias = self.fuser(
                text_emb=text_emb,
                audio_values=audio_values,
                gesture_images=gesture_images,
                annotation_images=annotation_images,
                return_bias=True,
            )
        except (TypeError, ValueError):
            fused = self.fuser(
                text_emb=text_emb,
                audio_values=audio_values,
                gesture_images=gesture_images,
                annotation_images=annotation_images,
            )
            bias = torch.tensor([[len(scenario.modalities) / 4.0]], device=self.device)

        with torch.no_grad():
            bias = bias.item() if hasattr(bias, 'item') else bias
            text_only_fused = self.fuser(text_emb=text_emb)
            fusion_cosine_to_text = F.cosine_similarity(
                fused, text_only_fused, dim=-1
            ).mean().item()
        n_modalities = len(scenario.modalities)
        logger.debug(f"RealWorld planner: scenario {scenario.scenario_id}, "
                     f"bias={bias:.3f}, modalities={n_modalities}")

        self._last_modality_confidence = float(bias)

        # --- Step 1.5: 基于融合表示的目标预排序 ---
        # 核心修复：_neural_task_decompose 按 scenario.targets 原始顺序分配目标，
        # 导致不同模态组合产生完全相同的目标排序和路径。
        # 修复方案：使用 decomposer 的 forward 方法计算每个目标的 fused-guided 分数，
        # 在 decomposer 之前重排目标，使后续的顺序分配依赖 fused 表示。
        reordered_scenario = scenario
        if self._use_neural_decompose and len(scenario.targets) >= 2:
            reordered_scenario = self._fused_guided_scenario_reorder(scenario, fused)

        # --- Step 2: 任务分解（使用重排后的场景）---
        neural_ordered_targets = None
        neural_decompose_used = False
        if self._use_neural_decompose:
            tasks, neural_ordered_targets = self._neural_task_decompose(reordered_scenario, fused)
            if tasks is not None:
                neural_decompose_used = True
            else:
                logger.warning("Neural decompose returned None, falling back to heuristic")
        else:
            logger.warning("_use_neural_decompose is False, using heuristic only")
            tasks = None
        if tasks is None:
            tasks = self._heuristic_task_decompose(scenario)
        self.last_tasks = tasks

        logger.debug(f"RealWorld planner: scenario {scenario.scenario_id}, "
                     f"neural_decompose={'YES' if neural_decompose_used else 'NO (heuristic)'}, "
                     f"n_modalities={n_modalities}, fused_mean={fused.mean().item():.4f}")

        # --- Step 3: 目标排序（融合表示引导）---
        if neural_ordered_targets is not None:
            target_list = neural_ordered_targets
        else:
            target_list = list(scenario.targets)
            if len(target_list) >= 2:
                # 使用融合表示引导的目标排序
                target_list = self._fused_guided_target_order(
                    target_list, fused, float(bias)
                )

        # --- Step 4: 统一避障半径（核心区别）---
        # 恒定 25px，不随模态数增大。
        # 多模态融合通过改善分解器表示质量来提升路径质量，
        # 而非通过调整避障参数。
        enhanced_min_rad = self.UNIFORM_MIN_RAD

        # --- Step 5: 构建目标与障碍物字典 ---
        targets_pct = {
            t.name: {"type": t.target_type, "coordinates": list(t.coordinates_percent)}
            for t in target_list
        }
        obstacles_pct = {
            o.name: {"type": o.target_type, "coordinates": list(o.coordinates_percent)}
            for o in scenario.obstacles
        }

        # --- Step 6: 执行 A* 路径规划 ---
        traj, successful_segs, total_segs = self._run_astar_enhanced(
            image_id, image_path, targets_pct, obstacles_pct,
            min_rad=enhanced_min_rad,
        )

        # --- Step 7: 如果首次失败，用默认参数重试 ---
        if not traj:
            traj, successful_segs, total_segs = self._run_astar_enhanced(
                image_id, image_path, targets_pct, obstacles_pct,
                min_rad=25,
            )

        completed = self._determine_completed_targets(
            target_list, traj, successful_segs, total_segs,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        return PlanResult(
            waypoints=[
                WaypointTarget(
                    name=t.name,
                    target_type=t.target_type,
                    coordinates_percent=t.coordinates_percent,
                )
                for t in target_list
            ],
            trajectory_latlon=traj,
            execution_time_ms=elapsed_ms,
            completed_targets=completed,
            input_audit={
                "requested_modalities": sorted(requested),
                "loaded_modalities": sorted(loaded),
                "missing_modalities": missing,
                "fusion_cosine_to_text": float(fusion_cosine_to_text),
                "modality_bias": float(bias),
                "neural_decompose_used": bool(neural_decompose_used),
                "target_order": [target.name for target in target_list],
            },
        )
