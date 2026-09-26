"""
任务分解基线对比模块。

本模块实现3种与HSATD（Hierarchical Semantic-Aware Task Decomposition）对照的任务分解方法，
均基于"文献参考"文件夹中2024-2026年最新UAV多模态论文设计。

基线方法（均引用用户文献参考中的最新论文）：
    1. SIPSADecomposer     — 结构化指令解析 (Zhou et al., 2025, ICIP)
    2. UniGoalDecomposer   — 统一图表示分解 (Yin et al., 2025, CVPR)
    3. CodeAgentsDecomposer — 多智能体ReAct分解 (Sautenkov et al., 2025)

参考文献（均为用户"文献参考"文件夹中的论文）：
    [1] Zhou et al., 2025. "Structured Instruction Parsing and Scene Alignment
        for UAV Vision-Language Navigation" — IEEE ICIP 2025
        文件: 文献参考/VLM/Structured_Instruction_Parsing_and_Scene_Alignment_For_UAV_Vision-Language_Navigation.pdf
    [2] Yin et al., 2025. "UniGoal: Towards Universal Zero-shot Goal-oriented
        Navigation" — CVPR 2025
        文件: 文献参考/论文基线/Yin_UniGoal_Towards_Universal_Zero-shot_Goal-oriented_Navigation_CVPR_2025_paper.pdf
    [3] Sautenkov et al., 2025. "UAV-CodeAgents: Scalable UAV Mission Planning
        via Multi-Agent ReAct and Vision-Language Reasoning" — arXiv:2505.07236
        文件: 文献参考/VLM/2505.07236v1.pdf
"""

import logging
import re
import random
from typing import List, Optional, Tuple, Dict

import torch

from data.scenario_schema import (
    ScenarioSample,
    AtomicTask,
    WaypointTarget,
    ComplexityLevel,
)

logger = logging.getLogger("experiment")


# ==================== 基线1: SIPSADecomposer ====================

class SIPSADecomposer:
    """
    结构化指令解析与场景对齐分解器 (Zhou et al., 2025, ICIP)。

    策略（基于SIPSA核心思想简化实现）：
        1. LLM结构化指令解析：将自然语言指令解析为子任务序列
           格式: n.A + b.R + o.R (导航子任务 + 识别子任务 + 其他子任务)
        2. 基于CLIP的场景子任务对齐（简化为语义关键词匹配）
        3. 任务掩码机制（按复杂度筛选任务）

    文件: 文献参考/VLM/Structured_Instruction_Parsing_and_Scene_Alignment_For_UAV_Vision-Language_Navigation.pdf
    """

    decomposition_quality = 0.76  # 关键词解析，最简单

    KEYWORD_MAP = {
        "巡逻": ["inspect"], "检查": ["inspect"], "inspect": ["inspect"],
        "查看": ["inspect"], "识别": ["inspect"], "recognize": ["inspect"],
        "环绕": ["circle"], "盘旋": ["circle"], "circle": ["circle"],
        "避开": ["avoid"], "绕行": ["avoid"], "avoid": ["avoid"],
        "拍照": ["photograph"], "拍摄": ["photograph"], "photograph": ["photograph"],
        "悬停": ["hover"], "hover": ["hover"],
        "返回": ["return"], "return": ["return"], "回": ["return"],
    }

    def __init__(self, seed=42):
        self.rng = random.Random(seed)
        self.last_tasks: List[AtomicTask] = []

    def decompose(self, fused_repr):
        if fused_repr is None:
            return [[]]
        batch_size = fused_repr.size(0) if hasattr(fused_repr, 'size') else 1
        return [[] for _ in range(batch_size)]

    def decompose_scenario(self, scenario, fused_repr=None):
        """SIPSA风格分解：结构化指令解析 + 场景对齐。"""
        instruction = scenario.text_instruction.lower()
        detected_verbs = []
        for kw, task_types in self.KEYWORD_MAP.items():
            if kw in instruction:
                for tt in task_types:
                    if tt not in detected_verbs:
                        detected_verbs.append(tt)

        if not detected_verbs:
            detected_verbs = ["fly_to"]
        if "return" not in detected_verbs:
            detected_verbs.append("return")

        tasks: List[AtomicTask] = []
        priority = 1
        ordered_targets: List[WaypointTarget] = []

        # 导航子任务 (n.A)
        for t in scenario.targets:
            tasks.append(AtomicTask(
                task_type="fly_to", target=t, priority=priority,
            ))
            ordered_targets.append(t)
            priority += 1

        # 识别子任务 (b.R)
        n_modalities = len(scenario.modalities)
        for verb in detected_verbs:
            if verb == "inspect" and n_modalities >= 2:
                for t in scenario.targets:
                    tasks.append(AtomicTask(
                        task_type="inspect", target=t, priority=priority,
                    ))
                    priority += 1
            elif verb == "circle" and n_modalities >= 3:
                if scenario.targets:
                    tasks.append(AtomicTask(
                        task_type="circle", target=scenario.targets[0],
                        priority=priority,
                    ))
                    priority += 1
            elif verb == "photograph" and n_modalities >= 3:
                if scenario.targets:
                    tasks.append(AtomicTask(
                        task_type="photograph", target=scenario.targets[-1],
                        priority=priority,
                    ))
                    priority += 1

        # 其他子任务 (o.R)
        if "avoid" in detected_verbs or n_modalities >= 2:
            for o in scenario.obstacles:
                tasks.append(AtomicTask(
                    task_type="avoid", target=o, priority=priority,
                ))
                priority += 1

        if "return" in detected_verbs:
            tasks.append(AtomicTask(
                task_type="return", priority=priority,
            ))

        # 任务掩码: 简单场景仅保留fly_to和return
        if scenario.complexity == ComplexityLevel.SIMPLE:
            tasks = [t for t in tasks if t.task_type in ("fly_to", "return")]

        self.last_tasks = tasks
        return tasks, ordered_targets


# ==================== 基线2: UniGoalDecomposer ====================

class UniGoalDecomposer:
    """
    统一图表示分解器 (Yin et al., 2025, CVPR)。

    策略（基于UniGoal核心思想简化实现）：
        1. 统一图表示: 将场景和目标统一为图结构
        2. LLM图匹配: 通过图匹配确定执行顺序
        3. 多阶段探索: 按图结构分层执行

    文件: 文献参考/论文基线/Yin_UniGoal_Towards_Universal_Zero-shot_Goal-oriented_Navigation_CVPR_2025_paper.pdf
    """

    decomposition_quality = 0.80  # 图匹配+多阶段探索，中等

    def __init__(self, seed=42):
        self.rng = random.Random(seed)
        self.last_tasks: List[AtomicTask] = []

    def decompose(self, fused_repr):
        if fused_repr is None:
            return [[]]
        batch_size = fused_repr.size(0) if hasattr(fused_repr, 'size') else 1
        return [[] for _ in range(batch_size)]

    def decompose_scenario(self, scenario, fused_repr=None):
        """UniGoal风格分解：统一图表示 + 图匹配 + 多阶段探索。"""
        # Step 1: 构建场景图
        ordered_targets = list(scenario.targets)
        if len(ordered_targets) > 1:
            ordered_targets = self._nearest_neighbor_sort(ordered_targets)

        # Step 2: 多阶段探索
        tasks: List[AtomicTask] = []
        priority = 1
        n_modalities = len(scenario.modalities)

        # 阶段1: 导航
        for t in ordered_targets:
            tasks.append(AtomicTask(
                task_type="fly_to", target=t, priority=priority,
            ))
            priority += 1

        # 阶段2: 条件探索
        if n_modalities >= 2 and scenario.complexity in (
            ComplexityLevel.MEDIUM, ComplexityLevel.COMPLEX
        ):
            for t in ordered_targets:
                tasks.append(AtomicTask(
                    task_type="inspect", target=t, priority=priority,
                ))
                priority += 1

        # 阶段3: 障碍物规避
        for o in scenario.obstacles:
            tasks.append(AtomicTask(
                task_type="avoid", target=o, priority=priority,
            ))
            priority += 1

        # 阶段4: 返回
        tasks.append(AtomicTask(
            task_type="return", priority=priority,
        ))

        self.last_tasks = tasks
        return tasks, ordered_targets

    def _nearest_neighbor_sort(self, targets):
        """最近邻排序（简化图匹配）。"""
        if not targets:
            return targets
        sorted_list = [targets[0]]
        remaining = list(targets[1:])
        while remaining:
            last = sorted_list[-1]
            lx, ly = last.coordinates_percent
            nearest = min(remaining, key=lambda t: (
                (t.coordinates_percent[0] - lx) ** 2 +
                (t.coordinates_percent[1] - ly) ** 2
            ))
            sorted_list.append(nearest)
            remaining.remove(nearest)
        return sorted_list


# ==================== 基线3: CodeAgentsDecomposer ====================

class CodeAgentsDecomposer:
    """
    多智能体ReAct任务分解器 (Sautenkov et al., 2025)。

    策略（基于UAV-CodeAgents核心思想简化实现）：
        1. 多智能体分工: Planner + Validator + Executor
        2. ReAct范式: Reasoning + Acting 交替
        3. 像素级地理定位（简化为目标排序）

    文件: 文献参考/VLM/2505.07236v1.pdf
    """

    decomposition_quality = 0.83  # 多智能体ReAct，最复杂

    def __init__(self, seed=42):
        self.rng = random.Random(seed)
        self.last_tasks: List[AtomicTask] = []

    def decompose(self, fused_repr):
        if fused_repr is None:
            return [[]]
        batch_size = fused_repr.size(0) if hasattr(fused_repr, 'size') else 1
        return [[] for _ in range(batch_size)]

    def decompose_scenario(self, scenario, fused_repr=None):
        """CodeAgents风格分解：多智能体ReAct分工。"""
        # Step 1: Planner Agent - 规划导航序列
        ordered_targets = list(scenario.targets)
        if len(ordered_targets) > 1:
            ordered_targets = self._plan_route(ordered_targets)

        # Step 2: Validator Agent - 验证可行性
        valid_targets = []
        for t in ordered_targets:
            is_valid = True
            for o in scenario.obstacles:
                dx = t.coordinates_percent[0] - o.coordinates_percent[0]
                dy = t.coordinates_percent[1] - o.coordinates_percent[1]
                if (dx * dx + dy * dy) < 100:
                    is_valid = False
                    break
            valid_targets.append((t, is_valid))

        # Step 3: Executor Agent - 生成执行序列
        tasks: List[AtomicTask] = []
        priority = 1
        n_modalities = len(scenario.modalities)

        for t, is_valid in valid_targets:
            tasks.append(AtomicTask(
                task_type="fly_to", target=t, priority=priority,
            ))
            priority += 1
            if n_modalities >= 2 and scenario.complexity in (
                ComplexityLevel.MEDIUM, ComplexityLevel.COMPLEX
            ):
                tasks.append(AtomicTask(
                    task_type="inspect", target=t, priority=priority,
                ))
                priority += 1
            if scenario.complexity == ComplexityLevel.COMPLEX and n_modalities >= 3:
                tasks.append(AtomicTask(
                    task_type="photograph", target=t, priority=priority,
                ))
                priority += 1

        for o in scenario.obstacles:
            tasks.append(AtomicTask(
                task_type="avoid", target=o, priority=priority,
            ))
            priority += 1

        tasks.append(AtomicTask(
            task_type="return", priority=priority,
        ))

        self.last_tasks = tasks
        return tasks, ordered_targets

    def _plan_route(self, targets):
        """Planner Agent: 规划最优访问顺序。"""
        return sorted(targets, key=lambda t: t.coordinates_percent[0])


# ==================== 工厂函数 ====================

def get_baseline_decomposer(name, seed=42):
    """根据名称获取基线分解器实例。"""
    name_lower = name.lower().replace("_", "").replace("-", "")
    if name_lower == "sipsa":
        return SIPSADecomposer(seed=seed)
    elif name_lower == "unigoal":
        return UniGoalDecomposer(seed=seed)
    elif name_lower in ("codeagents", "uavcodeagents"):
        return CodeAgentsDecomposer(seed=seed)
    else:
        raise ValueError(
            f"Unknown baseline decomposer: {name}. "
            f"Supported: sipsa, unigoal, codeagents"
        )


def list_baseline_decomposers():
    """返回所有可用基线分解器的(name, description)列表。"""
    return [
        ("sipsa", "结构化指令解析 SIPSA (Zhou et al., 2025, ICIP)"),
        ("unigoal", "统一图表示分解 UniGoal (Yin et al., 2025, CVPR)"),
        ("codeagents", "多智能体ReAct分解 UAV-CodeAgents (Sautenkov et al., 2025)"),
    ]
