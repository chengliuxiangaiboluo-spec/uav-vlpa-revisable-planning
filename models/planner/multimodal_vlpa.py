"""
多模态视觉-语言-路径规划（VLPA）系统主模块。

本模块实现顶层多模态VLPA系统，
集成所有子模块并提供统一的接口用于处理场景、运行交互式会话和获取基线结果。

核心功能:
    1. process_scenario: 运行增强型多模态规划器
    2. get_baseline_result: 运行原始文本单模态基线规划器
    3. process_interactive: 模拟飞行中的动态指令更新交互会话

系统架构:
    - MultimodalFuser: 多模态特征融合
    - TaskDecomposer: 任务分解器
    - EnhancedPlanner: 增强型规划器
    - BaselinePlanner: 基线规划器
    - DynamicConditionHandler: 动态条件处理器
    - KnowledgeIntegrator: 知识整合器

参考文献:
    Vaswani et al., 2017. "Attention Is All You Need"
    Baltrusaitis et al., 2019. "Multimodal Machine Learning: A Survey and Taxonomy"
    Sautenkov et al., 2025. "UAV-VLPA*: Vision-Language-Path Planning for Autonomous Drones"
"""

# 标准库导入
import logging  # 日志记录模块
from typing import List, Dict, Any, Optional  # 类型提示支持

# 项目模块导入
from configs.experiment_config import BaseConfig, ModelConfig  # 实验配置
from data.scenario_schema import ScenarioSample, PlanResult  # 场景数据结构
from models.fusion.multimodal_fuser import MultimodalFuser  # 多模态融合器
from models.reasoning.task_decomposer import TaskDecomposer  # 任务分解器
from models.reasoning.condition_handler import DynamicConditionHandler  # 动态条件处理器
from models.reasoning.knowledge_integrator import KnowledgeIntegrator  # 知识整合器
from models.planner.baseline_planner import BaselinePlanner  # 基线规划器
from models.planner.enhanced_planner import EnhancedPlanner  # 增强型规划器

# 获取实验日志记录器
logger = logging.getLogger("experiment")


class MultimodalVLPA:
    """
    多模态VLPA系统顶层类。

    该类集成所有多模态子系统，提供统一的API接口。

    主要组件:
        - fuser: 多模态融合器
        - decomposer: 任务分解器
        - condition_handler: 动态条件处理器
        - knowledge_integrator: 知识整合器
        - enhanced_planner: 增强型规划器
        - baseline_planner: 基线规划器

    提供的接口:
        - process_scenario(): 运行增强型多模态规划器
        - get_baseline_result(): 运行文本单模态基线规划器
        - process_interactive(): 模拟交互式飞行会话
    """

    def __init__(self, base_cfg: BaseConfig, model_cfg: ModelConfig):
        """
        初始化多模态VLPA系统。

        Args:
            base_cfg: 基础配置对象
            model_cfg: 模型配置对象
        """
        self.base_cfg = base_cfg
        self.device = base_cfg.device

        # 构建子模块
        self.fuser = MultimodalFuser(
            fusion_dim=model_cfg.fusion_dim,
            attention_heads=model_cfg.attention_heads,
            attention_layers=model_cfg.attention_layers,
            ffn_dim=model_cfg.ffn_dim,
            dropout=model_cfg.dropout,
            audio_model=model_cfg.audio_model,
            gesture_backbone=model_cfg.gesture_backbone,
            text_model=model_cfg.text_model,
        )
        self.decomposer = TaskDecomposer(
            d_model=model_cfg.fusion_dim,
            n_layers=model_cfg.decomposer_layers,
            max_subtasks=model_cfg.max_subtasks,
        )
        self.condition_handler = DynamicConditionHandler()
        self.knowledge_integrator = KnowledgeIntegrator()

        self.enhanced_planner = EnhancedPlanner(
            benchmark_dir=base_cfg.benchmark_dir,
            fuser=self.fuser,
            decomposer=self.decomposer,
            condition_handler=self.condition_handler,
            knowledge_integrator=self.knowledge_integrator,
            device=self.device,
        )
        self.baseline_planner = BaselinePlanner(
            benchmark_dir=base_cfg.benchmark_dir,
        )

        logger.info("MultimodalVLPA system initialised on device=%s.", self.device)

    def process_scenario(self, scenario: ScenarioSample) -> PlanResult:
        """
        运行增强型多模态规划器。

        该方法执行完整的多模态规划流水线，
        包括多模态融合、任务分解、路径规划等步骤。

        Args:
            scenario: 场景样本对象

        Returns:
            PlanResult: 规划结果对象
        """
        return self.enhanced_planner.plan(scenario)

    def get_baseline_result(self, scenario: ScenarioSample) -> PlanResult:
        """
        运行原始文本单模态基线规划器。

        该方法执行基线版本的规划，只使用文本输入，
        不使用音频、手势和标注等其他模态信息。

        Args:
            scenario: 场景样本对象

        Returns:
            PlanResult: 规划结果对象
        """
        return self.baseline_planner.plan(scenario)

    def process_interactive(
        self,
        scenario: ScenarioSample,
        updates: List[Dict[str, Any]],
    ) -> List[PlanResult]:
        """
        模拟交互式飞行任务中的动态指令更新。

        该方法模拟无人机在飞行过程中接收新的指令或约束条件，
        并执行快速重规划。

        Args:
            scenario: 初始场景对象
            updates: 更新列表，每个元素包含:
                - new_instruction: 新的文本指令字符串
                - new_constraints: 新的约束条件字典（可选）

        Returns:
            results: 规划结果列表（初始结果 + 每次更新的结果）
        """
        results = [self.enhanced_planner.plan(scenario)]

        state = {
            "image_id": scenario.image_id,
            "targets_pct": {
                t.name: {"type": t.target_type, "coordinates": list(t.coordinates_percent)}
                for t in scenario.targets
            },
            "obstacles_pct": {
                o.name: {"type": o.target_type, "coordinates": list(o.coordinates_percent)}
                for o in scenario.obstacles
            },
            "original_instruction": scenario.text_instruction,
        }

        for update in updates:
            new_constraints = update.get("new_constraints", {})
            result = self.condition_handler.trigger_replan(
                remaining_tasks=[],
                new_constraints=new_constraints,
                scenario_state=state,
            )
            if result is not None:
                results.append(result)

        return results
