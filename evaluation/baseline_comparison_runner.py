"""
基线对比运行器模块。
版本: v42.0-geometric (all methods are evaluated from realised trajectories)

本模块实现完整的基线对比实验框架，将多模态融合基线、任务分解基线、
路径规划基线和UAV系统基线整合到统一的评估框架中，生成对比报告。

支持的对比模式:
    1. fusion_comparison: 多模态融合方法对比
       对比对象: AFFNet / MAFTNet / SCAL / LPANet / MultimodalFuser (本项目)
    2. decomposition_comparison: 任务分解方法对比
       对比对象: SIPSA / UniGoal / UAV-CodeAgents / HSATD (本项目)
    3. path_planning_comparison: 路径规划方法对比
       对比对象: iKap / CSGLSO / A* (本项目)
    4. system_comparison: 多模态UAV系统对比
       对比对象: UAV-VLA-style / UAV-VLN-style / AerialVLN-style /
                 CityNav-style / Enhanced-VLPA (本项目)

输出:
    - results/baseline_comparison/ 目录下的图表、CSV、JSON报告
    - 包含统计检验、效应量、置信区间
"""

import os
import logging
import json
from math import erf, sqrt
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict

import numpy as np
import torch

from data.scenario_schema import (
    ScenarioSample,
    PlanResult,
    MetricsResult,
    ExpertPath,
    AtomicTask,
    WaypointTarget,
)
from evaluation.metrics import MetricsCalculator
from evaluation.task_completion import TaskCompletionEvaluator
from evaluation.instruction_accuracy import InstructionAccuracyEvaluator
from evaluation.instruction_accuracy_fine import FineGrainedIAEvaluator
from evaluation.statistical_analysis import StatisticalAnalyzer
from evaluation.comparison_runner import GroupResult, ComparisonResult

logger = logging.getLogger("experiment")


@dataclass
class BaselineComparisonResult:
    """基线对比结果聚合类。"""
    comparison_mode: str = ""  # fusion | decomposition | path_planning | system
    groups: Dict[str, GroupResult] = field(default_factory=dict)
    scenarios: List[ScenarioSample] = field(default_factory=list)
    statistical_tests: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)  # 方法名称、描述等


class BaselineComparisonRunner:
    """
    基线对比运行器 — 支持四种对比模式的统一执行框架。

    使用方式:
        # 多模态融合对比
        runner = BaselineComparisonRunner(
            mode='fusion',
            benchmark_dir=benchmark_dir,
            reference_planner=enhanced_planner,
        )
        runner.add_baseline('early_mlp', EarlyFusionMLP(...))
        runner.add_baseline('tfn', TensorFusionNetwork(...))
        result = runner.run(test_scenarios)

    对比结果:
        - metrics: 每个方法的逐场景指标
        - statistical_tests: 配对t检验、Cohen's d、95% CI
        - 与reference_planner (Enhanced-VLPA) 的对比
    """

    def __init__(
        self,
        mode: str,
        benchmark_dir: str,
        reference_planner=None,
        visit_threshold_m: float = 50.0,
        avoid_distance_m: float = 30.0,
        alpha: float = 0.05,
    ):
        """
        Args:
            mode: 对比模式 (fusion|decomposition|path_planning|system)
            benchmark_dir: 基准数据目录
            reference_planner: 参照规划器 (通常为EnhancedPlanner)
            visit_threshold_m: 目标访问阈值(米)
            avoid_distance_m: 障碍物避让距离(米)
            alpha: 显著性水平
        """
        if mode not in (
            "fusion", "decomposition", "path_planning", "system"
        ):
            raise ValueError(
                f"Unknown comparison mode: {mode}. "
                f"Supported: fusion, decomposition, path_planning, system"
            )

        self.mode = mode
        self.benchmark_dir = benchmark_dir
        self.reference_planner = reference_planner
        self.metrics_calc = MetricsCalculator()
        self.completion_eval = TaskCompletionEvaluator(
            visit_threshold_m, avoid_distance_m
        )
        self.accuracy_eval = InstructionAccuracyEvaluator()
        self.fine_ia_eval = FineGrainedIAEvaluator()
        self.stat_analyzer = StatisticalAnalyzer()
        self.alpha = alpha

        # 基线方法注册表: {name: (planner, description)}
        self._baselines: Dict[str, Tuple[Any, str]] = {}

    def add_baseline(self, name: str, planner: Any, description: str = "") -> None:
        """注册一个基线方法。"""
        if name in self._baselines:
            logger.warning("Baseline %s already exists, overwriting.", name)
        self._baselines[name] = (planner, description)

    def list_baselines(self) -> List[Tuple[str, str]]:
        """列出所有已注册的基线方法。"""
        return [(name, desc) for name, (_, desc) in self._baselines.items()]

    # ==================== 主执行入口 ====================

    def run(
        self, test_scenarios: List[ScenarioSample],
        reference_group: Optional[Any] = None,
    ) -> BaselineComparisonResult:
        """
        执行完整的基线对比实验。

        Args:
            test_scenarios: 测试场景列表
            reference_group: 预计算的参照组结果(GroupResult)。
                若提供则直接使用，跳过内部参照评估，
                确保所有模式使用完全相同的参照结果。

        Returns:
            BaselineComparisonResult: 完整对比结果
        """
        result = BaselineComparisonResult(
            comparison_mode=self.mode,
            scenarios=test_scenarios,
        )

        # 评估参照方法 (Enhanced-VLPA)
        if reference_group is not None:
            # 使用预计算的参照结果（保证跨模式一致性）
            result.groups["reference"] = reference_group
            result.metadata["reference"] = {
                "name": "Enhanced-VLPA",
                "description": "本项目核心系统: CrossModalAttn+MGHA+HSATD+A*",
            }
        elif self.reference_planner is not None:
            logger.info("[BaselineComparison] Evaluating reference: Enhanced-VLPA")
            ref_metrics, ref_plans = self._evaluate_planner(
                self.reference_planner, test_scenarios, "reference"
            )
            result.groups["reference"] = GroupResult(
                group_name="Enhanced-VLPA (reference)",
                metrics=ref_metrics,
                plan_results=ref_plans,
            )
            result.metadata["reference"] = {
                "name": "Enhanced-VLPA",
                "description": "本项目核心系统: CrossModalAttn+MGHA+HSATD+A*",
            }

        # 评估每个基线方法
        for name, (planner, desc) in self._baselines.items():
            logger.info("[BaselineComparison] Evaluating baseline: %s", name)
            try:
                base_metrics, base_plans = self._evaluate_planner(
                    planner, test_scenarios, name
                )
                result.groups[name] = GroupResult(
                    group_name=name,
                    metrics=base_metrics,
                    plan_results=base_plans,
                )
                result.metadata[name] = {"name": name, "description": desc}
            except Exception as e:
                logger.error("[BaselineComparison] Failed for %s: %s", name, e)
                continue

        # 执行统计检验
        result.statistical_tests = self._run_statistical_tests(result)

        return result

    # ==================== 规划器评估 ====================

    def _evaluate_planner(
        self,
        planner: Any,
        scenarios: List[ScenarioSample],
        group_name: str,
    ) -> Tuple[List[MetricsResult], List[PlanResult]]:
        """评估单个规划器在所有场景上的性能。

        评估策略：
        所有方法（参照和基线）使用完全相同的评估流程，
        不施加任何人为惩罚。差异仅来源于模型架构和训练权重本身。
        基线融合器经过与参照系统相同的训练数据和损失函数训练，
        确保公平对比。
        """
        metrics_list: List[MetricsResult] = []
        plan_list: List[PlanResult] = []

        for i, sc in enumerate(scenarios, 1):
            logger.debug("[%s] Scenario %d/%d: %s",
                         group_name, i, len(scenarios), sc.scenario_id)
            try:
                plan = planner.plan(sc)
            except Exception as e:
                logger.error("[%s] plan failed for %s: %s",
                             group_name, sc.scenario_id, e)
                plan = PlanResult()

            # 真值路径
            gt_paths = sc.ground_truth_paths
            gt = gt_paths[0] if gt_paths else ExpertPath()
            gt_tasks = self._ensure_atomic_tasks(sc.expert_atomic_tasks)

            # 计算指标
            m = self.metrics_calc.compute_all(plan, [gt])
            m.scenario_id = sc.scenario_id

            # TCR is always calculated from the realised trajectory. Never
            # apply a method-dependent, hand-tuned probability correction.
            m.task_completion_rate = self.completion_eval.evaluate(plan, sc)

            # Efficiency
            if m.trajectory_length_km > 0:
                m.efficiency_ratio = m.task_completion_rate / m.trajectory_length_km
            else:
                m.efficiency_ratio = 0.0

            # IA（细粒度指令文本级评估）
            if hasattr(planner, 'last_tasks') and planner.last_tasks:
                pseudo_tasks = self._ensure_atomic_tasks(planner.last_tasks)
                if gt_tasks:
                    try:
                        m.instruction_accuracy = self.fine_ia_eval.evaluate(
                            pseudo_tasks, gt_tasks,
                            instruction_text=sc.text_instruction,
                            targets=sc.targets, obstacles=sc.obstacles,
                        )
                    except Exception:
                        m.instruction_accuracy = 0.5
                else:
                    m.instruction_accuracy = 0.5
            else:
                m.instruction_accuracy = 0.5

            metrics_list.append(m)
            plan_list.append(plan)

        return metrics_list, plan_list

    @staticmethod
    def _ensure_atomic_tasks(tasks: List[Any]) -> List[AtomicTask]:
        """确保任务列表中的元素都是AtomicTask对象。"""
        result = []
        for t in tasks:
            if isinstance(t, AtomicTask):
                result.append(t)
            elif isinstance(t, dict):
                target = None
                if t.get("target") and isinstance(t["target"], dict):
                    target = WaypointTarget(**t["target"])
                elif t.get("target") and isinstance(t["target"], WaypointTarget):
                    target = t["target"]
                result.append(AtomicTask(
                    task_type=t.get("task_type", "fly_to"),
                    target=target,
                    priority=t.get("priority", 1),
                    conditions=t.get("conditions", []),
                ))
        return result

    # ==================== 统计检验 ====================

    def _run_statistical_tests(
        self, result: BaselineComparisonResult
    ) -> Dict[str, Any]:
        """对每个基线与参照方法做配对t检验。"""
        if "reference" not in result.groups:
            return {}

        ref_metrics = result.groups["reference"].metrics
        tests = {}

        metric_keys = [
            ("task_completion_rate", "TCR"),
            ("instruction_accuracy", "IA"),
            ("efficiency_ratio", "Efficiency"),
            ("trajectory_length_km", "TrajLen"),
            ("dtw_rmse", "DTW_RMSE"),
            ("sequential_rmse", "Seq_RMSE"),
            ("knn_rmse", "KNN_RMSE"),
        ]

        for baseline_name, group in result.groups.items():
            if baseline_name == "reference":
                continue
            base_metrics = group.metrics

            for attr, label in metric_keys:
                entry = {}
                ref_vals = [getattr(m, attr, np.nan) for m in ref_metrics]
                base_vals = [getattr(m, attr, np.nan) for m in base_metrics]
                # 清理NaN
                ref_clean = [v for v in ref_vals if not (v != v)]
                base_clean = [v for v in base_vals if not (v != v)]

                # 描述统计
                for gname, gvals in [("reference", ref_clean),
                                     (baseline_name, base_clean)]:
                    if gvals:
                        arr = np.array(gvals)
                        entry[f"{gname}_mean"] = float(np.mean(arr))
                        entry[f"{gname}_std"] = (
                            float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
                        )
                        ci = self.stat_analyzer.confidence_interval(gvals)
                        entry[f"{gname}_ci_lower"] = ci[0]
                        entry[f"{gname}_ci_upper"] = ci[1]
                    else:
                        for suffix in ["_mean", "_std", "_ci_lower", "_ci_upper"]:
                            entry[f"{gname}{suffix}"] = float("nan")

                # 配对t检验
                if ref_clean and base_clean:
                    n = min(len(ref_clean), len(base_clean))
                    r = ref_clean[:n]
                    b = base_clean[:n]
                    try:
                        tt = self.stat_analyzer.paired_ttest(r, b)
                        cd = self.stat_analyzer.effect_size_cohens_d(r, b)  # d>0 = reference更好
                        entry["p_value"] = tt["p_value"]
                        entry["cohens_d"] = cd
                        entry["significant"] = tt["p_value"] < self.alpha
                    except Exception:
                        entry["p_value"] = float("nan")
                        entry["cohens_d"] = float("nan")
                        entry["significant"] = False
                else:
                    entry["p_value"] = float("nan")
                    entry["cohens_d"] = float("nan")
                    entry["significant"] = False

                tests[f"{baseline_name}|{label}"] = entry

        return tests
