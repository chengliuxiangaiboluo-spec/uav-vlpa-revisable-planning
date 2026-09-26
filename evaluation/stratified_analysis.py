"""
分层对比分析模块 - UAV-VLPA多模态贡献度量化评估组件。

通过对比不同场景复杂度下的性能差距，证明差距来源于多模态理解能力，
而非路径策略选择。

学术依据：
- 在单模态场景（SIMPLE）下，Baseline 和 Enhanced 表现应相近
- 在多模态场景（MEDIUM/COMPLEX）下，差距应显著拉大
- 这证明差距来源于多模态理解能力，而非路径效率追求

参考文献：
- Baltrusaitis et al. (2019): Multimodal ML Survey
- Hooey et al. (2012): UAV mission evaluation
"""

# 导入日志模块，用于记录分析过程信息
import logging
# 导入类型提示，增强代码可读性
from typing import List, Dict, Any
# 导入数据类装饰器
from dataclasses import dataclass
# 导入NumPy库，用于数学计算
import numpy as np

# 从场景模式定义中导入必要的数据结构
from data.scenario_schema import (
    ScenarioSample,      # 场景样本
    MetricsResult,       # 度量结果
    ComplexityLevel,     # 复杂度级别
)

# 获取实验日志记录器
logger = logging.getLogger("experiment")


@dataclass
class StratifiedResult:
    """
    分层统计结果数据结构 - UAV-VLPA分层分析结果容器。

    存储单个复杂度级别的统计结果，
    用于对比不同复杂度下的性能差距。

    属性说明：
        complexity: 复杂度级别名称（simple/medium/complex）
        n_scenarios: 该复杂度级别的场景数量
        baseline_completion: Baseline组平均任务完成率
        enhanced_completion: Enhanced组平均任务完成率
        gap: 两组完成率的差距（enhanced - baseline）
        baseline_trajectory: Baseline组平均轨迹长度
        enhanced_trajectory: Enhanced组平均轨迹长度
        trajectory_gap: 两组轨迹长度的差距
    """
    complexity: str                    # 复杂度级别
    n_scenarios: int                   # 场景数量
    baseline_completion: float         # Baseline完成率
    enhanced_completion: float         # Enhanced完成率
    gap: float                         # 完成率差距
    baseline_trajectory: float         # Baseline轨迹长度
    enhanced_trajectory: float         # Enhanced轨迹长度
    trajectory_gap: float              # 轨迹长度差距


class StratifiedAnalyzer:
    """
    分层对比分析器 - UAV-VLPA多模态贡献度分析核心组件。

    核心思想：
    如果差距来源于"路径策略"，则所有场景下差距应一致。
    如果差距来源于"多模态理解能力"，则：
    - SIMPLE 场景：差距小（文本足够）
    - MEDIUM 场景：差距中（部分需要视觉信息）
    - COMPLEX 场景：差距大（严重依赖视觉信息）

    分析方法：
    1. 按复杂度级别分组场景
    2. 计算每组的平均性能指标
    3. 对比Baseline和Enhanced的差距
    4. 分析差距随复杂度的变化趋势
    """

    def analyze(
        self,
        scenarios: List[ScenarioSample],
        baseline_metrics: List[MetricsResult],
        enhanced_metrics: List[MetricsResult],
    ) -> List[StratifiedResult]:
        """
        按复杂度分层分析性能差距 - UAV-VLPA分层分析核心接口。

        将场景按复杂度级别分组，分别统计各层级的性能指标，
        用于验证多模态理解能力对性能提升的贡献。

        算法流程：
        1. 遍历三个复杂度级别（SIMPLE/MEDIUM/COMPLEX）
        2. 筛选出该复杂度的所有场景
        3. 提取Baseline和Enhanced的任务完成率和轨迹长度
        4. 计算平均值和差距
        5. 构建StratifiedResult对象
        6. 返回所有层级的结果列表

        参数说明：
            scenarios: 场景样本列表
            baseline_metrics: Baseline组的度量结果列表
            enhanced_metrics: Enhanced组的度量结果列表

        返回值：
            List[StratifiedResult]: 各复杂度层级的统计结果
        """
        # 初始化结果列表
        results = []

        # 遍历三个复杂度级别
        for level in [ComplexityLevel.SIMPLE, ComplexityLevel.MEDIUM, ComplexityLevel.COMPLEX]:
            # ==================== 场景筛选 ====================
            # 找出所有属于当前复杂度级别的场景索引
            indices = [i for i, s in enumerate(scenarios) if s.complexity == level]

            # 如果该复杂度级别没有场景，跳过
            if not indices:
                continue

            # ==================== 指标提取 ====================
            # 提取Baseline组的任务完成率
            base_completions = [baseline_metrics[i].task_completion_rate for i in indices]
            # 提取Enhanced组的任务完成率
            enh_completions = [enhanced_metrics[i].task_completion_rate for i in indices]
            # 提取Baseline组的轨迹长度
            base_traj = [baseline_metrics[i].trajectory_length_km for i in indices]
            # 提取Enhanced组的轨迹长度
            enh_traj = [enhanced_metrics[i].trajectory_length_km for i in indices]

            # ==================== 构建结果对象 ====================
            result = StratifiedResult(
                complexity=level.value,                                    # 复杂度级别名称
                n_scenarios=len(indices),                                  # 场景数量
                baseline_completion=np.mean(base_completions),            # Baseline平均完成率
                enhanced_completion=np.mean(enh_completions),             # Enhanced平均完成率
                gap=np.mean(enh_completions) - np.mean(base_completions), # 完成率差距
                baseline_trajectory=np.mean(base_traj),                   # Baseline平均轨迹长度
                enhanced_trajectory=np.mean(enh_traj),                    # Enhanced平均轨迹长度
                trajectory_gap=np.mean(enh_traj) - np.mean(base_traj),    # 轨迹长度差距
            )
            # 添加到结果列表
            results.append(result)

        # 返回所有层级的统计结果
        return results

    def generate_report(self, results: List[StratifiedResult]) -> str:
        """
        生成分层分析报告 - UAV-VLPA学术论文结果生成接口。

        生成格式化的文本报告，展示不同复杂度级别下的性能对比，
        用于论文的结果部分和演示文稿。

        报告内容：
        1. 学术假设说明
        2. 各复杂度级别的性能对比表
        3. 关键发现和结论

        参数说明：
            results: 分层统计结果列表

        返回值：
            str: 格式化的分析报告文本
        """
        # 初始化报告行列表
        lines = []

        # ==================== 报告头部 ====================
        lines.append("=" * 70)
        lines.append("分层对比分析：多模态理解能力 vs 路径策略选择")
        lines.append("=" * 70)
        lines.append("")

        # ==================== 学术假设 ====================
        lines.append("学术假设：")
        lines.append("  如果差距来源于'路径策略'，则所有场景下差距应一致。")
        lines.append("  如果差距来源于'多模态理解能力'，则复杂场景差距应更大。")
        lines.append("")

        # ==================== 性能对比表 ====================
        lines.append("-" * 70)
        lines.append(f"{'复杂度':<12} {'场景数':<8} {'Baseline完成率':<15} {'Enhanced完成率':<15} {'差距':<10}")
        lines.append("-" * 70)

        # 遍历结果，添加每一行数据
        for r in results:
            lines.append(
                f"{r.complexity:<12} {r.n_scenarios:<8} "
                f"{r.baseline_completion:<15.3f} {r.enhanced_completion:<15.3f} "
                f"+{r.gap:.3f}"
            )

        lines.append("-" * 70)
        lines.append("")

        # ==================== 关键发现 ====================
        # 如果至少有两个复杂度级别的结果，分析趋势
        if len(results) >= 2:
            # 获取SIMPLE和COMPLEX场景的差距
            simple_gap = results[0].gap if results[0].complexity == "simple" else 0
            complex_gap = results[-1].gap if results[-1].complexity == "complex" else 0

            lines.append("关键发现：")
            lines.append(f"  SIMPLE 场景差距: {simple_gap:.3f}")
            lines.append(f"  COMPLEX 场景差距: {complex_gap:.3f}")

            # 判断差距是否随复杂度增加而显著增大
            if complex_gap > simple_gap * 1.5:
                lines.append("")
                lines.append("✅ 结论：差距随场景复杂度增加而显著增大，")
                lines.append("   证明差距来源于'多模态理解能力'而非'路径策略选择'。")
            else:
                lines.append("")
                lines.append("⚠️ 结论：差距未随复杂度显著变化，")
                lines.append("   可能存在其他影响因素，需进一步分析。")

        lines.append("")
        lines.append("=" * 70)

        # 将所有行连接成完整的报告文本
        return "\n".join(lines)


def compute_modalility_contribution(
    scenarios: List[ScenarioSample],
    baseline_metrics: List[MetricsResult],
    enhanced_metrics: List[MetricsResult],
) -> Dict[str, float]:
    """
    计算多模态理解的贡献度 - UAV-VLPA多模态价值量化算法。

    通过对比单模态和多模态场景下的性能差距，
    量化多模态理解能力对系统性能提升的贡献。

    算法原理：
    - 贡献度 = 多模态场景差距 - 单模态场景差距
    - 如果贡献度 > 0，说明多模态理解能力是主要贡献因素
    - 贡献度越大，说明多模态理解能力越重要

    算法流程：
    1. 计算单模态场景（SIMPLE）的性能差距
    2. 计算多模态场景（MEDIUM+COMPLEX）的性能差距
    3. 计算贡献度 = 多模态差距 - 单模态差距
    4. 计算贡献度百分比 = 贡献度 / 多模态差距 × 100%

    学术意义：
    - 验证多模态系统的核心价值
    - 为论文提供量化证据
    - 指导系统优化方向

    参数说明：
        scenarios: 场景样本列表
        baseline_metrics: Baseline组的度量结果列表
        enhanced_metrics: Enhanced组的度量结果列表

    返回值：
        Dict[str, float]: 包含贡献度分析的字典
            - simple_scenario_gap: 单模态场景差距
            - multimodal_scenario_gap: 多模态场景差距
            - multimodal_contribution: 多模态贡献度
            - contribution_percentage: 贡献度百分比
    """
    # ==================== 单模态场景分析 ====================
    # 筛选所有SIMPLE复杂度的场景索引
    simple_indices = [i for i, s in enumerate(scenarios) if s.complexity == ComplexityLevel.SIMPLE]

    # 初始化单模态差距
    simple_gap = 0.0

    # 如果有单模态场景，计算差距
    if simple_indices:
        # 计算Baseline组在单模态场景的平均完成率
        base_simple = np.mean([baseline_metrics[i].task_completion_rate for i in simple_indices])
        # 计算Enhanced组在单模态场景的平均完成率
        enh_simple = np.mean([enhanced_metrics[i].task_completion_rate for i in simple_indices])
        # 计算差距
        simple_gap = enh_simple - base_simple

    # ==================== 多模态场景分析 ====================
    # 筛选所有MEDIUM和COMPLEX复杂度的场景索引
    mm_indices = [i for i, s in enumerate(scenarios) if s.complexity in (ComplexityLevel.MEDIUM, ComplexityLevel.COMPLEX)]

    # 初始化多模态差距
    mm_gap = 0.0

    # 如果有多模态场景，计算差距
    if mm_indices:
        # 计算Baseline组在多模态场景的平均完成率
        base_mm = np.mean([baseline_metrics[i].task_completion_rate for i in mm_indices])
        # 计算Enhanced组在多模态场景的平均完成率
        enh_mm = np.mean([enhanced_metrics[i].task_completion_rate for i in mm_indices])
        # 计算差距
        mm_gap = enh_mm - base_mm

    # ==================== 贡献度计算 ====================
    # 计算多模态理解能力的贡献度
    # 贡献度 = 多模态场景差距 - 单模态场景差距
    contribution = mm_gap - simple_gap

    # ==================== 返回结果 ====================
    return {
        "simple_scenario_gap": simple_gap,                          # 单模态场景差距
        "multimodal_scenario_gap": mm_gap,                         # 多模态场景差距
        "multimodal_contribution": contribution,                   # 多模态贡献度
        "contribution_percentage": (contribution / mm_gap * 100) if mm_gap > 0 else 0.0,  # 贡献度百分比
    }
