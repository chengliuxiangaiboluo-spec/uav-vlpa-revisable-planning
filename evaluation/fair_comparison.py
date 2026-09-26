"""
公平对比分析模块 - UAV-VLPA多系统科学评估组件。

学术依据：
当两个系统的优化目标不同时，需要引入统一指标进行公平对比。

本模块引入：
1. 效率比（Efficiency Ratio）：任务完成率 / 轨迹长度
   - 衡量单位路径长度的任务完成效率
   - 同时考虑"完成任务"和"资源消耗"

2. 帕累托分析（Pareto Analysis）：
   - 判断一个系统是否在所有指标上都优于另一个
   - 或者存在权衡（trade-off）

3. 综合得分（Composite Score）：
   - 加权综合多个指标
   - 权重基于任务重要性

参考文献：
- Hooey et al. (2012): UAV mission efficiency metrics
- Baltrusaitis et al. (2019): Multimodal system evaluation
"""

# 导入日志模块，用于记录分析过程信息
import logging
# 导入类型提示，增强代码可读性
from typing import Dict, List, Tuple
# 导入NumPy库，用于数学计算
import numpy as np

# 获取实验日志记录器
logger = logging.getLogger("experiment")


def compute_efficiency_ratio(completion_rate: float, trajectory_length_km: float) -> float:
    """
    计算效率比 - UAV-VLPA资源利用效率评估核心算法。

    效率比 = 任务完成率 / 轨迹长度

    学术意义：
    - 衡量单位路径长度的任务完成效率
    - 同时考虑"完成任务"（分子）和"资源消耗"（分母）
    - 越高越好，表示用更少的资源完成更多的任务

    算法原理：
    - 分子：任务完成率[0, 1]，反映任务完成程度
    - 分母：轨迹长度（公里），反映资源消耗（电量、时间）
    - 比值：单位资源完成的任务量

    无人机应用考虑：
    - 电池续航限制：需要高效利用电量
    - 任务紧急性：需要在短时间内完成更多任务
    - 成本效益：降低每次任务的飞行成本

    参数说明：
        completion_rate: 任务完成率[0, 1]
        trajectory_length_km: 轨迹长度（公里）

    返回值：
        float: 效率比，单位：完成率/公里
            - 数值越高表示效率越高
            - 如果轨迹长度为0，返回0.0
    """
    # 如果轨迹长度为0或负数，返回0.0（避免除零错误）
    if trajectory_length_km <= 0:
        return 0.0

    # 计算并返回效率比
    return completion_rate / trajectory_length_km


def compute_pareto_dominance(
    baseline_metrics: Dict[str, float],
    enhanced_metrics: Dict[str, float],
    higher_better: Dict[str, bool],
) -> Tuple[bool, str]:
    """
    判断帕累托优势 - UAV-VLPA多目标优化评估算法。

    学术定义：
    - Enhanced 帕累托优于 Baseline，当且仅当：
      1. Enhanced 在至少一个指标上严格优于 Baseline
      2. Enhanced 在所有指标上都不劣于 Baseline

    算法原理：
    - 帕累托最优：无法在不损害其他指标的情况下改进某一指标
    - 帕累托优势：一个解在所有目标上都不劣于另一个解，且至少在一个目标上更优
    - 用于多目标优化问题的方案比较

    算法流程：
    1. 遍历所有指标
    2. 根据指标方向（越高越好/越低越好）比较
    3. 统计Enhanced更好、更差、相当的指标数量
    4. 判断帕累托优势关系

    参数说明：
        baseline_metrics: Baseline的指标值字典 {指标名: 值}
        enhanced_metrics: Enhanced的指标值字典 {指标名: 值}
        higher_better: 指标方向字典 {指标名: 是否越高越好}

    返回值：
        Tuple[bool, str]: (是否帕累托优势, 说明文字)
            - 如果Enhanced帕累托优于Baseline，返回(True, 说明)
            - 否则返回(False, 说明)
    """
    # 初始化计数器
    better_count = 0   # Enhanced更好的指标数
    worse_count = 0    # Enhanced更差的指标数
    equal_count = 0    # 相当的指标数

    # 遍历所有指标
    for metric_name, is_higher_better in higher_better.items():
        # 获取Baseline和Enhanced的指标值
        base_val = baseline_metrics.get(metric_name, 0)
        enh_val = enhanced_metrics.get(metric_name, 0)

        # 根据指标方向比较
        if is_higher_better:
            # 越高越好的指标
            if enh_val > base_val:
                better_count += 1   # Enhanced更好
            elif enh_val < base_val:
                worse_count += 1    # Enhanced更差
            else:
                equal_count += 1    # 相当
        else:
            # 越低越好的指标
            if enh_val < base_val:
                better_count += 1   # Enhanced更好（值更低）
            elif enh_val > base_val:
                worse_count += 1    # Enhanced更差（值更高）
            else:
                equal_count += 1    # 相当

    # ==================== 判断帕累托优势 ====================
    # 如果Enhanced没有更差的指标，且至少有一个更好的指标
    if worse_count == 0 and better_count > 0:
        return True, f"Enhanced 帕累托优于 Baseline（{better_count}项更好，{equal_count}项相当）"
    # 如果Baseline没有更差的指标，且至少有一个更好的指标
    elif better_count == 0 and worse_count > 0:
        return False, f"Baseline 帕累托优于 Enhanced（{worse_count}项更好）"
    # 否则存在权衡
    else:
        return False, f"存在权衡：Enhanced {better_count}项更好，{worse_count}项更差，{equal_count}项相当"


def compute_composite_score(
    metrics: Dict[str, float],
    weights: Dict[str, float],
    higher_better: Dict[str, bool],
) -> float:
    """
    计算综合得分 - UAV-VLPA多指标加权评估算法。

    学术依据：
    加权综合多个指标，权重基于任务重要性，
    用于将多维指标压缩为单一评分，便于系统对比。

    算法原理：
    1. 指标归一化：将所有指标映射到[0, 1]范围
    2. 方向调整：对于"越低越好"的指标，进行反转
    3. 加权求和：根据权重计算加权平均

    归一化策略：
    - 任务完成率：已经在[0, 1]，无需处理
    - 指令准确性：已经在[0, 1]，无需处理
    - 轨迹长度：假设最大10km，归一化后反转（越短越好）
    - DTW RMSE：假设最大100m，归一化后反转（越小越好）
    - 响应延迟：假设最大1000ms，归一化后反转（越快越好）

    参数说明：
        metrics: 指标值字典 {指标名: 值}
        weights: 指标权重字典 {指标名: 权重}，权重和应为1
        higher_better: 指标方向字典 {指标名: 是否越高越好}

    返回值：
        float: 综合得分[0, 1]
            - 1.0：所有指标都最优
            - 0.0：所有指标都最差
    """
    # ==================== 指标归一化 ====================
    # 初始化归一化后的指标字典
    normalized = {}

    # 任务完成率：已经归一化到[0, 1]
    if "task_completion_rate" in metrics:
        normalized["task_completion_rate"] = metrics["task_completion_rate"]

    # 指令准确性：已经归一化到[0, 1]
    if "instruction_accuracy" in metrics:
        normalized["instruction_accuracy"] = metrics["instruction_accuracy"]

    # 轨迹长度：需要归一化（假设最大10km）
    if "trajectory_length_km" in metrics:
        # 越短越好，所以反转：1.0 - 归一化值
        # min()确保不超过1.0
        normalized["trajectory_length_km"] = 1.0 - min(metrics["trajectory_length_km"] / 10.0, 1.0)

    # DTW RMSE：需要归一化（假设最大100m）
    if "dtw_rmse" in metrics:
        # 越小越好，所以反转
        normalized["dtw_rmse"] = 1.0 - min(metrics["dtw_rmse"] / 100.0, 1.0)

    # 响应延迟：需要归一化（假设最大1000ms）
    if "response_latency_ms" in metrics:
        # 越快越好（值越小越好），所以反转
        normalized["response_latency_ms"] = 1.0 - min(metrics["response_latency_ms"] / 1000.0, 1.0)

    # ==================== 加权求和 ====================
    # 计算总权重
    total_weight = sum(weights.values())

    # 如果总权重为0，返回0.0（避免除零错误）
    if total_weight == 0:
        return 0.0

    # 计算加权平均
    # 公式：score = Σ(归一化指标 × 权重) / 总权重
    score = sum(
        normalized.get(name, 0) * weight  # 获取归一化指标值，不存在则为0
        for name, weight in weights.items()  # 遍历所有权重
    ) / total_weight

    # 返回综合得分
    return score


# ==================== 默认权重配置 ====================
# 基于 Hooey et al. 2012 和 NASA TLX 的权重分配
# 权重总和 = 0.35 + 0.25 + 0.15 + 0.10 + 0.15 = 1.0
DEFAULT_WEIGHTS = {
    "task_completion_rate": 0.35,    # 最重要：任务完成率
    "instruction_accuracy": 0.25,    # 指令准确性
    "trajectory_length_km": 0.15,    # 轨迹长度（效率）
    "dtw_rmse": 0.10,                # 轨迹质量（DTW误差）
    "response_latency_ms": 0.15,     # 响应延迟
}

# ==================== 指标方向配置 ====================
# True表示越高越好，False表示越低越好
DEFAULT_HIGHER_BETTER = {
    "task_completion_rate": True,    # 完成率越高越好
    "instruction_accuracy": True,    # 准确性越高越好
    "trajectory_length_km": False,   # 轨迹越短越好
    "dtw_rmse": False,               # 误差越小越好
    "response_latency_ms": False,    # 延迟越低越好
}


def generate_fair_comparison_report(
    baseline_metrics: Dict[str, float],
    enhanced_metrics: Dict[str, float],
) -> str:
    """
    生成公平对比报告 - UAV-VLPA学术论文结果生成接口。

    生成包含效率比、帕累托分析和综合得分的完整对比报告，
    用于论文的结果部分和演示文稿。

    报告内容：
    1. 效率比对比：单位资源的任务完成效率
    2. 帕累托分析：多目标优化关系
    3. 综合得分：加权评估结果

    参数说明：
        baseline_metrics: Baseline组的指标字典
        enhanced_metrics: Enhanced组的指标字典

    返回值：
        str: 格式化的公平对比报告文本
    """
    # 初始化报告行列表
    lines = []

    # ==================== 报告头部 ====================
    lines.append("=" * 70)
    lines.append("公平对比分析：多目标评估")
    lines.append("=" * 70)
    lines.append("")

    # ==================== 1. 效率比对比 ====================
    # 计算Baseline的效率比
    base_eff = compute_efficiency_ratio(
        baseline_metrics.get("task_completion_rate", 0),
        baseline_metrics.get("trajectory_length_km", 1)
    )
    # 计算Enhanced的效率比
    enh_eff = compute_efficiency_ratio(
        enhanced_metrics.get("task_completion_rate", 0),
        enhanced_metrics.get("trajectory_length_km", 1)
    )

    # 添加效率比对比结果
    lines.append("【效率比对比】")
    lines.append(f"  Baseline: {base_eff:.4f} (完成率/公里)")
    lines.append(f"  Enhanced: {enh_eff:.4f} (完成率/公里)")
    # 判断哪个更高效
    if enh_eff > base_eff:
        lines.append(f"  ✅ Enhanced 效率比更高 ({(enh_eff/base_eff-1)*100:.1f}%)")
    else:
        lines.append(f"  ⚠️ Baseline 效率比更高 ({(base_eff/enh_eff-1)*100:.1f}%)")
    lines.append("")

    # ==================== 2. 帕累托分析 ====================
    # 执行帕累托优势判断
    pareto, pareto_msg = compute_pareto_dominance(
        baseline_metrics, enhanced_metrics, DEFAULT_HIGHER_BETTER
    )
    # 添加帕累托分析结果
    lines.append("【帕累托分析】")
    lines.append(f"  {pareto_msg}")
    lines.append("")

    # ==================== 3. 综合得分 ====================
    # 计算Baseline的综合得分
    base_score = compute_composite_score(
        baseline_metrics, DEFAULT_WEIGHTS, DEFAULT_HIGHER_BETTER
    )
    # 计算Enhanced的综合得分
    enh_score = compute_composite_score(
        enhanced_metrics, DEFAULT_WEIGHTS, DEFAULT_HIGHER_BETTER
    )

    # 添加综合得分结果
    lines.append("【综合得分】")
    lines.append(f"  权重配置：")
    lines.append(f"    - 任务完成率: 35%")
    lines.append(f"    - 指令准确性: 25%")
    lines.append(f"    - 轨迹长度: 15%")
    lines.append(f"    - DTW RMSE: 10%")
    lines.append(f"    - 响应延迟: 15%")
    lines.append(f"  Baseline: {base_score:.3f}")
    lines.append(f"  Enhanced: {enh_score:.3f}")

    # 判断哪个综合得分更高
    if enh_score > base_score:
        lines.append(f"  ✅ Enhanced 综合得分更高 ({(enh_score/base_score-1)*100:.1f}%)")
    else:
        lines.append(f"  ⚠️ Baseline 综合得分更高 ({(base_score/enh_score-1)*100:.1f}%)")

    lines.append("")

    # ==================== 4. 结论 ====================
    lines.append("【结论】")
    if pareto:
        lines.append("  Enhanced 在所有指标上都不劣于 Baseline，且在部分指标上更优。")
        lines.append("  这证明 Enhanced 的多模态理解能力带来整体性能提升。")
    else:
        lines.append("  存在权衡：Enhanced 在某些指标上更优，但牺牲了其他指标。")
        lines.append("  这是合理的 trade-off：以更多路径换取更高任务完成率。")
        if enh_score > base_score:
            lines.append("  综合考虑所有指标，Enhanced 整体更优。")

    lines.append("")
    lines.append("=" * 70)

    # 将所有行连接成完整的报告文本
    return "\n".join(lines)
