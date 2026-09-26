"""
统计分析工具模块 - UAV-VLPA实验数据科学评估组件。

提供配对t检验、置信区间、效应量、汇总表等统计方法，
是UAV-VLPA系统中科学评估和论文撰写的核心工具。

核心功能：
- 配对t检验：比较两组实验的统计显著性
- 置信区间：计算均值的置信区间
- 效应量计算：Cohen's d效应量
- 汇总表生成：生成三组对比的统计摘要表

UAV-VLPA系统集成：
- 为实验结果提供统计学支持
- 支持学术论文中的显著性检验
- 生成标准化的对比分析表格
"""

# 导入日志模块，用于记录分析过程信息
import logging
# 导入类型提示，增强代码可读性
from typing import List, Tuple, Dict, Any

# 导入NumPy库，用于数组操作和数学计算
import numpy as np
# 导入scipy.stats，用于统计检验
from scipy import stats
# 导入pandas，用于数据框操作和表格生成
import pandas as pd

# 获取实验日志记录器
logger = logging.getLogger("experiment")


# =========================================================================
# 全项目统一统计检验入口
# =========================================================================

def one_sample_test(
    values_a: List[float],
    values_b: List[float] = None,
    popmean: float = 0.0,
) -> Dict[str, float]:
    """
    统一单样本t检验 - 全项目唯一的显著性检验方法。

    所有实验（合成基准、消融、组件替换、真实数据验证）的
    显著性检验均通过本函数完成，检验改进量 Δ 的均值是否显著
    不同于 popmean（默认 0）：

        H0: E[Δ] = popmean

    两种使用方式（数学上完全等价于配对t检验）：
        1. 配对比较：传入 values_a（我方）与 values_b（基线），
           内部计算 Δ = a - b，检验 E[Δ] = 0。
           适用于同一场景/同一种子下两种方法的对比。
        2. 固定基线比较：传入 values_a 与 popmean=固定基线值，
           检验 E[a] = popmean。适用于 Table VII 组件替换对比
           （基线为固定值，无逐种子样本）。

    效应量统一采用 Cohen's d = mean(Δ - popmean) / std(Δ)。

    参数说明：
        values_a: 样本值列表（我方方法，或改进后的值）
        values_b: 配对基线值列表（可选）；None 表示与 popmean 比较
        popmean: 原假设下的总体均值，默认 0.0

    返回值：
        Dict[str, float]:
            - t_statistic: t统计量
            - p_value: 双尾p值
            - df: 自由度
            - cohens_d: 效应量
    """
    a = np.asarray(values_a, dtype=float)
    if values_b is not None:
        b = np.asarray(values_b, dtype=float)
        n = min(len(a), len(b))
        deltas = a[:n] - b[:n]
    else:
        deltas = a

    n = len(deltas)
    if n < 2:
        return {"t_statistic": 0.0, "p_value": 1.0, "df": 0.0, "cohens_d": 0.0}

    centered = deltas - popmean
    mean_d = float(np.mean(centered))
    std_d = float(np.std(centered, ddof=1))

    if std_d == 0:
        return {
            "t_statistic": 0.0,
            "p_value": 1.0 if mean_d == 0 else 0.0,
            "df": float(n - 1),
            "cohens_d": 0.0,
        }

    # 单样本t检验（双尾）
    t_stat, p_val = stats.ttest_1samp(deltas, popmean)

    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_val),
        "df": float(n - 1),
        "cohens_d": mean_d / std_d,
    }


class StatisticalAnalyzer:
    """
    统计分析器 - UAV-VLPA实验组间统计对比核心组件。

    提供实验组之间的统计比较方法，
    用于验证多模态系统的性能提升是否具有统计显著性。

    统计方法：
    - 配对t检验：检验两组配对样本的均值差异
    - 置信区间：估计总体均值的可能范围
    - Cohen's d：衡量效应大小（实际意义）
    - 汇总表：整合所有统计结果的对比表格
    """

    @staticmethod
    def paired_ttest(
        group_a: List[float],
        group_b: List[float],
    ) -> Dict[str, float]:
        """
        执行配对双尾t检验 - UAV-VLPA统计显著性检验核心算法。

        比较两组配对样本，检验它们的均值是否存在统计显著性差异。

        算法原理：
        - 统一采用单样本t检验框架：对配对差值 Δ = a - b
          检验 H0: E[Δ] = 0（与配对t检验数学上完全等价）
        - 双尾检验：检测任意方向的差异（不限于单方向）
        - 全部实验共用 one_sample_test 唯一入口，保证口径一致

        无人机应用考虑：
        - 用于比较baseline vs enhanced的性能差异
        - p < 0.05表示差异具有统计显著性
        - 为学术论文提供统计学支持

        参数说明：
            group_a: 第一组样本值列表
                - 例如：baseline方法的指标值
            group_b: 第二组样本值列表
                - 例如：enhanced方法的指标值

        返回值：
            Dict[str, float]: 包含统计结果的字典
                - t_statistic: t统计量
                - p_value: p值（显著性水平）
                - df: 自由度
        """
        # 统一走 one_sample_test（配对模式：检验 E[Δ]=0）
        result = one_sample_test(group_a, group_b)

        # 返回统计结果字典
        return {"t_statistic": result["t_statistic"],
                "p_value": result["p_value"],
                "df": result["df"]}

    @staticmethod
    def confidence_interval(
        data: List[float],
        confidence: float = 0.95,
    ) -> Tuple[float, float]:
        """
        计算均值的置信区间 - UAV-VLPA参数估计核心算法。

        返回均值在指定置信水平下的置信区间（下限，上限）。

        算法原理：
        - 置信区间：估计总体均值的可能范围
        - 95%置信水平：如果重复抽样100次，约95次的区间会包含真实均值
        - 使用t分布：适用于小样本（n < 30）

        算法流程：
        1. 计算样本均值
        2. 计算标准误（SEM）
        3. 查找t分布的临界值
        4. 计算误差范围 = SEM × t临界值
        5. 返回（均值-误差范围，均值+误差范围）

        参数说明：
            data: 样本数据列表
            confidence: 置信水平，默认0.95（95%）

        返回值：
            Tuple[float, float]: (下限, 上限)
                - 均值在该区间内的概率为confidence
        """
        # 将数据转换为NumPy数组
        arr = np.array(data)
        # 获取样本大小
        n = len(arr)
        # 计算样本均值
        mean = np.mean(arr)
        # 计算标准误（Standard Error of Mean）
        # 如果样本数大于1，计算SEM；否则为0
        se = stats.sem(arr) if n > 1 else 0.0

        # 计算误差范围
        # h = SEM × t临界值
        # t临界值由置信水平和自由度决定
        h = se * stats.t.ppf((1 + confidence) / 2, n - 1) if n > 1 else 0.0

        # 返回置信区间（下限，上限）
        return (float(mean - h), float(mean + h))

    @staticmethod
    def effect_size_cohens_d(
        group_a: List[float],
        group_b: List[float],
    ) -> float:
        """
        计算Cohen's d效应量 - UAV-VLPA实际意义评估算法。

        Cohen's d衡量两组之间的标准化均值差异，
        用于评估差异的实际意义（而不仅仅是统计显著性）。

        算法原理（配对样本版本）：
        - 适用于配对设计（同一场景，不同方法）
        - Cohen's d = mean(diff) / std(diff)
        - 其中 diff = group_a - group_b

        效应量解释：
          * 0.2 = 小效应
          * 0.5 = 中等效应
          * 0.8 = 大效应

        参数说明：
            group_a: 第一组样本值列表（配对样本）
                - 例如：enhanced方法的指标值
            group_b: 第二组样本值列表（配对样本）
                - 例如：baseline方法的指标值

        返回值：
            float: Cohen's d效应量
                - 正值表示A > B
                - 负值表示A < B
                - 绝对值越大，效应越强
        """
        # 将输入列表转换为NumPy数组
        a = np.array(group_a)
        b = np.array(group_b)

        # 统一走 one_sample_test 的 Cohen's d（配对模式）
        return one_sample_test(a, b)["cohens_d"]

    def generate_summary_table(
        self,
        baseline_metrics: Dict[str, List[float]],
        enhanced_metrics: Dict[str, List[float]],
        human_metrics: Dict[str, List[float]],
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """
        生成三组对比的汇总数据框 - UAV-VLPA实验报告核心接口。

        生成一个对比baseline、enhanced和human三组的统计摘要DataFrame。

        表格列：
        - Metric: 指标名称
        - Baseline: 基线组均值 ± 置信区间
        - Enhanced: 增强组均值 ± 置信区间
        - Human: 人类组均值 ± 置信区间
        - p-value: baseline vs enhanced的p值
        - Cohen's d: baseline vs enhanced的效应量
        - Significant: 是否具有统计显著性

        算法流程：
        1. 遍历所有指标
        2. 计算每组的置信区间
        3. 执行配对t检验（baseline vs enhanced）
        4. 计算效应量（Cohen's d）
        5. 判断显著性（p < alpha）
        6. 构建结果行
        7. 返回DataFrame

        参数说明：
            baseline_metrics: 基线组指标字典 {指标名: 值列表}
            enhanced_metrics: 增强组指标字典 {指标名: 值列表}
            human_metrics: 人类组指标字典 {指标名: 值列表}
            alpha: 显著性水平，默认0.05

        返回值：
            pd.DataFrame: 包含所有统计结果的汇总表
        """
        # 初始化结果行列表
        rows = []

        # 遍历所有指标
        for metric_name in baseline_metrics:
            # 获取三组的指标值
            base_vals = baseline_metrics[metric_name]
            enh_vals = enhanced_metrics.get(metric_name, [])
            hum_vals = human_metrics.get(metric_name, [])

            # ==================== 计算置信区间 ====================
            # 计算每组的置信区间（如果数据存在）
            base_ci = self.confidence_interval(base_vals) if base_vals else (0, 0)
            enh_ci = self.confidence_interval(enh_vals) if enh_vals else (0, 0)
            hum_ci = self.confidence_interval(hum_vals) if hum_vals else (0, 0)

            # ==================== 统计检验 ====================
            # 如果baseline和enhanced都有数据，执行统计检验
            if base_vals and enh_vals:
                # 执行配对t检验
                tt = self.paired_ttest(base_vals, enh_vals)
                # 计算Cohen's d效应量
                cd = self.effect_size_cohens_d(enh_vals, base_vals)
            else:
                # 否则返回NaN
                tt = {"p_value": float("nan")}
                cd = float("nan")

            # ==================== 构建结果行 ====================
            rows.append({
                # 指标名称
                "Metric": metric_name,
                # Baseline组：均值 [置信区间]
                "Baseline": f"{np.mean(base_vals):.3f} [{base_ci[0]:.3f}, {base_ci[1]:.3f}]" if base_vals else "N/A",
                # Enhanced组：均值 [置信区间]
                "Enhanced": f"{np.mean(enh_vals):.3f} [{enh_ci[0]:.3f}, {enh_ci[1]:.3f}]" if enh_vals else "N/A",
                # Human组：均值 [置信区间]
                "Human": f"{np.mean(hum_vals):.3f} [{hum_ci[0]:.3f}, {hum_ci[1]:.3f}]" if hum_vals else "N/A",
                # p值（baseline vs enhanced）
                "p-value": f"{tt['p_value']:.4f}",
                # Cohen's d效应量
                "Cohen's d": f"{cd:.3f}" if not np.isnan(cd) else "N/A",
                # 是否显著（p < alpha）
                "Significant": "Yes" if tt["p_value"] < alpha else "No",
            })

        # 将结果列表转换为pandas DataFrame并返回
        return pd.DataFrame(rows)
