"""
主观评分模拟模块 - UAV-VLPA用户体验评估组件。

基于客观指标模拟Likert 1-5级用户满意度评分。
在真实实验中，这些评分将来自人类评估员。

核心功能：
- 多指标加权：综合任务完成率、效率、准确性和响应延迟
- 分数映射：将[0,1]标准化分数映射到[1,5]Likert量表
- 噪声模拟：添加高斯噪声模拟人类评分的 variability

UAV-VLPA系统集成：
- 为评估报告提供主观体验维度
- 与客观指标形成互补，全面评估系统性能
- 支持A/B测试中用户体验的量化比较
"""

# 导入NumPy库，用于数学计算和随机噪声生成
import numpy as np

# 从场景模式定义中导入度量结果数据结构
from data.scenario_schema import MetricsResult


class SubjectiveScoreSimulator:
    """
    主观评分模拟器 - UAV-VLPA用户体验量化核心组件。

    基于客观指标近似计算Likert量表(1-5级)的主观评分，
    是UAV-VLPA系统中用户体验评估的关键组件。

    权重设定依据：
    - 任务完成率 (0.35): 核心指标，决定任务是否成功
    - 效率/轨迹质量 (0.25): 影响资源消耗和执行时间
    - 指令准确性 (0.25): 反映系统理解能力
    - 响应时延 (0.15): 影响用户体验，但在离线规划场景中权重较低

    算法原理：
    1. 指标标准化：将各客观指标归一化到[0,1]范围
    2. 加权求和：根据权重计算综合得分
    3. 分数映射：将[0,1]映射到[1,5]Likert量表
    4. 噪声添加：添加高斯噪声模拟人类评分变异性

    参考文献：
    - Hooey et al. (2012) 对无人机操作员满意度研究
    - NASA TLX (Task Load Index) 权重分配方法
    """

    def __init__(
        self,
        w_completion: float = 0.35,
        w_efficiency: float = 0.25,
        w_accuracy: float = 0.25,
        w_latency: float = 0.15,
    ):
        """
        初始化主观评分模拟器。

        该构造函数配置各客观指标的权重，
        这些权重基于学术文献和无人机操作经验确定。

        参数说明：
            w_completion: 任务完成率权重，默认0.35
                - 最重要的指标，直接影响任务成功与否
            w_efficiency: 效率权重，默认0.25
                - 反映资源利用效率
            w_accuracy: 准确性权重，默认0.25
                - 反映系统理解能力
            w_latency: 延迟权重，默认0.15
                - 影响用户体验，但离线场景中重要性较低
        """
        # 保存任务完成率权重
        self.w_completion = w_completion
        # 保存效率权重
        self.w_efficiency = w_efficiency
        # 保存准确性权重
        self.w_accuracy = w_accuracy
        # 保存延迟权重
        self.w_latency = w_latency

    def simulate(self, metrics: MetricsResult) -> float:
        """
        模拟主观评分 - UAV-VLPA用户体验量化核心接口。

        基于客观度量指标计算模拟的主观评分，
        分数范围为[1, 5]，符合Likert量表标准。

        算法流程：
        1. 指标提取：从metrics对象中提取各项客观指标
        2. 标准化处理：将各指标归一化到[0,1]范围
        3. 加权计算：根据权重计算综合得分
        4. 分数映射：将[0,1]映射到[1,5]
        5. 噪声添加：添加高斯噪声增加真实性

        无人机应用考虑：
        - 完成率和准确性越高，评分越高
        - 轨迹效率越高（RMSE越低），评分越高
        - 响应延迟越低，评分越高

        参数说明：
            metrics: MetricsResult对象
                - 包含所有客观度量指标
                - 来自评估流水线的输出

        返回值：
            float: 模拟的主观评分[1, 5]
                - 1.0：非常不满意
                - 3.0：中性评价
                - 5.0：非常满意
        """
        # ==================== 指标标准化 ====================
        # 任务完成率标准化：限制最大值为1.0
        comp = min(metrics.task_completion_rate, 1.0)

        # 指令准确性标准化：限制最大值为1.0
        acc = min(metrics.instruction_accuracy, 1.0)

        # 轨迹效率计算：RMSE越低越好
        # 使用DTW RMSE作为轨迹质量指标
        # 设置上限500米：0米→1.0（完美），500米→0.0（最差）
        rmse = metrics.dtw_rmse if not np.isnan(metrics.dtw_rmse) else 250.0
        eff = max(0.0, 1.0 - rmse / 500.0)

        # 响应延迟评分：<500ms→1.0（优秀），>5000ms→0.0（差）
        # 使用线性插值计算延迟分数
        lat = metrics.response_latency_ms
        lat_score = max(0.0, 1.0 - (lat - 500) / 4500) if lat > 500 else 1.0

        # ==================== 加权求和 ====================
        # 计算加权综合得分
        raw = (
            self.w_completion * comp        # 任务完成率贡献
            + self.w_efficiency * eff        # 效率贡献
            + self.w_accuracy * acc          # 准确性贡献
            + self.w_latency * lat_score     # 延迟贡献
        )

        # ==================== 分数映射和噪声添加 ====================
        # 将[0,1]范围的raw分数映射到[1,5]Likert量表
        # 公式：score = 1.0 + 4.0 * raw
        # 添加高斯噪声（均值0，标准差0.15）模拟人类评分变异性
        score = 1.0 + 4.0 * raw + np.random.normal(0, 0.15)

        # 限制分数在[1.0, 5.0]范围内
        return float(np.clip(score, 1.0, 5.0))
