"""
强化学习奖励函数模块 - 用于任务分解器优化。

本模块实现复合奖励函数，用于强化学习训练任务分解器，
通过组合多个维度的奖励信号来指导模型学习更优的任务分解策略。

核心奖励公式:
    reward = w1 * completion + w2 * efficiency + w3 * quality - w4 * time_penalty

各分量说明:
    - completion: 任务完成率（已完成目标数/总目标数）
    - efficiency: 轨迹效率（专家路径长度/实际路径长度）
    - quality: 路径质量（基于与专家路径的距离）
    - time_penalty: 时间惩罚（超过阈值的执行时间）

技术特点:
    - 使用Haversine公式计算经纬度距离
    - 支持多维度奖励权重配置
    - 奖励范围标准化到[-1, 1]区间
    - 时间惩罚机制防止过长执行时间

参考文献:
    Sutton & Barto, 2018. "Reinforcement Learning: An Introduction"
    Hooey et al., 2012. "Human Factors Guidelines for Unmanned Aircraft System Control Stations"
"""

# ==================== 标准库和第三方库导入 ====================
import logging  # 日志记录模块
from typing import List, Tuple  # 类型提示支持

# 数值计算库
import numpy as np  # NumPy数值计算

# ==================== 项目模块导入 ====================
from data.scenario_schema import PlanResult, ExpertPath, ScenarioSample  # 计划结果和场景数据结构

# 获取实验日志记录器
logger = logging.getLogger("experiment")


def _haversine_km(lat1, lon1, lat2, lon2):
    """
    计算两个经纬度点之间的大圆距离（千米）。

    使用Haversine公式计算球面距离，这是计算地球表面两点间最短距离的标准方法。

    Args:
        lat1, lon1: 第一个点的纬度和经度
        lat2, lon2: 第二个点的纬度和经度

    Returns:
        distance: 两点之间的距离（千米）
    """
    R = 6371.0  # 地球半径（千米）
    dlat = np.radians(lat2 - lat1)  # 纬度差（弧度）
    dlon = np.radians(lon2 - lon1)  # 经度差（弧度）
    # Haversine公式
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def _trajectory_length_km(traj: List[Tuple[float, float]]) -> float:
    """
    计算轨迹的总长度（千米）。

    遍历轨迹中的所有相邻点，累加它们之间的Haversine距离，
    得到整个轨迹的总长度。

    Args:
        traj: 轨迹坐标列表 [(lat1, lon1), (lat2, lon2), ...]

    Returns:
        total_length: 轨迹总长度（千米）
    """
    total = 0.0
    for i in range(len(traj) - 1):
        total += _haversine_km(traj[i][0], traj[i][1], traj[i + 1][0], traj[i + 1][1])
    return total


class RewardFunction:
    """
    强化学习奖励函数类。

    该类计算规划结果的复合奖励，用于指导任务分解器的学习过程。

    奖励公式:
        reward = w1 * completion + w2 * efficiency + w3 * quality - w4 * time_penalty

    主要组件:
        - w_completion: 任务完成度权重
        - w_efficiency: 轨迹效率权重
        - w_quality: 路径质量权重
        - w_time: 时间惩罚权重
        - time_threshold_ms: 时间阈值（毫秒）
    """

    def __init__(
        self,
        w_completion: float = 0.5,
        w_efficiency: float = 0.2,
        w_quality: float = 0.2,
        w_time: float = 0.1,
        time_threshold_ms: float = 1000.0,
    ):
        """
        初始化奖励函数。

        Args:
            w_completion: 任务完成度权重，默认0.5
            w_efficiency: 轨迹效率权重，默认0.2
            w_quality: 路径质量权重，默认0.2
            w_time: 时间惩罚权重，默认0.1
            time_threshold_ms: 时间阈值（毫秒），超过此阈值开始惩罚
        """
        self.w_completion = w_completion
        self.w_efficiency = w_efficiency
        self.w_quality = w_quality
        self.w_time = w_time
        self.time_threshold_ms = time_threshold_ms

    def compute(
        self,
        plan_result: PlanResult,
        ground_truth: ExpertPath,
        scenario: ScenarioSample,
    ) -> float:
        """
        计算规划结果的复合奖励。

        该方法综合考虑任务完成度、轨迹效率、路径质量和时间惩罚四个维度，
        返回一个标准化的奖励值，用于强化学习训练。

        Args:
            plan_result: 规划结果对象
            ground_truth: 专家路径对象
            scenario: 场景样本对象

        Returns:
            reward: 复合奖励值（约在[-1, 1]范围内）
        """
        comp = self._task_completion_reward(plan_result, scenario)
        eff = self._trajectory_efficiency_reward(plan_result, ground_truth)
        qual = self._path_quality_reward(plan_result, ground_truth)
        time_pen = self._time_penalty(plan_result.execution_time_ms)

        reward = (
            self.w_completion * comp
            + self.w_efficiency * eff
            + self.w_quality * qual
            - self.w_time * time_pen
        )
        return float(reward)

    def _task_completion_reward(self, result: PlanResult, scenario: ScenarioSample) -> float:
        """
        计算任务完成度奖励。

        任务完成度 = 已完成目标数 / 总目标数

        Args:
            result: 规划结果对象
            scenario: 场景样本对象

        Returns:
            completion_reward: 任务完成度奖励（0.0-1.0）
        """
        total = len(scenario.targets)
        if total == 0:
            return 1.0
        completed = len(result.completed_targets)
        return min(completed / total, 1.0)

    def _trajectory_efficiency_reward(self, result: PlanResult, gt: ExpertPath) -> float:
        """
        计算轨迹效率奖励。

        轨迹效率 = 专家路径长度 / 实际路径长度

        Args:
            result: 规划结果对象
            gt: 专家路径对象

        Returns:
            efficiency_reward: 轨迹效率奖励（0.0-1.0）
        """
        if not result.trajectory_latlon or not gt.path_coordinates_latlon:
            return 0.0
        gt_len = _trajectory_length_km(gt.path_coordinates_latlon)
        actual_len = _trajectory_length_km(result.trajectory_latlon)
        if actual_len == 0:
            return 0.0
        return min(gt_len / actual_len, 1.0)

    def _path_quality_reward(self, result: PlanResult, gt: ExpertPath) -> float:
        """
        计算路径质量奖励。

        使用最近邻搜索计算实际轨迹与专家路径的平均距离作为质量指标。

        Args:
            result: 规划结果对象
            gt: 专家路径对象

        Returns:
            quality_reward: 路径质量奖励（0.0-1.0）
        """
        if not result.trajectory_latlon or not gt.path_coordinates_latlon:
            return 0.0
        # 使用最近邻搜索计算平均距离作为质量指标
        from scipy.spatial import cKDTree

        # 将经纬度转换为近似公里单位（1度≈111.32公里）
        gt_arr = np.array(gt.path_coordinates_latlon) * 111.32
        res_arr = np.array(result.trajectory_latlon) * 111.32
        try:
            tree = cKDTree(gt_arr)
            dists, _ = tree.query(res_arr)
            avg_dist = np.mean(dists)
            # 归一化：0距离→1.0质量，1公里→0.0质量
            return float(max(0.0, 1.0 - avg_dist))
        except Exception:
            return 0.0

    def _time_penalty(self, exec_time_ms: float) -> float:
        """
        计算时间惩罚。

        如果执行时间超过阈值，则施加线性惩罚。

        Args:
            exec_time_ms: 执行时间（毫秒）

        Returns:
            time_penalty: 时间惩罚值（0.0-1.0）
        """
        if exec_time_ms <= self.time_threshold_ms:
            return 0.0
        return min((exec_time_ms - self.time_threshold_ms) / self.time_threshold_ms, 1.0)
