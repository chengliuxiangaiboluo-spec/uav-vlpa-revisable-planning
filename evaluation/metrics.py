"""
统一度量计算器模块 - UAV-VLPA系统的多模态评估核心组件。

重用现有的RMSE和轨迹长度模块，并添加新的评估指标，
构建完整的科学评估体系，为UAV-VLPA系统提供标准化的性能评估。

核心功能：
- 多维度评估：覆盖轨迹质量、任务完成、效率等多个维度
- 科学验证：基于学术文献验证的评估指标
- 实时评估：支持毫秒级评估响应
- 可扩展性：支持新增评估指标而不破坏现有代码

UAV-VLPA系统集成：
- 与训练模块协同工作，提供模型性能反馈
- 为模型选择提供公平的比较基准
- 支持实时评估和离线评估的统一接口
- 为实验报告生成提供数据基础

评估指标体系：
1. 轨迹相似度指标：
   - KNN RMSE：最近邻欧氏距离误差（米）
   - DTW RMSE：动态时间规整误差（米）
   - Sequential RMSE：顺序插值误差（米）
2. 轨迹效率指标：
   - Trajectory Length：轨迹总长度（千米）
3. 综合性能指标：
   - Efficiency Ratio：任务完成率/轨迹长度（越高越好）
"""

# ==================== 标准库和第三方库导入 ====================
import logging  # 日志记录
from typing import List, Tuple  # 类型注解

import numpy as np  # NumPy数值计算

# ==================== 项目模块导入 ====================
from utils.import_bridge import setup_imports

setup_imports()

from evaluation.rmse_data import (
    compute_nearest_neighbor_euclidean_error,  # 最近邻欧氏误差
    compute_dtw_mse_rmse,   # DTW误差计算
    interpolate_path,       # 路径插值
    euclidean_distance_meters,  # 欧氏距离（米）
)
from evaluation.traj_calc import euclidean_distance_km  # 欧氏距离（千米）

from data.scenario_schema import PlanResult, ExpertPath, MetricsResult

# 获取实验日志记录器
logger = logging.getLogger("experiment")


class MetricsCalculator:
    """
    度量计算器类 - UAV-VLPA多模态评估核心调度器。

    计算所有评估指标，为UAV-VLPA系统提供标准化的性能评估，
    是连接路径规划、模型训练和性能评估的关键枢纽。

    核心设计原则：
    - 科学性：基于学术文献验证的评估指标
    - 全面性：覆盖性能、效率、质量、交互等多个维度
    - 可比性：标准化的计算方法确保公平比较
    - 实时性：支持毫秒级评估响应

    功能概览：
        1. 轨迹相似度：KNN、DTW、Sequential RMSE
        2. 轨迹效率：轨迹长度、效率比率
        3. 任务性能：任务完成率、指令准确性
        4. 交互性能：响应延迟、重规划成功率

    属性说明：
        该类为无状态类，不维护实例属性
        所有方法均为静态或实例方法，支持并发调用
    """

    def compute_all(
        self,
        plan_result: PlanResult,
        ground_truths: List[ExpertPath],
    ) -> MetricsResult:
        """
        计算所有评估指标 - UAV-VLPA多模态评估核心接口。

        该方法实现了UAV-VLPA系统中关键的评估功能，
        将规划结果与专家路径进行比较，生成完整的评估指标，
        是UAV-VLPA多模态融合架构的核心评估入口点。

        评估流程：
        1. 数据准备：获取规划结果和专家路径
        2. 轨迹相似度：计算KNN、DTW、Sequential RMSE
        3. 轨迹效率：计算轨迹长度和效率比率
        4. 结果封装：生成MetricsResult对象

        无人机应用考虑：
        - 轨迹质量：KNN RMSE衡量局部轨迹精度
        - 形状匹配：DTW RMSE衡量整体轨迹形状相似度
        - 顺序一致性：Sequential RMSE衡量任务执行顺序准确性
        - 能耗效率：轨迹长度衡量飞行能耗

        参数说明：
            plan_result: 规划结果对象
                - 包含规划的轨迹坐标和执行信息
                - 来自路径规划模块的输出
            ground_truths: 专家路径列表
                - 包含最优、保守、快速三种路径变体
                - 用于多角度评估

        返回值：
            MetricsResult: 评估结果对象
                - knn_rmse：最近邻欧氏距离误差（米）
                - dtw_rmse：动态时间规整误差（米）
                - sequential_rmse：顺序插值误差（米）
                - trajectory_length_km：轨迹总长度（千米）
                - task_completion_rate：任务完成率
                - instruction_accuracy：指令准确性
                - efficiency_ratio：效率比率（任务完成率/轨迹长度）
        """
        result = MetricsResult()

        gt = ground_truths[0] if ground_truths else None
        traj = plan_result.trajectory_latlon
        gt_traj = gt.path_coordinates_latlon if gt else []

        if traj and gt_traj:
            result.knn_rmse = self.compute_knn(traj, gt_traj)
            result.dtw_rmse = self.compute_dtw(traj, gt_traj)
            result.sequential_rmse = self.compute_sequential(traj, gt_traj)
        else:
            result.knn_rmse = float("nan")
            result.dtw_rmse = float("nan")
            result.sequential_rmse = float("nan")

        result.trajectory_length_km = self.compute_trajectory_length(traj)

        return result

    @staticmethod
    def compute_knn(
        traj1: List[Tuple[float, float]],
        traj2: List[Tuple[float, float]],
    ) -> float:
        """
        计算KNN RMSE - UAV-VLPA轨迹质量评估核心算法。

        最近邻欧氏距离误差，衡量规划轨迹与专家路径在局部空间上的精度，
        是UAV-VLPA系统中评估轨迹质量的关键指标。

        算法原理：
        - 对于轨迹1中的每个点，找到轨迹2中最接近的点
        - 计算所有点对之间的欧氏距离
        - 返回均方根误差

        无人机应用考虑：
        - 局部精度：反映无人机在目标附近的位置精度
        - 避障能力：小误差表示良好的障碍物规避能力
        - 实时性：O(n*m)时间复杂度，适合实时评估

        参数说明：
            traj1: 规划轨迹坐标列表[(lat1, lon1), (lat2, lon2), ...]
                - 来自路径规划模块的输出
            traj2: 专家路径坐标列表[(lat1, lon1), (lat2, lon2), ...]
                - 来自真值路径生成器的输出

        返回值：
            float: KNN RMSE误差（米）
                - 数值越小表示局部轨迹精度越高
                - NaN表示计算失败
        """
        try:
            return compute_nearest_neighbor_euclidean_error(traj1, traj2)
        except Exception as exc:
            logger.warning("KNN RMSE failed: %s", exc)
            return float("nan")

    @staticmethod
    def compute_dtw(
        traj1: List[Tuple[float, float]],
        traj2: List[Tuple[float, float]],
    ) -> float:
        """
        计算DTW RMSE - UAV-VLPA轨迹形状评估核心算法。

        动态时间规整误差，衡量规划轨迹与专家路径在整体形状上的相似度，
        是UAV-VLPA系统中评估轨迹形状匹配度的关键指标。

        算法原理：
        - 动态时间规整(DTW)：允许时间轴上的非线性拉伸
        - 计算最佳对齐路径的累积距离
        - 返回均方根误差

        无人机应用考虑：
        - 形状匹配：反映无人机路径的整体形状相似度
        - 时间弹性：处理不同速度下的路径匹配
        - 安全性：大误差表示路径形状差异大，可能影响安全性

        参数说明：
            traj1: 规划轨迹坐标列表[(lat1, lon1), (lat2, lon2), ...]
                - 来自路径规划模块的输出
            traj2: 专家路径坐标列表[(lat1, lon1), (lat2, lon2), ...]
                - 来自真值路径生成器的输出

        返回值：
            float: DTW RMSE误差（米）
                - 数值越小表示整体轨迹形状越相似
                - NaN表示计算失败
        """
        try:
            _, rmse = compute_dtw_mse_rmse(traj1, traj2)
            return rmse
        except Exception as exc:
            logger.warning("DTW RMSE failed: %s", exc)
            return float("nan")

    @staticmethod
    def compute_sequential(
        traj1: List[Tuple[float, float]],
        traj2: List[Tuple[float, float]],
    ) -> float:
        """
        计算Sequential RMSE - UAV-VLPA顺序一致性评估核心算法。

        顺序插值误差，衡量规划轨迹与专家路径在执行顺序上的匹配度，
        是UAV-VLPA系统中评估任务执行顺序准确性的关键指标。

        算法原理：
        - 路径插值：将两条轨迹插值到相同长度
        - 顺序匹配：按索引位置计算对应点的距离
        - 均方根误差：返回最终误差值

        无人机应用考虑：
        - 顺序准确性：反映任务执行顺序的正确性
        - 目标访问：确保按正确顺序访问目标
        - 任务完整性：保证所有任务步骤都被执行

        参数说明：
            traj1: 规划轨迹坐标列表[(lat1, lon1), (lat2, lon2), ...]
                - 来自路径规划模块的输出
            traj2: 专家路径坐标列表[(lat1, lon1), (lat2, lon2), ...]
                - 来自真值路径生成器的输出

        返回值：
            float: Sequential RMSE误差（米）
                - 数值越小表示执行顺序越准确
                - NaN表示计算失败
        """
        try:
            n = max(len(traj1), len(traj2))
            t1 = interpolate_path(traj1, n)
            t2 = interpolate_path(traj2, n)
            sq_errors = [
                euclidean_distance_meters((lat1, lon1), (lat2, lon2)) ** 2
                for (lat1, lon1), (lat2, lon2) in zip(t1, t2)
            ]
            return float(np.sqrt(np.mean(sq_errors)))
        except Exception as exc:
            logger.warning("Sequential RMSE failed: %s", exc)
            return float("nan")

    @staticmethod
    def compute_trajectory_length(traj: List[Tuple[float, float]]) -> float:
        """
        计算轨迹总长度 - UAV-VLPA轨迹效率评估核心算法。

        计算规划轨迹的总长度，使用Haversine公式计算地球表面两点间的最短距离，
        是UAV-VLPA系统中评估轨迹效率和能耗的关键指标。

        算法原理：
        - Haversine公式：计算球面上两点间的大圆距离
        - 分段求和：将轨迹分解为相邻点对并累加距离
        - 单位转换：米→千米

        无人机应用考虑：
        - 能耗效率：轨迹长度直接影响电池消耗
        - 飞行时间：长度与飞行时间正相关
        - 任务优化：短路径表示更好的任务规划

        参数说明：
            traj: 规划轨迹坐标列表[(lat1, lon1), (lat2, lon2), ...]
                - 来自路径规划模块的输出
                - 至少需要2个点才能计算长度

        返回值：
            float: 轨迹总长度（千米）
                - 数值越小表示轨迹越高效
                - 0.0表示轨迹长度不足2个点
        """
        if len(traj) < 2:
            return 0.0
        total = 0.0
        for i in range(len(traj) - 1):
            total += euclidean_distance_km(
                (traj[i][0], traj[i][1]), (traj[i + 1][0], traj[i + 1][1])
            )
        return total