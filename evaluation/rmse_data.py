"""
RMSE和轨迹指标计算工具模块 - UAV-VLPA轨迹质量评估核心组件。

提供计算路径间距离指标的函数，
是UAV-VLPA系统中评估轨迹相似性和质量的基础工具。

核心功能：
- Haversine距离计算：计算地理坐标间的地表距离
- 最近邻误差：计算两条路径间的平均最近邻距离
- DTW对齐误差：使用动态时间规整计算轨迹相似度
- 路径插值：将路径插值到指定点数

UAV-VLPA系统集成：
- 为轨迹质量评估提供多种距离度量方法
- 支持DTW、KNN、Sequential等多种RMSE计算
- 用于轨迹相似度比较和规划器性能评估
"""

# 导入NumPy库，用于数组操作和数学计算
import numpy as np
# 导入类型提示，增强代码可读性
from typing import List, Tuple
# 导入fastdtw库，用于动态时间规整算法
from fastdtw import fastdtw
# 导入数学库，用于三角函数计算
import math


# 地球半径（米）
# 6371000米是国际公认的平均地球半径
EARTH_RADIUS_M = 6_371_000


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    使用Haversine公式计算两个经纬度坐标之间的地表距离（米）- UAV-VLPA地理距离计算核心算法。

    Haversine公式用于计算球面上两点间的最短距离（大圆距离），
    是UAV-VLPA系统中所有地理空间计算的基础。

    算法原理：
    - 球面三角学：在球面上计算两点间的大圆距离
    - 公式：a = sin²(Δφ/2) + cos(φ1)⋅cos(φ2)⋅sin²(Δλ/2)
      c = 2⋅atan2(√a, √(1−a))
      d = R⋅c
    - 单位：米

    参数说明：
        lat1, lon1: 第一个点的纬度和经度（度）
            - WGS84坐标系
        lat2, lon2: 第二个点的纬度和经度（度）
            - WGS84坐标系

    返回值：
        float: 两点之间的地表距离（米）
            - 数值越大表示距离越远
    """
    # 计算纬度差值并转换为弧度
    dlat = math.radians(lat2 - lat1)
    # 计算经度差值并转换为弧度
    dlon = math.radians(lon2 - lon1)

    # 计算Haversine公式的中间变量a
    # a = sin²(Δφ/2) + cos(φ1)⋅cos(φ2)⋅sin²(Δλ/2)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)

    # 计算并返回最终距离
    # d = R⋅2⋅atan2(√a, √(1−a))
    return EARTH_RADIUS_M * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def compute_nearest_neighbor_euclidean_error(
    path1: List[Tuple[float, float]],
    path2: List[Tuple[float, float]]
) -> float:
    """
    计算两条路径间的最近邻欧几里得误差 - UAV-VLPA轨迹相似度评估算法。

    对path1中的每个点，在path2中找到最近的点并计算距离，
    然后返回平均距离。用于评估两条轨迹的整体相似度。

    算法流程：
    1. 遍历path1中的每个点
    2. 对每个点，计算与path2中所有点的距离
    3. 找到最小距离（最近邻）
    4. 计算所有最小距离的平均值

    无人机应用考虑：
    - 使用Haversine距离确保地理精度
    - 对路径点密度不敏感
    - 适合评估轨迹的整体覆盖度

    参数说明：
        path1: 第一条路径，(lat, lon)坐标列表
        path2: 第二条路径，(lat, lon)坐标列表

    返回值：
        float: 平均最近邻距离（米）
            - 数值越小表示路径越相似
            - 如果任一路径为空，返回inf
    """
    # 如果任一路径为空，返回无穷大（无法计算）
    if not path1 or not path2:
        return float('inf')

    # 初始化距离列表
    distances = []

    # 遍历path1中的每个点
    for lat1, lon1 in path1:
        # 找到该点与path2中所有点的最小距离
        min_dist = min(
            _haversine_m(lat1, lon1, lat2, lon2)
            for lat2, lon2 in path2
        )
        # 添加最小距离到列表
        distances.append(min_dist)

    # 返回平均最近邻距离
    return np.mean(distances)


def compute_dtw_mse_rmse(
    path1: List[Tuple[float, float]],
    path2: List[Tuple[float, float]]
) -> Tuple[float, float]:
    """
    计算两条路径间基于DTW的MSE和RMSE - UAV-VLPA轨迹对齐评估核心算法。

    使用动态时间规整（Dynamic Time Warping）算法对齐两条路径，
    然后计算均方误差（MSE）和均方根误差（RMSE）。

    算法原理：
    - DTW算法：找到两条路径间的最优对齐方式
    - 允许不同长度的路径进行比较
    - 考虑时间序列的非线性对齐

    算法流程：
    1. 定义Haversine距离函数作为DTW的距离度量
    2. 使用fastdtw计算最优对齐路径
    3. 计算对齐后的MSE = 总距离 / 对齐路径长度
    4. 计算RMSE = sqrt(MSE)

    无人机应用考虑：
    - 支持不同采样频率的轨迹比较
    - 使用Haversine距离确保地理精度
    - 适合评估轨迹的形状相似度

    参数说明：
        path1: 第一条路径，(lat, lon)坐标列表
        path2: 第二条路径，(lat, lon)坐标列表

    返回值：
        Tuple[float, float]: (MSE, RMSE)，单位为米
            - MSE：均方误差
            - RMSE：均方根误差
            - 如果任一路径为空，返回(inf, inf)
    """
    # 如果任一路径为空，返回无穷大
    if not path1 or not path2:
        return float('inf'), float('inf')

    # ==================== 定义距离函数 ====================
    # 使用自定义Haversine距离函数
    def haversine_dist(p1, p2):
        # 调用_haversine_m计算两点间的Haversine距离
        return _haversine_m(p1[0], p1[1], p2[0], p2[1])

    # ==================== 数据转换 ====================
    # 将路径列表转换为NumPy数组
    arr1 = np.array(path1)
    arr2 = np.array(path2)

    # ==================== 计算DTW对齐 ====================
    # 使用Haversine距离计算DTW对齐
    # distance: DTW总距离
    # path: 最优对齐路径
    distance, path = fastdtw(arr1, arr2, dist=haversine_dist)

    # ==================== 计算MSE和RMSE ====================
    # 计算均方误差：总距离 / 对齐路径长度
    mse = distance / len(path)
    # 计算均方根误差：sqrt(MSE)
    rmse = np.sqrt(mse)

    # 返回MSE和RMSE
    return mse, rmse


def interpolate_path(
    path: List[Tuple[float, float]],
    num_points: int
) -> List[Tuple[float, float]]:
    """
    将路径插值到指定点数 - UAV-VLPA轨迹重采样工具函数。

    使用线性插值将路径重采样到精确的点数，
    用于统一不同路径的采样密度。

    算法原理：
    - 线性插值：在原始路径点之间进行线性插值
    - 保持路径的几何形状
    - 均匀分布新的采样点

    算法流程：
    1. 检查边界条件（点数不足）
    2. 创建旧索引和新索引
    3. 分别对纬度和经度进行插值
    4. 组合回坐标元组列表

    无人机应用考虑：
    - 统一轨迹采样频率
    - 便于轨迹比较和可视化
    - 保持原始路径的几何特征

    参数说明：
        path: 原始路径，(lat, lon)坐标列表
        num_points: 期望的点数

    返回值：
        List[Tuple[float, float]]: 插值后的路径，包含num_points个点
    """
    # 边界检查：如果路径点太少或期望点数太少，直接返回原路径
    if len(path) < 2 or num_points < 2:
        return path

    # ==================== 数据转换 ====================
    # 将路径列表转换为NumPy数组
    arr = np.array(path)

    # ==================== 创建插值索引 ====================
    # 创建旧索引：[0, 1, 2, ..., n-1]
    old_indices = np.arange(len(arr))
    # 创建新索引：从0到n-1均匀分布的num_points个点
    new_indices = np.linspace(0, len(arr) - 1, num_points)

    # ==================== 执行插值 ====================
    # 对纬度进行线性插值
    lat_interp = np.interp(new_indices, old_indices, arr[:, 0])
    # 对经度进行线性插值
    lon_interp = np.interp(new_indices, old_indices, arr[:, 1])

    # ==================== 组合结果 ====================
    # 将插值后的纬度和经度组合回元组列表
    return list(zip(lat_interp, lon_interp))


def euclidean_distance_meters(
    coord1: Tuple[float, float],
    coord2: Tuple[float, float]
) -> float:
    """
    计算两个(lat, lon)坐标间的距离（米）- UAV-VLPA距离计算便捷接口。

    使用Haversine公式计算地表距离，
    是对_haversine_m函数的便捷封装。

    参数说明：
        coord1: 第一个坐标 (lat, lon)
        coord2: 第二个坐标 (lat, lon)

    返回值：
        float: 距离（米）
    """
    # 调用_haversine_m函数计算Haversine距离
    return _haversine_m(coord1[0], coord1[1], coord2[0], coord2[1])
