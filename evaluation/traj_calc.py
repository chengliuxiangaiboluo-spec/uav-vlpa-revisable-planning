"""
轨迹计算工具模块 - UAV-VLPA地理空间距离计算核心组件。

提供用于计算地理坐标间距离的函数，
是UAV-VLPA系统中评估轨迹质量和路径效率的基础工具。

核心功能：
- Haversine距离计算：计算地球表面两点间的最短距离
- 单位转换：支持千米和米两种单位
- 地理精度：使用WGS84坐标系和标准地球半径

UAV-VLPA系统集成：
- 为轨迹质量评估提供距离计算基础
- 用于路径效率指标（trajectory_length_km）的计算
- 支持DTW、KNN等轨迹相似度算法的距离计算
"""

# 导入数学库，用于三角函数和弧度转换计算
import math
# 导入类型提示，增强代码可读性和IDE支持
from typing import Tuple


def euclidean_distance_km(
    coord1: Tuple[float, float],
    coord2: Tuple[float, float]
) -> float:
    """
    计算两个经纬度坐标间的欧几里得距离（千米）- UAV-VLPA轨迹长度计算核心算法。

    使用Haversine公式计算大圆距离（great-circle distance），
    这是地球表面两点间的最短路径距离。

    算法原理：
    - 球面三角学：在球面上计算两点间的大圆距离
    - 公式：a = sin²(Δφ/2) + cos(φ1)⋅cos(φ2)⋅sin²(Δλ/2)
      c = 2⋅asin(√a)
      d = R⋅c
    - 单位：千米

    无人机应用考虑：
    - 地理精度：6371.0千米地球半径
    - 数值稳定性：避免大角度计算误差
    - 实时性能：轻量级计算适合实时评估

    参数说明：
        coord1: 第一个坐标 (纬度, 经度)，单位为十进制度
            - WGS84坐标系
            - 来自轨迹坐标列表
        coord2: 第二个坐标 (纬度, 经度)，单位为十进制度
            - WGS84坐标系
            - 来自轨迹坐标列表

    返回值：
        float: 两点间的距离（千米）
            - 数值越大表示距离越远
            - 用于轨迹长度和效率评估
    """
    # 将第一个坐标的纬度和经度从十进制度转换为弧度
    # math.radians将角度转换为弧度，用于三角函数计算
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])

    # 将第二个坐标的纬度和经度从十进制度转换为弧度
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])

    # ==================== Haversine公式核心计算 ====================
    # 计算纬度差值（弧度）
    dlat = lat2 - lat1

    # 计算经度差值（弧度）
    dlon = lon2 - lon1

    # 计算Haversine公式的中间变量a
    # a = sin²(Δφ/2) + cos(φ1)⋅cos(φ2)⋅sin²(Δλ/2)
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2

    # 计算角距离c（弧度）
    # c = 2⋅asin(√a)
    c = 2 * math.asin(math.sqrt(a))

    # 地球半径（千米）
    # 6371.0千米是国际公认的平均地球半径
    r = 6371.0

    # 计算并返回最终距离（千米）
    # d = R⋅c
    return r * c


def euclidean_distance_m(
    coord1: Tuple[float, float],
    coord2: Tuple[float, float]
) -> float:
    """
    计算两个经纬度坐标间的欧几里得距离（米）- UAV-VLPA精确距离计算接口。

    基于euclidean_distance_km函数，将千米转换为米，
    用于需要更高精度的距离计算场景。

    算法原理：
    - 复用Haversine公式计算
    - 单位转换：1千米 = 1000米

    无人机应用考虑：
    - 精确评估：米级精度适合障碍物规避和目标访问判断
    - 阈值比较：与50米访问阈值和30米规避距离直接比较

    参数说明：
        coord1: 第一个坐标 (纬度, 经度)，单位为十进制度
            - WGS84坐标系
            - 来自轨迹坐标列表
        coord2: 第二个坐标 (纬度, 经度)，单位为十进制度
            - WGS84坐标系
            - 来自轨迹坐标列表

    返回值：
        float: 两点间的距离（米）
            - 数值越大表示距离越远
            - 用于目标访问和障碍物规避判断
    """
    # 调用euclidean_distance_km计算千米距离，然后转换为米
    # 1千米 = 1000米
    return euclidean_distance_km(coord1, coord2) * 1000.0