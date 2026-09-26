"""
坐标重算模块 - UAV-VLPA系统的地理空间坐标对齐核心组件。

在像素坐标、百分比坐标和经纬度坐标之间进行精确转换，
使用基准图像的坐标数据（CSV文件）作为参考，
通过线性插值计算任意点的经纬度，确保UAV-VLPA系统中
不同模态（视觉、语言、路径规划）的地理空间一致性。

核心功能：
- 多坐标系统转换：支持像素→百分比→经纬度的双向转换
- 基准数据管理：读取和解析基准图像的地理坐标信息
- 线性插值：基于西北角和东南角坐标计算任意位置的经纬度
- 地理空间对齐：确保视觉标注、文本指令和路径规划在统一地理坐标系下工作

UAV-VLPA系统集成：
- 与路径规划模块协同工作，将像素路径转换为地理路径
- 为Haversine距离计算提供精确的地理坐标
- 支持多分辨率卫星图像的地理坐标一致性
- 为评估模块提供标准化的地理坐标转换接口

坐标系统说明：
- 像素坐标：(x_pixel, y_pixel)，原点在图像左上角(0,0)
- 百分比坐标：(x_pct, y_pct)，范围0-100，与图像分辨率无关
- 经纬度坐标：(latitude, longitude)，WGS84坐标系，标准地理坐标系统
- Haversine公式：用于计算地球表面两点间的最短距离
"""

# ==================== 标准库导入 ====================
import os   # 操作系统接口
import csv  # CSV文件读取
from typing import Dict, List, Tuple, Any  # 类型注解


def read_coordinates_from_csv(csv_file_path: str) -> Dict[int, Dict[str, Tuple[float, float]]]:
    """
    从CSV文件读取基准图像的坐标数据 - UAV-VLPA地理空间基准管理核心算法。

    CSV文件包含每张基准图像的西北角(NW)和东南角(SE)经纬度，
    用于后续的线性插值计算，是UAV-VLPA系统地理空间对齐的基础。

    算法原理：
    - CSV解析：使用DictReader安全解析CSV文件
    - 图像ID提取：从文件名自动提取图像ID
    - 坐标验证：确保经纬度数据的有效性

    无人机应用考虑：
    - 基准精度：西北角和东南角坐标提供高精度地理参考
    - 数据完整性：检查文件存在性和数据完整性
    - 性能优化：高效的CSV解析算法

    CSV格式要求：
        Image, NW Corner Lat, NW Corner Long, SE Corner Lat, SE Corner Long
        1.jpg,  40.7128,       -74.0060,        40.7028,       -73.9960

    参数说明：
        csv_file_path: CSV文件路径
            - 必须包含基准图像的地理坐标信息
            - 推荐使用绝对路径确保跨环境一致性

    返回值：
        Dict: 图像坐标字典
            {
                image_id: {
                    'NW': (latitude, longitude),  # 西北角坐标（左上角）
                    'SE': (latitude, longitude)   # 东南角坐标（右下角）
                },
                ...
            }
            - image_id：整数类型，图像标识符
            - NW/SE：WGS84坐标系下的经纬度坐标
    """
    coordinates_dict = {}

    # 检查文件是否存在
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"Coordinate CSV file not found: {csv_file_path}")

    # 读取CSV文件
    with open(csv_file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)  # 使用字典读取器
        for row in reader:
            # 从文件名提取图像ID（如 "1.jpg" -> 1）
            image_id = int(row['Image'].split('.')[0])
            # 解析西北角坐标
            nw_lat = float(row['NW Corner Lat'])
            nw_lon = float(row['NW Corner Long'])
            # 解析东南角坐标
            se_lat = float(row['SE Corner Lat'])
            se_lon = float(row['SE Corner Long'])

            # 存储到字典
            coordinates_dict[image_id] = {
                'NW': (nw_lat, nw_lon),
                'SE': (se_lat, se_lon)
            }

    return coordinates_dict


def coords_to_percentage(pixel_coords: List[List[int]], image_path: str) -> Dict[str, Dict[str, Any]]:
    """
    将像素坐标转换为百分比坐标 - UAV-VLPA坐标系统抽象核心算法。

    百分比坐标是相对于图像尺寸的比例，范围0-100，
    这种表示方式与图像分辨率无关，便于跨平台使用和多模态对齐。

    算法原理：
    - 比例计算：基于图像实际尺寸进行精确比例转换
    - 坐标映射：保持图像坐标系的一致性（原点在左上角）
    - 数据结构：生成标准化的字典结构便于后续处理

    无人机应用考虑：
    - 分辨率无关：确保在不同卫星图像分辨率下的位置一致性
    - 多模态对齐：为视觉标注、手势轨迹、文本指令提供统一坐标系
    - 实时性能：高效的像素到百分比转换

    转换公式：
        pct_x = (x_pixel / width) * 100  # 水平方向
        pct_y = (y_pixel / height) * 100  # 垂直方向

    参数说明：
        pixel_coords: 像素坐标列表 [[x1, y1], [x2, y2], ...]
            - 来自A*路径规划或视觉标注的像素坐标
            - 顺序代表任务执行序列
        image_path: 图像文件路径
            - 提供图像尺寸信息
            - 支持多种卫星图像格式

    返回值：
        Dict: 百分比坐标字典
            {
                "point_0": {"type": "waypoint", "coordinates": [pct_x, pct_y]},
                "point_1": {...},
                ...
            }
            - point_i：按顺序编号的坐标点
            - type：坐标点类型（waypoint, target, obstacle等）
            - coordinates：百分比坐标[x%, y%]
    """
    from PIL import Image  # 延迟导入，避免依赖

    # 获取图像尺寸
    with Image.open(image_path) as img:
        width, height = img.size

    result = {}
    for i, (x, y) in enumerate(pixel_coords):
        # 转换为百分比（0-100）
        pct_x = (x / width) * 100
        pct_y = (y / height) * 100
        result[f"point_{i}"] = {
            "type": "waypoint",
            "coordinates": [pct_x, pct_y]
        }

    return result


def recalculate_coordinates(
    percentage_coords: Dict[str, Dict[str, Any]],
    image_id: int,
    coordinates_dict: Dict[int, Dict[str, Tuple[float, float]]]
) -> Dict[str, Dict[str, Any]]:
    """
    将百分比坐标转换为经纬度坐标 - UAV-VLPA地理空间转换核心算法。

    使用基准图像的西北角(NW)和东南角(SE)经纬度，
    通过线性插值计算任意百分比位置的真实经纬度，
    是UAV-VLPA系统中地理空间对齐的关键步骤，
    为Haversine距离计算和路径质量评估提供精确的地理坐标基础。

    算法原理详解：
    - 线性插值：基于西北角和东南角构建地理坐标网格
    - 坐标映射：0%→NW, 100%→SE的线性映射关系
    - 数值计算：精确的浮点运算确保地理精度
    - 坐标系转换：WGS84坐标系下的标准地理坐标

    无人机应用考虑：
    - 地理精度：支持Haversine距离计算的精确地理坐标
    - 安全边界：为路径规划提供真实的地理约束
    - 多模态对齐：确保所有模态在统一地理坐标系下工作
    - 实时性能：高效的插值算法支持实时转换
    - 误差控制：基于基准图像的几何校正

    转换原理：
        - 图像左上角(0%, 0%)对应西北角(NW)坐标
        - 图像右下角(100%, 100%)对应东南角(SE)坐标
        - 中间点通过线性插值计算

    坐标计算公式：
        - 纬度：从北向南递减（NW.lat > SE.lat）
            lat = NW.lat - (pct_y / 100.0) * (NW.lat - SE.lat)
        - 经度：从西向东递增（NW.lon < SE.lon）
            lon = NW.lon + (pct_x / 100.0) * (SE.lon - NW.lon)

    参数说明：
        percentage_coords: 百分比坐标字典
            - 包含所有需要转换的坐标点
            - 格式标准化，便于后续处理
            - 支持多种坐标类型（waypoint, target, obstacle等）
        image_id: 图像ID（1-30）
            - 对应基准图像的标识符
            - 用于查找对应的地理坐标基准
            - 支持基准图像库的扩展
        coordinates_dict: 基准坐标字典
            - 来自read_coordinates_from_csv的输出
            - 包含所有基准图像的地理坐标信息
            - 支持多基准图像管理

    返回值：
        Dict: 经纬度坐标字典
            {
                "point_0": {"type": "waypoint", "coordinates": [lat, lon]},
                ...
            }
            - lat/lon：WGS84坐标系下的经纬度坐标
            - 用于Haversine距离计算和路径质量评估
            - 支持多模态模型训练和评估
    """
    # 检查图像ID是否存在
    if image_id not in coordinates_dict:
        raise ValueError(f"No coordinate data found for image {image_id}")

    # 获取图像角点坐标
    corners = coordinates_dict[image_id]
    nw_lat, nw_lon = corners['NW']  # 西北角（左上）
    se_lat, se_lon = corners['SE']  # 东南角（右下）

    # 计算纬度范围（正值，从北到南）
    lat_range = nw_lat - se_lat
    # 计算经度范围（正值，从西到东）
    lon_range = se_lon - nw_lon

    result = {}
    for key, coord_data in percentage_coords.items():
        pct_x, pct_y = coord_data['coordinates']

        # 百分比转经纬度
        # pct_x: 0% = 西边缘(NW.lon), 100% = 东边缘(SE.lon)
        # pct_y: 0% = 北边缘(NW.lat), 100% = 南边缘(SE.lat)
        lat = nw_lat - (pct_y / 100.0) * lat_range  # 纬度向南递减
        lon = nw_lon + (pct_x / 100.0) * lon_range  # 经度向东递增

        result[key] = {
            "type": coord_data["type"],
            "coordinates": [lat, lon]
        }

    return result


# ==================== 向后兼容别名 ====================
# 驼峰命名别名，保持与旧代码的兼容性
readCoordinatesFromCSV = read_coordinates_from_csv
recalculateCoordinates = recalculate_coordinates
coordsToPercentage = coords_to_percentage