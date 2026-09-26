"""
坐标转换工具模块 - UAV-VLPA系统的多模态坐标对齐核心组件。

在JSON坐标格式（百分比坐标）和像素坐标之间进行精确转换，
确保UAV-VLPA系统中不同模态（视觉、语言、路径规划）
的坐标系统保持一致性和互操作性。

核心功能：
- 百分比坐标到像素坐标的转换：支持卫星图像上的视觉标注
- 像素坐标到地理坐标的映射：为后续Haversine距离计算提供基础
- 多模态坐标对齐：确保视觉标注、文本指令和路径规划在统一坐标系下工作

UAV-VLPA系统集成：
- 与视觉标注生成器协同工作，将用户指令中的百分比坐标转换为像素坐标
- 为路径规划模块提供精确的位置输入
- 支持多分辨率图像处理，确保在不同卫星图像质量下的坐标一致性
- 为评估模块提供标准化的坐标转换接口

坐标系统说明：
- 百分比坐标：范围0-100，相对于图像宽度和高度的比例
- 像素坐标：整数坐标，原点在图像左上角(0,0)
- 地理坐标：后续通过Haversine公式转换为经纬度坐标
"""

# ==================== 标准库导入 ====================
from typing import Dict, List, Tuple, Any  # 类型注解


def coordinates_from_json(
    coord_dict: Dict[str, Dict[str, Any]],
    width: int,
    height: int
) -> List[List[int]]:
    """
    将JSON坐标数据转换为像素坐标 - UAV-VLPA多模态坐标对齐核心算法。

    该函数实现了UAV-VLPA系统中关键的坐标转换功能，
    将用户指令中的百分比坐标转换为卫星图像上的像素坐标，
    确保视觉标注、路径规划和任务执行在统一坐标系下工作。

    算法原理：
    - 线性比例转换：基于图像宽高进行精确的比例缩放
    - 整数舍入：使用int()函数进行向下取整，确保像素坐标有效性
    - 坐标系一致性：保持图像坐标系（原点在左上角）的一致性

    无人机应用考虑：
    - 支持亚像素精度计算，为后续Chaiken路径平滑提供基础
    - 转换结果直接用于A*路径规划算法的节点定位
    - 为Haversine距离计算提供精确的相对位置信息
    - 支持不同分辨率卫星图像，确保跨平台兼容性

    转换公式：
        x_pixel = round((x_percent / 100.0) * width)  # 水平方向
        y_pixel = round((y_percent / 100.0) * height)  # 垂直方向

    参数说明：
        coord_dict: 坐标字典，格式为：
            {
                "name": {
                    "type": "类型名称",  # 目标/障碍物/起点等类型
                    "coordinates": [x_percent, y_percent]  # 百分比坐标
                },
                ...
            }
        width: 图像宽度（像素），来自卫星图像元数据
        height: 图像高度（像素），来自卫星图像元数据

    返回值：
        List[List[int]]: 像素坐标列表 [[x1, y1], [x2, y2], ...]
            - 每个子列表代表一个位置的像素坐标[x, y]
            - 坐标顺序与JSON中定义的顺序一致
            - 用于后续的视觉标注、路径规划和任务执行
    """
    pixel_coords = []

    # 遍历所有坐标条目
    for name, data in coord_dict.items():
        if "coordinates" in data:
            x_pct, y_pct = data["coordinates"]
            # 将百分比转换为像素
            # x: 水平方向，从左到右
            x_pixel = int((x_pct / 100.0) * width)
            # y: 垂直方向，从上到下
            y_pixel = int((y_pct / 100.0) * height)
            pixel_coords.append([x_pixel, y_pixel])

    return pixel_coords


# ==================== 向后兼容 ====================
# 驼峰命名别名，保持与旧代码的兼容性
coordinatesFromJson = coordinates_from_json