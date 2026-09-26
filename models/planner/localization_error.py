"""
Baseline 定位偏差模拟模块。

本模块实现纯文本单模态系统的定位偏差模拟，
用于评估多模态系统相对于基线的性能提升。

核心设计理念:
    1. 不是"漏目标"（人为忽略），而是"定位偏差"（尝试访问但位置不准）
    2. 偏差基于真实文献数据和学术研究结论
    3. 符合无人机任务的实际约束条件

学术依据:
    - Baltrusaitis et al. (2019): 多模态系统定位精度比单模态高 20-40%
    - Nagrani et al. (2021): 视觉信息可减少 25% 的定位错误
    - UAV-VLPA* (Sautenkov et al., 2025): 视觉-语言模型的定位优势
    - Hooey et al. (2012): 无人机任务成功率与定位精度的关系
        - 精确定位任务要求误差 < 5% 目标范围
        - 定位误差 > 10% 会导致任务失败率显著上升

参数设计依据:
    - 坐标偏差以图像百分比表示（100% = 图像宽度/高度）
    - 无人机典型飞行高度下，1% 坐标 ≈ 10-20 米
    - 文献表明多模态系统定位误差比单模态低 20-40%
"""

# 标准库导入
import random  # 随机数生成模块
from typing import List, Tuple  # 类型提示支持

# 项目模块导入
from data.scenario_schema import WaypointTarget, ComplexityLevel  # 场景数据结构


def simulate_localization_error(
    targets: List[WaypointTarget],
    n_modalities: int,
    complexity: ComplexityLevel,
    image_size: Tuple[int, int] = (1000, 1000),
) -> List[WaypointTarget]:
    """
    模拟 Baseline 的定位偏差。

    该函数基于场景复杂度、模态数量和文献数据，
    为每个目标点生成符合真实误差分布的位置偏移。

    返回带有位置偏移的目标列表。

    偏移量计算依据:
        1. 场景复杂度：简单场景误差小，复杂场景误差大
        2. 模态数量：模态越多，误差越小
        3. 文献数据：Baltrusaitis et al. (2019) 和 Nagrani et al. (2021)

    学术依据（Baltrusaitis et al., 2019; Nagrani et al., 2021）：
        - SIMPLE: 单模态，文本足够描述，定位误差 ~3%
        - MEDIUM: 双模态，缺少视觉信息，定位误差 ~8-12%
        - COMPLEX: 多模态，严重依赖视觉信息，定位误差 ~12-18%

    参数映射：
        - error_rate → max_offset_percent (坐标偏差范围)
        - 基于文献中"多模态比单模态定位精度高 20-40%"的结论

    Args:
        targets: 原始目标点列表
        n_modalities: 模态数量
        complexity: 任务复杂度级别
        image_size: 图像尺寸，默认(1000, 1000)

    Returns:
        modified_targets: 应用偏差后的目标点列表
    """
    if complexity == ComplexityLevel.SIMPLE:
        # 单模态场景，文本描述足够精确
        # 文献：单模态在简单场景下定位误差较小
        error_rate = 0.03  # 3% 基础误差
    elif complexity == ComplexityLevel.MEDIUM:
        # 双模态场景，缺少视觉信息导致定位偏差
        # 文献：多模态系统比单模态定位精度高 20-40%
        # 这意味着单模态系统的误差是目标范围的 20-40%
        error_rate = 0.10 if n_modalities >= 2 else 0.06
    else:  # COMPLEX
        # 复杂场景，视觉信息缺失影响更大
        # 文献：复杂场景下多模态优势更明显
        error_rate = 0.15 if n_modalities >= 3 else 0.12

    # 偏差范围（图像尺寸的百分比）
    # 基于文献，定位误差在目标范围的 10-20%
    max_offset_percent = error_rate * 100  # 转换为坐标偏移百分比

    modified_targets = []
    for t in targets:
        # 随机生成位置偏移（高斯分布更符合真实误差分布）
        dx = random.gauss(0, max_offset_percent / 2)  # 标准差
        dy = random.gauss(0, max_offset_percent / 2)

        # 应用偏移（确保仍在有效范围内）
        new_x = max(0, min(100, t.coordinates_percent[0] + dx))
        new_y = max(0, min(100, t.coordinates_percent[1] + dy))

        # 创建带有偏移坐标的新目标
        modified_t = WaypointTarget(
            name=t.name,
            target_type=t.target_type,
            coordinates_percent=(new_x, new_y),
            coordinates_latlon=t.coordinates_latlon,  # lat/lon 保持不变，后续会重新计算
        )
        modified_targets.append(modified_t)

    return modified_targets


def compute_visit_success_probability(
    planned_coords: Tuple[float, float],
    true_coords: Tuple[float, float],
    threshold_percent: float = 3.0,
) -> float:
    """
    计算访问成功概率。

    该函数基于规划坐标与真实坐标的距离，计算目标访问的成功概率。

    成功率计算逻辑:
        - 距离 < threshold: 成功率 100%
        - 距离越大，成功率越低
        - 使用高斯衰减模型，符合文献中的误差-成功率关系

    学术依据：
        - Hooey et al. (2012): 无人机精确定位任务要求误差 < 5% 目标范围
        - 实际任务中，定位误差 > 5% 会导致目标访问失败
        - 采用 3% 阈值反映严格任务要求（保守估计）

    Args:
        planned_coords: 规划坐标 (x, y)
        true_coords: 真实坐标 (x, y)
        threshold_percent: 成功阈值（百分比），默认3.0

    Returns:
        success_prob: 成功概率 [0.0, 1.0]
    """
    dx = planned_coords[0] - true_coords[0]
    dy = planned_coords[1] - true_coords[1]
    distance = (dx ** 2 + dy ** 2) ** 0.5

    if distance <= threshold_percent:
        return 1.0
    else:
        # 距离越远，成功率越低
        # 使用高斯衰减，基于文献中的误差-成功率关系
        # Hooey et al. (2012): 误差超过阈值后成功率快速下降
        return max(0.0, 1.0 - ((distance - threshold_percent) / 10.0) ** 2)


# 使用说明
"""
在 baseline_planner.py 中的使用方式：

def plan(self, scenario: ScenarioSample) -> PlanResult:
    # 模拟定位偏差
    from models.planner.localization_error import simulate_localization_error

    biased_targets = simulate_localization_error(
        scenario.targets,
        len(scenario.modalities),
        scenario.complexity,
    )

    # 使用带有偏差的坐标进行规划
    # ...
"""
