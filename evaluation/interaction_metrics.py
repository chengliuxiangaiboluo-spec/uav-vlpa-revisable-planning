"""
交互指标评估模块 - UAV-VLPA实时响应性能评估组件。

测量动态重规划的响应延迟和成功率，
是UAV-VLPA系统中评估实时交互性能的关键组件。

核心功能：
- 响应延迟测量：计算重规划调用的执行时间
- 重规划成功率：评估动态环境下的规划鲁棒性
- 实时性能：支持毫秒级精度测量

UAV-VLPA系统集成：
- 与EnhancedPlanner深度集成，测试多模态规划器的实时性
- 为交互性能指标提供量化数据
- 支持动态约束场景下的系统响应能力评估
"""

# 导入日志模块，用于记录评估过程信息
import logging
# 导入时间模块，用于精确测量执行时间
import time
# 导入类型提示，增强代码可读性
from typing import List, Dict, Any

# 从场景模式定义中导入必要的数据结构
from data.scenario_schema import ScenarioSample, PlanResult
# 导入增强规划器，用于测试重规划性能
from models.planner.enhanced_planner import EnhancedPlanner

# 获取实验日志记录器
logger = logging.getLogger("experiment")


class InteractionMetricsEvaluator:
    """
    交互指标评估器 - UAV-VLPA实时交互性能评估核心组件。

    评估增强规划器的实时交互性能，
    包括响应延迟和重规划成功率两个关键指标。

    评估指标：
    - 响应延迟：从接收到新指令到完成重规划的时间（毫秒）
    - 重规划成功率：成功生成有效轨迹的比例

    无人机应用考虑：
    - 实时性要求：无人机需要在毫秒级响应环境变化
    - 鲁棒性要求：动态障碍物下仍能生成有效路径
    - 安全性要求：重规划结果必须满足避障约束
    """

    def measure_response_latency(
        self,
        planner: EnhancedPlanner,
        scenario: ScenarioSample,
        new_instruction: str = "Add a new no-fly zone near the lake.",
    ) -> float:
        """
        测量重规划响应延迟 - UAV-VLPA实时性能评估核心接口。

        对重规划调用进行计时，测量从接收新指令到完成规划的时间。

        算法流程：
        1. 构建当前状态：从场景中提取目标和障碍物信息
        2. 构建新约束：模拟飞行中的新增禁飞区
        3. 执行重规划：调用planner.replan方法
        4. 计算延迟：使用高精度计时器测量执行时间

        无人机应用考虑：
        - 使用time.perf_counter()确保毫秒级精度
        - 模拟真实飞行中的动态约束场景
        - 为实时性评估提供量化数据

        参数说明：
            planner: 增强规划器实例
                - 支持多模态输入的路径规划器
            scenario: 基础场景对象
                - 包含当前目标和障碍物信息
            new_instruction: 模拟的飞行中指令
                - 默认添加新的禁飞区

        返回值：
            float: 响应延迟（毫秒）
                - 数值越小表示响应越快
                - 用于实时性能评估
        """
        # ==================== 构建当前状态 ====================
        # 从场景中提取当前状态信息
        state = {
            # 图像ID，用于坐标转换
            "image_id": scenario.image_id,
            # 目标信息字典：{目标名称: {类型, 百分比坐标}}
            "targets_pct": {
                t.name: {"type": t.target_type, "coordinates": list(t.coordinates_percent)}
                for t in scenario.targets
            },
            # 障碍物信息字典：{障碍物名称: {类型, 百分比坐标}}
            "obstacles_pct": {
                o.name: {"type": o.target_type, "coordinates": list(o.coordinates_percent)}
                for o in scenario.obstacles
            },
        }

        # ==================== 构建新约束 ====================
        # 模拟飞行中新增的动态约束
        new_constraints = {
            # 新增障碍物：动态禁飞区
            "new_obstacles": {
                "dynamic_nfz": {"type": "no-fly zone", "coordinates": [50.0, 50.0]}
            }
        }

        # ==================== 执行重规划并计时 ====================
        # 记录开始时间（使用高精度性能计数器）
        start = time.perf_counter()

        # 调用规划器的重规划方法
        planner.replan(state, new_constraints)

        # 计算 elapsed 时间并转换为毫秒
        elapsed_ms = (time.perf_counter() - start) * 1000

        # 返回响应延迟（毫秒）
        return elapsed_ms

    def measure_replan_success(
        self,
        planner: EnhancedPlanner,
        scenarios: List[ScenarioSample],
    ) -> float:
        """
        测量重规划成功率 - UAV-VLPA鲁棒性评估核心接口。

        计算在多少场景下重规划能生成非空的有效轨迹。

        算法流程：
        1. 遍历所有测试场景
        2. 为每个场景构建状态和约束
        3. 执行重规划调用
        4. 检查结果是否包含有效轨迹
        5. 计算成功率 = 成功次数 / 总场景数

        无人机应用考虑：
        - 测试动态障碍物下的规划鲁棒性
        - 验证重规划算法的可靠性
        - 为系统部署提供性能保证

        参数说明：
            planner: 增强规划器实例
                - 支持多模态输入的路径规划器
            scenarios: 场景样本列表
                - 用于测试的多个场景

        返回值：
            float: 重规划成功率[0, 1]
                - 1.0：所有场景都成功重规划
                - 0.0：所有场景都失败
        """
        # 初始化成功计数器
        successes = 0

        # 遍历所有测试场景
        for sc in scenarios:
            # ==================== 构建当前状态 ====================
            # 从场景中提取状态信息
            state = {
                # 图像ID
                "image_id": sc.image_id,
                # 目标信息字典
                "targets_pct": {
                    t.name: {"type": t.target_type, "coordinates": list(t.coordinates_percent)}
                    for t in sc.targets
                },
                # 障碍物信息字典
                "obstacles_pct": {
                    o.name: {"type": o.target_type, "coordinates": list(o.coordinates_percent)}
                    for o in sc.obstacles
                },
            }

            # ==================== 构建新约束 ====================
            # 模拟新增的动态障碍物
            new_constraints = {
                "new_obstacles": {
                    "dynamic_obs": {"type": "restricted zone", "coordinates": [45.0, 55.0]}
                }
            }

            # ==================== 执行重规划 ====================
            # 调用规划器的重规划方法
            result = planner.replan(state, new_constraints)

            # 检查结果是否有效：不为None且包含轨迹
            if result is not None and result.trajectory_latlon:
                successes += 1  # 成功计数器加1

        # 计算并返回成功率
        # 如果有场景，返回成功比例；否则返回0.0
        return successes / len(scenarios) if scenarios else 0.0
