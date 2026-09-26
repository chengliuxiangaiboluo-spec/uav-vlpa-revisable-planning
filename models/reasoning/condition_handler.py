"""
动态条件处理器模块。

本模块实现无人机任务中的动态指令更新和实时重规划功能，
支持在飞行过程中接收新的指令或约束条件，并触发快速重规划。

核心功能:
    1. 动态指令更新：合并新指令并重新分解任务
    2. 条件检查：评估环境状态并触发相应动作
    3. 重规划触发：根据新约束执行快速路径重规划

技术特点:
    - 支持循环导入避免：通过set_planner方法后期设置规划器引用
    - 规则基础的条件评估：基于关键词匹配环境状态
    - 低时延重规划：使用EnhancedPlanner的快速重规划接口

参考文献:
    Hooey et al., 2012. "Human Factors Guidelines for Unmanned Aircraft System Control Stations"
    Baltrusaitis et al., 2019. "Multimodal Machine Learning: A Survey and Taxonomy"
"""

# 标准库导入
import logging  # 日志记录模块
import time      # 时间处理模块
from typing import List, Dict, Any, Optional  # 类型提示支持

# 项目模块导入
from data.scenario_schema import AtomicTask, PlanResult  # 原子任务和计划结果结构

# 获取实验日志记录器
logger = logging.getLogger("experiment")


class DynamicConditionHandler:
    """
    动态条件处理器类。

    该类负责处理无人机飞行过程中的动态指令更新和条件触发，
    支持实时指令修改、环境条件检查和快速路径重规划。

    主要组件:
        - planner: 增强型规划器引用
        - _active_conditions: 活跃条件列表
    """

    def __init__(self, planner_ref=None):
        """
        初始化动态条件处理器。

        Args:
            planner_ref: 增强型规划器引用（用于避免循环导入）
        """
        self.planner = planner_ref
        self._active_conditions: List[str] = []

    def set_planner(self, planner):
        """
        设置规划器引用。

        该方法用于后期设置规划器引用，避免模块间的循环导入问题。

        Args:
            planner: 增强型规划器实例
        """
        self.planner = planner

    def update_instruction(
        self,
        new_instruction: str,
        current_state: Dict[str, Any],
    ) -> List[AtomicTask]:
        """
        合并新指令并重新分解任务。

        该方法将新的文本指令与原始指令合并，然后调用规划器
        的任务分解功能生成新的原子任务序列。

        Args:
            new_instruction: 新的/修改后的指令文本
            current_state: 当前状态字典，至少包含:
                - completed_tasks: 已完成任务名称列表
                - current_position: 当前位置(lat, lon)元组

        Returns:
            updated_tasks: 更新后的原子任务列表
        """
        logger.info("Dynamic instruction update: %s", new_instruction)

        if self.planner is None:
            logger.warning("No planner attached; returning empty task list.")
            return []

        # 合并指令：原始指令 + 新指令
        merged = f"{current_state.get('original_instruction', '')} Additionally: {new_instruction}"
        current_state["text_instruction"] = merged
        # 调用规划器的任务分解功能
        tasks = self.planner.decompose_instruction(merged)
        return tasks

    def check_conditions(
        self,
        conditions: List[str],
        environment_state: Dict[str, Any],
    ) -> List[bool]:
        """
        评估条件列表与当前环境状态的匹配情况。

        这是一个简化的基于规则的条件评估器，每个条件字符串
        通过关键词匹配环境状态变量来判断是否满足。

        Args:
            conditions: 条件字符串列表
            environment_state: 环境状态字典

        Returns:
            results: 布尔值列表，表示每个条件是否满足
        """
        results = []
        for cond in conditions:
            cond_lower = cond.lower()
            triggered = False

            # 检查禁飞区条件
            if "no-fly zone" in cond_lower or "restriction" in cond_lower:
                triggered = environment_state.get("new_nfz", False)
            # 检查风速条件
            elif "wind" in cond_lower:
                wind = environment_state.get("wind_speed_kmh", 0)
                triggered = wind > 20
            # 检查能见度条件
            elif "visibility" in cond_lower:
                vis = environment_state.get("visibility_m", 9999)
                triggered = vis < 500
            # 检查新航点条件
            elif "new waypoint" in cond_lower or "new target" in cond_lower:
                triggered = environment_state.get("new_waypoint_available", False)
            # 检查障碍物区域条件
            elif "obstacle" in cond_lower and "area" in cond_lower:
                area = environment_state.get("obstacle_area_sqm", 0)
                triggered = area > 1000
            else:
                # 默认：随机触发30%的时间（用于模拟）
                triggered = environment_state.get("random_trigger", False)

            results.append(triggered)

        return results

    def trigger_replan(
        self,
        remaining_tasks: List[AtomicTask],
        new_constraints: Dict[str, Any],
        scenario_state: Dict[str, Any],
    ) -> Optional[PlanResult]:
        """
        触发带有更新约束的完整重规划。

        该方法调用增强型规划器的replan接口，执行快速路径重规划。

        Args:
            remaining_tasks: 剩余任务列表
            new_constraints: 新的约束条件字典
            scenario_state: 场景状态字典

        Returns:
            result: 如果重规划成功则返回新的PlanResult，否则返回None
        """
        if self.planner is None:
            logger.warning("Cannot replan: no planner attached.")
            return None

        logger.info("Triggering replan with %d remaining tasks, new constraints: %s",
                     len(remaining_tasks), new_constraints)

        start = time.perf_counter()
        try:
            # 调用增强型规划器的快速重规划接口
            result = self.planner.replan(scenario_state, new_constraints)
            elapsed_ms = (time.perf_counter() - start) * 1000
            if result is not None:
                result.execution_time_ms = elapsed_ms
            return result
        except Exception as exc:
            logger.error("Replan failed: %s", exc)
            return None
