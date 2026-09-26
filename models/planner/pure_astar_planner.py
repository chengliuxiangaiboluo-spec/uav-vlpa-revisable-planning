"""
纯A*规划器模块 - UAV-VLPA外部验证基线。

本模块实现最简化的路径规划器，仅使用A*算法在网格图上寻路，
不包含任何指令理解、任务分解、多模态融合或动态重规划功能。

作为外部验证基线，用于证明：
    1. 即使最基础的A*寻路也能产生可行路径
    2. 但缺乏指令理解导致任务完成率和指令准确性显著低于基线系统
    3. 多模态增强系统的优势不仅来自路径算法，更来自指令理解能力

设计原则:
    - 最小化: 仅A*寻路，无偏差、无校准、无融合
    - 无理解: 不解析指令语义，仅按场景目标顺序寻路
    - 无避障增强: 不扩展避障半径，仅依赖A*图构建时的基础避障
    - 可复现: 固定min_rad=25，无随机因素
"""

import os
import time
import logging
from typing import List

from utils.import_bridge import setup_imports
setup_imports()

from data.recalculate_to_latlon import (
    read_coordinates_from_csv,
    recalculate_coordinates,
    coords_to_percentage,
)
from data.coordinates_list import coordinates_from_json
from data import Astar
from data.scenario_schema import (
    ScenarioSample,
    PlanResult,
    WaypointTarget,
    AtomicTask,
)

from models.planner.enhanced_planner import (
    _image_discretization,
    _graph_creation,
    _adjacency_list_creation,
)

logger = logging.getLogger("experiment")


class PureAStarPlanner:
    """
    纯A*规划器 - UAV-VLPA外部验证基线。

    仅使用A*最短路径算法，不做任何指令理解或任务分解。
    目标按场景定义顺序访问，无智能排序。

    与BaselinePlanner的区别:
    - 无定位偏差（Baseline通过_compute_biased_targets模拟单模态不确定性）
    - 无任务分解（Baseline至少能识别fly_to+return）
    - 无数据驱动校准（Baseline使用校准参数调整避障半径）
    - 固定避障半径min_rad=25（Baseline使用校准后的半径）
    """

    def __init__(self, benchmark_dir: str):
        """
        初始化纯A*规划器。

        Args:
            benchmark_dir: 基准数据目录路径
        """
        self.benchmark_dir = benchmark_dir
        self.coordinates_dict = read_coordinates_from_csv(
            os.path.join(benchmark_dir, "parsed_coordinates.csv")
        )
        self.images_dir = os.path.join(benchmark_dir, "images")
        self.last_tasks = []  # 保留接口兼容性

    def plan(self, scenario: ScenarioSample) -> PlanResult:
        """
        纯A*路径规划 - 无指令理解，仅按目标顺序寻路。

        算法流程:
        1. 读取场景图像，构建离散化网格
        2. 按场景定义的目标顺序（无排序优化），依次A*寻路
        3. 固定避障半径，不做任何增强
        4. 返回原始路径（无平滑）

        Args:
            scenario: 场景样本对象

        Returns:
            PlanResult: 规划结果
        """
        start = time.perf_counter()
        self.last_tasks = []

        image_id = scenario.image_id
        image_path = os.path.join(self.images_dir, f"{image_id}.jpg")

        if not os.path.exists(image_path):
            logger.warning("PureA* image not found: %s", image_path)
            return PlanResult(
                waypoints=[],
                trajectory_latlon=[],
                execution_time_ms=(time.perf_counter() - start) * 1000,
                completed_targets=[],
            )

        # 坐标转换: 目标和障碍物
        targets_pct = {
            t.name: {"type": t.target_type, "coordinates": list(t.coordinates_percent)}
            for t in scenario.targets
        }
        obstacles_pct = {
            o.name: {"type": o.target_type, "coordinates": list(o.coordinates_percent)}
            for o in scenario.obstacles
        }
        targets_with_home = {"home": {"type": "home", "coordinates": [10, 10]}}
        targets_with_home.update(targets_pct)

        try:
            # 离散化
            pts, w, h = _image_discretization(image_path, step=10)
            fly_pixels = coordinates_from_json(targets_with_home, w, h)
            avoid_pixels = coordinates_from_json(obstacles_pct, w, h)

            for xy in fly_pixels:
                pts.append(xy)

            # 固定min_rad=25，不做任何增强或校准
            min_rad = 25
            graph, coordinates, N = _graph_creation(pts, avoid_pixels, min_rad=min_rad)
            adj = _adjacency_list_creation(graph, N)

            all_path_px = []
            successful_segments = 0
            total_segments = 0

            # A*路径规划: 按目标顺序，home → target1 → target2 → ... → home
            for i in range(len(fly_pixels), 1, -1):
                total_segments += 1
                try:
                    path_indices = Astar.Graph(adj).find_path(
                        coordinates[:N - i + 3], N - i + 2
                    )
                    for v in path_indices:
                        all_path_px.append(coordinates[v])
                    successful_segments += 1
                except Exception:
                    continue

            if all_path_px:
                pct_json = coords_to_percentage(all_path_px, image_path)
                latlon = recalculate_coordinates(pct_json, image_id, self.coordinates_dict)
                traj = [(v["coordinates"][0], v["coordinates"][1]) for v in latlon.values()]
            else:
                traj = []

        except Exception as exc:
            logger.warning("PureA* planner failed for image %d: %s", image_id, exc)
            traj = []
            successful_segments = 0
            total_segments = 1

        elapsed_ms = (time.perf_counter() - start) * 1000

        # 简单完成判断：成功段数/总段数
        completion_rate = successful_segments / max(total_segments, 1)
        completed = []
        if completion_rate >= 0.8:
            completed = [t.name for t in scenario.targets]
        elif completion_rate >= 0.5:
            completed = [t.name for t in scenario.targets[:len(scenario.targets) // 2 + 1]]

        # 生成最简任务列表（仅fly_to + return）
        priority = 1
        for t in scenario.targets:
            self.last_tasks.append(AtomicTask(
                task_type="fly_to",
                target=t,
                priority=priority,
            ))
            priority += 1
        self.last_tasks.append(AtomicTask(
            task_type="return",
            priority=priority,
        ))

        return PlanResult(
            waypoints=[
                WaypointTarget(
                    name=t.name,
                    target_type=t.target_type,
                    coordinates_percent=t.coordinates_percent,
                )
                for t in scenario.targets
            ],
            trajectory_latlon=traj,
            execution_time_ms=elapsed_ms,
            completed_targets=completed,
        )
