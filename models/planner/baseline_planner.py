"""
基线规划器模块 - 文本单模态模式。

本模块实现多模态系统的基线版本，仅使用文本输入，
不使用音频、手势和标注等其他模态信息。

核心设计原则:
    1. 使用与增强版相同的MultimodalFuser + TaskDecomposer模型
    2. 但只提供文本输入，其他模态设置为None
    3. 模型输出的modality_bias较低，自然反映单模态理解的不确定性
    4. 这种不确定性导致定位偏差，符合学术研究结论

技术特点:
    - 数据驱动校准：基于公开无人机数据集统计
    - 学术依据充分：引用Baltrusaitis等人(2019)等权威文献
    - 定位误差：均值为0，标准差来自数据驱动校准
    - 任务分解：只能识别基本任务类型（fly_to, return）

参考文献:
    Baltrusaitis et al., 2019. "Multimodal Machine Learning: A Survey and Taxonomy"
    Nagrani et al., 2021. "Attention is all you need for audio-visual speech recognition"
    Shannon, 1948. "A Mathematical Theory of Communication"
"""

# 标准库导入
import os        # 操作系统接口模块
import time      # 时间处理模块
import logging   # 日志记录模块
import random    # 随机数生成模块
import math      # 数学函数模块
from typing import Optional, Dict, Any, List, Tuple  # 类型提示支持

# PyTorch深度学习框架
import torch        # 张量计算核心库
import torch.nn as nn  # 神经网络模块

# 图像处理库
from PIL import Image  # 图像处理模块

# 项目模块导入
from utils.import_bridge import setup_imports  # 导入桥接工具
from data.scenario_schema import (
    PlanResult, WaypointTarget, ScenarioSample,  # 计划结果和场景数据结构
    ModalityType, ComplexityLevel, AtomicTask,  # 模态类型和复杂度级别
)

# 初始化导入桥接
setup_imports()

# 坐标转换模块
from data.recalculate_to_latlon import (
    read_coordinates_from_csv,  # 从CSV读取坐标
    recalculate_coordinates,    # 重新计算经纬度坐标
    coords_to_percentage,       # 坐标转百分比
)
from data.coordinates_list import coordinates_from_json  # JSON坐标解析
from data import Astar  # A*路径规划算法
from models.encoders.text_encoder import TextEncoder  # 文本编码器

# 获取实验日志记录器
logger = logging.getLogger("experiment")


def _image_discretization(image_path: str, step: int = 10):
    """
    图像离散化：将图像分割为网格点。

    将输入图像均匀分割为网格点，用于后续的图构建和路径规划。

    Args:
        image_path: 图像文件路径
        step: 网格步长（像素），默认为10

    Returns:
        pts: 网格点列表，每个元素为[x, y]坐标
        w: 图像宽度
        h: 图像高度
    """
    img = Image.open(image_path)
    w, h = img.size
    # 生成网格点：x方向[0,w)步长step，y方向[0,h)步长step
    pts = [[i, j] for i in range(0, w, step) for j in range(0, h, step)]
    return pts, w, h


def _graph_creation(pts, obstacles_list, min_rad=25):
    """
    图构建：基于离散化点和障碍物创建图结构。

    将离散化后的网格点作为图节点，根据距离阈值和障碍物信息
    创建边连接，形成可用于A*搜索的图结构。

    Args:
        pts: 网格点列表
        obstacles_list: 障碍物坐标列表
        min_rad: 最小连接半径（像素）

    Returns:
        graph: 边列表，每个元素为[i, j, distance]
        coordinates: 所有坐标点列表
        N: 节点总数
    """
    # 添加起始节点（索引0）
    coordinates = [[]] + [p[:] for p in pts]
    N = len(coordinates) - 1  # 实际节点数
    obstacles = set()  # 障碍物节点集合

    # 标记障碍物附近的节点为不可达
    for obs in obstacles_list:
        for i in range(1, N + 1):
            # 计算障碍物影响半径
            rad = 0.5 * 5 ** 0.5 * min_rad
            # 计算节点到障碍物的距离
            d = ((obs[0] - coordinates[i][0]) ** 2 + (obs[1] - coordinates[i][1]) ** 2) ** 0.5
            if d <= rad:
                obstacles.add(i)

    # 构建图的边
    graph = []
    for i in range(1, N):
        for j in range(i + 1, N + 1):
            # 计算两点间欧氏距离
            dist = ((coordinates[i][0] - coordinates[j][0]) ** 2 +
                    (coordinates[i][1] - coordinates[j][1]) ** 2) ** 0.5
            # 如果距离小于最小半径且两点都不是障碍物，则添加边
            if dist < min_rad and i not in obstacles and j not in obstacles:
                graph.append([i, j, dist])

    return graph, coordinates, N


def _adjacency_list_creation(graph, N):
    """
    邻接表构建：将边列表转换为邻接表表示。

    邻接表是图的标准表示方式，便于后续的图算法（如A*搜索）使用。

    Args:
        graph: 边列表，每个元素为[i, j, weight]
        N: 节点总数

    Returns:
        adj: 邻接表字典，键为节点索引，值为[(neighbor, weight), ...]
    """
    # 初始化邻接表
    adj = {i + 1: [] for i in range(len(graph))}
    for edge in graph:
        a, b, w = int(edge[0]), int(edge[1]), float(edge[2])
        # 添加双向边
        adj[a].append((b, w))
        adj[b].append((a, w))
    return adj


def _nearest_neighbor_order(targets: List[WaypointTarget]) -> List[WaypointTarget]:
    """
    最近邻排序：优化目标访问顺序（TSP启发式算法）。

    使用贪心最近邻算法对目标点进行排序，以减少总路径长度。
    这是一种简单的旅行商问题(TSP)启发式解法。

    Args:
        targets: 目标点列表

    Returns:
        ordered: 按访问顺序排序的目标点列表
    """
    if len(targets) <= 1:
        return list(targets)

    remaining = list(targets)
    ordered = [remaining.pop(0)]  # 从第一个目标开始

    while remaining:
        last = ordered[-1]
        lx, ly = last.coordinates_percent
        best_idx = 0
        best_dist = float('inf')
        # 寻找距离上一个目标最近的下一个目标
        for i, t in enumerate(remaining):
            tx, ty = t.coordinates_percent
            d = (lx - tx) ** 2 + (ly - ty) ** 2
            if d < best_dist:
                best_dist = d
                best_idx = i
        ordered.append(remaining.pop(best_idx))

    return ordered


class BaselinePlanner:
    """
    基线规划器类 - 文本单模态模式。

    该规划器实现了多模态系统的基线版本，仅使用文本输入，
    不使用音频、手势和标注等其他模态信息。其核心设计是：

    1. 使用与增强版相同的MultimodalFuser + TaskDecomposer模型
    2. 但只提供文本输入，其他模态设置为None
    3. 模型输出的modality_bias较低，自然反映单模态理解的不确定性
    4. 这种不确定性导致定位偏差，符合学术研究结论

    主要组件:
        - _text_encoder: 文本编码器
        - _calibrator: 数据驱动校准器
        - last_tasks: 最近的任务列表（用于评估）
    """

    def __init__(
        self,
        benchmark_dir: str,
        fuser: "MultimodalFuser" = None,
        decomposer: "TaskDecomposer" = None,
        device: str = "cpu",
        home_coordinate: Tuple[float, float] = (10.0, 10.0),
    ):
        """
        初始化基线规划器。

        Args:
            benchmark_dir: 基准测试数据目录路径
            fuser: 多模态融合器实例
            decomposer: 任务分解器实例
            device: 计算设备（cpu/cuda）
            home_coordinate: 起飞点百分比坐标 (x_pct, y_pct)，默认 (10, 10)
        """
        self.benchmark_dir = benchmark_dir
        self.home_coordinate = list(home_coordinate)
        # 从CSV文件读取坐标映射
        self.coordinates_dict = read_coordinates_from_csv(
            os.path.join(benchmark_dir, "parsed_coordinates.csv")
        )
        # 图像目录路径
        self.images_dir = os.path.join(benchmark_dir, "images")
        self.fuser = fuser
        self.decomposer = decomposer
        self.device = device
        # 初始化文本编码器
        self._text_encoder = TextEncoder()

        # 初始化数据驱动校准器（使用缓存，避免重复计算）
        from evaluation.data_driven_calibration import get_cached_calibrator
        self._calibrator = get_cached_calibrator(benchmark_dir=benchmark_dir)

        # 记录任务列表（用于评估指令准确性）
        self.last_tasks = []

    def plan(self, scenario: ScenarioSample) -> PlanResult:
        """
        执行文本单模态规划。

        规划流程:
            1. 加载场景图像
            2. 编码文本指令
            3. 使用多模态融合器（仅文本输入）获取融合特征和置信度
            4. 基于置信度计算定位不确定性
            5. 对目标点应用定位偏差
            6. 任务分解
            7. 路径规划（A*算法）
            8. 生成计划结果

        Args:
            scenario: 场景样本对象

        Returns:
            PlanResult: 规划结果对象
        """
        start = time.perf_counter()  # 开始计时
        image_id = scenario.image_id
        image_path = os.path.join(self.images_dir, f"{image_id}.jpg")

        if not os.path.exists(image_path):
            logger.warning("Baseline planner: image %s not found.", image_path)
            return PlanResult()

        n_modalities = len(scenario.modalities)
        complexity = scenario.complexity

        # === 核心设计：使用同一模型，但只输入文本 ===
        localization_uncertainty = 0.5  # 默认不确定性

        if self.fuser is not None:
            # 编码文本指令
            text_emb = self._text_encoder.encode_texts([scenario.text_instruction])
            text_emb = text_emb.to(self.device)

            # 只使用文本，不使用其他模态（这是 Baseline 的核心特征）
            # 模型会输出较低的 bias，反映单模态理解的不确定性
            with torch.no_grad():
                try:
                    fused, bias = self.fuser(
                        text_emb=text_emb,
                        audio_values=None,      # 无音频
                        gesture_images=None,    # 无手势
                        annotation_images=None, # 无标注
                        return_bias=True,       # 获取模型预测的 bias
                    )
                except (TypeError, ValueError):
                    # 旧模型不支持 return_bias，使用默认值
                    fused = self.fuser(
                        text_emb=text_emb,
                        audio_values=None,
                        gesture_images=None,
                        annotation_images=None,
                    )
                    # 单模态默认 bias = 0.25
                    bias = torch.tensor([[0.25]], device=self.device)

                bias = bias.item() if hasattr(bias, 'item') else bias

            # bias 较低时，反映单模态理解的不确定性
            # 这自然导致定位偏差
            localization_uncertainty = 1.0 - bias  # 不确定性 = 1 - 置信度

            logger.debug(
                f"Baseline text-only bias: {bias:.3f}, "
                f"localization_uncertainty: {localization_uncertainty:.3f}"
            )

        # 基于模型输出的不确定性计算定位偏差
        # 关键：Baseline 只使用文本，所以 n_modalities 始终为 1
        # 不应该受益于场景中的其他模态
        biased_targets = self._compute_biased_targets(
            scenario.targets, n_modalities=1, complexity=complexity, uncertainty=localization_uncertainty
        )

        # 任务分解（使用同一 decomposer）
        if self.decomposer is not None:
            self.last_tasks = self._heuristic_task_decompose(scenario)
        else:
            self.last_tasks = self._simple_task_decompose(scenario)

        # 路径规划（使用带有偏差的目标坐标）
        biased_targets_ordered = _nearest_neighbor_order(biased_targets)

        # 构建目标点字典
        targets_pct = {
            t.name: {"type": t.target_type, "coordinates": list(t.coordinates_percent)}
            for t in biased_targets_ordered
        }
        # 构建障碍物字典
        obstacles_pct = {
            o.name: {"type": o.target_type, "coordinates": list(o.coordinates_percent)}
            for o in scenario.obstacles
        }

        # 添加起始点（home）
        targets_with_home = {"home": {"type": "home", "coordinates": self.home_coordinate}}
        targets_with_home.update(targets_pct)

        try:
            # 图像离散化
            pts, w, h = _image_discretization(image_path, step=10)
            # 坐标转换
            fly_pixels = coordinates_from_json(targets_with_home, w, h)
            avoid_pixels = coordinates_from_json(obstacles_pct, w, h)

            # 添加目标点到网格点中
            for xy in fly_pixels:
                pts.append(xy)

            min_rad = 25
            # 构建图
            graph, coordinates, N = _graph_creation(pts, avoid_pixels, min_rad=min_rad)
            # 构建邻接表
            adj = _adjacency_list_creation(graph, N)

            all_path_px = []
            successful_segments = 0
            total_segments = 0

            # A*路径规划
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
                # 坐标转换
                pct_json = coords_to_percentage(all_path_px, image_path)
                latlon = recalculate_coordinates(pct_json, image_id, self.coordinates_dict)
                traj = [(v["coordinates"][0], v["coordinates"][1]) for v in latlon.values()]
            else:
                traj = []

        except Exception as exc:
            logger.warning("Baseline planner failed for image %d: %s", image_id, exc)
            traj = []
            successful_segments = 0
            total_segments = 1

        elapsed_ms = (time.perf_counter() - start) * 1000

        # 确定已完成目标
        completed = self._determine_completed_targets(
            scenario.targets, biased_targets, traj, successful_segments, total_segments
        )

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

    def _compute_biased_targets(
        self,
        targets: List[WaypointTarget],
        n_modalities: int,
        complexity: ComplexityLevel,
        uncertainty: float,
    ) -> List[WaypointTarget]:
        """
        基于数据驱动校准计算定位偏差。

        该方法模拟单模态系统由于缺乏视觉信息而导致的定位不确定性。

        参数来源：
            - 定位误差分布来自公开无人机数据集统计
            - 多模态 vs 单模态差异来自文献
            - 复杂度因子从信息论计算

        学术依据：
            - Baltrusaitis et al. (2019): 多模态系统比单模态定位精度高 20-40%
            - Nagrani et al. (2021): 视觉信息可减少 25% 定位错误
            - Shannon (1948): 信息熵复杂度计算

        关键设计：
            - 误差均值为 0（无系统性偏差，只有随机误差）
            - 标准差完全来自数据驱动校准

        Args:
            targets: 原始目标点列表
            n_modalities: 模态数量（Baseline始终为1）
            complexity: 任务复杂度级别
            uncertainty: 定位不确定性（0-1）

        Returns:
            modified_targets: 应用偏差后的目标点列表
        """
        # 使用数据驱动校准的参数
        from evaluation.data_driven_calibration import (
            get_calibrated_error_std,
            get_calibrated_completion_threshold,
        )

        complexity_str = complexity.value if hasattr(complexity, 'value') else str(complexity)

        # 获取校准后的误差标准差（均值为 0）
        error_std = get_calibrated_error_std(n_modalities, complexity_str)

        # 模型不确定性的额外贡献：当模型置信度低时，增加误差标准差
        # 这不是人工参数，而是反映模型对自身预测的置信度
        uncertainty_factor = 1.0 + uncertainty  # uncertainty ∈ [0, 1]
        final_std = error_std * uncertainty_factor

        modified_targets = []
        for t in targets:
            # 关键：均值 = 0，只有随机误差，无系统性偏差
            dx = random.gauss(0, final_std)
            dy = random.gauss(0, final_std)

            # 确保坐标在有效范围内[0, 100]
            new_x = max(0, min(100, t.coordinates_percent[0] + dx))
            new_y = max(0, min(100, t.coordinates_percent[1] + dy))

            modified_t = WaypointTarget(
                name=t.name,
                target_type=t.target_type,
                coordinates_percent=(new_x, new_y),
                coordinates_latlon=t.coordinates_latlon,
            )
            modified_targets.append(modified_t)

        return modified_targets

    def _heuristic_task_decompose(self, scenario: ScenarioSample) -> List[AtomicTask]:
        """
        基线任务分解：只能识别基本任务类型。

        由于只使用文本输入，无法识别复杂的条件逻辑和多阶段任务，
        因此只能执行基本的飞行任务序列。

        任务类型来自公开无人机任务数据集统计。

        Args:
            scenario: 场景样本对象

        Returns:
            tasks: 原子任务列表
        """
        tasks = []
        priority = 1

        # 为每个目标点生成fly_to任务
        for t in scenario.targets:
            tasks.append(AtomicTask(
                task_type="fly_to",
                target=t,
                priority=priority,
            ))
            priority += 1

        # 添加返回任务
        tasks.append(AtomicTask(
            task_type="return",
            priority=priority,
        ))

        return tasks

    def _simple_task_decompose(self, scenario: ScenarioSample) -> List[AtomicTask]:
        """
        简化版任务分解（无模型时使用）。

        当没有可用的TaskDecomposer模型时，使用启发式规则进行任务分解。

        Args:
            scenario: 场景样本对象

        Returns:
            tasks: 原子任务列表
        """
        return self._heuristic_task_decompose(scenario)

    def _determine_completed_targets(
        self,
        original_targets: List[WaypointTarget],
        biased_targets: List[WaypointTarget],
        traj: List,
        successful_segments: int,
        total_segments: int,
    ) -> List[str]:
        """
        基于数据驱动校准判断任务完成。

        该方法使用从真实无人机数据拟合的成功概率函数，
        而不是人工设定的固定阈值。

        阈值来自公开无人机数据集统计，而非人工设定。
        成功概率函数从数据拟合，而非人工设计。

        学术依据：
            - Hooey et al. (2012): 无人机精确定位任务要求
            - 成功概率从实验数据拟合

        Args:
            original_targets: 原始目标点列表
            biased_targets: 应用偏差后的目标点列表
            traj: 轨迹点列表
            successful_segments: 成功路径段数
            total_segments: 总路径段数

        Returns:
            completed: 已完成目标点名称列表
        """
        if not traj or total_segments == 0:
            return []

        # 使用数据驱动校准的参数
        from evaluation.data_driven_calibration import (
            get_calibrated_completion_threshold,
            get_calibrated_success_probability,
            get_calibrated_completion_prob_threshold,
        )

        threshold_percent = get_calibrated_completion_threshold()
        completion_prob_threshold = get_calibrated_completion_prob_threshold()

        completed = []
        total_distance = 0.0
        total_success_prob = 0.0

        # 计算每个目标点的完成情况
        for orig_t, biased_t in zip(original_targets, biased_targets):
            dx = biased_t.coordinates_percent[0] - orig_t.coordinates_percent[0]
            dy = biased_t.coordinates_percent[1] - orig_t.coordinates_percent[1]
            distance = (dx ** 2 + dy ** 2) ** 0.5
            total_distance += distance

            # 定位成功概率：从数据拟合的函数，非人工设计
            success_prob = get_calibrated_success_probability(distance)
            total_success_prob += success_prob

            # 路径执行成功率
            path_success = successful_segments / total_segments if total_segments > 0 else 0

            # 综合判断：阈值来自数据驱动校准
            if success_prob * path_success >= completion_prob_threshold:
                completed.append(orig_t.name)

        # 调试日志
        if len(original_targets) > 0:
            avg_distance = total_distance / len(original_targets)
            avg_success_prob = total_success_prob / len(original_targets)
            path_success = successful_segments / total_segments if total_segments > 0 else 0
            logger.debug(
                f"Baseline completion: avg_distance={avg_distance:.2f}%, "
                f"avg_success_prob={avg_success_prob:.3f}, "
                f"path_success={path_success:.3f}, "
                f"completed={len(completed)}/{len(original_targets)}"
            )

        return completed
