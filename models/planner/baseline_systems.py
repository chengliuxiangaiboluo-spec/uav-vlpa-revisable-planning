"""
多模态UAV系统对比模块。

本模块实现4种与本项目Enhanced MultimodalVLPA系统对照的多模态UAV系统，
用于系统级对比实验，验证本项目核心融合策略、任务分解策略和路径规划策略
的协同有效性。

系统对比组合（均基于用户"文献参考"文件夹中的2023-2026年最新论文）：
    1. UAV-VLA风格 — MAFTNet + CodeAgents + A*
       (MAFTNet: Liu et al., 2026, IEEE Sensors;
        CodeAgents: Sautenkov et al., 2025, arXiv:2505.07236)
    2. UAV-VLN风格 — AFFNet + SIPSA + A*
       (AFFNet: Tang et al., 2026, IEEE TIP;
        SIPSA: Zhou et al., 2025, IEEE ICIP)
    3. AerialVLN风格 — SCAL + UniGoal + A*
       (SCAL: Chen et al., 2026, IEEE JSTARS;
        UniGoal: Yin et al., 2025, CVPR)
    4. CityNav风格 — LPANet + SIPSA + iKap
       (LPANet: Wu et al., 2026, IEEE TIP;
        SIPSA: Zhou et al., 2025, IEEE ICIP;
        iKap: Li et al., 2025, IEEE ICRA)

每个基线系统使用不同的（融合器、分解器、规划器）组合，
构建可公平对比的端到端UAV多模态系统。

参考文献（均为用户"文献参考"文件夹中的论文）：
    [1] Tang et al., 2026. "Adaptive Fine-Grained Fusion Network for
        Multimodal UAV Object Detection" — IEEE TIP
        文件: 文献参考/UAV多模态导航文献/Adaptive_Fine-Grained_Fusion_Network_for_Multimodal_UAV_Object_Detection.pdf
    [2] Liu et al., 2026. "MAFTNet: Multimodal Adaptive Fusion-Based
        Transformer Network" — IEEE Sensors Journal
        文件: 文献参考/UAV多模态导航文献/MAFTNet_Multimodal_Adaptive_Fusion-Based_Transformer_Network_for_Infrared_and_Visible_Image_UAV_Object_Detection.pdf
    [3] Chen et al., 2026. "SCAL: A Semantic-Consistent Adaptive
        Alignment Learning Framework" — IEEE JSTARS
        文件: 文献参考/架构/SCAL_A_Semantic-Consistent_Adaptive_Alignment_Learning_Framework_for_UAV_Remote_Sensing_Cross-Modal_Retrieval.pdf
    [4] Wu et al., 2026. "Large Language Model Guided Progressive Feature
        Alignment for Multimodal UAV Object Detection" — IEEE TIP
        文件: 文献参考/UAV多模态导航文献/Large_Language_Model_Guided_Progressive_Feature_Alignment_for_Multimodal_UAV_Object_Detection.pdf
    [5] Zhou et al., 2025. "Structured Instruction Parsing and Scene Alignment
        for UAV Vision-Language Navigation" — IEEE ICIP 2025
        文件: 文献参考/VLM/Structured_Instruction_Parsing_and_Scene_Alignment_For_UAV_Vision-Language_Navigation.pdf
    [6] Yin et al., 2025. "UniGoal: Towards Universal Zero-shot Goal-oriented
        Navigation" — CVPR 2025
        文件: 文献参考/论文基线/Yin_UniGoal_Towards_Universal_Zero-shot_Goal-oriented_Navigation_CVPR_2025_paper.pdf
    [7] Sautenkov et al., 2025. "UAV-CodeAgents: Scalable UAV Mission Planning
        via Multi-Agent ReAct and Vision-Language Reasoning" — arXiv:2505.07236
        文件: 文献参考/VLM/2505.07236v1.pdf
    [8] Li et al., 2025. "iKap: Kinematics-aware Planning with Imperative
        Learning" — IEEE ICRA 2025
        文件: 文献参考/路径规划/iKap_Kinematics-Aware_Planning_with_Imperative_Learning.pdf
"""

import os
import time
import logging
import random
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    ModalityType,
    ComplexityLevel,
)

from models.planner.enhanced_planner import (
    _image_discretization,
    _graph_creation,
    _adjacency_list_creation,
    _nearest_neighbor_order,
)
from models.encoders.text_encoder import TextEncoder

logger = logging.getLogger("experiment")


# ==================== 通用系统组件适配器 ====================

class ModularUAVSystem:
    """
    模块化UAV系统 — 通过组合不同基线模块构建对比系统。

    与EnhancedPlanner保持相同的接口，但其内部组件可任意替换：
        - fuser: 多模态融合器 (MultimodalFuser / baseline_fusers)
        - decomposer: 任务分解器 (TaskDecomposer / baseline_decomposers)
        - path_planner: 路径规划器 (A*/iKap/CSGLSO)

    设计原则:
        - 模块独立: 各组件可独立替换，便于系统级对比
        - 接口统一: 与EnhancedPlanner.plan(scenario)接口一致
        - 端到端可执行: 提供完整规划流水线
        - 文献映射: 每个组合对应一个已知UAV-VL系统
    """

    def __init__(
        self,
        system_name: str,
        benchmark_dir: str,
        fuser: Optional[nn.Module] = None,
        decomposer: Optional[Any] = None,
        path_planner_name: str = "astar",  # astar / ikap / csglso
        device: str = "cpu",
        seed: int = 42,
        use_neural_decompose: bool = False,
        home_coordinate: Tuple[float, float] = (10.0, 10.0),
        quality_override: Optional[float] = None,
    ):
        """
        Args:
            system_name: 系统名称（用于报告标识）
            benchmark_dir: 基准数据目录
            fuser: 多模态融合器实例
            decomposer: 任务分解器实例（可能为None → 仅用启发式）
            path_planner_name: 路径规划器名 ("astar" | "ikap" | "csglso")
            device: 计算设备
            seed: 随机种子
            use_neural_decompose: 是否使用神经分解器（默认False，使用启发式）
            home_coordinate: 起飞点百分比坐标
        """
        self.system_name = system_name
        self.benchmark_dir = benchmark_dir
        self.fuser = fuser
        self.decomposer = decomposer
        self.path_planner_name = path_planner_name
        self.device = device
        self._use_neural_decompose = use_neural_decompose
        self.home_coordinate = list(home_coordinate)
        self.rng = random.Random(seed)
        self._quality_override = quality_override

        # 坐标映射与图像路径
        self.coordinates_dict = read_coordinates_from_csv(
            os.path.join(benchmark_dir, "parsed_coordinates.csv")
        )
        self.images_dir = os.path.join(benchmark_dir, "images")
        self._text_encoder = TextEncoder()

        # 路径规划器: 仅在需要时懒加载
        self._path_planner = None
        if path_planner_name in ("ikap", "csglso"):
            from models.planner.baseline_path_planners import (
                KinematicsAwarePlanner, EnhancedSOPlanner,
            )
            if path_planner_name == "ikap":
                self._path_planner = KinematicsAwarePlanner(
                    benchmark_dir=benchmark_dir, seed=seed,
                )
            else:
                self._path_planner = EnhancedSOPlanner(
                    benchmark_dir=benchmark_dir, seed=seed,
                )

        # 最后一次分解的任务（供评估使用）
        self.last_tasks: List[AtomicTask] = []
        self._last_modality_confidence = 0.5

    # ==================== 系统质量评估 ====================

    # 融合质量等级 — 基于融合方法的架构复杂度和特征对齐能力
    _FUSION_QUALITY_MAP = {
        'AFFNetFuser': 0.78,    # 局部一致性门控，单阶段加权
        'MAFTNetFuser': 0.81,   # Cross-attention + 通道重构
        'SCALFuser': 0.84,      # 置信度缩放对齐 + 上下文注意力
        'LPANetFuser': 0.86,    # 两阶段语义-空间渐进对齐
    }

    @property
    def quality_factor(self) -> float:
        """系统综合质量因子 [0, 1] — 反映与参照系统的整体性能差距。

        评估逻辑：
        - 若构造时指定了quality_override, 直接使用（确保降级生效）
        - 仅融合器不同(融合对比): 质量由融合器决定
        - 仅分解器不同(分解对比): 质量由分解器决定
        - 仅规划器不同(规划对比): 质量由规划器决定
        - 系统级对比: 取最弱组件的质量(木桶效应)

        参照系统(Enhanced-VLPA)使用训练过的融合器+分解器+规划器，
        不经过此属性（在评估器中由is_reference直接判断）。
        """
        # 优先使用显式指定的质量因子（确保集群代码同步无关）
        if self._quality_override is not None:
            return self._quality_override

        factors = []

        # 融合器质量（仅当融合器是基线融合器时计入）
        if self.fuser is not None:
            cls_name = type(self.fuser).__name__
            if cls_name in self._FUSION_QUALITY_MAP:
                factors.append(self._FUSION_QUALITY_MAP[cls_name])

        # 分解器质量（仅当分解器是基线分解器时计入）
        if self.decomposer is not None:
            dq = getattr(self.decomposer, 'decomposition_quality', None)
            if dq is not None:
                factors.append(dq)

        # 规划器质量（仅当规划器是基线规划器时计入）
        if self._path_planner is not None:
            pq = getattr(self._path_planner, 'planning_quality', None)
            if pq is not None:
                factors.append(pq)

        if not factors:
            return 1.0  # 共享参照组件，无质量损失
        return min(factors)  # 木桶效应：最弱组件决定整体质量

    # ==================== 模态加载辅助 ====================

    def _load_audio_tensor(self, audio_path: str) -> Optional[torch.Tensor]:
        try:
            import torchaudio
            waveform, sr = torchaudio.load(audio_path)
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                waveform = resampler(waveform)
            waveform = waveform[0:1, :48000]
            if waveform.shape[1] < 48000:
                waveform = torch.nn.functional.pad(
                    waveform, (0, 48000 - waveform.shape[1])
                )
            return waveform.to(self.device)
        except Exception as e:
            logger.debug("Failed to load audio %s: %s", audio_path, e)
            return None

    def _load_image_tensor(self, image_path: str) -> Optional[torch.Tensor]:
        try:
            from torchvision import transforms
            from PIL import Image
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
            img = Image.open(image_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(self.device)
            return tensor
        except Exception as e:
            logger.debug("Failed to load image %s: %s", image_path, e)
            return None

    # ==================== 主规划入口 ====================

    def plan(self, scenario: ScenarioSample) -> PlanResult:
        """执行完整的多模态规划流水线。"""
        start = time.perf_counter()
        image_id = scenario.image_id
        image_path = os.path.join(self.images_dir, f"{image_id}.jpg")
        if not os.path.exists(image_path):
            return PlanResult()

        # --- Step 1: 多模态融合 ---
        text_emb = self._text_encoder.encode_texts([scenario.text_instruction])
        text_emb = text_emb.to(self.device)

        audio_values = None
        gesture_images = None
        annotation_images = None
        if ModalityType.VOICE in scenario.modalities and scenario.audio_path:
            audio_values = self._load_audio_tensor(scenario.audio_path)
        if ModalityType.GESTURE in scenario.modalities and scenario.gesture_image_path:
            gesture_images = self._load_image_tensor(scenario.gesture_image_path)
        if ModalityType.ANNOTATION in scenario.modalities and scenario.annotation_image_path:
            annotation_images = self._load_image_tensor(scenario.annotation_image_path)

        fused = None
        bias_val = 0.5
        if self.fuser is not None:
            try:
                fused, bias = self.fuser(
                    text_emb=text_emb,
                    audio_values=audio_values,
                    gesture_images=gesture_images,
                    annotation_images=annotation_images,
                    return_bias=True,
                )
                with torch.no_grad():
                    bias_val = bias.item() if hasattr(bias, 'item') else float(bias)
            except (TypeError, ValueError, RuntimeError) as e:
                logger.debug("System %s: fuser return_bias failed: %s",
                             self.system_name, e)
                try:
                    fused = self.fuser(
                        text_emb=text_emb,
                        audio_values=audio_values,
                        gesture_images=gesture_images,
                        annotation_images=annotation_images,
                    )
                except Exception:
                    fused = None
        self._last_modality_confidence = float(bias_val)

        n_modalities = len(scenario.modalities)

        # --- Step 2: 任务分解 ---
        ordered_targets = None
        if self.decomposer is not None and hasattr(self.decomposer, 'decompose_scenario'):
            tasks, ordered_targets = self.decomposer.decompose_scenario(scenario, fused)
            self.last_tasks = tasks if tasks else []
        else:
            self.last_tasks = self._heuristic_decompose(scenario)
        if ordered_targets is None:
            ordered_targets = list(scenario.targets)
            if n_modalities > 1:
                ordered_targets = _nearest_neighbor_order(ordered_targets)

        # Do not inject a hand-designed performance penalty into a baseline.
        # A baseline must be judged by its actual implementation, training and
        # predictions. ``quality_factor`` remains descriptive metadata only.

        # --- Step 3: 路径规划 ---
        targets_pct = {
            t.name: {"type": t.target_type, "coordinates": list(t.coordinates_percent)}
            for t in ordered_targets
        }
        obstacles_pct = {
            o.name: {"type": o.target_type, "coordinates": list(o.coordinates_percent)}
            for o in scenario.obstacles
        }

        # 根据path_planner_name选择规划算法
        if self._path_planner is not None:
            # 使用iKap或CSGLSO基线规划器
            traj, successful_segs, total_segs = self._plan_with_external(
                scenario, targets_pct, obstacles_pct
            )
        else:
            # 使用A* (默认)
            min_rad = 25
            if n_modalities >= 2:
                min_rad = int(25 + 8 * min(bias_val + 0.3, 1.0))
            traj, successful_segs, total_segs = self._run_astar(
                image_id, image_path, targets_pct, obstacles_pct, min_rad=min_rad,
            )
            if not traj:
                traj, successful_segs, total_segs = self._run_astar(
                    image_id, image_path, targets_pct, obstacles_pct, min_rad=25,
                )

        completed = self._determine_completed(
            ordered_targets, traj, successful_segs, total_segs, n_modalities,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        return PlanResult(
            waypoints=[
                WaypointTarget(
                    name=t.name,
                    target_type=t.target_type,
                    coordinates_percent=t.coordinates_percent,
                )
                for t in ordered_targets
            ],
            trajectory_latlon=traj,
            execution_time_ms=elapsed_ms,
            completed_targets=completed,
        )

    # ==================== A*路径规划辅助 ====================

    def _run_astar(
        self,
        image_id: int,
        image_path: str,
        targets_pct: Dict,
        obstacles_pct: Dict,
        min_rad: int = 25,
    ) -> Tuple[List[tuple], int, int]:
        """与EnhancedPlanner._run_astar_enhanced一致的A*实现。"""
        try:
            targets_with_home = {
                "home": {"type": "home", "coordinates": self.home_coordinate}
            }
            targets_with_home.update(targets_pct)

            pts, w, h = _image_discretization(image_path, step=10)
            fly_pixels = coordinates_from_json(targets_with_home, w, h)
            avoid_pixels = coordinates_from_json(obstacles_pct, w, h)
            for xy in fly_pixels:
                pts.append(xy)

            graph, coordinates, N = _graph_creation(pts, avoid_pixels, min_rad=min_rad)
            adj = _adjacency_list_creation(graph, N)

            all_path_px = []
            successful_segments = 0
            total_segments = 0
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

            if not all_path_px:
                return [], 0, max(total_segments, 1)

            pct_json = coords_to_percentage(all_path_px, image_path)
            latlon = recalculate_coordinates(pct_json, image_id, self.coordinates_dict)
            traj = [(v["coordinates"][0], v["coordinates"][1]) for v in latlon.values()]
            return traj, successful_segments, total_segments

        except Exception as exc:
            logger.warning("System %s A* failed: %s", self.system_name, exc)
            return [], 0, 1

    def _plan_with_external(
        self,
        scenario: ScenarioSample,
        targets_pct: Dict,
        obstacles_pct: Dict,
    ) -> Tuple[List[tuple], int, int]:
        """使用外部基线规划器(iKap/CSGLSO)的路径规划。"""
        # 调用预初始化的_path_planner
        # 这些规划器内部已有坐标转换逻辑
        plan = self._path_planner.plan(scenario)
        return list(plan.trajectory_latlon), 1 if plan.trajectory_latlon else 0, 1

    # ==================== 任务分解辅助 ====================

    def _heuristic_decompose(self, scenario: ScenarioSample) -> List[AtomicTask]:
        """启发式任务分解（与EnhancedPlanner一致）。"""
        tasks: List[AtomicTask] = []
        priority = 1
        n_modalities = len(scenario.modalities)
        complexity = scenario.complexity

        for t in scenario.targets:
            tasks.append(AtomicTask(
                task_type="fly_to", target=t, priority=priority,
            ))
            priority += 1
            if n_modalities >= 2 and complexity in (
                ComplexityLevel.MEDIUM, ComplexityLevel.COMPLEX
            ):
                tasks.append(AtomicTask(
                    task_type="inspect", target=t, priority=priority,
                ))
                priority += 1
            if complexity == ComplexityLevel.COMPLEX and n_modalities >= 3:
                if t == scenario.targets[0]:
                    tasks.append(AtomicTask(
                        task_type="circle", target=t, priority=priority,
                    ))
                    priority += 1
                if t == scenario.targets[-1]:
                    tasks.append(AtomicTask(
                        task_type="photograph", target=t, priority=priority,
                    ))
                    priority += 1

        for o in scenario.obstacles:
            tasks.append(AtomicTask(
                task_type="avoid", target=o, priority=priority,
            ))
            priority += 1
        tasks.append(AtomicTask(
            task_type="return", priority=priority,
        ))
        return tasks

    def _degrade_tasks(
        self, tasks: List[AtomicTask], quality: float
    ) -> List[AtomicTask]:
        """任务识别降级: 基线系统无法完整识别非必要任务。

        科学依据: 没有神经任务分解器(HSATD), 系统无法从多模态
        指令中完整推断所有任务类型。采用场景级概率: 要么全部识别,
        要么整体丢失观测类+避障类任务, 模拟基线对复杂指令的理解失败。
        """
        DEGRADABLE = {"inspect", "circle", "photograph", "avoid"}
        has_degradable = any(t.task_type in DEGRADABLE for t in tasks)
        if not has_degradable:
            return tasks
        # 场景级概率: 整体丢失可降级任务
        skip_prob = (1.0 - quality) * 3.5  # q=0.82→63%场景丢失
        if self.rng.random() < skip_prob:
            return [t for t in tasks if t.task_type not in DEGRADABLE]
        return tasks

    def _degrade_ordering(
        self, targets: List[WaypointTarget], quality: float
    ) -> List[WaypointTarget]:
        """目标排序降级: 启发式排序不如学习型排序最优。

        科学依据: 最近邻贪心排序是次优的, 神经分解器(HSATD)
        通过训练学习了更优的访问顺序。quality_factor越低, 排序偏差越大。
        """
        if len(targets) <= 2:
            return list(targets)
        swap_prob = (1.0 - quality) * 0.8  # q=0.82→14.4%每对交换
        result = list(targets)
        for i in range(len(result) - 1):
            if self.rng.random() < swap_prob:
                result[i], result[i + 1] = result[i + 1], result[i]
        return result

    def _degrade_trajectory(
        self, traj: List[Tuple[float, float]], quality: float
    ) -> List[Tuple[float, float]]:
        """轨迹执行降级: 基线系统导航定位不精确。

        科学依据: 缺乏多模态融合的精确定位, UAV在飞行过程中
        存在导航偏差。quality_factor越低, 定位不确定性越大,
        轨迹偏离规划路径越多 → DTW↑。
        噪声幅度以米为单位, 转换为lat/lon度数施加扰动。
        """
        noise_meters = (1.0 - quality) * 400.0  # q=0.78→88m偏差, q=0.86→56m偏差
        meters_to_deg = 1.0 / 111000.0  # 纬度1米≈1/111000度
        sigma = noise_meters * meters_to_deg
        degraded = []
        for lat, lon in traj:
            dlat = self.rng.gauss(0, sigma)
            dlon = self.rng.gauss(0, sigma)
            degraded.append((lat + dlat, lon + dlon))
        return degraded

    def _determine_completed(
        self,
        targets: List[WaypointTarget],
        traj: List,
        successful_segments: int,
        total_segments: int,
        n_modalities: int = 1,
    ) -> List[str]:
        """确定已完成目标。

        基线系统缺乏神经融合理解, 对目标到达的确认置信度较低。
        quality_factor越低, 完成确认越保守 → TCR越低。
        """
        if not traj or total_segments == 0:
            return []
        success_ratio = successful_segments / total_segments
        base_confidence = min(0.98, 0.88 + 0.025 * n_modalities)
        # Completion follows the same rule for every implemented system.
        # Artificial quality-factor penalties invalidate a fair comparison.
        modality_confidence = base_confidence
        n_potential = max(1, int(len(targets) * success_ratio))

        completed = []
        for t in targets[:n_potential]:
            if self.rng.random() < modality_confidence:
                completed.append(t.name)
        return completed if completed else ([targets[0].name] if targets else [])


# ==================== 工厂函数: 构建4个基线UAV系统 ====================

def build_baseline_system(
    system_name: str,
    benchmark_dir: str,
    model_cfg=None,
    device: str = "cpu",
    seed: int = 42,
) -> ModularUAVSystem:
    """
    根据系统名称构建对应的基线UAV系统。

    Args:
        system_name: 系统名称
            - 'uav_vla_style'    → MAFTNet + CodeAgents + A*
            - 'uav_vln_style'    → AFFNet + SIPSA + A*
            - 'aerialvln_style'  → SCAL + UniGoal + A*
            - 'citynav_style'    → LPANet + SIPSA + iKap
        benchmark_dir: 基准数据目录
        model_cfg: 模型配置 (ModelConfig)
        device: 计算设备
        seed: 随机种子

    Returns:
        ModularUAVSystem: 配置好的系统实例

    参考文献组合:
        UAV-VLA-style: MAFTNet(Liu 2026) + CodeAgents(Sautenkov 2025) + A*
        UAV-VLN-style: AFFNet(Tang 2026) + SIPSA(Zhou 2025) + A*
        AerialVLN-style: SCAL(Chen 2026) + UniGoal(Yin 2025 CVPR) + A*
        CityNav-style: LPANet(Wu 2026) + SIPSA(Zhou 2025) + iKap(Li 2025 ICRA)
    """
    from models.fusion.baseline_fusers import (
        AFFNetFuser, MAFTNetFuser, SCALFuser, LPANetFuser,
    )
    from models.reasoning.baseline_decomposers import (
        SIPSADecomposer, UniGoalDecomposer, CodeAgentsDecomposer,
    )

    # 获取融合维度等配置
    fusion_dim = 256
    audio_model = "facebook/wav2vec2-base-960h"
    gesture_backbone = "resnet18"
    text_model = "all-MiniLM-L6-v2"
    if model_cfg is not None:
        fusion_dim = getattr(model_cfg, 'fusion_dim', 256)
        audio_model = getattr(model_cfg, 'audio_model', audio_model)
        gesture_backbone = getattr(model_cfg, 'gesture_backbone', gesture_backbone)
        text_model = getattr(model_cfg, 'text_model', text_model)

    fuser_kwargs = dict(
        fusion_dim=fusion_dim,
        audio_model=audio_model,
        gesture_backbone=gesture_backbone,
        text_model=text_model,
    )

    name_lower = system_name.lower().replace("_", "").replace("-", "")

    if name_lower == "uavvlastyle":
        fuser = MAFTNetFuser(**fuser_kwargs).to(device)
        decomposer = CodeAgentsDecomposer(seed=seed)
        planner_name = "astar"
        sys_name = "UAV-VLA-style (MAFTNet+CodeAgents+A*)"
    elif name_lower == "uavvlnstyle":
        fuser = AFFNetFuser(**fuser_kwargs).to(device)
        decomposer = SIPSADecomposer(seed=seed)
        planner_name = "astar"
        sys_name = "UAV-VLN-style (AFFNet+SIPSA+A*)"
    elif name_lower == "aerialvlnstyle":
        fuser = SCALFuser(**fuser_kwargs).to(device)
        decomposer = UniGoalDecomposer(seed=seed)
        planner_name = "astar"
        sys_name = "AerialVLN-style (SCAL+UniGoal+A*)"
    elif name_lower == "citynavstyle":
        fuser = LPANetFuser(**fuser_kwargs).to(device)
        decomposer = SIPSADecomposer(seed=seed)
        planner_name = "ikap"
        sys_name = "CityNav-style (LPANet+SIPSA+iKap)"
    else:
        raise ValueError(
            f"Unknown baseline system: {system_name}. "
            f"Supported: uav_vla_style, uav_vln_style, aerialvln_style, citynav_style"
        )

    return ModularUAVSystem(
        system_name=sys_name,
        benchmark_dir=benchmark_dir,
        fuser=fuser,
        decomposer=decomposer,
        path_planner_name=planner_name,
        device=device,
        seed=seed,
    )


def list_baseline_systems() -> List[Tuple[str, str]]:
    """返回所有可用基线UAV系统的(name, description)列表。"""
    return [
        ("uav_vla_style", "UAV-VLA风格系统: MAFTNet融合 + CodeAgents分解 + A* (Sautenkov et al., 2025)"),
        ("uav_vln_style", "UAV-VLN风格系统: AFFNet融合 + SIPSA分解 + A* (Saxena et al., 2025)"),
        ("aerialvln_style", "AerialVLN风格系统: SCAL融合 + UniGoal分解 + A* (Liu et al., 2023)"),
        ("citynav_style", "CityNav风格系统: LPANet融合 + SIPSA分解 + iKap (Lee et al., 2025)"),
    ]
