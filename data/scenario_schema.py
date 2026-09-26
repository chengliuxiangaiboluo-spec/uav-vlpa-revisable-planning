"""
场景模式模块 - UAV-VLPA系统的多模态实验框架核心数据结构。

定义了UAV-VLPA系统中所有核心数据结构，包括模态类型、
复杂度级别、航点目标、原子任务、专家路径、场景样本、
规划结果和评估指标等，是整个多模态实验框架的数据基础。

核心功能：
- 数据标准化：定义统一的数据结构确保跨模块一致性
- 序列化支持：提供JSON序列化和反序列化接口
- 类型安全：使用Python类型提示确保数据完整性
- 扩展性设计：支持新增模态类型和复杂度级别

UAV-VLPA系统集成：
- 与数据生成模块协同工作，构建完整的多模态场景
- 为模型训练提供标准化的输入数据格式
- 为评估模块提供统一的指标计算接口
- 支持实时推理和离线评估的统一数据结构
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Optional, Tuple, Dict, Any


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ModalityType(Enum):
    """
    模态类型枚举 - UAV-VLPA多模态融合架构的核心分类器。

    定义了UAV-VLPA系统支持的所有输入模态类型，
    是多模态融合和跨模态注意力机制的基础。

    核心设计原则：
    - 单一职责：每个模态类型代表独立的感知通道
    - 可扩展性：支持新增模态类型而无需修改核心逻辑
    - 兼容性：与现有数据生成和模型训练流程无缝集成

    模态类型说明：
        TEXT: 文本指令模态
            - 自然语言文本指令
            - 基础模态，所有场景必需
        VOICE: 语音指令模态
            - 语音识别生成的文本指令
            - 支持自然交互
        GESTURE: 手势轨迹模态
            - 手绘风格的手势轨迹
            - 视觉空间指令
        ANNOTATION: 图像标注模态
            - 箭头、圆圈、区域高亮等视觉标注
            - 多模态语义增强
        MIXED: 混合模态
            - 多种模态的组合
            - 支持复杂的多模态交互
    """
    TEXT = "text"
    VOICE = "voice"
    GESTURE = "gesture"
    ANNOTATION = "annotation"
    MIXED = "mixed"


class ComplexityLevel(Enum):
    """
    复杂度级别枚举 - UAV-VLPA课程学习和评估基准的核心维度。

    定义了UAV-VLPA系统中任务复杂度的三个级别，
    是课程学习(Curriculum Learning)和公平评估的基础。

    核心设计原则：
    - 学术驱动：基于UAV-VLPA*论文(Sautenkov et al., 2025)的实证研究
    - 工程实用：考虑实际无人机部署的资源约束
    - 分层设计：支持从简单到复杂的渐进式学习

    复杂度级别说明：
        SIMPLE: 简单级别
            - 单模态（仅文本）
            - 目标数量：<3个
            - 无条件约束
            - 用于基础功能验证
        MEDIUM: 中等级别
            - 双模态（文本 + 1个额外模态）
            - 目标数量：2-5个
            - 简单条件约束
            - 用于验证多模态融合优势
        COMPLEX: 复杂级别
            - 多模态（文本 + 2-3个额外模态）
            - 目标数量：3-8个
            - 复杂动态约束
            - 用于充分展示系统性能
    """
    SIMPLE = "simple"       # Single modality, <3 targets
    MEDIUM = "medium"       # Dual modality, conditional logic
    COMPLEX = "complex"     # Multi-modal, multi-stage, dynamic constraints


# ---------------------------------------------------------------------------
# Waypoint / Task primitives
# ---------------------------------------------------------------------------

@dataclass
class WaypointTarget:
    """
    航点目标类 - UAV-VLPA地理空间建模的核心实体。

    表示地图上的一个航点或目标位置，是路径规划和任务分解的基本单元。

    核心设计原则：
    - 多坐标系统：支持百分比坐标和地理坐标
    - 类型丰富：支持多种目标类型
    - 扩展性：支持新增目标属性

    属性说明：
        name: 目标名称
            - 唯一标识符，如'target_1', 'obstacle_2'
            - 用于任务序列和路径规划
        target_type: 目标类型
            - 如'building', 'lake', 'tree', 'road'
            - 影响路径规划和避障策略
        coordinates_percent: 百分比坐标
            - (x%, y%) 在卫星图像上的相对位置
            - 与图像分辨率无关，便于跨平台使用
        coordinates_latlon: 地理坐标
            - (latitude, longitude) WGS84坐标系
            - 用于Haversine距离计算和真实世界映射
    """
    name: str
    target_type: str                          # e.g. "building", "lake"
    coordinates_percent: Tuple[float, float]  # (x%, y%) on satellite image
    coordinates_latlon: Tuple[float, float] = (0.0, 0.0)


@dataclass
class AtomicTask:
    """
    原子任务类 - UAV-VLPA任务分解的核心输出单元。

    表示由任务分解器生成的最小可执行任务单元，
    是UAV-VLPA系统中任务规划和执行的基本粒度。

    核心设计原则：
    - 原子性：每个任务都是最小可执行单元
    - 可组合性：多个原子任务组合成完整任务序列
    - 可扩展性：支持新增任务类型

    属性说明：
        task_type: 任务类型
            - "fly_to": 飞向目标位置
            - "circle": 在目标周围盘旋
            - "avoid": 规避障碍物
            - "inspect": 检查目标
            - "return": 返回起点
            - "hover": 悬停
            - "photograph": 拍摄照片
            - "EOS": 任务结束
        target: 目标对象
            - 可选，某些任务类型不需要目标
            - 如'fly_to'需要目标，'hover'不需要
        priority: 优先级
            - 数值越小优先级越高
            - 用于多任务调度和冲突解决
        conditions: 条件列表
            - 动态约束条件
            - 如'if wind_speed > 20km/h, circle at safe distance'
            - 用于强化学习的奖励信号
    """
    task_type: str          # "fly_to", "circle", "avoid", "inspect", "return"
    target: Optional[WaypointTarget] = None
    priority: int = 1       # lower = higher priority
    conditions: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Expert / ground-truth path
# ---------------------------------------------------------------------------

@dataclass
class ExpertPath:
    """
    专家路径类 - UAV-VLPA评估基准的核心数据结构。

    表示为单个场景生成的一个专家标注路径，
    是UAV-VLPA系统中性能评估和模型训练的监督信号来源。

    核心设计原则：
    - 多策略：提供最优、保守、快速三种路径变体
    - 可复现：确定性算法确保结果一致性
    - 可解释：路径生成过程透明可追溯

    属性说明：
        waypoints: 航点列表
            - 按访问顺序排列的目标序列
            - 用于任务分解和路径规划
        path_coordinates_latlon: 路径坐标列表
            - [(lat1, lon1), (lat2, lon2), ...]
            - 用于Haversine距离计算和轨迹质量评估
        variant_label: 变体标签
            - "optimal": 标准A*算法生成的最优路径
            - "conservative": 增大障碍物缓冲区的保守路径
            - "fast": 目标子集的快速路径
            - 用于多角度评估和模型诊断
    """
    waypoints: List[WaypointTarget] = field(default_factory=list)
    path_coordinates_latlon: List[Tuple[float, float]] = field(default_factory=list)
    variant_label: str = "optimal"  # "optimal", "conservative", "fast"


# ---------------------------------------------------------------------------
# Scenario sample (single experiment data point)
# ---------------------------------------------------------------------------

@dataclass
class ScenarioSample:
    """
    场景样本类 - UAV-VLPA多模态实验框架的核心数据容器。

    表示一个完整的多模态场景样本，是UAV-VLPA系统中
    数据生成、模型训练、路径规划和性能评估的基本单元。

    核心设计原则：
    - 完整性：包含所有必要的多模态信息
    - 灵活性：支持不同复杂度和模态组合
    - 可扩展性：支持新增字段和元数据

    属性说明：
        scenario_id: 场景ID
            - 唯一标识符，格式'scenario_{number}'
            - 用于数据追踪和实验管理
        image_id: 图像ID
            - 1-30，对应基准卫星图像索引
            - 用于坐标转换和地理映射
        complexity: 复杂度级别
            - ComplexityLevel枚举值
            - 决定任务难度和评估标准
        modalities: 模态列表
            - ModalityType枚举值列表
            - 决定场景的多模态配置

        指令输入（根据模态选择性填充）：
            text_instruction: 文本指令
                - 自然语言文本指令
                - 所有场景的基础输入
            audio_path: 语音文件路径
                - WAV格式语音指令
                - 用于语音模态
            gesture_image_path: 手势图像路径
                - PNG格式手势轨迹
                - 用于手势模态
            annotation_image_path: 标注图像路径
                - PNG格式标注图像
                - 用于标注模态

        目标与障碍物：
            targets: 目标列表
                - WaypointTarget对象列表
                - 任务执行的目标位置
            obstacles: 障碍物列表
                - WaypointTarget对象列表
                - 需要规避的危险区域

        条件约束：
            conditions: 条件列表
                - 动态约束条件字符串
                - 用于强化学习和条件规划

        真值数据：
            ground_truth_paths: 专家路径列表
                - 3种专家路径变体
                - 用于评估和监督学习
            expert_atomic_tasks: 专家原子任务列表
                - 专家定义的任务序列
                - 用于指令准确性和任务分解评估

        元数据：
            metadata: 额外元数据字典
                - 实验配置、时间戳、版本信息等
                - 用于实验追踪和调试
    """
    scenario_id: str
    image_id: int                                    # 1-30, benchmark image index

    complexity: ComplexityLevel = ComplexityLevel.SIMPLE
    modalities: List[ModalityType] = field(default_factory=lambda: [ModalityType.TEXT])

    # Instruction inputs (not all are filled; depends on modalities)
    text_instruction: str = ""
    audio_path: Optional[str] = None
    gesture_image_path: Optional[str] = None
    annotation_image_path: Optional[str] = None

    # Targets & obstacles
    targets: List[WaypointTarget] = field(default_factory=list)
    obstacles: List[WaypointTarget] = field(default_factory=list)

    # Conditional / dynamic constraints
    conditions: List[str] = field(default_factory=list)

    # Ground truth (3 expert paths per scenario)
    ground_truth_paths: List[ExpertPath] = field(default_factory=list)

    # Expert atomic task sequence (ground truth for instruction accuracy)
    expert_atomic_tasks: List[AtomicTask] = field(default_factory=list)

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ---- serialisation helpers -------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """
        序列化为字典 - UAV-VLPA数据持久化核心接口。

        将ScenarioSample对象序列化为字典格式，用于JSON序列化和存储，
        是UAV-VLPA系统中数据持久化和跨模块通信的关键接口。

        算法原理：
        - 数据结构转换：将dataclass对象转换为字典
        - 枚举处理：将ComplexityLevel和ModalityType枚举转换为字符串值
        - 类型安全：保持原始数据类型的完整性

        无人机应用考虑：
        - JSON兼容：确保生成的字典可直接序列化为JSON
        - 向后兼容：支持不同版本的数据格式
        - 性能优化：高效的序列化算法

        返回值：
            Dict[str, Any]: 序列化后的字典
                - 包含所有ScenarioSample属性
                - complexity: ComplexityLevel枚举的字符串值
                - modalities: ModalityType枚举列表的字符串值
                - targets: 目标列表的字典表示
                - obstacles: 障碍物列表的字典表示
                - ground_truth_paths: 专家路径列表的字典表示
                - expert_atomic_tasks: 专家原子任务列表的字典表示
        """
        d = asdict(self)
        d["complexity"] = self.complexity.value
        d["modalities"] = [m.value for m in self.modalities]
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ScenarioSample":
        """
        从字典反序列化 - UAV-VLPA数据加载核心接口。

        将字典格式的数据反序列化为ScenarioSample对象，
        用于从JSON文件或网络传输中加载场景数据，
        是UAV-VLPA系统中数据加载和恢复的关键接口。

        算法原理：
        - 数据结构重建：将字典转换回dataclass对象
        - 枚举重建：将字符串值转换为ComplexityLevel和ModalityType枚举
        - 嵌套对象：递归重建WaypointTarget、ExpertPath、AtomicTask等嵌套对象

        无人机应用考虑：
        - 错误处理：优雅处理缺失字段和数据类型错误
        - 向后兼容：支持不同版本的数据格式迁移
        - 性能优化：高效的反序列化算法

        参数说明：
            d: 字典格式的数据
                - 包含ScenarioSample的所有属性
                - complexity: ComplexityLevel枚举的字符串值
                - modalities: ModalityType枚举列表的字符串值
                - targets: 目标列表的字典表示
                - obstacles: 障碍物列表的字典表示
                - ground_truth_paths: 专家路径列表的字典表示
                - expert_atomic_tasks: 专家原子任务列表的字典表示

        返回值：
            ScenarioSample: 反序列化后的场景样本对象
                - 完整重建所有嵌套对象
                - 保持原始数据的完整性和一致性
        """
        d["complexity"] = ComplexityLevel(d["complexity"])
        d["modalities"] = [ModalityType(m) for m in d["modalities"]]
        d["targets"] = [WaypointTarget(**t) for t in d.get("targets", [])]
        d["obstacles"] = [WaypointTarget(**o) for o in d.get("obstacles", [])]
        d["ground_truth_paths"] = [
            ExpertPath(
                waypoints=[WaypointTarget(**w) for w in gp.get("waypoints", [])],
                path_coordinates_latlon=[
                    tuple(c) for c in gp.get("path_coordinates_latlon", [])
                ],
                variant_label=gp.get("variant_label", "optimal"),
            )
            for gp in d.get("ground_truth_paths", [])
        ]
        # 反序列化 expert_atomic_tasks
        d["expert_atomic_tasks"] = [
            AtomicTask(
                task_type=t.get("task_type", "fly_to"),
                target=WaypointTarget(**t["target"]) if t.get("target") else None,
                priority=t.get("priority", 1),
                conditions=t.get("conditions", []),
            )
            for t in d.get("expert_atomic_tasks", [])
        ]
        return cls(**d)


# ---------------------------------------------------------------------------
# Plan result (output of a planner)
# ---------------------------------------------------------------------------

@dataclass
class PlanResult:
    """
    规划结果类 - UAV-VLPA路径规划模块的核心输出。

    表示规划器为单个场景生成的路径规划结果，
    是UAV-VLPA系统中路径执行和性能评估的基础。

    核心设计原则：
    - 完整性：包含所有必要的规划信息
    - 实时性：支持毫秒级规划响应
    - 可扩展性：支持新增规划指标

    属性说明：
        waypoints: 航点列表
            - 按执行顺序排列的目标序列
            - 用于任务分解和路径跟踪
        trajectory_latlon: 轨迹坐标列表
            - [(lat1, lon1), (lat2, lon2), ...]
            - 用于Haversine距离计算和轨迹质量评估
        execution_time_ms: 执行时间（毫秒）
            - 规划算法的运行时间
            - 用于实时性评估
        completed_targets: 已完成目标列表
            - 成功访问的目标名称
            - 用于任务完成率计算
        trajectory_length_km: 轨迹长度（千米）
            - Haversine距离计算的总路径长度
            - 用于效率评估
    """
    waypoints: List[WaypointTarget] = field(default_factory=list)
    trajectory_latlon: List[Tuple[float, float]] = field(default_factory=list)
    execution_time_ms: float = 0.0
    completed_targets: List[str] = field(default_factory=list)
    trajectory_length_km: float = 0.0
    # Runtime evidence for strict external-validation protocols.  Kept empty
    # for ordinary planners so existing callers remain backward compatible.
    input_audit: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Aggregated metrics for one scenario evaluation
# ---------------------------------------------------------------------------

@dataclass
class MetricsResult:
    """
    评估指标类 - UAV-VLPA科学评估体系的核心数据结构。

    定义了UAV-VLPA系统中完整的科学评估指标体系，
    是公平比较不同算法和模型性能的基础。

    核心设计原则：
    - 科学性：基于学术文献验证的评估指标
    - 全面性：覆盖性能、效率、质量、交互等多个维度
    - 可比性：标准化的计算方法确保公平比较

    科学评估指标体系：
    1. 核心性能指标：
       - task_completion_rate: 任务完成率
           * 成功完成的目标数量 / 总目标数量
           * 衡量基本功能可靠性
       - instruction_accuracy: 指令准确性
           * 正确解析的指令数量 / 总指令数量
           * 衡量多模态理解能力

    2. 轨迹效率指标：
       - trajectory_length_km: 轨迹长度
           * Haversine距离计算的总路径长度
           * 衡量能耗效率
       - efficiency_ratio: 效率比率
           * 任务完成率 / 轨迹长度
           * 综合性能指标，越高越好

    3. 轨迹相似度指标：
       - dtw_rmse: DTW误差
           * 动态时间规整均方根误差
           * 衡量轨迹形状相似度
       - knn_rmse: KNN误差
           * K近邻均方根误差
           * 衡量局部轨迹质量
       - sequential_rmse: 顺序误差
           * 顺序匹配均方根误差
           * 衡量任务执行顺序准确性

    4. 交互性能指标：
       - response_latency_ms: 响应延迟
           * 从指令接收到路径规划完成的时间
           * 衡量实时性
       - replan_success_rate: 重规划成功率
           * 成功重规划次数 / 总重规划次数
           * 衡量环境适应能力

    5. 主观评估指标：
       - subjective_score: 主观评分
           * 专家人工评分（模拟）
           * 衡量综合用户体验

    学术依据：
    - Hooey et al. (2012): 无人机任务评估指标体系
    - Baltrusaitis et al. (2019): 多模态系统评估方法
    - Sautenkov et al. (2025): UAV-VLPA*论文的评估框架
    """
    scenario_id: str = ""

    # Trajectory quality (reused from existing modules)
    knn_rmse: float = 0.0
    dtw_rmse: float = 0.0
    sequential_rmse: float = 0.0
    trajectory_length_km: float = 0.0

    # Task-level
    task_completion_rate: float = 0.0
    instruction_accuracy: float = 0.0

    # Efficiency metrics (科学对比指标)
    efficiency_ratio: float = 0.0  # 任务完成率 / 轨迹长度

    # Interaction
    response_latency_ms: float = 0.0
    replan_success_rate: float = 0.0

    # Subjective (simulated)
    subjective_score: float = 0.0
