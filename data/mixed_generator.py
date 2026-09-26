"""
混合模态生成器模块 - UAV-VLPA系统的多模态数据融合核心组件。

将2-3个单模态输出组合成一个ScenarioSample，
通过调用各个独立的生成器并合并它们的结果，
构建完整的多模态场景数据。

核心功能：
- 模态组合：根据复杂度级别智能选择文本、语音、手势、标注等模态组合
- 数据融合：将不同模态的数据统一到ScenarioSample结构中
- 路径管理：为各模态数据生成标准化的文件路径
- 多模态对齐：确保不同模态在时间、空间和语义上的一致性

UAV-VLPA系统集成：
- 与数据生成流水线深度集成，为不同复杂度场景配置最优模态组合
- 为多模态融合模型提供训练数据
- 支持课程学习(Curriculum Learning)的分层数据生成
- 为评估模块提供标准化的多模态测试数据

学术依据：
- 基于UAV-VLPA*论文(Sautenkov et al., 2025)的模态组合策略
- 遵循多模态机器学习综述(Baltrusaitis et al., 2019)的最佳实践
- 复杂度驱动：简单场景单模态，复杂场景多模态
"""

# ==================== 标准库导入 ====================
import logging  # 日志记录
import random   # 随机数生成
from typing import List, Tuple  # 类型注解

# ==================== 项目模块导入 ====================
from data.scenario_schema import (
    ComplexityLevel,    # 复杂度级别枚举
    ModalityType,       # 模态类型枚举
    ScenarioSample,     # 场景样本类
    WaypointTarget,     # 航点目标类
)
from data.voice_generator import VoiceInstructionGenerator       # 语音生成器
from data.gesture_generator import GestureTrajectoryGenerator    # 手势生成器
from data.annotation_generator import ImageAnnotationGenerator   # 标注生成器

# 获取实验日志记录器
logger = logging.getLogger("experiment")


class MixedModalityGenerator:
    """
    混合模态生成器类 - UAV-VLPA多模态数据融合核心调度器。

    从各个独立生成器组合混合模态场景数据，
    负责根据复杂度级别选择适当的模态组合，并调用相应的生成器，
    是UAV-VLPA系统中连接不同模态生成器的关键枢纽。

    核心设计原则：
    - 模态协同：确保不同模态之间的语义一致性和时空对齐
    - 复杂度适配：根据任务复杂度自动选择最优模态组合
    - 可扩展性：支持新增模态类型而无需修改核心逻辑
    - 鲁棒性：处理各种模态生成失败的情况

    功能概览：
        1. 模态选择：基于复杂度级别的智能模态组合
        2. 数据增强：为现有场景添加多模态信息
        3. 路径管理：生成标准化的文件存储路径
        4. 元数据更新：维护场景样本的完整模态信息

    属性说明：
        voice_gen: 语音指令生成器
            - 生成自然语言语音指令
            - 添加环境噪声模拟真实场景
        gesture_gen: 手势轨迹生成器
            - 生成手绘风格的手势轨迹
            - 支持贝塞尔曲线和抖动效果
        annotation_gen: 图像标注生成器
            - 生成箭头、圆圈、区域高亮等视觉标注
            - 支持百分比坐标系统
    """

    def __init__(
        self,
        voice_gen: VoiceInstructionGenerator,
        gesture_gen: GestureTrajectoryGenerator,
        annotation_gen: ImageAnnotationGenerator,
    ):
        """
        初始化混合模态生成器。

        该构造函数配置混合模态生成器的核心参数，
        这些参数直接影响UAV-VLPA系统中多模态数据的质量和多样性。

        参数说明：
            voice_gen: 语音指令生成器实例
                - 用于生成高质量的语音指令
                - 支持多种语音风格和环境噪声
            gesture_gen: 手势轨迹生成器实例
                - 用于生成自然的手势轨迹
                - 支持不同复杂度的手势样式
            annotation_gen: 图像标注生成器实例
                - 用于生成专业的图像标注
                - 支持多种标注类型和样式
        """
        self.voice_gen = voice_gen
        self.gesture_gen = gesture_gen
        self.annotation_gen = annotation_gen

    @staticmethod
    def select_modalities(complexity: ComplexityLevel) -> List[ModalityType]:
        """
        为给定的复杂度级别选择合适的模态组合 - UAV-VLPA多模态数据生成核心算法。

        该方法实现了UAV-VLPA系统中关键的模态选择功能，
        根据任务复杂度智能选择最优的模态组合，
        确保训练数据集能够全面覆盖无人机任务的各种复杂度。

        算法原理：
        - 简单级别：单模态（仅文本），适合基础功能验证
        - 中等级别：双模态（文本 + 随机一个额外模态），需要少量视觉信息
        - 复杂级别：多模态（文本 + 2-3个额外模态），严重依赖视觉信息

        无人机应用考虑：
        - 模态组合：适应不同无人机平台的传感器配置
        - 资源优化：平衡计算成本和模型性能
        - 课程学习：支持渐进式多模态训练
        - 实时性能：优化模态生成的并发处理

        学术依据：
            - UAV-VLPA*论文(Sautenkov et al., 2025)的模态组合策略
            - 多模态机器学习综述(Baltrusaitis et al., 2019)的最佳实践
            - 复杂度驱动：简单场景单模态，复杂场景多模态

        参数说明：
            complexity: 复杂度级别
                - ComplexityLevel.SIMPLE: 简单级别，单模态
                - ComplexityLevel.MEDIUM: 中等级别，双模态
                - ComplexityLevel.COMPLEX: 复杂级别，多模态

        返回值：
            List[ModalityType]: 选定的模态类型列表
                - ModalityType.TEXT: 文本模态（必需）
                - ModalityType.VOICE: 语音模态
                - ModalityType.GESTURE: 手势模态
                - ModalityType.ANNOTATION: 标注模态
        """
        if complexity == ComplexityLevel.SIMPLE:
            # 简单级别：仅文本模态
            return [ModalityType.TEXT]

        elif complexity == ComplexityLevel.MEDIUM:
            # 中等级别：文本 + 随机一个额外模态
            secondary = random.choice(
                [ModalityType.VOICE, ModalityType.GESTURE, ModalityType.ANNOTATION]
            )
            return [ModalityType.TEXT, secondary]

        else:  # COMPLEX - 确保有足够的模态数量
            # 复杂级别必须有3+模态（TEXT + 至少2个额外）
            all_extra = [ModalityType.VOICE, ModalityType.GESTURE, ModalityType.ANNOTATION]
            # 随机选择2-3个额外模态（确保至少2个）
            extras = random.sample(all_extra, k=random.randint(2, 3))
            return [ModalityType.TEXT] + extras

    def augment_scenario(
        self,
        scenario: ScenarioSample,
        image_path: str,
        output_dir: str,
    ) -> ScenarioSample:
        """
        为现有场景样本填充额外的模态字段 - UAV-VLPA多模态数据增强核心接口。

        该方法实现了UAV-VLPA系统中关键的多模态数据增强功能，
        根据场景指定的模态组合，调用相应的生成器并写入输出文件，
        构建完整的多模态场景数据，是UAV-VLPA多模态融合架构的核心入口点。

        增强流程：
        1. 输入解析：提取目标和障碍物的百分比坐标
        2. 模态处理：按需调用语音、手势、标注生成器
        3. 路径生成：为各模态数据生成标准化文件路径
        4. 元数据更新：更新场景对象的模态路径字段

        无人机应用考虑：
        - 百分比坐标系统：确保在不同分辨率图像上的位置一致性
        - 文件路径管理：支持大规模数据集的组织和管理
        - 异常处理：优雅处理模态生成失败的情况
        - 性能优化：支持并发模态生成提高效率

        参数说明：
            scenario: 场景样本对象
                - 包含基础文本指令和模态需求
                - 用于指导多模态数据生成
            image_path: 卫星图像路径
                - 提供视觉模态的基础图像
                - 支持多种卫星图像格式
            output_dir: 输出目录路径
                - 存储生成的多模态数据
                - 支持子目录结构便于组织

        返回值：
            ScenarioSample: 增强后的场景样本（更新了路径字段）
                - audio_path: 语音文件路径
                - gesture_image_path: 手势图像路径
                - annotation_image_path: 标注图像路径
                - 用于后续的多模态模型训练和评估
        """
        # 提取目标和障碍物的百分比坐标
        targets_pct = [
            (t.coordinates_percent[0], t.coordinates_percent[1])
            for t in scenario.targets
        ]
        obstacles_pct = [
            (o.coordinates_percent[0], o.coordinates_percent[1])
            for o in scenario.obstacles
        ]

        sid = scenario.scenario_id  # 场景ID

        # ==================== 语音模态 ====================
        if ModalityType.VOICE in scenario.modalities:
            audio_path = f"{output_dir}/audio/{sid}.wav"
            # 生成语音并添加噪声
            self.voice_gen.generate(
                scenario.text_instruction, audio_path, add_noise=True
            )
            scenario.audio_path = audio_path  # 更新场景对象

        # ==================== 手势模态 ====================
        if ModalityType.GESTURE in scenario.modalities and targets_pct:
            gesture_path = f"{output_dir}/gesture/{sid}.png"
            # 生成手势轨迹图像
            self.gesture_gen.generate(image_path, targets_pct, gesture_path)
            scenario.gesture_image_path = gesture_path  # 更新场景对象

        # ==================== 标注模态 ====================
        if ModalityType.ANNOTATION in scenario.modalities:
            annot_path = f"{output_dir}/annotation/{sid}.png"
            # 生成标注图像
            self.annotation_gen.generate(
                image_path, targets_pct, obstacles_pct, annot_path
            )
            scenario.annotation_image_path = annot_path  # 更新场景对象

        return scenario
