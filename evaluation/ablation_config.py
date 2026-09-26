"""
消融实验配置模块 - UAV-VLPA系统消融研究核心配置。

定义所有消融实验组（B0-B9）的配置，控制各模块的启用/禁用状态。

消融组设计：
    B0: Baseline-TextOnly        仅文本输入，最弱对照
    B1: Enhanced-Full             完整模型，全模块开启（参照组）
    B2: -Voice                    移除语音模态
    B3: -Gesture                  移除手势模态
    B4: -Annotation               移除图像标注模态
    B5a: Text+Voice               文本+语音
    B5b: Text+Gesture             文本+手势
    B5c: Text+Annotation          文本+图像标注
    B6: -CrossModalAttention      跨模态注意力替换为简单拼接融合
    B7: -DataDrivenCalibration    固定默认参数替换数据驱动校准
    B8: -DynamicReplan            关闭动态重规划
    B9: -RefinedTaskDecomposition 简化任务分解（仅fly_to + return）
"""

from dataclasses import dataclass, field
from typing import List, Set
from enum import Enum

from data.scenario_schema import ModalityType


class AblationGroupID(Enum):
    """消融组标识符枚举。"""
    B0_BASELINE_TEXT_ONLY = "B0"
    B1_ENHANCED_FULL = "B1"
    B2_NO_VOICE = "B2"
    B3_NO_GESTURE = "B3"
    B4_NO_ANNOTATION = "B4"
    B5A_TEXT_VOICE = "B5a"
    B5B_TEXT_GESTURE = "B5b"
    B5C_TEXT_ANNOTATION = "B5c"
    B6_NO_CROSS_ATTENTION = "B6"
    B7_NO_CALIBRATION = "B7"
    B8_NO_REPLAN = "B8"
    B9_SIMPLE_DECOMPOSE = "B9"
    B10_NO_SEMANTIC_CONSTRAINT = "B10"
    B11_NO_VLM_GROUNDING = "B11"


@dataclass
class AblationGroupConfig:
    """
    单个消融组的配置。

    属性:
        group_id: 消融组标识符
        name: 消融组名称（用于显示和报告）
        description: 描述（中文）
        allowed_modalities: 允许使用的模态集合
        use_cross_modal_attention: 是否使用跨模态注意力（B6为False）
        use_data_driven_calibration: 是否使用数据驱动校准（B7为False）
        enable_dynamic_replan: 是否启用动态重规划（B8为False）
        use_refined_decomposition: 是否使用精细任务分解（B9为False）
        is_baseline_mode: 是否为基线模式（B0为True）
    """
    group_id: AblationGroupID
    name: str
    description: str
    allowed_modalities: Set[ModalityType] = field(default_factory=lambda: {
        ModalityType.TEXT, ModalityType.VOICE,
        ModalityType.GESTURE, ModalityType.ANNOTATION,
    })
    use_cross_modal_attention: bool = True
    use_data_driven_calibration: bool = True
    enable_dynamic_replan: bool = True
    use_refined_decomposition: bool = True
    use_semantic_constraint: bool = True
    use_vlm_grounding: bool = True
    is_baseline_mode: bool = False


def get_all_ablation_configs() -> List[AblationGroupConfig]:
    """
    获取所有消融组的配置列表。

    返回 B0-B9（含B5a/b/c），共12个消融组。
    """
    ALL_MODALITIES = {
        ModalityType.TEXT, ModalityType.VOICE,
        ModalityType.GESTURE, ModalityType.ANNOTATION,
    }

    configs = [
        # B0: Baseline-TextOnly - 仅文本输入
        AblationGroupConfig(
            group_id=AblationGroupID.B0_BASELINE_TEXT_ONLY,
            name="B0: Baseline-TextOnly",
            description="仅文本输入；用于建立最弱对照",
            allowed_modalities={ModalityType.TEXT},
            enable_dynamic_replan=False,  # BaselinePlanner 无 replan 方法
            is_baseline_mode=True,
        ),

        # B1: Enhanced-Full - 完整模型
        AblationGroupConfig(
            group_id=AblationGroupID.B1_ENHANCED_FULL,
            name="B1: Enhanced-Full",
            description="文本+语音+手势+图像标注，全模块开启（参照组）",
            allowed_modalities=ALL_MODALITIES.copy(),
        ),

        # B2: -Voice - 移除语音模态
        AblationGroupConfig(
            group_id=AblationGroupID.B2_NO_VOICE,
            name="B2: -Voice",
            description="移除语音模态，仅保留文本+手势+图像标注",
            allowed_modalities={ModalityType.TEXT, ModalityType.GESTURE, ModalityType.ANNOTATION},
        ),

        # B3: -Gesture - 移除手势模态
        AblationGroupConfig(
            group_id=AblationGroupID.B3_NO_GESTURE,
            name="B3: -Gesture",
            description="移除手势模态，仅保留文本+语音+图像标注",
            allowed_modalities={ModalityType.TEXT, ModalityType.VOICE, ModalityType.ANNOTATION},
        ),

        # B4: -Annotation - 移除图像标注模态
        AblationGroupConfig(
            group_id=AblationGroupID.B4_NO_ANNOTATION,
            name="B4: -Annotation",
            description="移除图像标注模态，仅保留文本+语音+手势",
            allowed_modalities={ModalityType.TEXT, ModalityType.VOICE, ModalityType.GESTURE},
        ),

        # B5a: Text+Voice
        AblationGroupConfig(
            group_id=AblationGroupID.B5A_TEXT_VOICE,
            name="B5a: Text+Voice",
            description="文本+语音单模态组合",
            allowed_modalities={ModalityType.TEXT, ModalityType.VOICE},
        ),

        # B5b: Text+Gesture
        AblationGroupConfig(
            group_id=AblationGroupID.B5B_TEXT_GESTURE,
            name="B5b: Text+Gesture",
            description="文本+手势单模态组合",
            allowed_modalities={ModalityType.TEXT, ModalityType.GESTURE},
        ),

        # B5c: Text+Annotation
        AblationGroupConfig(
            group_id=AblationGroupID.B5C_TEXT_ANNOTATION,
            name="B5c: Text+Annotation",
            description="文本+图像标注单模态组合",
            allowed_modalities={ModalityType.TEXT, ModalityType.ANNOTATION},
        ),

        # B6: -CrossModalAttention - 替换跨模态注意力为简单融合
        AblationGroupConfig(
            group_id=AblationGroupID.B6_NO_CROSS_ATTENTION,
            name="B6: -CrossModalAttention",
            description="保留多模态输入，但将跨模态注意力替换为简单拼接/平均融合",
            allowed_modalities=ALL_MODALITIES.copy(),
            use_cross_modal_attention=False,
        ),

        # B7: -DataDrivenCalibration - 使用固定默认参数
        AblationGroupConfig(
            group_id=AblationGroupID.B7_NO_CALIBRATION,
            name="B7: -DataDrivenCalibration",
            description="使用固定默认参数替换数据驱动校准模块",
            allowed_modalities=ALL_MODALITIES.copy(),
            use_data_driven_calibration=False,
        ),

        # B8: -DynamicReplan - 关闭动态重规划
        AblationGroupConfig(
            group_id=AblationGroupID.B8_NO_REPLAN,
            name="B8: -DynamicReplan",
            description="关闭动态条件触发的重规划功能",
            allowed_modalities=ALL_MODALITIES.copy(),
            enable_dynamic_replan=False,
        ),

        # B9: -RefinedTaskDecomposition - 简化任务分解
        AblationGroupConfig(
            group_id=AblationGroupID.B9_SIMPLE_DECOMPOSE,
            name="B9: -RefinedTaskDecomposition",
            description="使用简化任务分解（仅fly_to + return），验证复杂任务分解贡献",
            allowed_modalities=ALL_MODALITIES.copy(),
            use_refined_decomposition=False,
        ),

        # B10: -SemanticConstraint - 禁用语义转移约束矩阵 T (A3)
        AblationGroupConfig(
            group_id=AblationGroupID.B10_NO_SEMANTIC_CONSTRAINT,
            name="B10: -SemanticConstraint",
            description="禁用 HSATD 语义转移约束矩阵 T 的 logits 掩码，验证约束注入贡献",
            allowed_modalities=ALL_MODALITIES.copy(),
            use_semantic_constraint=False,
        ),

        # B11: -VLMGrounding - 以启发式坐标提取替代 Molmo VLM 定位
        AblationGroupConfig(
            group_id=AblationGroupID.B11_NO_VLM_GROUNDING,
            name="B11: -VLMGrounding",
            description="以启发式文本解析+噪声定位替代 Molmo VLM 坐标提取，量化 VLM grounding 贡献",
            allowed_modalities=ALL_MODALITIES.copy(),
            use_vlm_grounding=False,
        ),
    ]

    return configs


def get_ablation_config_by_id(group_id: AblationGroupID) -> AblationGroupConfig:
    """根据ID获取消融组配置。"""
    for cfg in get_all_ablation_configs():
        if cfg.group_id == group_id:
            return cfg
    raise ValueError(f"Unknown ablation group: {group_id}")
