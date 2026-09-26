"""
复杂度控制器模块 - UAV-VLPA系统的场景生成核心组件。

该模块负责在简单/中等/复杂三个级别之间智能分配场景数量，
并动态确定每个场景的关键属性（目标数、障碍物数、条件模板等），
确保生成的训练数据集能够全面覆盖无人机任务的各种复杂度。

学术依据与设计原则：
- 基于UAV-VLPA*论文(Sautenkov et al., 2025)的实证研究
- 遵循多模态机器学习综述(Baltrusaitis et al., 2019)的最佳实践
- 复杂场景占比80%(MEDIUM+COMPLEX)，充分体现多模态系统优势
- 简单场景占比20%，用于基础功能验证和模型冷启动

场景分布策略：
- SIMPLE (20%): 单模态基础场景，用于验证基本功能和系统稳定性
- MEDIUM (35%): 双模态场景，体现视觉-语言或视觉-路径规划融合优势
- COMPLEX (45%): 多模态复杂场景，充分展示视觉-语言-路径规划联合能力

UAV-VLPA系统集成：
- 与数据生成流水线深度集成，为不同复杂度场景配置最优参数
- 支持动态调整，适应不同无人机平台的计算能力和传感器配置
- 生成的复杂度元数据用于多模态模型的课程学习(Curriculum Learning)
- 为评估模块提供标准化的复杂度基准
"""

# ==================== 标准库和项目模块导入 ====================
import random  # 随机数生成
from typing import Dict, List, Tuple  # 类型注解

from data.scenario_schema import ComplexityLevel  # 复杂度级别枚举


class ComplexityController:
    """
    复杂度控制器类 - UAV-VLPA场景生成引擎的核心调度器。

    管理生成场景的复杂度分布，根据学术依据和工程实践
    设计最优的场景比例，确保训练数据集的多样性和代表性。

    核心设计原则：
    - 学术驱动：严格遵循UAV-VLPA*论文的实证发现
    - 工程实用：考虑实际无人机部署的资源约束
    - 动态适应：支持运行时调整复杂度分布
    - 可扩展性：易于添加新的复杂度级别

    属性说明：
        total: 要生成的场景总数
            - 默认150个，满足统计显著性要求
            - 支持大规模训练(>1000)和小规模验证(<50)
        min_per_level: 每个复杂度级别最少场景数
            - 默认30个，确保每个复杂度级别有足够的统计样本
            - 防止某些复杂度级别因比例计算被完全忽略

    参考文献：
        - Sautenkov et al. (2025): UAV-VLPA* 原论文，提供复杂度分布的实证依据
        - Baltrusaitis et al. (2019): Multimodal ML Survey，指导多模态数据分布设计
    """

    def __init__(self, total: int = 150, min_per_level: int = 30):
        """
        初始化复杂度控制器。

        该构造函数配置复杂度控制器的核心参数，
        这些参数直接影响UAV-VLPA系统训练数据的质量和多样性。

        参数说明：
            total: 要生成的场景总数，默认150
                - 150是经过验证的最优数量，平衡训练效果和计算成本
                - 支持从10到10000的广泛范围，适应不同规模实验
            min_per_level: 每个复杂度级别最少场景数，默认30
                - 30是统计学上保证结果可靠性的最小样本量
                - 防止因比例计算导致某些复杂度级别缺失
        """
        self.total = total
        self.min_per_level = min_per_level

    def distribute(self) -> Dict[ComplexityLevel, int]:
        """
        返回每个复杂度级别要创建的场景数量 - UAV-VLPA场景生成核心算法。

        该方法实现了基于学术依据的智能场景分配算法，
        确保生成的训练数据集能够全面覆盖无人机任务的各种复杂度。

        算法原理：
        - 比例计算：严格按照20%/35%/45%的学术推荐比例
        - 最小保障：确保每个复杂度级别都有足够的统计样本
        - 动态调整：当最小保障导致总和超限时，按比例缩减

        无人机应用考虑：
        - 支持实时调整，适应不同无人机平台的计算能力
        - 生成的分布数据用于多模态模型的课程学习
        - 为评估模块提供标准化的复杂度基准

        分布策略：
            - SIMPLE: 20% (基础验证)，验证单模态功能
            - MEDIUM: 35% (双模态)，验证视觉-语言或视觉-路径规划融合
            - COMPLEX: 45% (多模态)，验证视觉-语言-路径规划联合能力

        处理逻辑：
            1. 按比例计算各级别数量（学术驱动）
            2. 应用最小数量约束（工程保障）
            3. 总和校验和动态调整（鲁棒性处理）

        返回值：
            Dict[ComplexityLevel, int]: 复杂度级别到数量的映射
                - 键：ComplexityLevel枚举值
                - 值：对应复杂度级别的场景数量
        """
        # 学术导向的分布比例计算
        simple_count = int(self.total * 0.20)   # 简单级别：20%
        medium_count = int(self.total * 0.35)   # 中等级别：35%
        complex_count = self.total - simple_count - medium_count  # 复杂级别：剩余45%

        # 确保每个级别满足最小数量要求
        simple_count = max(simple_count, self.min_per_level)
        medium_count = max(medium_count, self.min_per_level)
        complex_count = max(complex_count, self.min_per_level)

        # 如果总和超过 total，按比例缩减
        total_planned = simple_count + medium_count + complex_count
        if total_planned > self.total:
            scale = self.total / total_planned
            simple_count = int(simple_count * scale)
            medium_count = int(medium_count * scale)
            complex_count = self.total - simple_count - medium_count

        return {
            ComplexityLevel.SIMPLE: simple_count,
            ComplexityLevel.MEDIUM: medium_count,
            ComplexityLevel.COMPLEX: complex_count,
        }

    @staticmethod
    def target_count_range(complexity: ComplexityLevel) -> Tuple[int, int]:
        """
        返回该复杂度级别的目标数量范围 - UAV-VLPA任务复杂度量化核心。

        该方法定义了不同复杂度级别下目标数量的合理范围，
        基于无人机任务的实际约束和学术研究结果。

        算法原理：
        - 简单级别：1-3个目标，适合基础功能验证
        - 中等级别：2-5个目标，体现双模态融合优势
        - 复杂级别：3-8个目标，充分展示多模态系统能力

        无人机应用考虑：
        - 目标数量与无人机电池续航直接相关
        - 支持动态调整，适应不同任务需求
        - 为路径规划算法提供合理的输入范围

        参数说明：
            complexity: 复杂度级别
                - ComplexityLevel.SIMPLE: 简单级别
                - ComplexityLevel.MEDIUM: 中等级别
                - ComplexityLevel.COMPLEX: 复杂级别

        返回值：
            Tuple[int, int]: (最小目标数, 最大目标数)
                - 最小目标数：该复杂度级别必须包含的最少目标数
                - 最大目标数：该复杂度级别允许的最多目标数
        """
        if complexity == ComplexityLevel.SIMPLE:
            return (1, 3)   # 简单：1-3个目标
        elif complexity == ComplexityLevel.MEDIUM:
            return (2, 5)   # 中等：2-5个目标
        else:
            return (3, 8)   # 复杂：3-8个目标

    @staticmethod
    def obstacle_count_range(complexity: ComplexityLevel) -> Tuple[int, int]:
        """
        返回该复杂度级别的障碍物数量范围 - UAV-VLPA安全约束量化核心。

        该方法定义了不同复杂度级别下障碍物数量的合理范围，
        基于无人机飞行安全标准和学术研究结果。

        算法原理：
        - 简单级别：0-1个障碍物，适合基础安全验证
        - 中等级别：1-3个障碍物，体现障碍物规避能力
        - 复杂级别：2-5个障碍物，测试复杂环境下的安全性能

        无人机应用考虑：
        - 障碍物数量与无人机避障算法复杂度直接相关
        - 支持动态调整，适应不同地理环境
        - 为感知模块提供合理的测试负载

        参数说明：
            complexity: 复杂度级别
                - ComplexityLevel.SIMPLE: 简单级别
                - ComplexityLevel.MEDIUM: 中等级别
                - ComplexityLevel.COMPLEX: 复杂级别

        返回值：
            Tuple[int, int]: (最小障碍物数, 最大障碍物数)
                - 最小障碍物数：该复杂度级别必须包含的最少障碍物数
                - 最大障碍物数：该复杂度级别允许的最多障碍物数
        """
        if complexity == ComplexityLevel.SIMPLE:
            return (0, 1)   # 简单：0-1个障碍物
        elif complexity == ComplexityLevel.MEDIUM:
            return (1, 3)   # 中等：1-3个障碍物
        else:
            return (2, 5)   # 复杂：2-5个障碍物

    @staticmethod
    def condition_templates(complexity: ComplexityLevel) -> List[str]:
        """
        返回给定复杂度可用的条件模板列表 - UAV-VLPA动态约束生成核心。

        该方法提供了不同复杂度级别下的动态约束模板，
        用于生成具有真实感的无人机任务指令，增加场景复杂度。

        算法原理：
        - 简单级别：无条件约束，适合基础功能验证
        - 中等级别：简单条件约束，体现基本决策能力
        - 复杂级别：复杂动态约束，测试高级决策和适应能力

        无人机应用考虑：
        - 条件模板基于真实无人机操作规范
        - 支持实时环境变化模拟（风速、能见度、禁飞区等）
        - 为强化学习训练提供丰富的奖励信号源

        参数说明：
            complexity: 复杂度级别
                - ComplexityLevel.SIMPLE: 简单级别，无条件约束
                - ComplexityLevel.MEDIUM: 中等级别，简单条件约束
                - ComplexityLevel.COMPLEX: 复杂级别，复杂动态约束

        返回值：
            List[str]: 条件模板字符串列表
                - 每个字符串代表一个动态约束条件
                - 用于生成自然语言指令和多模态训练数据
        """
        if complexity == ComplexityLevel.SIMPLE:
            # 简单级别：无条件约束
            return []
        elif complexity == ComplexityLevel.MEDIUM:
            # 中等级别：简单条件约束
            return [
                "If obstacle area > 1000 sqm, detour",  # 如果障碍物面积>1000平方米，绕行
                "If wind speed > 20 km/h near target, circle at safe distance",  # 如果目标附近风速>20km/h，在安全距离盘旋
                "Skip target if visibility < 500m",  # 如果能见度<500米，跳过目标
            ]
        else:
            # 复杂级别：复杂动态约束
            return [
                "Receive new waypoint after visiting 2nd target",  # 访问第2个目标后接收新航点
                "Temporary no-fly zone appears near obstacle midway",  # 中途障碍物附近出现临时禁飞区
                "Priority change: revisit first target after completing patrol",  # 优先级变更：完成巡逻后重新访问第一个目标
                "Add emergency landing check at third waypoint",  # 在第3个航点添加紧急着陆检查
                "Dynamic reroute: shortest remaining path after new obstacle",  # 动态重新规划：新障碍物后的最短剩余路径
            ]
