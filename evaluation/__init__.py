"""
评估模块初始化文件 - UAV-VLPA系统的多模态评估接口层。

该文件定义了evaluation包的公共API接口，通过重新导出核心评估函数和类，
为UAV-VLPA系统中其他模块提供统一、简洁的评估访问方式。

核心功能：
- 接口抽象：隐藏内部实现细节，提供稳定的公共API
- 依赖管理：控制外部模块对evaluation包的依赖关系
- 版本兼容：支持向后兼容的API演进
- 可扩展性：便于添加新的评估指标而不破坏现有代码

UAV-VLPA系统集成：
- 与训练模块协同工作，提供标准化的模型评估接口
- 为模型选择提供公平的比较基准
- 支持实时评估和离线评估的统一接口
- 为实验报告生成提供数据基础
"""

# 导入核心评估模块
from .metrics import MetricsCalculator
from .task_completion import TaskCompletionEvaluator
from .instruction_accuracy import InstructionAccuracyEvaluator
from .instruction_accuracy_fine import FineGrainedIAEvaluator
from .traj_calc import euclidean_distance_km
from .fair_comparison import generate_fair_comparison_report
from .report_generator import ReportGenerator

# 评估模块的公共API
__all__ = [
    'MetricsCalculator',
    'TaskCompletionEvaluator',
    'InstructionAccuracyEvaluator',
    'FineGrainedIAEvaluator',
    'euclidean_distance_km',
    'generate_fair_comparison_report',
    'ReportGenerator',
]