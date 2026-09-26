"""
数据模块初始化文件 - UAV-VLPA系统的多模态数据接口层。

该文件定义了data包的公共API接口，通过重新导出核心数据结构，
为UAV-VLPA系统中其他模块提供统一、简洁的数据访问方式。

核心功能：
- 接口抽象：隐藏内部实现细节，提供稳定的公共API
- 依赖管理：控制外部模块对data包的依赖关系
- 版本兼容：支持向后兼容的API演进
- 可扩展性：便于添加新的数据结构而不破坏现有代码

UAV-VLPA系统集成：
- 与模型模块协同工作，提供标准化的数据结构
- 为训练模块提供统一的数据输入接口
- 为评估模块提供标准化的指标计算接口
- 支持实时推理和离线评估的统一数据访问
"""

from .scenario_schema import (
    ModalityType,
    ComplexityLevel,
    WaypointTarget,
    AtomicTask,
    ExpertPath,
    ScenarioSample,
    PlanResult,
    MetricsResult,
)
