"""
实验配置文件 - 使用数据类（dataclasses）集中管理所有超参数和路径。

该模块定义了无人机视觉-语言-路径规划（UAV-VLPA）实验的所有配置类，
包括基础配置、数据集配置、模型配置、训练配置和评估配置。

设计原则：
    1. 集中管理：所有参数统一在此文件定义，便于修改和复现
    2. 类型安全：使用 Python 类型注解确保参数类型正确
    3. 默认值：提供合理的默认值，简化配置流程
"""

# ==================== 标准库和第三方库导入 ====================
import os      # 操作系统接口，用于路径操作
import torch   # PyTorch深度学习框架，用于检测GPU可用性
from dataclasses import dataclass, field  # 数据类装饰器和字段工厂
from typing import Tuple, List  # 类型注解工具


def _default_project_root() -> str:
    """
    获取默认项目根目录路径。

    在独立项目中，项目根目录是 Experiment 目录本身。
    该函数通过当前文件位置向上回溯一级来确定根目录。

    Returns:
        str: 项目根目录的绝对路径
    """
    # __file__ 获取当前文件的绝对路径
    # os.path.dirname 获取文件所在目录（configs/）
    # os.path.join(..., "..") 向上一级到项目根目录
    # os.path.abspath 转换为绝对路径
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# ==================== 基础配置类 ====================
@dataclass
class BaseConfig:
    """
    基础配置类 - 定义共享路径和硬件设置。

    属性：
        project_root: 项目根目录路径
        device: 计算设备（cuda 或 cpu）
        seed: 随机种子，确保实验可复现
        num_workers: 数据加载的工作进程数
    """

    project_root: str = ""  # 项目根目录，空字符串表示使用默认值
    device: str = "cuda" if torch.cuda.is_available() else "cpu"  # 自动检测GPU
    seed: int = 42  # 随机种子，42是机器学习中常用的默认值
    num_workers: int = 4  # 数据加载的工作进程数

    def __post_init__(self):
        """
        初始化后处理函数。

        如果 project_root 未设置，则使用默认的项目根目录。
        这是 dataclass 的特殊方法，在对象创建后自动调用。
        """
        if not self.project_root:
            self.project_root = _default_project_root()

    # ==================== 路径属性（动态计算） ====================
    @property
    def benchmark_dir(self) -> str:
        """基准数据集目录路径。"""
        return os.path.join(self.project_root, "benchmark-UAV-VLPA-nano-30")

    @property
    def pathplan_dir(self) -> str:
        """路径规划目录（在独立项目中未使用，保留用于兼容性）。"""
        return os.path.join(self.project_root, "pathplan")

    @property
    def tsp_dir(self) -> str:
        """TSP目录（在独立项目中未使用，保留用于兼容性）。"""
        return os.path.join(self.project_root, "tsp_only")

    @property
    def experiment_dir(self) -> str:
        """实验目录路径。在独立模式下，项目根目录就是实验目录。"""
        return self.project_root

    @property
    def benchmark_images_dir(self) -> str:
        """基准图像目录路径。"""
        return os.path.join(self.benchmark_dir, "images")

    @property
    def benchmark_csv(self) -> str:
        """基准数据集CSV文件路径。"""
        return os.path.join(self.benchmark_dir, "parsed_coordinates.csv")


# ==================== 数据集配置类 ====================
@dataclass
class DatasetConfig:
    """
    数据集配置类 - 定义数据生成参数。

    该配置控制合成数据集的生成过程，包括场景数量、复杂度分布、
    各种模态数据的生成参数以及数据集划分比例。
    """

    # ==================== 场景生成参数 ====================
    total_scenarios: int = 150  # 要生成的总场景数
    min_per_complexity: int = 50  # 每种复杂度级别最少生成的场景数
    expert_paths_per_scenario: int = 3  # 每个场景的专家路径数量
    num_benchmark_images: int = 30  # 基准图像数量（用于循环使用）

    # ==================== 语音生成参数 ====================
    voice_snr_range: Tuple[float, float] = (10.0, 30.0)  # 信噪比范围（dB）
    voice_sample_rate: int = 16000  # 音频采样率（Hz），16kHz是语音标准

    # ==================== 手势生成参数 ====================
    gesture_jitter_sigma: float = 3.0  # 手势轨迹抖动标准差（像素）
    gesture_line_width: int = 3  # 手势线条宽度（像素）
    gesture_color: Tuple[int, int, int, int] = (255, 0, 0, 180)  # 手势颜色（RGBA）

    # ==================== 标注生成参数 ====================
    annotation_arrow_color: Tuple[int, int, int] = (0, 200, 0)  # 箭头颜色（RGB）
    annotation_obstacle_color: Tuple[int, int, int, int] = (255, 50, 50, 150)  # 障碍物颜色（RGBA）
    annotation_target_color: Tuple[int, int, int, int] = (50, 200, 50, 150)  # 目标颜色（RGBA）

    # ==================== 输出配置 ====================
    output_dir: str = ""  # 输出目录，空字符串表示使用默认路径

    # ==================== 数据集划分比例 ====================
    train_ratio: float = 0.70  # 训练集比例（70%）
    val_ratio: float = 4 / 30  # 验证集比例（≈13.3%, 3000样本时400个）
    test_ratio: float = 5 / 30  # 测试集比例（≈16.7%, 3000样本时500个）


# ==================== 模型配置类 ====================
@dataclass
class ModelConfig:
    """
    模型配置类 - 定义模型架构超参数。

    该配置包含：
        1. 预训练模型标识符（HuggingFace模型名称）
        2. 多模态融合层的维度参数
        3. 任务分解器的配置
        4. 标注编码器的特征维度
    """

    # ==================== 预训练模型标识符 ====================
    # 这些是从 HuggingFace Hub 加载的预训练模型名称
    vlm_model: str = "cyan2k/molmo-7B-O-bnb-4bit"  # 视觉语言模型（4-bit量化版）
    audio_model: str = "facebook/wav2vec2-base-960h"  # 音频特征提取模型
    gesture_backbone: str = "resnet18"  # 手势图像骨干网络
    text_model: str = "sentence-transformers/all-MiniLM-L6-v2"  # 文本编码模型
    tts_model: str = "microsoft/speecht5_tts"  # 文本转语音模型

    # ==================== 多模态融合层参数 ====================
    fusion_dim: int = 256  # 融合层特征维度（A2实验确认256为最优）
    attention_heads: int = 8  # 注意力头数
    attention_layers: int = 2  # 注意力层数
    ffn_dim: int = 2048  # 前馈网络维度
    dropout: float = 0.1  # Dropout比率（防止过拟合）

    # ==================== 任务分解器参数 ====================
    max_subtasks: int = 10  # 最大子任务数量
    decomposer_layers: int = 2  # 分解器层数

    # ==================== 标注编码器参数 ====================
    annotation_feature_dim: int = 128  # 标注特征维度


# ==================== 训练配置类 ====================
@dataclass
class TrainingConfig:
    """
    训练配置类 - 定义所有训练阶段的超参数。

    该配置包含三个主要训练阶段：
        1. 融合模块训练：训练多模态融合层
        2. VLM微调（LoRA）：使用低秩适配微调视觉语言模型
        3. RL任务分解器（PPO）：使用强化学习训练任务分解器
    """

    # ==================== 融合模块训练参数 ====================
    fusion_lr: float = 1e-4  # 融合模块学习率
    fusion_epochs: int = 20  # 融合模块训练轮数
    fusion_batch_size: int = 16  # 融合模块批次大小
    fusion_weight_decay: float = 1e-2  # 融合模块权重衰减（L2正则化）

    # ==================== VLM微调参数（LoRA） ====================
    vlm_lr: float = 2e-5  # VLM微调学习率（较小，防止破坏预训练知识）
    vlm_epochs: int = 5  # VLM微调轮数
    vlm_batch_size: int = 1  # VLM批次大小（受显存限制）
    vlm_gradient_accumulation: int = 16  # 梯度累积步数（模拟大batch）
    # LoRA（低秩适配）参数
    lora_rank: int = 16  # LoRA秩（低秩矩阵的维度）
    lora_alpha: int = 32  # LoRA缩放参数
    lora_dropout: float = 0.05  # LoRA Dropout比率
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]  # 目标模块：查询和值投影层
    )

    # ==================== RL任务分解器参数（PPO） ====================
    rl_lr: float = 3e-4  # 强化学习学习率
    rl_episodes: int = 1500  # 训练回合数
    rl_gamma: float = 0.99  # 折扣因子（未来奖励的衰减率）
    rl_clip_eps: float = 0.2  # PPO裁剪参数（防止策略更新过大）
    rl_ppo_epochs: int = 4    # 每条轨迹的PPO更新次数（>1使clip_eps生效）
    rl_entropy_coeff: float = 0.2  # 熵正则化系数（鼓励探索，防止策略坍塌）
    rl_value_coeff: float = 0.5  # 价值函数损失系数
    rl_max_steps_per_episode: int = 20  # 每回合最大步数

    # ==================== 通用训练参数 ====================
    early_stopping_patience: int = 5  # 早停耐心值（验证集不改善的轮数）
    checkpoint_dir: str = ""  # 检查点保存目录
    log_interval: int = 10  # 日志记录间隔（步数）


# ==================== 评估配置类 ====================
@dataclass
class EvalConfig:
    """
    评估配置类 - 定义评估参数。

    该配置控制模型评估过程的行为，包括：
        - 目标访问判定阈值
        - 障碍物避让距离
        - 统计检验参数
        - 结果输出设置
    """

    target_visit_threshold_m: float = 50.0  # 目标访问判定阈值（米）
    obstacle_avoid_distance_m: float = 30.0  # 障碍物避让距离（米）
    t_test_alpha: float = 0.05  # t检验显著性水平（α=0.05对应95%置信度）
    confidence_level: float = 0.95  # 置信水平
    results_dir: str = ""  # 结果输出目录
    generate_latex: bool = True  # 是否生成LaTeX格式的结果表格


# ==================== 检查点目录解析 ====================
def _has_checkpoint_files(dir_path: str) -> bool:
    """
    判断目录内是否存在训练权重文件（.pt）。

    Args:
        dir_path: 待检查的目录路径

    Returns:
        bool: 目录存在且含至少一个 .pt 文件时为 True
    """
    if not os.path.isdir(dir_path):
        return False
    return any(name.endswith(".pt") for name in os.listdir(dir_path))


def resolve_checkpoint_dir(explicit: str = "", project_root: str = "") -> str:
    """
    解析评估/补充实验脚本使用的检查点目录。

    服务器目录结构中默认 checkpoints 目录可能不存在，而 location-split
    权重保存在 checkpointsLocation 中。解析优先级：
        1. 显式传入的目录（命令行 --checkpoint-dir）
        2. 默认 checkpoints 目录（仅当其内含 .pt 权重时）
        3. checkpointsLocation 目录（location-split 权重，目录存在即使用）
        4. 回退到默认 checkpoints 目录（由调用方给出缺失告警）

    Args:
        explicit: 命令行显式指定的检查点目录，空字符串表示未指定
        project_root: 项目根目录，空字符串表示使用默认值

    Returns:
        str: 解析后的检查点目录绝对/相对路径（与传入形式一致）
    """
    if explicit:
        return explicit
    root = project_root or _default_project_root()
    default_dir = os.path.join(root, "checkpoints")
    location_dir = os.path.join(root, "checkpointsLocation")
    if _has_checkpoint_files(default_dir):
        return default_dir
    if os.path.isdir(location_dir):
        return location_dir
    return default_dir


# ==================== 默认配置获取函数 ====================
def get_default_config() -> (
    Tuple[BaseConfig, DatasetConfig, ModelConfig, TrainingConfig, EvalConfig]
):
    """
    返回完整的默认配置集合。

    该函数创建所有配置类的实例，并设置默认的输出路径。
    返回的元组包含五个配置对象，按以下顺序：
        1. BaseConfig: 基础配置
        2. DatasetConfig: 数据集配置
        3. ModelConfig: 模型配置
        4. TrainingConfig: 训练配置
        5. EvalConfig: 评估配置

    Returns:
        Tuple: 包含五个配置对象的元组
    """
    # 创建基础配置（会自动设置项目根目录）
    base = BaseConfig()

    # 创建数据集配置，设置默认输出目录
    dataset = DatasetConfig(
        output_dir=os.path.join(base.experiment_dir, "data", "generated"),
    )

    # 创建模型配置（使用默认值）
    model = ModelConfig()

    # 创建训练配置，设置默认检查点目录
    training = TrainingConfig(
        checkpoint_dir=os.path.join(base.experiment_dir, "checkpoints"),
    )

    # 创建评估配置，设置默认结果目录
    evaluation = EvalConfig(
        results_dir=os.path.join(base.experiment_dir, "results"),
    )

    # 返回所有配置对象的元组
    return base, dataset, model, training, evaluation