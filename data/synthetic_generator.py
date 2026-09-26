"""
合成数据生成器模块 - UAV-VLPA多模态数据工厂核心组件。

负责根据实验配置生成完整的多模态场景数据集，
是UAV-VLPA系统中连接数据生成、模型训练和性能评估的关键枢纽。

核心功能：
1. 场景生成：创建完整的多模态场景样本
2. 数据增强：为场景添加语音、手势、标注等模态
3. 真值生成：创建专家路径和原子任务序列
4. 内存管理：优化大规模数据生成的内存使用

生成策略：
- 基于学术文献验证的数据生成方法
- 支持三种复杂度级别（SIMPLE/MEDIUM/COMPLEX）
- 支持四种模态类型（TEXT/VOICE/GESTURE/ANNOTATION）
- 自动分配模态组合，确保数据多样性

使用方式：
    generator = SyntheticGenerator(base_cfg, dataset_cfg)
    samples = generator.generate_all()
"""

# 导入标准库
import os         # 操作系统接口
import random     # 随机数生成
import logging    # 日志记录
import sys        # 系统功能
import traceback  # 异常追踪
import shutil     # 高级文件操作
import time       # 时间处理
import gc         # 垃圾回收（内存管理）
import torch      # PyTorch深度学习框架（内存管理）
from typing import List  # 类型提示

# 导入UAV-VLPA项目模块
from configs.experiment_config import BaseConfig, DatasetConfig              # 实验配置
from data.scenario_schema import (                                         # 场景数据结构
    ComplexityLevel,    # 复杂度级别
    ModalityType,       # 模态类型
    WaypointTarget,     # 航点目标
    ExpertPath,         # 专家路径
    ScenarioSample,     # 场景样本
    AtomicTask,         # 原子任务
)
from data.voice_generator import VoiceInstructionGenerator                 # 语音指令生成器
from data.gesture_generator import GestureTrajectoryGenerator              # 手势轨迹生成器
from data.annotation_generator import ImageAnnotationGenerator             # 图像标注生成器
from data.mixed_generator import MixedModalityGenerator                    # 混合模态生成器
from data.ground_truth_generator import GroundTruthPathGenerator           # 真值路径生成器
from data.complexity_controller import ComplexityController                # 复杂度控制器
from data.dataset_manager import DatasetManager                            # 数据集管理器

# 获取实验日志记录器
logger = logging.getLogger("experiment")

# 目标类型池 - UAV-VLPA地理空间建模的基础
TARGET_TYPES = ["building", "stadium", "school", "hospital", "warehouse", "parking lot", "bridge", "crossroad", "church", "factory"]
# 障碍物类型池 - UAV-VLPA安全约束建模的基础
OBSTACLE_TYPES = ["lake", "river", "pond", "restricted zone", "forest"]

class SyntheticGenerator:
    """
    合成数据生成器类 - UAV-VLPA系统的多模态数据工厂核心组件。

    负责根据实验配置生成完整的多模态场景数据集，
    是UAV-VLPA系统中连接数据生成、模型训练和性能评估的关键枢纽。

    核心设计原则：
    - 系统性：基于学术文献验证的数据生成策略
    - 可扩展性：支持新增目标类型、障碍物类型和任务类型
    - 鲁棒性：完善的错误处理和内存管理机制
    - 效率性：针对大规模数据生成的性能优化

    功能概览：
        1. 场景生成：创建完整的多模态场景样本
        2. 数据增强：为场景添加语音、手势、标注等模态
        3. 真值生成：创建专家路径和原子任务序列
        4. 内存管理：优化大规模数据生成的内存使用

    属性说明：
        base_cfg: 基础配置对象
            - 包含基准数据路径、环境配置等全局设置
        dataset_cfg: 数据集配置对象
            - 包含数据集规模、复杂度分布等具体参数
        voice_gen: 语音指令生成器
            - 生成自然语言语音指令
            - 支持环境噪声模拟
        gesture_gen: 手势轨迹生成器
            - 生成手绘风格的手势轨迹
            - 支持贝塞尔曲线和抖动效果
        annotation_gen: 图像标注生成器
            - 生成箭头、圆圈、区域高亮等视觉标注
            - 支持百分比坐标系统
        mixed_gen: 混合模态生成器
            - 组合不同模态生成器
            - 实现多模态数据融合
        gt_gen: 真值路径生成器
            - 生成专家标注路径
            - 提供监督学习信号
        complexity_ctrl: 复杂度控制器
            - 管理场景复杂度分布
            - 支持课程学习(Curriculum Learning)
    """
    def __init__(self, base_cfg: BaseConfig, dataset_cfg: DatasetConfig):
        """
        初始化合成数据生成器。

        该构造函数配置合成数据生成器的核心参数，
        这些参数直接影响UAV-VLPA系统中多模态数据的质量和多样性。

        参数说明：
            base_cfg: 基础配置对象
                - 包含基准数据路径、环境配置等全局设置
                - 用于真值路径生成和坐标转换
            dataset_cfg: 数据集配置对象
                - 包含数据集规模、复杂度分布等具体参数
                - 控制生成数据的统计特性
        """
        self.base_cfg = base_cfg
        self.dataset_cfg = dataset_cfg
        self.voice_gen = VoiceInstructionGenerator(sample_rate=dataset_cfg.voice_sample_rate, snr_range=dataset_cfg.voice_snr_range)
        self.gesture_gen = GestureTrajectoryGenerator(line_width=dataset_cfg.gesture_line_width, color=dataset_cfg.gesture_color, jitter_sigma=dataset_cfg.gesture_jitter_sigma)
        self.annotation_gen = ImageAnnotationGenerator(arrow_color=dataset_cfg.annotation_arrow_color, obstacle_color=dataset_cfg.annotation_obstacle_color, target_color=dataset_cfg.annotation_target_color)
        self.mixed_gen = MixedModalityGenerator(self.voice_gen, self.gesture_gen, self.annotation_gen)
        self.gt_gen = GroundTruthPathGenerator(benchmark_csv=base_cfg.benchmark_csv, benchmark_images_dir=base_cfg.benchmark_images_dir)
        self.complexity_ctrl = ComplexityController(total=dataset_cfg.total_scenarios, min_per_level=dataset_cfg.min_per_complexity)

    def generate_all(self) -> List[ScenarioSample]:
        """
        生成所有场景 - UAV-VLPA多模态数据工厂核心接口。

        该方法实现了UAV-VLPA系统中关键的多模态数据生成功能，
        根据配置生成完整的多模态场景数据集，是UAV-VLPA多模态融合架构的核心入口点。

        生成流程：
        1. 环境准备：清理输出目录，创建子目录结构
        2. 复杂度分配：根据复杂度控制器分配各级别场景数量
        3. 场景生成：按复杂度级别批量生成场景
        4. 内存管理：优化大规模数据生成的内存使用
        5. 进度监控：实时记录生成进度和状态
        6. 结果验证：确保所有场景生成完成

        核心优化：
        - 梯度禁用：使用torch.no_grad()避免内存泄漏
        - 垃圾回收：定期调用gc.collect()释放内存
        - GPU缓存清理：在CUDA可用时清空GPU缓存
        - 强制落盘：确保数据写入磁盘

        无人机应用考虑：
        - 大规模生成：支持数千个场景的高效生成
        - 内存优化：针对无人机任务的资源约束进行优化
        - 进度追踪：实时监控生成状态
        - 错误处理：完善的异常处理机制

        返回值：
            List[ScenarioSample]: 生成的场景样本列表
                - 包含完整的多模态场景信息
                - 可直接用于模型训练和评估
        """
        output_dir = self.dataset_cfg.output_dir

        # 【启动前清理】确保目录干净，防止文件名冲突
        if os.path.exists(output_dir):
            for sub in ("audio", "gesture", "annotation"):
                sub_path = os.path.join(output_dir, sub)
                if os.path.exists(sub_path):
                    shutil.rmtree(sub_path)

        os.makedirs(output_dir, exist_ok=True)
        for sub in ("audio", "gesture", "annotation"):
            os.makedirs(os.path.join(output_dir, sub), exist_ok=True)

        distribution = self.complexity_ctrl.distribute()
        samples: List[ScenarioSample] = []
        scenario_idx = 0

        # --- 核心优化部分 ---
        for complexity, count in distribution.items():
            for i in range(count):
                scenario_idx += 1
                image_id = ((scenario_idx - 1) % self.dataset_cfg.num_benchmark_images) + 1
                try:
                    # 关键修改 1：关闭梯度计算。这不影响生成精度，但能防止内存随着循环次数增加而无限堆积
                    with torch.no_grad():
                        sample = self._create_scenario(scenario_idx, image_id, complexity, output_dir)

                    samples.append(sample)

                    # 关键修改 2：手动触发垃圾回收。在 16 核环境下，确保 CPU 缓存的过期对象被立即清理
                    if scenario_idx % 5 == 0:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    if scenario_idx % 10 == 0:
                        logger.info("✅ VERIFIED PROGRESS: %d / %d", scenario_idx, self.dataset_cfg.total_scenarios)

                except Exception as e:
                    logger.error("❌ FATAL CRASH at scenario %d", scenario_idx)
                    traceback.print_exc()
                    sys.exit(1)

        # 【强制落盘】
        if sys.platform != "win32":
            os.sync()
        time.sleep(1)

        logger.info("Total scenarios completely generated: %d", len(samples))
        return samples

    def _create_scenario(self, idx: int, image_id: int, complexity: ComplexityLevel, output_dir: str) -> ScenarioSample:
        """
        创建单个场景 - UAV-VLPA多模态数据生成核心算法。

        该方法实现了UAV-VLPA系统中关键的单场景生成功能，
        根据复杂度级别和图像ID创建完整的多模态场景样本，
        是UAV-VLPA多模态融合架构的核心构建单元。

        生成步骤：
        1. 场景标识：生成唯一的场景ID
        2. 图像选择：选择对应的基准卫星图像
        3. 目标生成：根据复杂度级别生成目标位置
        4. 障碍物生成：根据复杂度级别生成障碍物位置
        5. 模态选择：根据复杂度级别选择合适的模态组合
        6. 文本生成：生成自然语言文本指令
        7. 场景构建：创建ScenarioSample对象
        8. 数据增强：为场景添加多模态数据
        9. 真值生成：创建专家路径和原子任务序列

        无人机应用考虑：
        - 目标分布：在图像安全区域内随机分布（10-90%范围）
        - 障碍物处理：支持多种障碍物类型
        - 模态协同：确保不同模态在时空上的一致性
        - 真值对齐：专家路径与文本指令保持语义一致性

        参数说明：
            idx: 场景索引
                - 用于生成唯一场景ID
            image_id: 图像ID
                - 对应基准卫星图像
                - 用于坐标转换和地理映射
            complexity: 复杂度级别
                - 决定场景难度和模态组合
            output_dir: 输出目录
                - 存储生成的多模态数据
                - 支持子目录结构便于组织

        返回值：
            ScenarioSample: 创建的场景样本对象
                - 包含完整的多模态场景信息
                - 可直接用于模型训练和评估
        """
        sid = f"scenario_{idx:04d}"
        image_path = os.path.join(self.base_cfg.benchmark_images_dir, f"{image_id}.jpg")
        tgt_range = self.complexity_ctrl.target_count_range(complexity)
        obs_range = self.complexity_ctrl.obstacle_count_range(complexity)
        targets = self._random_waypoints(random.randint(*tgt_range), TARGET_TYPES, prefix="target")
        obstacles = self._random_waypoints(random.randint(*obs_range), OBSTACLE_TYPES, prefix="obstacle")

        # 遵循文档的随机模态逻辑
        modalities = self.mixed_gen.select_modalities(complexity)
        text = self.voice_gen.generate_instruction_text([t.target_type for t in targets], [o.target_type for o in obstacles], complexity)

        sample = ScenarioSample(
            scenario_id=sid, image_id=image_id, complexity=complexity, modalities=modalities,
            text_instruction=text, targets=targets, obstacles=obstacles, conditions=[], metadata={}
        )
        # 存文件逻辑
        sample = self.mixed_gen.augment_scenario(sample, image_path, output_dir)

        # GT 路径逻辑
        gt_raw = self.gt_gen.generate_expert_paths(image_id, self._targets_to_pct_dict(targets), self._targets_to_pct_dict(obstacles), n_paths=self.dataset_cfg.expert_paths_per_scenario)
        sample.ground_truth_paths = [ExpertPath(waypoints=[], path_coordinates_latlon=[tuple(c) for c in gp.get("path_coordinates_latlon", [])], variant_label=gp.get("variant_label", "optimal")) for gp in gt_raw]

        # 生成专家原子任务序列（ground truth for instruction accuracy）
        sample.expert_atomic_tasks = self._generate_expert_atomic_tasks(targets, obstacles, complexity)
        return sample

    @staticmethod
    def _generate_expert_atomic_tasks(
        targets: List[WaypointTarget],
        obstacles: List[WaypointTarget],
        complexity: ComplexityLevel,
    ) -> List[AtomicTask]:
        """
        生成专家原子任务序列 - UAV-VLPA任务分解核心算法。

        根据场景的目标和障碍物生成专家定义的原子任务序列，
        是UAV-VLPA系统中任务分解和指令准确性评估的基础。

        生成策略：
        - SIMPLE级别：基础任务序列（fly_to + return）
        - MEDIUM级别：增强任务序列（fly_to + inspect + return）
        - COMPLEX级别：完整任务序列（fly_to + inspect + avoid + circle + photograph + return）

        算法原理：
        - 优先级调度：数值越小优先级越高
        - 任务组合：多个原子任务组合成完整任务序列
        - 复杂度适配：根据复杂度级别动态调整任务类型

        无人机应用考虑：
        - 安全约束：avoid任务确保障碍物规避
        - 任务完整性：circle和photograph任务确保任务质量
        - 顺序保证：priority字段确保执行顺序
        - 可扩展性：支持新增任务类型

        参数说明：
            targets: 目标列表
                - 需要访问的目标位置
                - 决定fly_to和inspect任务数量
            obstacles: 障碍物列表
                - 需要规避的危险区域
                - 决定avoid任务数量
            complexity: 复杂度级别
                - 决定是否添加circle、photograph等高级任务

        返回值：
            List[AtomicTask]: 专家原子任务序列
                - 按执行顺序排列
                - 包含所有必要的任务类型
                - 可直接用于任务分解评估
        """
        tasks: List[AtomicTask] = []
        priority = 1

        # 每个目标生成对应的 fly_to + inspect 任务
        for t in targets:
            tasks.append(AtomicTask(
                task_type="fly_to",
                target=t,
                priority=priority,
            ))
            priority += 1
            # 中等/复杂任务为每个目标添加 inspect
            if complexity in (ComplexityLevel.MEDIUM, ComplexityLevel.COMPLEX):
                tasks.append(AtomicTask(
                    task_type="inspect",
                    target=t,
                    priority=priority,
                ))
                priority += 1

        # 每个障碍物生成 avoid 任务
        for o in obstacles:
            tasks.append(AtomicTask(
                task_type="avoid",
                target=o,
                priority=priority,
            ))
            priority += 1

        # 复杂任务添加 circle 和 photograph
        if complexity == ComplexityLevel.COMPLEX and targets:
            tasks.append(AtomicTask(
                task_type="circle",
                target=targets[0],
                priority=priority,
            ))
            priority += 1
            tasks.append(AtomicTask(
                task_type="photograph",
                target=targets[-1],
                priority=priority,
            ))
            priority += 1

        # 最后返回基地
        tasks.append(AtomicTask(
            task_type="return",
            priority=priority,
        ))
        return tasks

    @staticmethod
    def _random_waypoints(count, type_pool, prefix):
        return [WaypointTarget(name=f"{prefix}_{i+1}", target_type=random.choice(type_pool), coordinates_percent=(round(random.uniform(10, 90), 1), round(random.uniform(10, 90), 1))) for i in range(count)]

    @staticmethod
    def _targets_to_pct_dict(waypoints):
        return {wp.name: {"type": wp.target_type, "coordinates": list(wp.coordinates_percent)} for wp in waypoints}