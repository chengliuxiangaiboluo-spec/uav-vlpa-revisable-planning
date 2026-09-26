"""
语音指令生成器模块 - UAV-VLPA系统的多模态语音交互核心组件。

增强版本：添加了内存管理、no_grad推理和写入验证功能，
专门针对无人机任务的语音交互需求进行优化。

该模块使用 Microsoft SpeechT5 模型将文本指令合成为语音，
支持添加噪声模拟真实环境，并提供回退机制确保稳定性，
是UAV-VLPA多模态融合架构中语音模态的关键组成部分。

核心功能：
- 文本到语音合成：使用SpeechT5模型生成高质量语音
- 噪声模拟：添加高斯噪声模拟真实环境干扰
- 内存管理：优化大规模语音生成的内存使用
- 回退机制：在TTS模型不可用时使用正弦波生成
- 写入验证：确保音频文件正确写入磁盘

UAV-VLPA系统集成：
- 与数据生成流水线深度集成，为不同复杂度场景生成语音指令
- 为多模态融合模型提供语音模态训练数据
- 支持实时语音交互和离线评估
- 为评估模块提供标准化的语音指令接口
"""

# ==================== 标准库和第三方库导入 ====================
import os       # 操作系统接口
import random   # 随机数生成
import logging  # 日志记录
import gc       # 垃圾回收
import torch    # PyTorch深度学习框架
import numpy as np  # NumPy数值计算
from typing import List, Tuple, Optional  # 类型注解

# ==================== 离线环境配置 ====================
# 必须在导入transformers之前配置离线环境
from utils.offline_config import setup_offline_environment
setup_offline_environment()

# 导入HuggingFace客户端补丁
from utils.hf_client_patch import patch_huggingface_client, safe_model_load
patch_huggingface_client()

# 尝试导入soundfile用于音频写入
try:
    import soundfile as sf
except ImportError:
    sf = None  # 如果未安装，使用numpy保存为.npy格式

# 导入Transformers相关组件
from transformers import pipeline, SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor
from data.scenario_schema import ComplexityLevel  # 复杂度级别枚举

# 获取实验日志记录器
logger = logging.getLogger("experiment")

# ==================== 指令模板库 ====================
# 根据复杂度级别定义不同的指令模板

# 简单级别模板：单目标、基础指令
SIMPLE_TEMPLATES = [
    "Fly around all {target} and return to base.",  # 环绕所有目标并返回基地
    "Inspect every {target} in the area and land at the take-off point.",  # 检查区域内每个目标并在起飞点降落
    "Visit all {target} at 100 meters altitude, then return home.",  # 在100米高度访问所有目标然后回家
    "Circle each {target} once and come back.",  # 每个目标盘旋一次然后返回
    "Survey the {target} from above and return to launch.",  # 从上方勘察目标并返回发射点
]

# 中等级别模板：双模态、条件逻辑
MEDIUM_TEMPLATES = [
    "Inspect {target}, avoid {obstacle}. If the area near {target} is clear, circle it twice.",  # 检查目标，避开障碍物。如果目标附近区域清晰，盘旋两次
    "Fly to all {target} but stay at least 50 meters from any {obstacle}.",  # 飞向所有目标但保持至少50米距离远离任何障碍物
    "Visit each {target} in order. If you detect {obstacle}, reroute around it.",  # 按顺序访问每个目标。如果检测到障碍物，绕行
]

# 复杂级别模板：多阶段、动态约束
COMPLEX_TEMPLATES = [
    "Phase 1: Fly to each {target} and photograph it. Phase 2: Reroute around newly reported {obstacle}. Phase 3: Return to base via the safest path.",  # 阶段1：飞向每个目标并拍照。阶段2：绕行新报告的障碍物。阶段3：通过最安全路径返回基地
    "Inspect {target} in priority order. Dynamically avoid {obstacle}. If a new no-fly zone is declared, adjust the remaining route.",  # 按优先级顺序检查目标。动态避开障碍物。如果宣布新的禁飞区，调整剩余路线
]

class VoiceInstructionGenerator:
    """
    语音指令生成器类 - UAV-VLPA多模态语音交互核心调度器。

    使用Microsoft SpeechT5模型将文本指令合成为语音，支持添加噪声模拟真实环境，
    提供回退机制，在TTS模型加载失败时使用正弦波生成音频，
    是UAV-VLPA系统中连接文本指令和语音模态的关键枢纽。

    核心设计原则：
    - 可靠性：完善的回退机制确保系统稳定性
    - 实时性：优化的内存管理和推理性能
    - 真实感：噪声模拟增强语音交互的真实感
    - 兼容性：支持多种音频格式和硬件配置

    功能概览：
        1. TTS合成：高质量文本到语音转换
        2. 噪声添加：模拟真实环境干扰
        3. 内存管理：优化大规模语音生成
        4. 写入验证：确保音频文件完整性
        5. 回退机制：保障系统可靠性

    属性说明：
        sample_rate: 音频采样率（Hz）
            - 默认16000Hz，平衡音质和文件大小
            - 支持不同采样率适配
        snr_range: 信噪比范围（dB）
            - 默认(10.0, 30.0)dB，模拟不同环境噪声水平
            - 控制噪声强度，影响语音清晰度
        _tts_pipeline: TTS管道对象（延迟加载）
            - 延迟加载提高启动速度
            - 支持离线环境
        _speaker_embeddings: 说话人嵌入向量（SpeechT5必需）
            - 默认零向量，支持自定义说话人
            - 影响语音音色和语调
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        snr_range: Tuple[float, float] = (10.0, 30.0),
    ):
        """
        初始化语音指令生成器。

        该构造函数配置语音指令生成器的核心参数，
        这些参数直接影响UAV-VLPA系统中语音模态的质量和可靠性。

        参数说明：
            sample_rate: 音频采样率，默认16000Hz
                - 16kHz是语音识别的标准采样率
                - 平衡音质和计算资源消耗
            snr_range: 信噪比范围（dB），默认(10.0, 30.0)
                - 10dB：强噪声环境（如风噪、引擎声）
                - 30dB：弱噪声环境（室内安静环境）
                - 随机选择模拟真实环境变化
        """
        self.sample_rate = sample_rate
        self.snr_range = snr_range
        self._tts_pipeline = None  # 延迟加载TTS管道
        # 预加载一个默认的 speaker embedding (SpeechT5 必需)
        # 使用零向量作为默认说话人
        self._speaker_embeddings = torch.zeros((1, 512))

    def _load_tts(self):
        """
        加载TTS模型和组件 - UAV-VLPA语音模态初始化核心算法。

        完全手动加载所有组件，彻底杜绝离线环境下的'猜测'或联网报错，
        使用绝对路径加载本地缓存的模型权重，
        是UAV-VLPA系统中语音模态可靠性的关键保障。

        算法原理：
        - 绝对路径加载：避免相对路径问题
        - 备用路径尝试：提高加载成功率
        - 安全加载：支持重试机制
        - CPU强制：确保显存不溢出

        无人机应用考虑：
        - 离线支持：专为无人机离线环境设计
        - 内存优化：CPU推理降低资源消耗
        - 故障恢复：完善的回退机制
        - 稳定性：防止模型加载失败导致系统崩溃

        如果加载失败，将回退到正弦波生成模式，
        确保系统在任何环境下都能正常运行。
        """
        # 如果已经加载过，直接返回
        if self._tts_pipeline is not None:
            return

        try:
            # Public release: use an explicit local cache or this checkout's
            # non-versioned Weights directory.
            absolute_weights_dir = os.environ.get(
                "UAV_VLPA_WEIGHTS_DIR",
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Weights"),
            )
            local_model_path = os.path.join(absolute_weights_dir, "speecht5_tts")

            # 尝试备用路径（拼写变体）
            if not os.path.exists(local_model_path):
                local_model_path = os.path.join(absolute_weights_dir, "speech5_tts")

            # 如果路径不存在，抛出错误
            if not os.path.exists(local_model_path):
                raise FileNotFoundError(f"CRITICAL: Model path not found: {local_model_path}")

            logger.info(f"[VoiceGen] Force loading TTS components from: {local_model_path}")

            def _load_tts_pipeline():
                """内部函数：加载TTS管道组件。"""
                # 加载处理器、模型和声码器
                processor = SpeechT5Processor.from_pretrained(local_model_path)
                model = SpeechT5ForTextToSpeech.from_pretrained(local_model_path)
                vocoder = SpeechT5HifiGan.from_pretrained(local_model_path)

                # 创建文本到音频管道
                return pipeline(
                    "text-to-audio",
                    model=model,
                    vocoder=vocoder,
                    tokenizer=processor.tokenizer,
                    feature_extractor=processor.feature_extractor,
                    device=-1  # 强制使用CPU确保显存不溢出
                )

            # 使用安全加载函数，支持重试
            self._tts_pipeline = safe_model_load(_load_tts_pipeline, max_retries=2)
            logger.info("✅ SUCCESS: TTS pipeline fully initialized.")

        except Exception as exc:
            # 加载失败，使用回退模式
            logger.error(f"❌ TTS absolute load failed: {exc}. Falling back to sine-wave.")
            self._tts_pipeline = "fallback"

    def generate(self, text: str, output_path: str, add_noise: bool = True) -> str:
        """
        合成语音并保存为WAV文件 - UAV-VLPA语音模态生成核心接口。

        该方法实现了UAV-VLPA系统中关键的语音生成功能，
        将文本指令转换为可执行的语音指令，是UAV-VLPA多模态融合架构的核心入口点。

        生成流程：
        1. 模型加载：确保TTS模型已加载
        2. 音频生成：使用TTS或回退方法生成音频
        3. 噪声添加：模拟真实环境干扰
        4. 归一化处理：防止音频削波
        5. 文件写入：保存为WAV格式
        6. 写入验证：确保文件完整性
        7. 内存清理：释放临时资源

        核心优化：
        - torch.no_grad()：显著降低CPU负载
        - gc.collect()：及时释放内存
        - 文件验证：确保音频文件正确写入
        - 错误处理：优雅处理各种异常情况

        无人机应用考虑：
        - 环境模拟：噪声添加增强真实感
        - 资源优化：CPU推理适应无人机硬件限制
        - 可靠性：完善的错误处理机制
        - 实时性能：优化的音频处理算法

        参数说明：
            text: 要合成的文本
                - 自然语言文本指令
                - 来自数据生成流水线
            output_path: 输出音频文件路径
                - WAV格式，标准音频格式
                - 支持子目录结构便于组织
            add_noise: 是否添加噪声，默认True
                - 模拟真实环境中的噪声干扰
                - 提高模型鲁棒性

        返回值：
            str: 输出文件路径
                - 用于后续的语音模态处理
                - 可直接用于模型训练和评估

        异常：
            RuntimeError: 如果文件写入失败
                - 确保音频文件正确写入磁盘
                - 防止数据丢失
        """
        # 确保TTS模型已加载
        self._load_tts()
        # 创建输出目录
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        audio = None
        sr = self.sample_rate

        # 使用TTS模型生成音频（如果不是回退模式）
        if self._tts_pipeline and self._tts_pipeline != "fallback":
            try:
                # 使用 torch.no_grad() 显著降低CPU负载并防止内存挂起
                with torch.no_grad():
                    result = self._tts_pipeline(
                        text,
                        forward_params={"speaker_embeddings": self._speaker_embeddings}
                    )
                # 提取音频数组和采样率
                audio = np.array(result["audio"], dtype=np.float32)
                sr = result.get("sampling_rate", self.sample_rate)
            except Exception as exc:
                # TTS运行时错误，使用回退方法
                logger.warning(f"TTS synthesis runtime error: {exc}")
                audio = self._synthesize_fallback(text)
        else:
            # 回退模式：使用正弦波
            audio = self._synthesize_fallback(text)

        # 添加噪声（如果启用）
        if add_noise:
            snr = random.uniform(*self.snr_range)  # 随机选择信噪比
            audio = self._add_noise(audio, snr)

        # 音频归一化：将峰值调整到0.9，防止削波
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / peak * 0.9

        # 写入音频文件
        if sf is not None:
            sf.write(output_path, audio, sr)  # 使用soundfile写入WAV
        else:
            np.save(output_path.replace(".wav", ".npy"), audio)  # 回退到numpy格式

        # 【核心校验】：确保文件真的写进去了，否则直接报错
        if not os.path.exists(output_path):
            raise RuntimeError(f"CRITICAL WRITE ERROR: Failed to save {output_path}")

        # 【核心回收】：每生成一条音频，强制清理一次内存
        del audio
        gc.collect()

        return output_path

    def _synthesize_fallback(self, text: str) -> np.ndarray:
        """
        回退合成方法：生成正弦波音频 - UAV-VLPA系统可靠性保障核心算法。

        当TTS模型不可用时使用，生成频率调制的正弦波，
        模拟语音的基本特征（有变化但无实际语义），
        是UAV-VLPA系统中确保语音模态可用性的关键保障。

        算法原理：
        - 频率调制：基础220Hz，随时间变化±100Hz
        - 时长计算：每个字符约0.06秒
        - 正弦波生成：模拟语音的基本周期性

        无人机应用考虑：
        - 离线支持：无需网络连接
        - 资源效率：极低的CPU和内存消耗
        - 快速响应：毫秒级生成速度
        - 系统稳定：防止TTS故障导致系统崩溃

        参数说明：
            text: 输入文本（仅用于确定音频长度）
                - 不解析文本内容
                - 仅用于计算音频时长

        返回值：
            np.ndarray: 音频数组
                - float32类型，标准音频格式
                - 可直接写入WAV文件
        """
        # 根据文本长度确定音频时长（每个字符约0.06秒）
        duration = max(1.0, len(text) * 0.06)
        # 生成时间轴
        t = np.linspace(0, duration, int(self.sample_rate * duration), dtype=np.float32)
        # 频率调制：基础220Hz，随时间变化±100Hz
        freq = 220 + 100 * np.sin(2 * np.pi * 0.5 * t)
        # 生成调频正弦波
        audio = 0.3 * np.sin(2 * np.pi * freq * t)
        return audio.astype(np.float32)

    def _add_noise(self, audio: np.ndarray, snr_db: float) -> np.ndarray:
        """
        向音频添加高斯噪声 - UAV-VLPA环境模拟核心算法。

        根据指定的信噪比（SNR）计算噪声功率，生成相应的高斯噪声，
        模拟无人机在真实环境中可能遇到的各种噪声干扰，
        是UAV-VLPA系统中提升语音模态鲁棒性的关键步骤。

        算法原理：
        - 信噪比计算：signal_power / noise_power = 10^(snr_db/10)
        - 高斯噪声：符合真实环境噪声分布
        - 功率控制：精确控制噪声强度

        无人机应用考虑：
        - 环境模拟：风噪、引擎声、电磁干扰等
        - 鲁棒性：提高语音识别准确率
        - 真实感：增强语音交互的真实体验
        - 可调性：SNR范围可配置

        参数说明：
            audio: 原始音频数组
                - float32类型，标准音频格式
                - 来自TTS合成或回退生成
            snr_db: 信噪比（dB）
                - 10dB：强噪声环境（如高速飞行）
                - 30dB：弱噪声环境（如低空悬停）
                - 随机选择模拟环境变化

        返回值：
            np.ndarray: 添加噪声后的音频
                - 与原始音频同尺寸
                - 可直接用于模型训练和评估
        """
        # 计算信号功率（均方值）
        signal_power = np.mean(audio ** 2)
        # 根据SNR计算噪声功率
        noise_power = signal_power / (10 ** (snr_db / 10))
        # 生成高斯噪声
        noise = np.random.normal(0, np.sqrt(noise_power), len(audio)).astype(np.float32)
        # 噪声叠加
        return audio + noise

    @staticmethod
    def generate_instruction_text(target_types, obstacle_types, complexity) -> str:
        """
        根据目标、障碍物和复杂度生成指令文本 - UAV-VLPA多模态指令生成核心算法。

        从预定义的模板库中选择合适的模板，并填充目标类型和障碍物类型，
        生成符合无人机任务需求的自然语言指令，
        是UAV-VLPA系统中连接视觉模态和语言模态的关键桥梁。

        生成策略：
        - SIMPLE级别：单目标、基础指令
        - MEDIUM级别：双模态、条件逻辑
        - COMPLEX级别：多阶段、动态约束

        算法原理：
        - 模板选择：随机选择同类模板
        - 字符串格式化：安全的目标和障碍物类型填充
        - 复杂度适配：根据复杂度级别选择模板

        无人机应用考虑：
        - 任务相关：指令内容与无人机任务高度相关
        - 环境感知：包含障碍物规避等安全要求
        - 语义丰富：支持复杂的多模态理解
        - 可扩展性：模板库易于扩展

        参数说明：
            target_types: 目标类型列表
                - 如['building', 'stadium']
                - 决定指令中的目标描述
            obstacle_types: 障碍物类型列表
                - 如['lake', 'forest']
                - 决定指令中的障碍物描述
            complexity: 复杂度级别
                - ComplexityLevel枚举值
                - 决定指令的复杂程度

        返回值：
            str: 生成的指令文本
                - 自然语言文本
                - 可直接用于TTS合成
                - 可用于多模态模型训练
        """
        # 格式化目标类型字符串
        target_str = ", ".join(target_types) if target_types else "buildings"
        # 格式化障碍物类型字符串
        obstacle_str = ", ".join(obstacle_types) if obstacle_types else "lakes"

        # 根据复杂度选择模板
        if complexity == ComplexityLevel.SIMPLE:
            tpl = random.choice(SIMPLE_TEMPLATES)
            return tpl.format(target=target_str)
        elif complexity == ComplexityLevel.MEDIUM:
            tpl = random.choice(MEDIUM_TEMPLATES)
            return tpl.format(target=target_str, obstacle=obstacle_str)
        else:
            tpl = random.choice(COMPLEX_TEMPLATES)
            return tpl.format(target=target_str, obstacle=obstacle_str)
