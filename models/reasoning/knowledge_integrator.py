"""
知识整合器模块。

本模块实现基于大语言模型（LLM）的常识推理，
用于丰富指令内容、解决语义歧义和提升任务规划质量。

核心功能:
    1. 指令丰富：将隐含约束显式化
    2. 歧义解析：澄清模糊术语在无人机任务中的具体含义
    3. 上下文感知：结合卫星图像上下文进行推理

技术特点:
    - 支持多种LLM后端（默认gpt-4o）
    - 自动API密钥管理（环境变量或参数）
    - 优雅降级：LLM不可用时返回原始输入
    - 安全设计：温度设置为0，确保确定性输出

参考文献:
    Brown et al., 2020. "Language Models are Few-Shot Learners"
    OpenAI, 2024. "GPT-4o Technical Report"
"""

# 标准库导入
import logging  # 日志记录模块
import os        # 操作系统接口模块
from typing import Optional  # 类型提示支持

# 获取实验日志记录器
logger = logging.getLogger("experiment")


class KnowledgeIntegrator:
    """
    基于大语言模型的知识整合器类。

    该类利用大语言模型进行常识推理，用于丰富指令内容、
    解决语义歧义和提升任务规划质量。

    主要组件:
        - _llm: 大语言模型实例
        - llm_model: 使用的LLM模型名称
        - api_key: API密钥
    """

    def __init__(self, llm_model: str = "gpt-4o", api_key: Optional[str] = None):
        """
        初始化知识整合器。

        Args:
            llm_model: 大语言模型名称，默认为"gpt-4o"
            api_key: API密钥，如果未提供则从环境变量获取
        """
        self.llm_model = llm_model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self._llm = None

    def _load_llm(self):
        """
        延迟加载大语言模型。

        采用延迟加载策略，仅在首次使用时加载模型，
        避免不必要的内存占用和加载时间。

        加载流程:
            1. 检查模型是否已加载
            2. 使用LangChain加载ChatOpenAI模型
            3. 加载失败时启用后备模式
        """
        if self._llm is not None:
            return
        try:
            from langchain_openai import ChatOpenAI

            self._llm = ChatOpenAI(
                api_key=self.api_key,
                model_name=self.llm_model,
                temperature=0,  # 温度设为0，确保确定性输出
            )
            logger.info("LLM loaded for knowledge integration: %s", self.llm_model)
        except Exception as exc:
            logger.warning("LLM unavailable (%s); knowledge integration disabled.", exc)
            self._llm = "fallback"  # 启用后备模式

    def enrich_instruction(self, instruction: str, context: str = "") -> str:
        """
        使用常识推理丰富指令内容。

        该方法将指令中的隐含约束显式化，提高指令的完整性和可执行性。

        示例:
            "Avoid all water bodies" → 包括湖泊、河流、池塘等

        如果大语言模型不可用，则返回原始指令不变。

        Args:
            instruction: 原始指令字符串
            context: 上下文信息（可选）

        Returns:
            enriched_instruction: 丰富后的指令字符串
        """
        self._load_llm()
        if not self._llm or self._llm == "fallback":
            return instruction

        prompt = (
            "You are a UAV mission planner assistant. "
            "Expand the following instruction by making implicit constraints explicit. "
            "Keep the output as a single instruction paragraph.\n\n"
            f"Context: {context}\n"
            f"Instruction: {instruction}\n"
            "Expanded instruction:"
        )
        try:
            response = self._llm.invoke(prompt)
            return response.content.strip()
        except Exception as exc:
            logger.warning("Instruction enrichment failed: %s", exc)
            return instruction

    def resolve_ambiguity(self, ambiguous_term: str, image_context: str = "") -> str:
        """
        解析模糊术语的具体含义。

        该方法结合卫星图像上下文，澄清模糊术语在无人机任务中的具体指代。

        如果大语言模型不可用，则返回原始术语。

        Args:
            ambiguous_term: 模糊术语
            image_context: 图像上下文信息（可选）

        Returns:
            clarification: 对模糊术语的澄清说明
        """
        self._load_llm()
        if not self._llm or self._llm == "fallback":
            return ambiguous_term

        prompt = (
            "You are a UAV mission planner. "
            f"The term '{ambiguous_term}' appeared in a satellite-image-based mission. "
            f"Image context: {image_context}. "
            "Provide a concise clarification of what this term refers to in the UAV mission context. "
            "Respond with one sentence."
        )
        try:
            response = self._llm.invoke(prompt)
            return response.content.strip()
        except Exception as exc:
            logger.warning("Ambiguity resolution failed: %s", exc)
            return ambiguous_term
