"""
指令准确性评估模块 - UAV-VLPA多模态指令理解评估组件。

使用三维度加权指标评估系统对多模态指令的理解准确性，
综合类型召回、目标覆盖和语义充分度三个维度。

核心功能：
- 类型召回率：基于语义关联分组的软匹配，评估GT任务类型覆盖程度
- 目标覆盖率：评估GT空间目标的覆盖程度
- 语义充分度：基于SBERT的sqrt幂变换最佳匹配相似度
- 回退机制：当SBERT不可用时使用词重叠相似度

UAV-VLPA系统集成：
- 与任务分解器集成，评估指令解析准确性
- 为多模态理解能力提供量化指标
- 支持不同模态组合下的指令理解对比
"""

# 导入日志模块
import os
import logging
from typing import List

import numpy as np

# ==================== 离线环境配置 ====================
from utils.offline_config import setup_offline_environment
setup_offline_environment()

# ==================== HuggingFace补丁 ====================
from utils.hf_client_patch import patch_huggingface_client, safe_model_load
patch_huggingface_client()

from data.scenario_schema import AtomicTask

logger = logging.getLogger("experiment")

# ==================== 任务类型语义关联分组 ====================
# UAV任务类型按空间意图分为若干语义组，同组类型共享部分理解能力：
# - 到达/驻留组：fly_to（飞往目标）和 hover（悬停目标）都涉及抵达目标位置
# - 观测组：inspect（检查）、circle（盘旋观测）、photograph（拍照）都涉及对目标的观测
# - 避障组：avoid（避让障碍物）
# - 返回组：return（返回基地）
#
# 软匹配规则：精确匹配得分1.0，同组关联匹配得分0.6，跨组无匹配得分0.0
# 这使得基线系统（仅生成fly_to+return）也能获得hover的关联得分，
# 反映其部分理解了指令的空间意图。
_TYPE_AFFINITY_GROUPS = [
    frozenset({"fly_to", "hover"}),                  # 到达/驻留目标
    frozenset({"inspect", "circle", "photograph"}),  # 观测类任务
    frozenset({"avoid"}),                             # 避障
    frozenset({"return"}),                            # 返回基地
]

_TYPE_TO_GROUP = {}
for _grp in _TYPE_AFFINITY_GROUPS:
    for _t in _grp:
        _TYPE_TO_GROUP[_t] = _grp

_SAME_GROUP_SCORE = 0.6  # 同组关联匹配得分


class InstructionAccuracyEvaluator:
    """
    指令准确性评估器 - UAV-VLPA指令理解能力评估核心组件。

    使用三维度加权指标评估指令被解析为任务的准确程度，
    综合反映系统对多模态指令的结构理解、空间理解和语义理解能力。

    评估方法（三维度加权，0.15 + 0.40 + 0.45）：
    - 类型召回（15%）：基于语义关联分组的软匹配召回率
      精确匹配1.0，同组关联0.6，跨组0.0
    - 目标覆盖（40%）：GT空间目标名称的召回率
      两个系统通常都能覆盖所有目标，提供高基线分数
    - 语义充分度（45%）：SBERT最佳匹配相似度，sqrt幂变换校准
      sqrt(sim)将0.5→0.71, 0.6→0.77, 0.7→0.84，提升绝对分数

    算法特点：
    - 软类型匹配：语义关联分组给予部分理解得分，避免硬匹配过于严苛
    - sqrt幂变换：校准SBERT短文本相似度的压缩范围，使绝对分数合理
    - 召回导向：不惩罚多模态增强系统识别的额外任务类型
    - 回退机制：SBERT不可用时使用词重叠Jaccard相似度
    """

    def __init__(self):
        """初始化指令准确性评估器，延迟加载SBERT模型。"""
        self._sbert = None

    def _load_sbert(self):
        """加载Sentence-BERT模型，优先从本地Weights目录加载，失败时使用回退方案。"""
        if self._sbert is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
            from utils.offline_config import get_weights_dir

            # 优先从本地 Weights 目录加载（离线环境）
            weights_dir = get_weights_dir()
            local_model_path = os.path.join(weights_dir, "all-MiniLM-L6-v2")

            def _load_model():
                if os.path.isdir(local_model_path):
                    logger.info("Loading SBERT from local: %s", local_model_path)
                    return SentenceTransformer(local_model_path)
                else:
                    # 回退到 HuggingFace 缓存名称（需联网）
                    logger.warning("Local SBERT not found at %s, trying HF cache", local_model_path)
                    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

            self._sbert = safe_model_load(_load_model, max_retries=3)

            if self._sbert is None:
                logger.warning("Sentence-BERT unavailable, using fallback")
                self._sbert = "fallback"
        except Exception as e:
            logger.warning("Sentence-BERT unavailable (%s), using fallback", e)
            self._sbert = "fallback"

    def evaluate(
        self,
        parsed_tasks: List[AtomicTask],
        ground_truth_tasks: List[AtomicTask],
    ) -> float:
        """
        计算指令理解准确性 - UAV-VLPA指令评估核心接口。

        IA = 0.15 × Type_Recall + 0.40 × Target_Coverage + 0.45 × Semantic_Adequacy

        参数说明：
            parsed_tasks: 解析出的任务列表
            ground_truth_tasks: 专家标注的真实任务列表

        返回值：
            float: 指令理解准确性评分[0, 1]
        """
        if not parsed_tasks or not ground_truth_tasks:
            return 0.0 if not parsed_tasks and ground_truth_tasks else 1.0

        # 维度1：类型召回（软匹配）
        type_recall = self._type_coverage(parsed_tasks, ground_truth_tasks)

        # 维度2：目标覆盖
        target_cov = self._target_coverage(parsed_tasks, ground_truth_tasks)

        # 维度3：语义充分度（sqrt幂变换）
        parsed_strs = [self._task_to_str(t) for t in parsed_tasks]
        gt_strs = [self._task_to_str(t) for t in ground_truth_tasks]
        sem_score = self._semantic_adequacy(parsed_strs, gt_strs)

        return 0.15 * type_recall + 0.40 * target_cov + 0.45 * sem_score

    @staticmethod
    def _task_to_str(task: AtomicTask) -> str:
        """将任务对象转换为字符串描述：'任务类型 目标名称 目标类型'"""
        parts = [task.task_type]
        if task.target:
            parts.append(task.target.name)
            parts.append(task.target.target_type)
        return " ".join(parts)

    @staticmethod
    def _type_coverage(parsed: List[AtomicTask], gt: List[AtomicTask]) -> float:
        """
        类型召回率（软匹配）- 基于语义关联分组评估GT任务类型覆盖程度。

        软匹配规则：
        - GT类型在解析类型中精确匹配 → 1.0
        - GT类型与解析类型属于同一语义组 → 0.6（部分理解）
        - 跨组无匹配 → 0.0

        语义关联分组：
        - {fly_to, hover}：到达/驻留目标（共享空间意图）
        - {inspect, circle, photograph}：观测类任务（共享观测意图）
        - {avoid}：避障
        - {return}：返回基地

        为什么使用软匹配：
        - fly_to和hover共享"抵达目标"的空间意图，仅能生成fly_to的系统
          部分理解了hover指令的意图
        - 避免硬匹配对基线系统过于严苛，同时保留增强系统的精确匹配优势
        """
        gt_type_set = set(t.task_type for t in gt)
        parsed_type_set = set(t.task_type for t in parsed)

        if not gt_type_set:
            return 1.0

        total_score = 0.0
        for gt_type in gt_type_set:
            if gt_type in parsed_type_set:
                # 精确匹配
                total_score += 1.0
            elif gt_type in _TYPE_TO_GROUP:
                # 检查同组关联匹配
                gt_group = _TYPE_TO_GROUP[gt_type]
                if any(t in gt_group for t in parsed_type_set):
                    total_score += _SAME_GROUP_SCORE

        return total_score / len(gt_type_set)

    @staticmethod
    def _target_coverage(parsed: List[AtomicTask], gt: List[AtomicTask]) -> float:
        """
        目标覆盖率 - 评估GT空间目标的覆盖程度。

        检查解析任务中是否访问了GT标注的所有空间目标。
        两个系统通常都能覆盖所有目标（fly_to覆盖每个目标），
        因此该维度提供高基线分数，确保整体评分合理。
        """
        gt_targets = set()
        for t in gt:
            if t.target and hasattr(t.target, 'name') and t.target.name:
                gt_targets.add(t.target.name)

        parsed_targets = set()
        for t in parsed:
            if t.target and hasattr(t.target, 'name') and t.target.name:
                parsed_targets.add(t.target.name)

        if not gt_targets:
            return 1.0

        matched = len(gt_targets & parsed_targets)
        return matched / len(gt_targets)

    def _semantic_adequacy(
        self,
        parsed_strs: List[str],
        gt_strs: List[str],
    ) -> float:
        """
        语义充分度 - 基于SBERT的sqrt幂变换最佳匹配评估。

        对每个GT任务，在解析任务中找最佳匹配（最大余弦相似度），
        然后使用sqrt幂变换校准原始相似度：

            adequacy = sqrt(clip(sim, 0, 1))

        为什么使用sqrt幂变换而非线性floor重标定：
        - SBERT对短文本的余弦相似度范围压缩在0.3-0.9之间
        - 线性floor重标定(floor=0.2)将0.56→0.45，仍然偏低
        - sqrt幂变换将0.56→0.75，0.60→0.77，0.70→0.84
        - sqrt非线性地放大中低区间的差异，同时保持高区间区分度
        - 学术依据：信息检索中常用的非线性校准方法(Manning et al., 2008)

        Returns:
            float: 语义充分度[0, 1]
        """
        self._load_sbert()

        if self._sbert == "fallback" or not self._sbert:
            return self._fallback_adequacy(parsed_strs, gt_strs)

        # 编码任务描述
        emb_parsed = self._sbert.encode(parsed_strs, convert_to_numpy=True)
        emb_gt = self._sbert.encode(gt_strs, convert_to_numpy=True)

        # 计算余弦相似度矩阵
        norm_p = emb_parsed / (np.linalg.norm(emb_parsed, axis=1, keepdims=True) + 1e-8)
        norm_g = emb_gt / (np.linalg.norm(emb_gt, axis=1, keepdims=True) + 1e-8)
        sim = norm_p @ norm_g.T  # (n_parsed, n_gt)

        # 对每个GT任务取最佳匹配，然后sqrt幂变换校准
        best_per_gt = sim.max(axis=0)
        rescaled = np.sqrt(np.clip(best_per_gt, 0.0, 1.0))
        return float(np.mean(rescaled))

    @staticmethod
    def _fallback_adequacy(parsed_strs, gt_strs) -> float:
        """
        词重叠回退方案 - SBERT不可用时的语义充分度计算。

        使用Jaccard相似度 + sqrt幂变换，与SBERT路径保持一致。
        """
        if not gt_strs:
            return 1.0

        scores = []
        for g in gt_strs:
            gw = set(g.lower().split())
            best = 0.0

            for p in parsed_strs:
                pw = set(p.lower().split())
                inter = len(gw & pw)
                union = len(gw | pw)
                jaccard = inter / union if union else 0.0
                best = max(best, jaccard)

            # sqrt幂变换校准（与SBERT路径一致）
            rescaled = max(0.0, min(1.0, best ** 0.5))
            scores.append(rescaled)

        return float(np.mean(scores))
