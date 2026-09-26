"""
消融实验报告生成器 - 生成论文可直接使用的消融实验图表和表格。

输出内容（对应实验设计7.5节）：
    1. 主消融表：B0-B9在全测试集上的均值、区间、显著性与效应量
    2. 模态贡献图：B1对比B2/B3/B4/B5的性能下降图
    3. 复杂度分层图：SIMPLE/MEDIUM/COMPLEX下B1与关键组对比曲线
    4. 案例可视化占位（轨迹差异）
"""

import os
import json
import logging
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluation.ablation_runner import AblationResult, AblationGroupResult
from evaluation.ablation_config import AblationGroupID
from evaluation.statistical_analysis import StatisticalAnalyzer

logger = logging.getLogger("experiment")

# 消融组颜色方案
GROUP_COLORS = {
    "B0": "#E74C3C",   # 红色 - 基线
    "B1": "#2ECC71",   # 绿色 - 完整模型
    "B2": "#3498DB",   # 蓝色
    "B3": "#9B59B6",   # 紫色
    "B4": "#E67E22",   # 橙色
    "B5a": "#1ABC9C",  # 青色
    "B5b": "#F39C12",  # 金色
    "B5c": "#D35400",  # 深橙
    "B6": "#8E44AD",   # 深紫
    "B7": "#2C3E50",   # 深灰蓝
    "B8": "#C0392B",   # 深红
    "B9": "#7F8C8D",   # 灰色
}


class AblationReportGenerator:
    """
    消融实验报告生成器。

    生成论文可直接使用的图表、表格和文本报告。
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.analyzer = StatisticalAnalyzer()

    def generate_full_report(self, ablation_result: AblationResult) -> str:
        """
        生成完整的消融实验报告。

        返回报告摘要路径。
        """
        logger.info("Generating ablation experiment report...")

        # 1. 主消融表
        summary_path = self._generate_main_ablation_table(ablation_result)

        # 2. 模态贡献图
        self._plot_modality_contribution(ablation_result)

        # 3. 模块贡献图
        self._plot_module_contribution(ablation_result)

        # 4. 复杂度分层图
        self._plot_stratified_comparison(ablation_result)

        # 5. 综合性能雷达图
        self._plot_performance_radar(ablation_result)

        # 6. 统计检验结果保存
        self._save_statistical_tests(ablation_result)

        # 7. JSON格式完整结果
        self._save_json_results(ablation_result)

        logger.info("Ablation report generated in %s", self.output_dir)
        return summary_path

    def _generate_main_ablation_table(self, result: AblationResult) -> str:
        """
        生成主消融表（对应7.5节第1项）。

        表格列：Group | Metric Mean [CI] | p-value | Cohen's d | Significant
        """
        b1_key = AblationGroupID.B1_ENHANCED_FULL.value
        metric_names = [
            ("task_completion_rate", "Task Completion Rate"),
            ("instruction_accuracy", "Instruction Accuracy"),
            ("trajectory_length_km", "Trajectory Length (km)"),
            ("sequential_rmse", "Sequential RMSE (m)"),
            ("response_latency_ms", "Response Latency (ms)"),
            ("replan_success_rate", "Replan Success Rate"),
            ("subjective_score", "Subjective Score (1-5)"),
        ]

        rows = []
        for gid, group_result in result.group_results.items():
            if not group_result.metrics:
                continue

            for attr, label in metric_names:
                vals = [getattr(m, attr, 0.0) for m in group_result.metrics]
                vals_clean = [v for v in vals if not np.isnan(v)]

                if not vals_clean:
                    continue

                mean_val = np.mean(vals_clean)
                ci = self.analyzer.confidence_interval(vals_clean) if len(vals_clean) > 1 else (mean_val, mean_val)

                # RMSE和比率指标不可为负，限制CI下限
                non_negative_metrics = {
                    "Sequential RMSE (m)",
                    "Task Completion Rate", "Instruction Accuracy",
                    "Replan Success Rate",
                    "Subjective Score (1-5)",
                }
                ci_lower = max(ci[0], 0.0) if label in non_negative_metrics else ci[0]

                # 统计检验信息
                test_info = result.statistical_tests.get(gid, {}).get(attr, {})
                p_val = test_info.get("p_value", float("nan"))
                cd = test_info.get("cohens_d", float("nan"))
                sig = test_info.get("significant", False)

                rows.append({
                    "Group": f"{gid}: {group_result.group_config.name}" if group_result.group_config else gid,
                    "Metric": label,
                    "Mean": f"{mean_val:.4f}",
                    "95% CI": f"[{ci_lower:.4f}, {ci[1]:.4f}]",
                    "p-value": f"{p_val:.4f}" if not np.isnan(p_val) else "ref",
                    "Cohen's d": f"{cd:.3f}" if not np.isnan(cd) else "ref",
                    "Significant": "Yes" if sig else ("ref" if gid == b1_key else "No"),
                })

        df = pd.DataFrame(rows)
        csv_path = os.path.join(self.output_dir, "ablation_main_table.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        # 打印到控制台
        print("\n" + "=" * 80)
        print("ABLATION STUDY - Main Results Table")
        print("=" * 80)
        print(df.to_string(index=False))
        print()

        logger.info("Main ablation table saved to %s", csv_path)
        return csv_path

    def _plot_modality_contribution(self, result: AblationResult):
        """
        模态贡献图（对应7.5节第2项）。

        展示B1对比B2/B3/B4/B5a/B5b/B5c的性能下降。
        """
        b1_key = AblationGroupID.B1_ENHANCED_FULL.value
        if b1_key not in result.group_results:
            return

        b1_metrics = result.group_results[b1_key].metrics
        if not b1_metrics:
            return

        b1_completion = np.mean([m.task_completion_rate for m in b1_metrics])
        b1_accuracy = np.mean([m.instruction_accuracy for m in b1_metrics])

        modality_groups = ["B2", "B3", "B4", "B5a", "B5b", "B5c"]
        modality_labels = [
            "-Voice", "-Gesture", "-Annotation",
            "Text+Voice", "Text+Gesture", "Text+Annotation",
        ]

        completion_drops = []
        accuracy_drops = []
        available_labels = []

        for gid, label in zip(modality_groups, modality_labels):
            if gid not in result.group_results:
                continue
            gm = result.group_results[gid].metrics
            if not gm:
                continue

            comp = np.mean([m.task_completion_rate for m in gm])
            acc = np.mean([m.instruction_accuracy for m in gm])

            completion_drops.append(b1_completion - comp)
            accuracy_drops.append(b1_accuracy - acc)
            available_labels.append(label)

        if not available_labels:
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        x = np.arange(len(available_labels))
        width = 0.5

        # 任务完成率下降
        colors = [GROUP_COLORS.get(g, "#888") for g in modality_groups[:len(available_labels)]]
        bars1 = axes[0].bar(x, completion_drops, width, color=colors, alpha=0.85)
        axes[0].set_ylabel("Performance Drop", fontsize=12)
        axes[0].set_title("Task Completion Rate Drop\n(vs B1: Enhanced-Full)", fontsize=13)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(available_labels, rotation=30, ha="right", fontsize=10)
        axes[0].axhline(y=0, color="black", linewidth=0.5)
        axes[0].grid(axis="y", linestyle="--", alpha=0.5)
        for bar, val in zip(bars1, completion_drops):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:+.3f}", ha="center", fontsize=9,
            )

        # 指令准确率下降
        bars2 = axes[1].bar(x, accuracy_drops, width, color=colors, alpha=0.85)
        axes[1].set_ylabel("Performance Drop", fontsize=12)
        axes[1].set_title("Instruction Accuracy Drop\n(vs B1: Enhanced-Full)", fontsize=13)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(available_labels, rotation=30, ha="right", fontsize=10)
        axes[1].axhline(y=0, color="black", linewidth=0.5)
        axes[1].grid(axis="y", linestyle="--", alpha=0.5)
        for bar, val in zip(bars2, accuracy_drops):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:+.3f}", ha="center", fontsize=9,
            )

        plt.suptitle("Modality Contribution Analysis (Ablation B2-B5)", fontsize=15, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "ablation_modality_contribution.png"), dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Modality contribution plot saved.")

    def _plot_module_contribution(self, result: AblationResult):
        """
        模块贡献图：B1对比B6/B7/B8/B9的性能下降。
        """
        b1_key = AblationGroupID.B1_ENHANCED_FULL.value
        if b1_key not in result.group_results:
            return

        b1_metrics = result.group_results[b1_key].metrics
        if not b1_metrics:
            return

        module_groups = ["B0", "B6", "B7", "B8", "B9"]
        module_labels = [
            "B0: TextOnly",
            "B6: -CrossAttn",
            "B7: -Calibration",
            "B8: -Replan",
            "B9: -TaskDecomp",
        ]

        metric_names = {
            "task_completion_rate": "Task Completion",
            "instruction_accuracy": "Inst. Accuracy",
        }

        fig, axes = plt.subplots(1, len(metric_names), figsize=(5 * len(metric_names), 6))

        for ax_idx, (attr, title) in enumerate(metric_names.items()):
            b1_val = np.mean([getattr(m, attr, 0) for m in b1_metrics])
            vals = []
            labels = []

            for gid, label in zip(module_groups, module_labels):
                if gid not in result.group_results:
                    continue
                gm = result.group_results[gid].metrics
                if not gm:
                    continue
                v = np.mean([getattr(m, attr, 0) for m in gm])
                vals.append(v)
                labels.append(label)

            if not vals:
                continue

            x = np.arange(len(labels) + 1)
            all_vals = [b1_val] + vals
            all_labels = ["B1: Full"] + labels
            colors = [GROUP_COLORS.get("B1", "#2ECC71")] + [
                GROUP_COLORS.get(g, "#888") for g in module_groups[:len(vals)]
            ]

            bars = axes[ax_idx].bar(x, all_vals, color=colors, alpha=0.85)
            axes[ax_idx].set_title(title, fontsize=12)
            axes[ax_idx].set_xticks(x)
            axes[ax_idx].set_xticklabels(all_labels, rotation=45, ha="right", fontsize=9)
            axes[ax_idx].grid(axis="y", linestyle="--", alpha=0.5)

            for bar, val in zip(bars, all_vals):
                axes[ax_idx].text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", fontsize=8,
                )

        plt.suptitle("Module Contribution Analysis (Ablation B6-B9)", fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "ablation_module_contribution.png"), dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Module contribution plot saved.")

    def _plot_stratified_comparison(self, result: AblationResult):
        """
        复杂度分层图（对应7.5节第3项）。

        SIMPLE/MEDIUM/COMPLEX下B1与关键消融组的对比曲线。
        """
        if not result.stratified_analysis:
            return

        key_groups = ["B0", "B1", "B6", "B7", "B8"]
        complexity_levels = ["simple", "medium", "complex"]
        complexity_labels = ["SIMPLE", "MEDIUM", "COMPLEX"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        x = np.arange(len(complexity_levels))

        # 任务完成率
        for gid in key_groups:
            if gid not in result.stratified_analysis:
                continue
            strat = result.stratified_analysis[gid]
            vals = [strat.get(c, {}).get("task_completion_rate", 0) for c in complexity_levels]
            color = GROUP_COLORS.get(gid, "#888")
            axes[0].plot(x, vals, "o-", color=color, linewidth=2, markersize=8, label=gid)

        axes[0].set_xticks(x)
        axes[0].set_xticklabels(complexity_labels, fontsize=11)
        axes[0].set_ylabel("Task Completion Rate", fontsize=12)
        axes[0].set_title("Task Completion by Complexity", fontsize=13)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, linestyle="--", alpha=0.5)
        axes[0].set_ylim(0, 1.05)

        # 指令准确性
        for gid in key_groups:
            if gid not in result.stratified_analysis:
                continue
            strat = result.stratified_analysis[gid]
            vals = [strat.get(c, {}).get("instruction_accuracy", 0) for c in complexity_levels]
            color = GROUP_COLORS.get(gid, "#888")
            axes[1].plot(x, vals, "s-", color=color, linewidth=2, markersize=8, label=gid)

        axes[1].set_xticks(x)
        axes[1].set_xticklabels(complexity_labels, fontsize=11)
        axes[1].set_ylabel("Instruction Accuracy", fontsize=12)
        axes[1].set_title("Instruction Accuracy by Complexity", fontsize=13)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, linestyle="--", alpha=0.5)
        axes[1].set_ylim(0, 1.05)

        plt.suptitle("Stratified Analysis: Key Ablation Groups by Complexity", fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "ablation_stratified_comparison.png"), dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Stratified comparison plot saved.")

    def _plot_performance_radar(self, result: AblationResult):
        """
        综合性能雷达图：B1 vs 主要消融组。
        """
        b1_key = AblationGroupID.B1_ENHANCED_FULL.value
        if b1_key not in result.group_results:
            return

        radar_metrics = [
            ("task_completion_rate", "Task Compl."),
            ("instruction_accuracy", "Inst. Acc."),
            ("subjective_score", "Subj. Score"),
        ]

        groups_to_plot = ["B0", "B1", "B6", "B9"]
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
        angles += angles[:1]

        for gid in groups_to_plot:
            if gid not in result.group_results:
                continue
            gm = result.group_results[gid].metrics
            if not gm:
                continue

            values = []
            for attr, _ in radar_metrics:
                v = np.mean([getattr(m, attr, 0) for m in gm])
                values.append(v)
            values += values[:1]

            color = GROUP_COLORS.get(gid, "#888")
            ax.plot(angles, values, "o-", linewidth=2, color=color, label=gid)
            ax.fill(angles, values, alpha=0.1, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([label for _, label in radar_metrics], fontsize=11)
        ax.set_title("Ablation Performance Radar", fontsize=14, pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "ablation_performance_radar.png"), dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Performance radar plot saved.")

    def _save_statistical_tests(self, result: AblationResult):
        """保存统计检验结果到CSV。"""
        rows = []
        for gid, tests in result.statistical_tests.items():
            for metric, info in tests.items():
                rows.append({
                    "Ablation Group": gid,
                    "Metric": metric,
                    "B1 Mean": info.get("b1_mean", ""),
                    "Ablation Mean": info.get("ablation_mean", ""),
                    "p-value": info.get("p_value", ""),
                    "Cohen's d": info.get("cohens_d", ""),
                    "B1 CI": str(info.get("b1_ci", "")),
                    "Ablation CI": str(info.get("ablation_ci", "")),
                    "Significant (p<0.05)": info.get("significant", ""),
                })

        if rows:
            df = pd.DataFrame(rows)
            path = os.path.join(self.output_dir, "ablation_statistical_tests.csv")
            df.to_csv(path, index=False, encoding="utf-8-sig")
            logger.info("Statistical tests saved to %s", path)

    def _save_json_results(self, result: AblationResult):
        """保存完整消融结果为JSON。"""
        output = {
            "summary": {},
            "statistical_tests": {},
            "stratified_analysis": result.stratified_analysis,
        }

        for gid, gr in result.group_results.items():
            if not gr.metrics:
                continue
            output["summary"][gid] = {
                "name": gr.group_config.name if gr.group_config else gid,
                "description": gr.group_config.description if gr.group_config else "",
                "n_scenarios": len(gr.metrics),
                "mean_task_completion_rate": float(np.mean([m.task_completion_rate for m in gr.metrics])),
                "mean_instruction_accuracy": float(np.mean([m.instruction_accuracy for m in gr.metrics])),
                "mean_trajectory_length_km": float(np.mean([m.trajectory_length_km for m in gr.metrics])),
                "mean_response_latency_ms": float(np.mean([m.response_latency_ms for m in gr.metrics])),
                "mean_replan_success_rate": float(np.mean([m.replan_success_rate for m in gr.metrics])),
                "mean_subjective_score": float(np.mean([m.subjective_score for m in gr.metrics])),
            }

        # 序列化统计检验结果（转换tuple为list）
        for gid, tests in result.statistical_tests.items():
            serializable_tests = {}
            for metric, info in tests.items():
                s_info = {}
                for k, v in info.items():
                    if isinstance(v, tuple):
                        s_info[k] = list(v)
                    elif isinstance(v, (np.floating, np.integer)):
                        s_info[k] = float(v)
                    else:
                        s_info[k] = v
                serializable_tests[metric] = s_info
            output["statistical_tests"][gid] = serializable_tests

        path = os.path.join(self.output_dir, "ablation_results.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False, default=str)
        logger.info("Full ablation results saved to %s", path)
