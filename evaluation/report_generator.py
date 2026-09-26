"""
报告生成器模块 - UAV-VLPA学术论文结果生成组件。

生成实验结果的图表、汇总表格和LaTeX格式输出，
用于学术论文的结果部分和演示文稿。

输出目录：根目录/results/
每次运行自动清空旧结果。

核心功能：
1. 任务完成率对比柱状图
2. 轨迹长度对比图
3. 训练损失曲线
4. 综合评估汇总表
5. LaTeX格式输出（可选）

使用方式：
    generator = ReportGenerator(output_dir="results")
    report = generator.generate_full_report(comparison_result)
"""

# 导入标准库
import os           # 操作系统接口
import shutil       # 高级文件操作
import logging      # 日志记录
from typing import List, Dict, Optional  # 类型提示

# 导入第三方库
import numpy as np  # 数值计算
import matplotlib   # 绘图库
matplotlib.use("Agg")  # 使用非交互式后端（服务器模式）
import matplotlib.pyplot as plt  # 绘图接口
import pandas as pd  # 数据处理

# 导入UAV-VLPA项目模块
from evaluation.comparison_runner import ComparisonResult, GroupResult  # 对比结果
from evaluation.statistical_analysis import StatisticalAnalyzer          # 统计分析
from data.scenario_schema import MetricsResult                           # 度量结果

# 获取实验日志记录器
logger = logging.getLogger("experiment")


class ReportGenerator:
    """
    报告生成器 - UAV-VLPA学术论文图表生成核心组件。

    从对比结果生成图表和表格，
    结果保存到根目录/results/，每次运行自动清空旧结果。

    核心功能：
    1. 生成任务完成率对比柱状图
    2. 生成轨迹长度对比图
    3. 生成训练损失曲线（可选）
    4. 生成综合评估汇总表
    5. 生成LaTeX格式输出

    学术用途：
    - 论文结果部分的图表生成
    - 演示文稿的数据可视化
    - 实验报告的自动化生成
    """

    def __init__(self, output_dir: str):
        """
        初始化报告生成器。

        参数说明：
            output_dir: 输出目录路径，默认为"results"
        """
        self.output_dir = output_dir

        # ==================== 清空旧结果 ====================
        # 如果输出目录已存在，删除它
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
            logger.info("Cleared previous results in %s", output_dir)

        # ==================== 创建新目录 ====================
        # 创建新的输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 初始化统计分析器
        self.analyzer = StatisticalAnalyzer()

    def generate_full_report(
        self,
        comparison: ComparisonResult,
        training_history: Optional[Dict[str, List[float]]] = None,
    ) -> str:
        """
        生成完整报告 - UAV-VLPA实验结果可视化主接口。

        生成所有图表和表格，返回报告摘要文本。

        生成内容：
        1. task_completion.png - 任务完成率对比柱状图
        2. trajectory_length.png - 轨迹长度对比图
        3. training_loss.png - 训练损失曲线（如果提供training_history）
        4. summary_table.csv - 综合评估汇总表
        5. summary.txt - 报告摘要文本

        参数说明：
            comparison: 三组对比结果（Baseline/Enhanced/Human）
            training_history: 训练历史记录（可选）

        返回值：
            str: 报告摘要文本
        """
        # ==================== 生成图表 ====================
        # 生成轨迹对比图
        self._plot_trajectory_comparison(comparison)
        # 生成RMSE箱线图
        self._plot_rmse_boxplot(comparison)
        # 生成任务完成率柱状图
        self._plot_completion_bar(comparison)

        # 新增：效率比对比图
        self._plot_efficiency_ratio(comparison)

        # 新增：分层分析图表
        self._plot_stratified_analysis(comparison)

        # 如果提供训练历史，生成训练曲线图
        if training_history:
            self._plot_training_curves(training_history)

        # ==================== 生成表格和报告 ====================
        # 生成汇总表格（含效率比）
        summary_path = self._generate_summary_csv(comparison)

        # 新增：生成分层分析报告
        self._generate_stratified_report(comparison)

        # 记录日志
        logger.info("Full report generated in %s", self.output_dir)

        # 返回汇总表格路径
        return summary_path

    # ---- charts ----------------------------------------------------------

    def _plot_trajectory_comparison(self, comp: ComparisonResult):
        """Bar chart comparing trajectory lengths across groups."""
        n = len(comp.scenarios)
        images = list(range(1, n + 1))
        base_len = [m.trajectory_length_km for m in comp.baseline.metrics]
        enh_len = [m.trajectory_length_km for m in comp.enhanced.metrics]
        hum_len = [m.trajectory_length_km for m in comp.human.metrics]

        x = np.arange(n)
        w = 0.25

        plt.figure(figsize=(max(12, n * 0.6), 6))
        plt.bar(x - w, base_len, w, label="Baseline", color="#3C78D8", alpha=0.8)
        plt.bar(x, enh_len, w, label="Enhanced", color="#6AA84F", alpha=0.8)
        plt.bar(x + w, hum_len, w, label="Human", color="#D47F27", alpha=0.8)
        plt.xticks(x, images, fontsize=9)
        plt.xlabel("Scenario", fontsize=14)
        plt.ylabel("Trajectory Length (km)", fontsize=14)
        plt.title("Trajectory Length Comparison", fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "trajectory_comparison.png"), dpi=150)
        plt.close()

    def _plot_rmse_boxplot(self, comp: ComparisonResult):
        """Box plot of RMSE metrics across groups."""
        groups = {"Baseline": comp.baseline, "Enhanced": comp.enhanced}
        metrics = {"KNN": "knn_rmse", "DTW": "dtw_rmse", "Sequential": "sequential_rmse"}

        fig, axes = plt.subplots(1, len(metrics), figsize=(14, 5))
        for ax, (label, attr) in zip(axes, metrics.items()):
            data = []
            labels = []
            for gname, group in groups.items():
                vals = [getattr(m, attr) for m in group.metrics if not np.isnan(getattr(m, attr))]
                if vals:
                    data.append(vals)
                    labels.append(gname)
            if data:
                bp = ax.boxplot(data, labels=labels, patch_artist=True)
                colors = ["#3C78D8", "#6AA84F"]
                for patch, color in zip(bp["boxes"], colors[: len(data)]):
                    patch.set_facecolor(color)
            ax.set_title(f"{label} RMSE", fontsize=13)
            ax.set_ylabel("Error (m)", fontsize=11)
            ax.grid(axis="y", linestyle="--", alpha=0.7)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "rmse_boxplot.png"), dpi=150)
        plt.close()

    def _plot_completion_bar(self, comp: ComparisonResult):
        """Bar chart of average task completion rate."""
        groups = {
            "Baseline": np.mean([m.task_completion_rate for m in comp.baseline.metrics]),
            "Enhanced": np.mean([m.task_completion_rate for m in comp.enhanced.metrics]),
            "Human": np.mean([m.task_completion_rate for m in comp.human.metrics]),
        }
        plt.figure(figsize=(6, 5))
        bars = plt.bar(groups.keys(), groups.values(), color=["#3C78D8", "#6AA84F", "#D47F27"], alpha=0.85)
        for bar, val in zip(bars, groups.values()):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{val:.2%}", ha="center", fontsize=12)
        plt.ylim(0, 1.15)
        plt.ylabel("Task Completion Rate", fontsize=13)
        plt.title("Average Task Completion Rate", fontsize=15)
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "completion_rate.png"), dpi=150)
        plt.close()

    def _plot_training_curves(self, history: Dict[str, List[float]]):
        """Plot training loss curves."""
        plt.figure(figsize=(8, 5))
        for key, vals in history.items():
            plt.plot(vals, label=key)
        plt.xlabel("Epoch / Episode")
        plt.ylabel("Loss / Reward")
        plt.title("Training Curves")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "training_curves.png"), dpi=150)
        plt.close()

    def _plot_efficiency_ratio(self, comp: ComparisonResult):
        """效率比对比图：效率比 = 任务完成率 / 轨迹长度"""
        base_eff = [m.efficiency_ratio for m in comp.baseline.metrics if getattr(m, 'efficiency_ratio', 0) > 0]
        enh_eff = [m.efficiency_ratio for m in comp.enhanced.metrics if getattr(m, 'efficiency_ratio', 0) > 0]
        if not base_eff or not enh_eff:
            return
        groups = {"Baseline": np.mean(base_eff), "Enhanced": np.mean(enh_eff)}
        plt.figure(figsize=(6, 5))
        bars = plt.bar(groups.keys(), groups.values(), color=["#3C78D8", "#6AA84F"], alpha=0.85)
        for bar, val in zip(bars, groups.values()):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002, f"{val:.4f}", ha="center", fontsize=11)
        plt.ylabel("Efficiency Ratio (completion/km)", fontsize=12)
        plt.title("Trajectory Efficiency Comparison\n(Higher is Better)", fontsize=14)
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "efficiency_ratio.png"), dpi=150)
        plt.close()

    def _plot_stratified_analysis(self, comp: ComparisonResult):
        """分层分析图表：按复杂度展示性能差距。"""
        stratified = comp.stratified_analysis.get("stratified_results", [])
        if not stratified:
            return
        complexities = [r.complexity.upper() for r in stratified]
        baseline_completion = [r.baseline_completion for r in stratified]
        enhanced_completion = [r.enhanced_completion for r in stratified]
        gaps = [r.gap for r in stratified]
        fig, ax1 = plt.subplots(figsize=(10, 6))
        x = np.arange(len(complexities))
        width = 0.35
        bars1 = ax1.bar(x - width/2, baseline_completion, width, label="Baseline", color="#3C78D8", alpha=0.8)
        bars2 = ax1.bar(x + width/2, enhanced_completion, width, label="Enhanced", color="#6AA84F", alpha=0.8)
        ax1.set_xlabel("Complexity Level", fontsize=12)
        ax1.set_ylabel("Task Completion Rate", fontsize=12)
        ax1.set_ylim(0, 1.1)
        ax1.set_xticks(x)
        ax1.set_xticklabels(complexities, fontsize=11)
        ax1.legend(loc="upper left", fontsize=10)
        ax2 = ax1.twinx()
        ax2.plot(x, gaps, "ro-", linewidth=2, markersize=8, label="Gap")
        ax2.set_ylabel("Performance Gap", fontsize=12, color="red")
        ax2.tick_params(axis="y", labelcolor="red")
        for i, (b1, b2, g) in enumerate(zip(bars1, bars2, gaps)):
            ax1.text(b1.get_x() + b1.get_width()/2, b1.get_height() + 0.02, f"{b1.get_height():.2f}", ha="center", fontsize=9)
            ax1.text(b2.get_x() + b2.get_width()/2, b2.get_height() + 0.02, f"{b2.get_height():.2f}", ha="center", fontsize=9)
            ax2.text(x[i], g + 0.01, f"+{g:.3f}", ha="center", fontsize=9, color="red")
        plt.title("Stratified Analysis: Task Completion Rate by Complexity\n(Gap increases with complexity = Multimodal understanding contribution)", fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "stratified_analysis.png"), dpi=150)
        plt.close()
        # 第二张图：轨迹长度对比
        fig, ax = plt.subplots(figsize=(10, 5))
        baseline_traj = [r.baseline_trajectory for r in stratified]
        enhanced_traj = [r.enhanced_trajectory for r in stratified]
        ax.bar(x - width/2, baseline_traj, width, label="Baseline", color="#3C78D8", alpha=0.8)
        ax.bar(x + width/2, enhanced_traj, width, label="Enhanced", color="#6AA84F", alpha=0.8)
        ax.set_xlabel("Complexity Level", fontsize=12)
        ax.set_ylabel("Trajectory Length (km)", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(complexities, fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.title("Trajectory Length by Complexity Level", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "stratified_trajectory.png"), dpi=150)
        plt.close()

    def _generate_stratified_report(self, comp: ComparisonResult):
        """生成分层分析文本报告。"""
        report = comp.stratified_analysis.get("stratified_report", "")
        contribution = comp.stratified_analysis.get("multimodal_contribution", {})
        if not report:
            return

        # 生成公平对比报告
        from evaluation.fair_comparison import generate_fair_comparison_report

        baseline_metrics = {
            "task_completion_rate": np.mean([m.task_completion_rate for m in comp.baseline.metrics]),
            "instruction_accuracy": np.mean([m.instruction_accuracy for m in comp.baseline.metrics]),
            "trajectory_length_km": np.mean([m.trajectory_length_km for m in comp.baseline.metrics]),
            "dtw_rmse": np.mean([m.dtw_rmse for m in comp.baseline.metrics if not np.isnan(m.dtw_rmse)]),
            "response_latency_ms": np.mean([m.response_latency_ms for m in comp.baseline.metrics]),
        }
        enhanced_metrics = {
            "task_completion_rate": np.mean([m.task_completion_rate for m in comp.enhanced.metrics]),
            "instruction_accuracy": np.mean([m.instruction_accuracy for m in comp.enhanced.metrics]),
            "trajectory_length_km": np.mean([m.trajectory_length_km for m in comp.enhanced.metrics]),
            "dtw_rmse": np.mean([m.dtw_rmse for m in comp.enhanced.metrics if not np.isnan(m.dtw_rmse)]),
            "response_latency_ms": np.mean([m.response_latency_ms for m in comp.enhanced.metrics]),
        }

        fair_report = generate_fair_comparison_report(baseline_metrics, enhanced_metrics)

        report_path = os.path.join(self.output_dir, "stratified_analysis.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
            f.write("\n\n")
            f.write("=" * 70 + "\n")
            f.write("Multimodal Understanding Contribution Analysis\n")
            f.write("=" * 70 + "\n")
            for key, val in contribution.items():
                f.write(f"{key}: {val:.4f}\n")
            f.write("\n\n")
            f.write(fair_report)

        # 单独保存公平对比报告
        fair_path = os.path.join(self.output_dir, "fair_comparison.txt")
        with open(fair_path, "w", encoding="utf-8") as f:
            f.write(fair_report)

        # 复杂度自适应评估报告
        try:
            from evaluation.complexity_adaptive_evaluator import ComplexityAdaptiveEvaluator
            adaptive_eval = ComplexityAdaptiveEvaluator()
            adaptive_report = adaptive_eval.generate_report(
                comp.scenarios, comp.baseline.metrics, comp.enhanced.metrics
            )
            adaptive_path = os.path.join(self.output_dir, "complexity_adaptive_evaluation.txt")
            with open(adaptive_path, "w", encoding="utf-8") as f:
                f.write(adaptive_report)
            logger.info("Complexity-adaptive evaluation saved to %s", adaptive_path)
        except Exception as e:
            logger.warning("Complexity-adaptive evaluation failed: %s", e)

        logger.info("Stratified analysis report saved to %s", report_path)
        logger.info("Fair comparison report saved to %s", fair_path)

        print("\n" + "=" * 70)
        print(report)
        print(fair_report)

    # ---- summary table ---------------------------------------------------

    def _generate_summary_csv(self, comp: ComparisonResult) -> str:
        """Generate a summary CSV with statistical analysis."""
        # 定义所有要输出的指标（与 MetricsResult 字段一致）
        metric_keys = [
            ("trajectory_length_km", "Trajectory Length (km)"),
            ("knn_rmse", "KNN RMSE (m)"),
            ("dtw_rmse", "DTW RMSE (m)"),
            ("sequential_rmse", "Sequential RMSE (m)"),
            ("task_completion_rate", "Task Completion Rate"),
            ("instruction_accuracy", "Instruction Accuracy"),
            ("efficiency_ratio", "Efficiency Ratio (/km)"),  # 新增
            ("response_latency_ms", "Response Latency (ms)"),
            ("replan_success_rate", "Replan Success Rate"),
            ("subjective_score", "Subjective Score (1-5)"),
        ]

        base_dict = {}
        enh_dict = {}
        hum_dict = {}

        for attr, label in metric_keys:
            base_dict[label] = [
                getattr(m, attr, np.nan) for m in comp.baseline.metrics
                if not np.isnan(getattr(m, attr, np.nan))
            ]
            enh_dict[label] = [
                getattr(m, attr, np.nan) for m in comp.enhanced.metrics
                if not np.isnan(getattr(m, attr, np.nan))
            ]
            hum_dict[label] = [
                getattr(m, attr, np.nan) for m in comp.human.metrics
                if not np.isnan(getattr(m, attr, np.nan))
            ]

        df = self.analyzer.generate_summary_table(base_dict, enh_dict, hum_dict)

        csv_path = os.path.join(self.output_dir, "summary_table.csv")
        df.to_csv(csv_path, index=False)
        logger.info("Summary table saved to %s", csv_path)

        # Print to console
        print("\n" + df.to_string(index=False) + "\n")

        return csv_path