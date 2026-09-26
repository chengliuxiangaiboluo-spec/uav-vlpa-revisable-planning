"""
训练工具模块：早停机制、检查点管理、学习率调度器。

本模块提供深度学习训练过程中的核心实用工具，
包括早停机制、模型检查点保存/加载和学习率调度。

主要组件:
    - EarlyStopping: 早停机制（当监控指标不再改善时停止训练）
    - CheckpointManager: 检查点管理器（保存和加载模型状态）
    - CosineAnnealingLR: 余弦退火学习率调度器（已集成在其他模块中）

技术特点:
    - 支持多种监控模式（最小化/最大化）
    - 自动创建检查点目录
    - 完整的模型状态保存（模型参数、优化器状态、元数据）
    - 最新检查点自动查找功能

参考文献:
    Loshchilov & Hutter, 2017. "SGDR: Stochastic Gradient Descent with Warm Restarts"
    Goodfellow et al., 2016. "Deep Learning"
"""

# 标准库导入
import os        # 操作系统接口模块
import logging   # 日志记录模块
from typing import Optional, Dict, Any  # 类型提示支持

# PyTorch深度学习框架
import torch      # 张量计算核心库

# 获取实验日志记录器
logger = logging.getLogger("experiment")


class EarlyStopping:
    """
    早停机制类。

    当监控的指标（如验证损失）不再改善时，提前终止训练，
    防止过拟合并节省计算资源。

    主要参数:
        - patience: 等待次数（指标连续多少次未改善后停止）
        - min_delta: 最小改善阈值（小于该值视为无改善）
        - mode: 监控模式（'min'表示最小化指标，'max'表示最大化指标）
    """

    def __init__(self, patience: int = 5, min_delta: float = 1e-4, mode: str = "min"):
        """
        初始化早停机制。

        Args:
            patience: 等待次数（指标连续多少次未改善后停止）
            min_delta: 最小改善阈值（小于该值视为无改善）
            mode: 监控模式（'min'表示最小化指标，'max'表示最大化指标）
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best: Optional[float] = None
        self.counter = 0

    def __call__(self, metric: float) -> bool:
        """
        调用早停机制。

        Args:
            metric: 当前监控指标值

        Returns:
            should_stop: 是否应该停止训练
        """
        if self.best is None:
            self.best = metric
            return False

        if self.mode == "min":
            improved = metric < self.best - self.min_delta
        else:
            improved = metric > self.best + self.min_delta

        if improved:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience


class CheckpointManager:
    """
    检查点管理器类。

    该类负责模型检查点的保存和加载，
    包括模型参数、优化器状态和训练元数据。

    主要功能:
        - save(): 保存检查点
        - load(): 加载检查点
        - find_latest(): 查找最新检查点
    """

    def __init__(self, checkpoint_dir: str):
        """
        初始化检查点管理器。

        Args:
            checkpoint_dir: 检查点保存目录
        """
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        epoch: int,
        metrics: Dict[str, float],
        name: str = "checkpoint",
    ) -> str:
        """
        保存模型检查点。

        Args:
            model: 待保存的模型
            optimizer: 待保存的优化器（可选）
            epoch: 当前训练轮数
            metrics: 监控指标字典
            name: 检查点名称前缀

        Returns:
            path: 检查点文件路径
        """
        path = os.path.join(self.checkpoint_dir, f"{name}_epoch{epoch}.pt")
        state = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
        }
        if optimizer is not None:
            state["optimizer_state_dict"] = optimizer.state_dict()

        torch.save(state, path)
        logger.info("Checkpoint saved: %s (epoch %d)", path, epoch)
        return path

    def load(
        self,
        model: torch.nn.Module,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Dict[str, Any]:
        """
        加载模型检查点。

        Args:
            model: 待加载的模型
            path: 检查点文件路径
            optimizer: 待加载的优化器（可选）

        Returns:
            checkpoint: 检查点字典（包含所有保存的信息）
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info("Checkpoint loaded: %s (epoch %d)", path, checkpoint.get("epoch", -1))
        return checkpoint

    def find_latest(self, name: str = "checkpoint") -> Optional[str]:
        """
        查找最新的检查点文件。

        Args:
            name: 检查点名称前缀

        Returns:
            latest_path: 最新检查点文件路径，或None（未找到）
        """
        candidates = [
            f for f in os.listdir(self.checkpoint_dir)
            if f.startswith(name) and f.endswith(".pt")
        ]
        if not candidates:
            return None
        candidates.sort()
        return os.path.join(self.checkpoint_dir, candidates[-1])
