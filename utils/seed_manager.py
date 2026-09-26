"""
可复现随机种子管理模块。

该模块提供全局随机种子设置功能，确保实验的可复现性。
通过统一设置 Python、NumPy 和 PyTorch 的随机种子，
使得相同代码在相同数据上产生相同的结果。
"""

# ==================== 随机数库导入 ====================
import random  # Python标准随机数模块
import numpy as np  # NumPy科学计算库
import torch  # PyTorch深度学习框架


def set_global_seed(seed: int = 42) -> None:
    """
    设置全局随机种子，确保实验可复现。

    该函数同时设置以下库的随机种子：
        1. Python标准库 random
        2. NumPy随机数生成器
        3. PyTorch CPU随机数生成器
        4. PyTorch CUDA随机数生成器（如果可用）

    此外，还配置 CUDA 后端的确定性行为：
        - cudnn.deterministic = True: 确保卷积操作确定性
        - cudnn.benchmark = False: 禁用自动优化（可能影响确定性）

    Args:
        seed: 随机种子值，默认为42（机器学习常用值）
    """
    # 设置Python标准库随机种子
    random.seed(seed)

    # 设置NumPy随机种子
    np.random.seed(seed)

    # 设置PyTorch CPU随机种子
    torch.manual_seed(seed)

    # 如果CUDA可用，设置CUDA相关随机种子和配置
    if torch.cuda.is_available():
        # 设置所有CUDA设备的随机种子
        torch.cuda.manual_seed_all(seed)
        # 确保CuDNN使用确定性算法（可能降低性能但保证可复现）
        torch.backends.cudnn.deterministic = True
        # 禁用CuDNN自动寻找最优算法（确保确定性）
        torch.backends.cudnn.benchmark = False
