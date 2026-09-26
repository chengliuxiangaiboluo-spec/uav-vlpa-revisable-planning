"""
离线环境配置模块 - UAV-VLPA项目。

该模块配置离线运行环境，确保在无网络环境下能够：
    1. 从本地路径加载预训练模型权重
    2. 禁用所有HuggingFace联网尝试
    3. 统一配置各种缓存目录
"""

# ==================== 标准库导入 ====================
import os      # 操作系统接口，用于环境变量和路径操作
import sys     # 系统相关功能
from pathlib import Path  # 面向对象的路径操作


def setup_offline_environment(server_mode: bool = False):
    """
    配置离线模型使用环境。

    The weights directory is set through ``UAV_VLPA_WEIGHTS_DIR``.  If the
    variable is unset, the release uses ``<checkout>/Weights``.  The directory
    is intentionally not included in this public repository.
    主要功能：
        1. 确定权重文件目录（优先使用验证过的绝对路径）
        2. 设置HuggingFace相关环境变量
        3. 启用严格的离线模式
        4. 禁用遥测和进度条等干扰项

    Args:
        server_mode: 服务器模式标志（当前未使用，保留用于扩展）

    Returns:
        str: 确定的权重目录路径
    """
    auto_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weights_dir = os.path.abspath(os.environ.get(
        "UAV_VLPA_WEIGHTS_DIR", os.path.join(auto_project_root, "Weights")
    ))

    # 强制创建目录（防止权限问题）
    try:
        os.makedirs(weights_dir, exist_ok=True)
    except Exception as e:
        print(f"[Offline Config] Warning: Could not create directory {weights_dir}: {e}")

    # ==================== HuggingFace 环境变量设置 ====================
    # 设置所有HuggingFace相关缓存目录，确保从本地加载模型
    os.environ["HF_HOME"] = weights_dir  # HuggingFace主目录
    os.environ["TRANSFORMERS_CACHE"] = weights_dir  # Transformers缓存
    os.environ["HF_HUB_CACHE"] = weights_dir  # Hub缓存
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = weights_dir  # Sentence-BERT缓存

    # ==================== 严格离线模式 ====================
    # 彻底切断所有联网尝试
    os.environ["TRANSFORMERS_OFFLINE"] = "1"  # Transformers离线模式
    os.environ["HF_HUB_OFFLINE"] = "1"  # Hub离线模式
    os.environ["HF_DATASETS_OFFLINE"] = "1"  # 数据集离线模式

    # ==================== 禁用干扰项 ====================
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"  # 禁用遥测（数据收集）
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"  # 禁用进度条
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "3600"  # 下载超时（实际上不会下载）

    # ==================== 打印配置信息 ====================
    print("[Offline Config] Mode: environment-or-checkout")
    print(f"[Offline Config] Weights directory: {weights_dir}")
    print(f"[Offline Config] Offline mode strictly enabled")

    return weights_dir


def get_weights_dir(server_mode: bool = False):
    """
    获取权重目录路径。

    Args:
        server_mode: 服务器模式标志（当前未使用）

    Returns:
        str: 权重目录路径
    """
    auto_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.abspath(os.environ.get(
        "UAV_VLPA_WEIGHTS_DIR", os.path.join(auto_project_root, "Weights")
    ))
