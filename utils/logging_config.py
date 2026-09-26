"""
统一日志配置模块 - 实验框架。

该模块提供统一的日志配置，支持同时输出到控制台和文件。
所有实验模块应使用此配置来保持日志格式的一致性。
"""

# ==================== 标准库导入 ====================
import logging  # Python标准日志模块
import os       # 操作系统接口，用于目录操作
import sys      # 系统相关功能，用于标准输出


def setup_logging(
    log_dir: str = "logs",           # 日志文件存放目录
    level: int = logging.INFO,       # 日志级别（DEBUG < INFO < WARNING < ERROR < CRITICAL）
    log_file: str = "experiment.log", # 日志文件名
) -> logging.Logger:
    """
    配置日志系统，同时输出到控制台和文件。

    该函数创建一个名为 "experiment" 的日志记录器，配置两个处理器：
        1. 控制台处理器：输出到标准输出（stdout）
        2. 文件处理器：输出到指定日志文件

    Args:
        log_dir: 日志文件存放目录，默认为 "logs"
        level: 日志级别，默认为 INFO（记录INFO及以上级别）
        log_file: 日志文件名，默认为 "experiment.log"

    Returns:
        logging.Logger: 配置好的日志记录器实例
    """
    # 创建日志目录（如果不存在）
    # exist_ok=True 表示目录已存在时不报错
    os.makedirs(log_dir, exist_ok=True)

    # 获取或创建名为 "experiment" 的日志记录器
    logger = logging.getLogger("experiment")
    # 设置日志记录器的级别
    logger.setLevel(level)

    # 如果记录器已有处理器，说明已经配置过，直接返回
    # 避免重复添加处理器导致日志重复输出
    if logger.handlers:
        return logger

    # 创建日志格式器
    # 格式：[时间] 级别 - 名称 - 消息
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",  # 时间格式：年-月-日 时:分:秒
    )

    # ==================== 控制台处理器 ====================
    # 创建输出到标准输出的处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)  # 设置处理器级别
    console_handler.setFormatter(fmt)  # 设置格式
    logger.addHandler(console_handler)  # 添加到记录器

    # ==================== 文件处理器 ====================
    # 创建输出到文件的处理器
    file_handler = logging.FileHandler(
        os.path.join(log_dir, log_file),  # 完整文件路径
        encoding="utf-8"  # 使用UTF-8编码，支持中文
    )
    file_handler.setLevel(level)  # 设置处理器级别
    file_handler.setFormatter(fmt)  # 设置格式
    logger.addHandler(file_handler)  # 添加到记录器

    return logger
