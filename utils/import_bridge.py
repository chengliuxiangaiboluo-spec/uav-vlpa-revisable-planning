"""
导入桥接模块 - UAV-VLPA外部模块兼容性接口。

为重用现有的pathplan/和tsp_only/模块提供导入桥接，
确保在独立项目模式下能够优雅地处理缺失的路径。

核心功能：
- 路径配置：将外部模块目录添加到sys.path
- 兼容性处理：优雅处理缺失的路径
- 导入管理：确保模块可以被直接导入

UAV-VLPA系统集成：
- 支持外部路径规划模块的集成
- 提供TSP（旅行商问题）求解器的访问接口
- 保持与原有代码库的兼容性
"""

# 导入系统模块，用于操作Python路径
import sys
# 导入操作系统接口，用于路径操作
import os


def _get_project_root() -> str:
    """
    获取独立实验项目的根目录路径 - UAV-VLPA路径配置辅助函数。

    基于当前文件的相对位置计算项目根目录，
    用于定位外部模块的路径。

    返回值：
        str: 项目根目录的绝对路径
    """
    # 获取当前文件的目录，向上一级得到项目根目录
    # os.path.dirname(__file__)获取当前文件所在目录
    # os.path.join(..., "..")向上一级
    # os.path.abspath()转换为绝对路径
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def setup_imports() -> str:
    """
    配置外部模块导入路径 - UAV-VLPA模块兼容性核心接口。

    将pathplan/和tsp_only/目录添加到sys.path中（如果它们存在），
    使得这些目录中的模块可以直接被导入。

    算法流程：
    1. 获取项目根目录
    2. 构建需要添加的路径列表
    3. 逆序插入sys.path（确保第一个路径在位置0）
    4. 跳过不存在的路径（独立模式）
    5. 返回项目根目录路径

    路径列表：
    - pathplan/: 路径规划模块目录
    - pathplan/experiments/: 路径规划实验目录
    - tsp_only/: TSP求解器目录
    - project_root: 项目根目录

    返回值：
        str: 项目根目录路径
    """
    # 获取项目根目录
    project_root = _get_project_root()

    # 构建需要添加到sys.path的路径列表
    paths_to_add = [
        os.path.join(project_root, "pathplan"),            # 路径规划模块
        os.path.join(project_root, "pathplan", "experiments"),  # 路径规划实验
        os.path.join(project_root, "tsp_only"),            # TSP求解器
        project_root,                                       # 项目根目录
    ]

    # ==================== 路径插入 ====================
    # 逆序插入，这样列表中的第一个路径最终会出现在位置0
    # 确保优先级：列表前面的路径优先被搜索
    for p in reversed(paths_to_add):
        # 如果路径不存在（独立模式下），跳过
        if not os.path.isdir(p):
            continue  # 跳过独立模式下不存在的路径

        # 如果路径已在sys.path中，先移除（避免重复）
        while p in sys.path:
            sys.path.remove(p)

        # 将路径插入到sys.path的开头（最高优先级）
        sys.path.insert(0, p)

    # 返回项目根目录路径
    return project_root
