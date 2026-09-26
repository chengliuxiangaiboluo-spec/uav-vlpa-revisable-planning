"""
服务器配置文件 - UAV-VLPA项目。

该文件包含服务器特定的路径配置，用于区分服务器环境和本地开发环境。
在多环境部署时，通过修改这些常量来适配不同的文件系统结构。
"""

import os
from pathlib import Path

# This public release deliberately contains no host-specific paths.  Configure
# a deployment through environment variables, or use the checkout directory.
PROJECT_ROOT = Path(os.environ.get("UAV_VLPA_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
WEIGHTS_DIR = Path(os.environ.get("UAV_VLPA_WEIGHTS_DIR", PROJECT_ROOT / "Weights")).resolve()

# Compatibility aliases for older experiment scripts.
SERVER_PROJECT_ROOT = str(PROJECT_ROOT)
SERVER_WEIGHTS_DIR = str(WEIGHTS_DIR)
LOCAL_PROJECT_ROOT = str(PROJECT_ROOT)
LOCAL_WEIGHTS_DIR = str(WEIGHTS_DIR)
