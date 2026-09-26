"""
图像标注编码器模块。

本模块实现基于颜色阈值的视觉标注检测器，
用于从无人机图像中检测箭头、圆圈、高亮等视觉标注，
并将其空间分布编码为固定维度的特征向量。

主要功能:
    1. 使用OpenCV HSV颜色空间检测红色/绿色标注区域
    2. 提取每个标注的中心坐标、面积和类型
    3. 将标注信息编码为128维特征向量
    4. 聚合所有标注生成最终的512维融合特征

检测原理:
    - 红色标注: 表示障碍物 (type_id=1)
    - 绿色标注: 表示目标点 (type_id=2)
    - 使用HSV颜色空间避免光照影响

参考文献:
    Bradski, 2000. "The OpenCV Library"
"""

# 标准库导入
import logging  # 日志记录模块
from typing import List, Dict, Any, Tuple  # 类型提示支持

# 数值计算和深度学习框架
import numpy as np  # 数值计算库
import torch  # 张量计算核心库
import torch.nn as nn  # 神经网络模块

# 获取日志记录器
logger = logging.getLogger("experiment")


def _detect_colour_regions(
    image_np: np.ndarray,
    lower_hsv: np.ndarray,
    upper_hsv: np.ndarray,
    min_area: int = 50,
) -> List[Dict[str, Any]]:
    """
    使用OpenCV HSV阈值检测连续的颜色区域。

    这是一个辅助函数，用于检测指定HSV颜色范围内的连通区域。

    Args:
        image_np: RGB格式的numpy图像数组，shape为(H, W, 3)
        lower_hsv: HSV颜色空间下限，shape为(3,)
        upper_hsv: HSV颜色空间上限，shape为(3,)
        min_area: 最小区域面积阈值（像素），用于过滤噪声

    Returns:
        检测到的区域列表，每个元素包含中心坐标(cx,cy)和面积(area)

    Note:
        如果OpenCV不可用，将跳过检测并返回空列表
    """
    try:
        import cv2
    except ImportError:
        logger.warning("OpenCV not available; annotation detection skipped.")
        return []

    # 转换到HSV颜色空间
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    # 创建颜色掩码
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    # 形态学闭运算：连接断裂区域，去除小孔
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 寻找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        regions.append({"cx": cx, "cy": cy, "area": area})

    return regions


class AnnotationEncoder(nn.Module):
    """
    图像标注编码器。

    该编码器使用颜色阈值技术从无人机图像中检测视觉标注，
    并将其空间分布编码为固定维度的特征向量。

    主要组件:
        - detect_annotations(): 检测红色(障碍物)和绿色(目标)标注
        - annotation_mlp(): 将每个标注的几何特征映射到128维
        - projection(): 聚合所有标注特征并投影到512维融合空间

    工作流程:
        1. 将输入图像转换为HSV颜色空间
        2. 使用预定义的HSV范围检测红色和绿色区域
        3. 计算每个区域的中心坐标、归一化面积和类型ID
        4. 通过MLP编码每个标注
        5. 对所有标注特征进行平均池化
        6. 投影到统一的融合维度
    """

    def __init__(self, proj_dim: int = 512, max_annotations: int = 20, annotation_feat_dim: int = 128):
        """
        初始化标注编码器。

        Args:
            proj_dim: 投影后的特征维度（用于多模态融合）
            max_annotations: 单张图像最多检测的标注数量
            annotation_feat_dim: 每个标注的中间特征维度
        """
        super().__init__()  # 调用父类构造函数
        self.max_annotations = max_annotations  # 最大标注数量
        self.annotation_feat_dim = annotation_feat_dim  # 标注特征维度

        # 每个标注的特征：归一化中心x、归一化中心y、归一化面积、类型ID → 共4维
        # MLP将4维几何特征映射到128维语义特征
        self.annotation_mlp = nn.Sequential(
            nn.Linear(4, annotation_feat_dim),  # 4 → 128
            nn.ReLU(),  # 非线性激活
            nn.Linear(annotation_feat_dim, annotation_feat_dim),  # 128 → 128
        )
        # 聚合所有标注特征并投影到融合维度
        self.projection = nn.Sequential(
            nn.Linear(annotation_feat_dim, proj_dim),  # 128 → 512
            nn.LayerNorm(proj_dim),  # 层归一化，稳定特征分布
        )
        logger.info("AnnotationEncoder initialised (proj=%d).", proj_dim)

    def detect_annotations(self, image_np: np.ndarray) -> List[Dict[str, Any]]:
        """
        在RGB图像中检测红色(障碍物)和绿色(目标)标注。

        使用HSV颜色空间进行鲁棒的颜色检测，避免光照变化影响。

        Args:
            image_np: RGB格式的numpy图像数组，shape为(H, W, 3)

        Returns:
            标注列表，每个元素是字典，包含以下键:
                - cx: 归一化中心x坐标 (0-1)
                - cy: 归一化中心y坐标 (0-1)
                - area: 归一化面积 (0-1)
                - type_id: 标注类型ID
                    1: 红色标注（障碍物）
                    2: 绿色标注（目标点）

        HSV颜色范围:
            - 红色: [0,80,80]→[10,255,255] 和 [160,80,80]→[180,255,255]
            - 绿色: [35,80,80]→[85,255,255]
        """
        h, w = image_np.shape[:2]  # 获取图像尺寸

        # 红色在HSV中跨越0度，需要两个范围处理色相环绕
        red1 = _detect_colour_regions(
            image_np, np.array([0, 80, 80]), np.array([10, 255, 255])
        )
        red2 = _detect_colour_regions(
            image_np, np.array([160, 80, 80]), np.array([180, 255, 255])
        )
        # 为红色标注添加类型ID和归一化
        for r in red1 + red2:
            r["type_id"] = 1  # 障碍物
            r["cx"] /= w  # 归一化x坐标
            r["cy"] /= h  # 归一化y坐标
            r["area"] /= (w * h)  # 归一化面积

        # 绿色标注
        green = _detect_colour_regions(
            image_np, np.array([35, 80, 80]), np.array([85, 255, 255])
        )
        # 为绿色标注添加类型ID和归一化
        for g in green:
            g["type_id"] = 2  # 目标点
            g["cx"] /= w  # 归一化x坐标
            g["cy"] /= h  # 归一化y坐标
            g["area"] /= (w * h)  # 归一化面积

        all_annots = red1 + red2 + green  # 合并所有标注
        return all_annots[: self.max_annotations]  # 限制最大数量

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        对一批标注图像进行编码。

        前向传播流程:
            1. 遍历批处理中的每张图像
            2. 将PyTorch张量转换为numpy数组
            3. 调用detect_annotations检测标注
            4. 将标注特征通过MLP编码
            5. 对所有标注特征进行平均池化
            6. 投影到融合维度

        Args:
            images: 归一化的RGB图像张量，shape为(batch, 3, H, W)
                   值域应在[0, 1]范围内

        Returns:
            编码后的特征张量，shape为(batch, proj_dim)

        Note:
            如果某张图像没有检测到标注，将使用零向量
        """
        batch_size = images.size(0)  # 批处理大小
        device = images.device  # 获取设备
        all_feats = []  # 存储所有图像的特征

        for i in range(batch_size):
            # 将单张图像从PyTorch张量转换为numpy数组
            img_np = (images[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            annots = self.detect_annotations(img_np)  # 检测标注

            if annots:
                # 构建标注特征矩阵：每行包含[cx, cy, area, type_id]
                raw = torch.tensor(
                    [[a["cx"], a["cy"], a["area"], a["type_id"]] for a in annots],
                    dtype=torch.float32,
                    device=device,
                )
                feats = self.annotation_mlp(raw)          # (n_annots, feat_dim)
                pooled = feats.mean(dim=0)                 # (feat_dim,) 平均池化
            else:
                # 没有标注时使用零向量
                pooled = torch.zeros(self.annotation_feat_dim, device=device)

            all_feats.append(pooled)

        stacked = torch.stack(all_feats, dim=0)  # (batch, feat_dim)
        return self.projection(stacked)           # (batch, proj_dim)
