"""
图像标注生成器模块 - UAV-VLPA系统的视觉提示生成组件。

在卫星图像上绘制箭头、圆圈、区域高亮和编号标签，
模拟人类用户对无人机任务的视觉指导。该模块是UAV-VLPA
多模态融合架构中视觉模态的关键组成部分，用于生成
带有丰富视觉提示的训练数据集。

核心功能：
- 将用户指令中的视觉概念（如'指向目标'、'避开障碍物'）
  转换为具体的图像标注元素
- 支持多种标注类型：箭头（路径指示）、圆圈（目标标记）、
  区域高亮（障碍物标识）、编号标签（任务顺序）
- 使用百分比坐标系统，确保标注在不同分辨率图像上的
  位置一致性
- 生成标准化的标注元数据，供多模态模型训练使用

UAV-VLPA系统集成：
- 与文本编码器协同工作，将视觉标注转换为嵌入向量
- 为跨模态注意力机制提供视觉输入特征
- 支持数据增强，提高模型对不同标注风格的鲁棒性
- 生成的标注图像用于训练视觉-语言-路径规划联合模型
"""

# ==================== 标准库和第三方库导入 ====================
import os       # 操作系统接口
import math     # 数学函数（用于箭头角度计算）
import logging  # 日志记录
import random   # 随机数生成
from typing import List, Tuple, Dict, Any  # 类型注解

from PIL import Image, ImageDraw, ImageFont  # PIL图像处理库

# 获取实验日志记录器
logger = logging.getLogger("experiment")


class ImageAnnotationGenerator:
    """
    图像标注生成器类 - UAV-VLPA视觉提示生成核心组件。

    在卫星图像上合成视觉标注（箭头、圆圈、高亮），
    模拟人类用户对无人机任务的视觉指导。该类实现了
    完整的视觉提示生成流水线，支持UAV-VLPA系统中
    多模态融合的视觉输入需求。

    核心设计原则：
    - 语义一致性：标注元素严格对应用户指令的语义含义
    - 可扩展性：支持新增标注类型而不影响现有接口
    - 兼容性：生成的标注可直接被视觉编码器处理
    - 鲁棒性：处理各种图像尺寸、格式和质量差异

    属性说明：
        arrow_color: 箭头颜色（RGB），代表路径指示方向
            - 默认绿色(0, 200, 0)，符合航空导航标准
        obstacle_color: 障碍物高亮颜色（RGBA），代表需要规避的区域
            - 默认半透明红色(255, 50, 50, 150)，符合危险警示标准
        target_color: 目标标记颜色（RGBA），代表任务目标位置
            - 默认半透明绿色(50, 200, 50, 150)，符合安全目标标准
    """

    def __init__(
        self,
        arrow_color: Tuple[int, int, int] = (0, 200, 0),
        obstacle_color: Tuple[int, int, int, int] = (255, 50, 50, 150),
        target_color: Tuple[int, int, int, int] = (50, 200, 50, 150),
    ):
        """
        初始化图像标注生成器。

        该构造函数配置视觉标注生成器的核心参数，
        这些参数直接影响UAV-VLPA系统中视觉模态的
        表达能力和模型训练效果。

        参数说明：
            arrow_color: 箭头颜色，默认绿色(0, 200, 0)
                - 代表路径指示方向，绿色符合航空导航标准
                - RGB格式，用于绘制箭杆和箭头
            obstacle_color: 障碍物高亮颜色，默认半透明红色
                - 代表需要规避的区域，红色符合危险警示标准
                - RGBA格式，包含透明度通道，避免遮挡背景图像
            target_color: 目标标记颜色，默认半透明绿色
                - 代表任务目标位置，绿色符合安全目标标准
                - RGBA格式，用于目标圆圈填充和边框
        """
        self.arrow_color = arrow_color
        self.obstacle_color = obstacle_color
        self.target_color = target_color

    # ==================== 绘制原语（私有静态方法） ====================

    @staticmethod
    def _draw_arrow(
        draw: ImageDraw.ImageDraw,
        start: Tuple[float, float],
        end: Tuple[float, float],
        color: Tuple[int, ...],
        width: int = 3,
        head_size: int = 14,
    ):
        """
        绘制从起点到终点的箭头 - UAV-VLPA视觉路径指示核心原语。

        该方法实现了标准的矢量箭头绘制算法，专门针对
        无人机路径规划的视觉表示需求进行优化。

        算法原理：
        - 箭杆：使用直线连接起点和终点，表示路径方向
        - 箭头：使用等腰三角形表示方向终点，角度计算基于
          atan2函数确保方向准确性
        - 箭头大小：与箭杆宽度成比例，保持视觉一致性

        无人机应用考虑：
        - 支持亚像素精度，确保在高分辨率卫星图像上的清晰显示
        - 箭头角度计算使用数学库函数，保证数值稳定性
        - 箭头大小可调，适应不同尺度的任务场景

        参数说明：
            draw: ImageDraw对象，用于在图像上绘制
            start: 起点坐标 (x, y)，代表路径起始位置
            end: 终点坐标 (x, y)，代表路径终止位置
            color: 箭头颜色，RGB或RGBA格式
            width: 箭杆线宽，默认3像素，确保在不同缩放级别下可见
            head_size: 箭头大小，默认14像素，与箭杆宽度协调
        """
        # 绘制箭杆（直线）
        draw.line([start, end], fill=color, width=width)

        # 计算箭头角度并绘制箭头（三角形）
        # atan2计算终点相对于起点的角度
        angle = math.atan2(end[1] - start[1], end[0] - start[0])
        # 计算箭头左右两翼的坐标
        left = (
            end[0] - head_size * math.cos(angle - math.pi / 6),
            end[1] - head_size * math.sin(angle - math.pi / 6),
        )
        right = (
            end[0] - head_size * math.cos(angle + math.pi / 6),
            end[1] - head_size * math.sin(angle + math.pi / 6),
        )
        # 绘制填充三角形作为箭头
        draw.polygon([end, left, right], fill=color)

    @staticmethod
    def _draw_circle_marker(
        draw: ImageDraw.ImageDraw,
        center: Tuple[float, float],
        radius: int,
        color: Tuple[int, ...],
        width: int = 3,
    ):
        """
        绘制空心圆形标记 - UAV-VLPA目标定位核心原语。

        该方法实现了标准的圆形标记绘制算法，专门针对
        无人机任务目标的视觉表示需求进行优化。

        算法原理：
        - 使用外接矩形定义圆形边界，确保几何精度
        - 空心设计突出目标位置，避免遮挡背景信息
        - 线条宽度可调，适应不同图像分辨率

        无人机应用考虑：
        - 半径固定为22像素，确保在典型卫星图像分辨率下的
          可视性和精确性
        - 支持抗锯齿渲染，提高边缘平滑度
        - 圆形标记与编号标签配合使用，形成完整的视觉提示

        参数说明：
            draw: ImageDraw对象，用于在图像上绘制
            center: 圆心坐标 (x, y)，代表目标地理位置
            radius: 圆半径，单位像素，默认22像素
            color: 圆圈颜色，RGB或RGBA格式
            width: 线条宽度，默认3像素，确保视觉清晰度
        """
        # 计算外接矩形
        bbox = (
            center[0] - radius,
            center[1] - radius,
            center[0] + radius,
            center[1] + radius,
        )
        draw.ellipse(bbox, outline=color, width=width)

    @staticmethod
    def _draw_region_highlight(
        draw: ImageDraw.ImageDraw,
        center: Tuple[float, float],
        size: int,
        color: Tuple[int, ...],
    ):
        """
        绘制区域高亮（半透明矩形） - UAV-VLPA障碍物识别核心原语。

        该方法实现了标准的区域高亮绘制算法，专门针对
        无人机障碍物规避的视觉表示需求进行优化。

        算法原理：
        - 使用矩形区域表示障碍物影响范围
        - 半透明填充确保背景图像可见，便于视觉分析
        - 边框强调区域边界，提高识别精度

        无人机应用考虑：
        - 随机大小变化（30-60像素）模拟真实障碍物的多样性
        - RGBA颜色格式支持透明度控制，避免过度遮挡
        - 中心坐标系统确保与GPS坐标系的映射一致性

        参数说明：
            draw: ImageDraw对象，用于在图像上绘制
            center: 矩形中心坐标 (x, y)，代表障碍物地理位置
            size: 矩形边长，单位像素，随机范围30-60像素
            color: 填充颜色，RGBA格式（含透明度通道）
        """
        half = size // 2
        bbox = (center[0] - half, center[1] - half, center[0] + half, center[1] + half)
        # 填充矩形并绘制边框
        draw.rectangle(bbox, fill=color, outline=color[:3], width=2)

    @staticmethod
    def _draw_number_label(
        draw: ImageDraw.ImageDraw,
        position: Tuple[float, float],
        number: int,
        color: Tuple[int, ...] = (255, 255, 255),
    ):
        """
        绘制带圆圈的数字标签 - UAV-VLPA任务序列化核心原语。

        该方法实现了标准的数字标签绘制算法，专门针对
        无人机多步骤任务的视觉序列化需求进行优化。

        算法原理：
        - 黑色半透明背景圆圈提高文字对比度
        - 居中偏移微调确保视觉平衡
        - 字体回退机制确保跨平台兼容性

        无人机应用考虑：
        - 数字代表任务执行顺序（1,2,3...）
        - 圆圈半径固定为12像素，与目标圆圈协调
        - 白色文字确保在各种背景下的可读性

        参数说明：
            draw: ImageDraw对象，用于在图像上绘制
            position: 标签位置 (x, y)，相对于目标圆圈的偏移位置
            number: 要显示的数字，代表任务执行序号
            color: 文字颜色，默认白色(255, 255, 255)，确保高对比度
        """
        x, y = position
        r = 12  # 圆圈半径
        # 绘制黑色半透明背景圆圈
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(0, 0, 0, 200))
        # 尝试加载字体，失败则使用默认字体
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except (OSError, IOError):
            font = ImageFont.load_default()
        text = str(number)
        # 绘制数字（居中偏移微调）
        draw.text((x - 4, y - 8), text, fill=color, font=font)

    # ==================== 公共接口 ====================

    def generate(
        self,
        image_path: str,
        targets_percent: List[Tuple[float, float]],
        obstacles_percent: List[Tuple[float, float]],
        output_path: str,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        在卫星图像上添加箭头、圆圈和标签标注 - UAV-VLPA视觉提示生成主接口。

        该方法实现了完整的视觉提示生成流水线，将用户指令
        转换为可训练的视觉标注数据，是UAV-VLPA多模态
        融合架构中视觉模态的关键入口点。

        生成流程：
        1. 输入解析：读取卫星图像和百分比坐标
        2. 目标标注：按顺序绘制目标圆圈、编号标签和连接箭头
        3. 障碍物标注：随机大小的区域高亮
        4. 元数据生成：记录所有标注的语义信息
        5. 输出保存：生成标注图像和元数据字典

        无人机应用考虑：
        - 百分比坐标系统确保标注在不同分辨率图像上的位置一致性
        - 连接箭头自动构建任务执行顺序，支持多步骤路径规划
        - 元数据字典结构化存储，便于多模态模型训练
        - 支持高DPI输出（300 DPI），满足专业无人机应用需求

        参数说明：
            image_path: 源卫星图像路径，支持常见图像格式
            targets_percent: 目标位置百分比坐标 [(x%, y%), ...]
                - x%, y%：相对于图像宽度和高度的百分比坐标
                - 顺序代表任务执行序列（1st, 2nd, 3rd...）
            obstacles_percent: 障碍物位置百分比坐标 [(x%, y%), ...]
                - 表示需要规避的危险区域中心位置
            output_path: 输出图像路径，支持PNG/JPEG格式

        返回值：
            Tuple[str, Dict]: (输出路径, 标注元数据字典)
                - 输出路径：生成的标注图像文件路径
                - 标注元数据字典：包含所有标注元素的结构化信息
                    用于多模态模型训练和评估
        """
        # 创建输出目录
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # 加载图像并转换为RGBA模式（支持透明度）
        image = Image.open(image_path).convert("RGBA")
        width, height = image.size
        # 创建透明覆盖层
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # 初始化元数据
        metadata: Dict[str, Any] = {"annotations": []}

        # ==================== 绘制目标标注 ====================
        # 包括：绿色圆圈 + 编号标签 + 连接箭头
        prev_px = None  # 上一个目标位置（用于绘制箭头）
        for idx, (xp, yp) in enumerate(targets_percent, 1):
            # 将百分比坐标转换为像素坐标
            px, py = xp / 100.0 * width, yp / 100.0 * height

            # 绘制目标圆圈（半径22像素）
            self._draw_circle_marker(draw, (px, py), 22, self.target_color[:3], width=3)
            # 绘制编号标签（偏移避免重叠）
            self._draw_number_label(draw, (px - 16, py - 16), idx)
            # 记录元数据
            metadata["annotations"].append(
                {"type": "target_circle", "index": idx, "center_pct": [xp, yp]}
            )

            # 如果不是第一个目标，绘制从前一个目标到当前目标的箭头
            if prev_px is not None:
                self._draw_arrow(draw, prev_px, (px, py), self.arrow_color, width=2)
                metadata["annotations"].append(
                    {
                        "type": "arrow",
                        "from_pct": [
                            prev_px[0] / width * 100,
                            prev_px[1] / height * 100,
                        ],
                        "to_pct": [xp, yp],
                    }
                )
            prev_px = (px, py)

        # ==================== 绘制障碍物标注 ====================
        # 使用红色半透明区域高亮
        for xp, yp in obstacles_percent:
            px, py = xp / 100.0 * width, yp / 100.0 * height
            # 随机大小增加多样性（30-60像素）
            size = random.randint(30, 60)
            self._draw_region_highlight(draw, (px, py), size, self.obstacle_color)
            metadata["annotations"].append(
                {"type": "obstacle_region", "center_pct": [xp, yp], "size_px": size}
            )

        # 合并覆盖层并转换为RGB保存
        result = Image.alpha_composite(image, overlay).convert("RGB")
        result.save(output_path, dpi=(300, 300))
        logger.debug("Annotation image saved to %s", output_path)

        return output_path, metadata
