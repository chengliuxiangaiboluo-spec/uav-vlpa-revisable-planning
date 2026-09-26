"""
手势编码器模块 - 使用预训练ResNet18骨干网络。

本模块实现基于预训练ResNet的手势图像编码器，
用于从带有手势标注的卫星图像中提取视觉特征。

主要功能:
    1. 加载预训练的ResNet骨干网络
    2. 提取512维图像特征向量
    3. 通过投影层映射到融合维度
    4. 可选的骨干网络微调

使用模型:
    ResNet18/ResNet34 (ImageNet预训练)
    - 特征维度: 512
    - 特点: 轻量级、高效能的图像编码器

参考文献:
    He et al., 2016. "Deep Residual Learning for Image Recognition"
"""

# PyTorch深度学习框架
import torch  # 张量计算核心库
import torch.nn as nn  # 神经网络模块
import logging  # 日志记录模块

# 获取日志记录器
logger = logging.getLogger("experiment")


class GestureEncoder(nn.Module):
    """
    基于ResNet的手势图像编码器。

    该编码器使用预训练的ResNet骨干网络从带有手势标注的
    卫星图像中提取视觉特征，并通过投影层映射到融合维度。

    属性:
        proj_dim (int): 投影后的特征维度
        features (nn.Sequential): ResNet骨干网络（去除最后FC层）
        projection (nn.Sequential): 投影层（512 → proj_dim）

    工作流程:
        1. 加载预训练ResNet骨干
        2. 可选冻结部分层（仅训练layer4）
        3. 提取图像特征向量
        4. 投影到融合维度

    Note:
        输入图像应经过ImageNet标准化预处理
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        proj_dim: int = 512,
        freeze_backbone: bool = True,
    ):
        """
        初始化手势编码器。

        Args:
            backbone: 骨干网络类型，可选"resnet18"或"resnet34"
            proj_dim: 投影后的特征维度（用于多模态融合）
            freeze_backbone: 是否冻结骨干网络参数
                            True: 仅训练投影层，保留预训练特征
                            False: 微调整个网络
        """
        super().__init__()  # 调用父类构造函数
        self.proj_dim = proj_dim  # 保存投影维度

        import torchvision.models as models  # 导入torchvision模型

        # 根据骨干类型选择预训练模型
        if backbone == "resnet18":
            base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            feat_dim = 512  # ResNet18输出特征维度
        elif backbone == "resnet34":
            base = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            feat_dim = 512  # ResNet34输出特征维度
        else:
            # 默认使用ResNet18
            base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            feat_dim = 512

        # 移除最后的全连接层，只保留特征提取部分
        # ResNet的结构：conv → bn → relu → maxpool → layer1-4 → avgpool → fc
        # 我们取到avgpool之前的层
        self.features = nn.Sequential(*list(base.children())[:-1])

        # 冻结骨干网络参数（如果需要）
        if freeze_backbone:
            for name, param in self.features.named_parameters():
                # 仅解冻最后一个残差块(layer4)，保留低层特征
                if "layer4" not in name:
                    param.requires_grad = False  # 禁止梯度更新

        # 投影层：展平 → 线性变换 → 层归一化
        self.projection = nn.Sequential(
            nn.Flatten(),  # 展平特征图: (batch, 512, 1, 1) → (batch, 512)
            nn.Linear(feat_dim, proj_dim),  # 线性投影: 512 → proj_dim
            nn.LayerNorm(proj_dim),  # 层归一化，稳定特征分布
        )
        logger.info("GestureEncoder initialised (%s, proj=%d).", backbone, proj_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        对手势标注图像进行编码。

        前向传播流程:
            1. 通过ResNet骨干提取特征
            2. 通过投影层映射到融合维度

        Args:
            images: 归一化后的图像张量，shape为(batch, 3, H, W)
                   应使用ImageNet均值和标准差进行归一化

        Returns:
            投影后的特征张量，shape为(batch, proj_dim)

        Note:
            推荐输入尺寸: 224×224（与ImageNet预训练一致）
        """
        feat = self.features(images)   # 提取特征: (batch, 512, 1, 1)
        return self.projection(feat)   # 投影: (batch, proj_dim)

    @staticmethod
    def preprocess(image) -> torch.Tensor:
        """
        对PIL图像应用标准ImageNet预处理变换。

        预处理流程:
            1. 调整尺寸至224×224
            2. 转换为PyTorch张量
            3. 使用ImageNet均值和标准差归一化

        Args:
            image: PIL图像对象

        Returns:
            预处理后的图像张量，shape为(1, 3, 224, 224)

        Note:
            ImageNet归一化参数:
            - 均值: [0.485, 0.456, 0.406] (RGB)
            - 标准差: [0.229, 0.224, 0.225] (RGB)
        """
        from torchvision import transforms

        # 定义预处理变换流水线
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 调整尺寸
            transforms.ToTensor(),  # 转换为张量，值域[0, 1]
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet RGB均值
                std=[0.229, 0.224, 0.225],  # ImageNet RGB标准差
            ),
        ])
        return transform(image).unsqueeze(0)  # 添加batch维度: (1, 3, 224, 224)
