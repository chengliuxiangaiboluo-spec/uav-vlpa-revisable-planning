"""
融合模块训练器模块。

本模块实现多模态融合器（MultimodalFuser）的训练，
使用对比对齐损失和MSE重建损失在合成多模态场景数据上进行训练。

核心训练策略:
    1. 模态丢弃训练（Modality Dropout）：随机丢弃部分模态，让模型学会区分单模态和多模态输入
    2. 对比对齐损失：确保不同模态的特征在语义空间中对齐
    3. MSE重建损失：重建文本嵌入作为监督信号
    4. 模态置信度损失：监督模型输出的modality_bias与实际模态数量匹配

技术特点:
    - 支持多模态数据加载（文本、音频、手势、标注）
    - 动态学习率调度（余弦退火）
    - 早停机制防止过拟合
    - 检查点管理

参考文献:
    Radford et al., 2021. "Learning Transferable Visual Models From Natural Language Supervision"
    Baltrusaitis et al., 2019. "Multimodal Machine Learning: A Survey and Taxonomy"
"""

# 标准库导入
import logging  # 日志记录模块
import time     # 墙钟时间，用于 Slurm 进度输出
import random   # 随机数生成模块
from pathlib import Path
from typing import List, Dict  # 类型提示支持

# PyTorch深度学习框架
import torch        # 张量计算核心库
import torch.nn as nn  # 神经网络模块
import torch.nn.functional as F  # 函数式接口
from torch.utils.data import Dataset, DataLoader  # 数据集和数据加载器

# 项目模块导入
from data.scenario_schema import ScenarioSample, ModalityType  # 场景数据结构
from models.fusion.multimodal_fuser import MultimodalFuser  # 多模态融合器
from models.encoders.text_encoder import TextEncoder  # 文本编码器
from training.training_utils import EarlyStopping, CheckpointManager  # 训练工具
from configs.experiment_config import TrainingConfig  # 训练配置

# 获取实验日志记录器
logger = logging.getLogger("experiment")


class ScenarioDataset(Dataset):
    """
    场景数据集类。

    该类包装ScenarioSample列表，用于多模态融合器的训练。

    主要功能:
        - 加载文本指令并编码为嵌入向量
        - 提供模态数量信息作为监督信号
        - 支持索引访问和批处理
    """

    def __init__(self, samples: List[ScenarioSample]):
        """
        初始化场景数据集。

        Args:
            samples: 场景样本列表
        """
        self.samples = samples
        self._text_encoder = TextEncoder()  # 文本编码器实例

    def __len__(self):
        """返回数据集大小。"""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        获取指定索引的数据样本。

        Args:
            idx: 样本索引

        Returns:
            sample_dict: 包含文本嵌入、场景ID、复杂度、模态数量等信息的字典
        """
        s = self.samples[idx]
        # 获取文本嵌入
        text_emb = self._text_encoder.encode_texts([s.text_instruction])[0]
        # 模态数量（用于监督信号）
        n_modalities = len(s.modalities)
        return {
            "text_emb": text_emb,
            "scenario_id": s.scenario_id,
            "complexity": s.complexity.value,
            "n_modalities": n_modalities,
            "idx": idx,  # 用索引来获取样本，避免 collate 问题
        }


class FusionModuleTrainer:
    """
    多模态融合器训练器类。

    该类负责训练多模态融合器，使用对比对齐损失和MSE重建损失，
    并引入模态置信度损失来监督模型的模态理解能力。

    主要组件:
        - fuser: 多模态融合器实例
        - ckpt_mgr: 检查点管理器
        - optimizer: 优化器
        - scheduler: 学习率调度器
        - stopper: 早停检查器
    """

    def __init__(
        self,
        config: TrainingConfig,
        fuser: MultimodalFuser,
        device: str = "cpu",
    ):
        """
        初始化融合模块训练器。

        Args:
            config: 训练配置对象
            fuser: 多模态融合器实例
            device: 计算设备（cpu/cuda）
        """
        self.config = config
        self.fuser = fuser.to(device)  # 将模型移动到指定设备
        self.device = device
        self.ckpt_mgr = CheckpointManager(config.checkpoint_dir)  # 检查点管理器

    def _load_modality_tensors(self, sample: ScenarioSample):
        """
        加载多模态张量。

        该方法根据场景样本中的模态类型，加载相应的音频、手势和标注图像张量。

        Args:
            sample: 场景样本对象

        Returns:
            audio_values: 音频波形张量（可选）
            gesture_images: 手势图像张量（可选）
            annotation_images: 标注图像张量（可选）
        """
        audio_values = None
        gesture_images = None
        annotation_images = None

        # 加载音频
        if ModalityType.VOICE in sample.modalities and sample.audio_path:
            try:
                import torchaudio
                waveform, sr = torchaudio.load(sample.audio_path)
                if sr != 16000:
                    resampler = torchaudio.transforms.Resample(sr, 16000)
                    waveform = resampler(waveform)
                waveform = waveform[0:1, :48000]
                if waveform.shape[1] < 48000:
                    waveform = F.pad(waveform, (0, 48000 - waveform.shape[1]))
                audio_values = waveform
            except Exception:
                pass

        # 加载手势图像
        if ModalityType.GESTURE in sample.modalities and sample.gesture_image_path:
            try:
                from torchvision import transforms
                from PIL import Image
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
                ])
                img = Image.open(sample.gesture_image_path).convert("RGB")
                gesture_images = transform(img).unsqueeze(0)
            except Exception:
                pass

        # 加载标注图像
        if ModalityType.ANNOTATION in sample.modalities and sample.annotation_image_path:
            try:
                from torchvision import transforms
                from PIL import Image
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
                ])
                img = Image.open(sample.annotation_image_path).convert("RGB")
                annotation_images = transform(img).unsqueeze(0)
            except Exception:
                pass

        return audio_values, gesture_images, annotation_images

    def train(
        self,
        train_samples: List[ScenarioSample],
        val_samples: List[ScenarioSample],
    ) -> Dict[str, List[float]]:
        """
        运行完整的训练循环，包括模态丢弃训练。

        训练流程:
            1. 构建训练和验证数据集
            2. 初始化优化器和学习率调度器
            3. 执行多轮训练和验证
            4. 保存最佳检查点
            5. 应用早停机制

        模态丢弃训练策略:
            - 随机丢弃部分模态，模拟单模态/少模态情况
            - 监督信号 = 实际使用模态数 / 最大模态数
            - 让模型学会：多模态 → 高 bias，单模态 → 低 bias

        Args:
            train_samples: 训练样本列表
            val_samples: 验证样本列表

        Returns:
            history: 包含训练和验证损失历史的字典
        """
        started_at = time.monotonic()
        print(
            "[Fusion] start | train=%d | val=%d | epochs=%d | batch_size=%d | device=%s"
            % (len(train_samples), len(val_samples), self.config.fusion_epochs,
               self.config.fusion_batch_size, self.device),
            flush=True,
        )
        train_ds = ScenarioDataset(train_samples)
        val_ds = ScenarioDataset(val_samples)

        train_loader = DataLoader(
            train_ds, batch_size=self.config.fusion_batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.config.fusion_batch_size
        )

        optimizer = torch.optim.AdamW(
            self.fuser.parameters(),
            lr=self.config.fusion_lr,
            weight_decay=self.config.fusion_weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.fusion_epochs
        )
        stopper = EarlyStopping(patience=self.config.early_stopping_patience)

        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        # Submission evaluation only needs one unambiguous final Fusion model.
        # Retaining a full model+optimizer checkpoint after every epoch consumed
        # tens of GB across concurrent Slurm jobs and caused filesystem write
        # failures.  Keep only the best model weights instead.
        best_checkpoint_loss = float("inf")

        # 最大模态数量（用于归一化 bias 监督信号）
        max_modalities = 4  # text + audio + gesture + annotation

        for epoch in range(1, self.config.fusion_epochs + 1):
            # --- PHASE: TRAIN ---
            self.fuser.train()
            epoch_train_loss = 0.0

            for batch in train_loader:
                text_emb = batch["text_emb"].to(self.device)
                n_modalities = batch["n_modalities"]  # 场景原始模态数量
                indices = batch["idx"]  # 样本索引

                optimizer.zero_grad()

                # === 模态丢弃训练 ===
                # 随机丢弃模态，让模型学习区分单模态和多模态
                batch_size = text_emb.size(0)
                total_loss = 0.0

                for i in range(batch_size):
                    idx = indices[i].item() if hasattr(indices[i], 'item') else indices[i]
                    sample = train_samples[idx]  # 通过索引获取样本
                    n_mod = n_modalities[i].item() if hasattr(n_modalities[i], 'item') else n_modalities[i]

                    # 加载多模态数据
                    audio_values, gesture_images, annotation_images = \
                        self._load_modality_tensors(sample)

                    # 模态丢弃：随机丢弃部分模态
                    dropout_prob = random.random()
                    if dropout_prob > 0.5 and n_mod > 1:
                        # 丢弃模态（模拟单模态情况）
                        audio_values = None
                        gesture_images = None
                        annotation_images = None
                        actual_modality_count = 1  # 只有文本
                    else:
                        # 保留所有模态
                        actual_modality_count = n_mod

                    # 移动到设备
                    if audio_values is not None:
                        audio_values = audio_values.to(self.device)
                    if gesture_images is not None:
                        gesture_images = gesture_images.to(self.device)
                    if annotation_images is not None:
                        annotation_images = annotation_images.to(self.device)

                    # 前向传播，返回 bias
                    text_emb_i = text_emb[i:i+1]
                    fused, bias = self.fuser(
                        text_emb=text_emb_i,
                        audio_values=audio_values,
                        gesture_images=gesture_images,
                        annotation_images=annotation_images,
                        return_bias=True,
                    )

                    # === 损失计算 ===

                    # 1. 多模态锚点对齐损失（余弦相似度）
                    # 关键改进：锚点 = 所有可用模态特征的平均，而非纯文本
                    # 这样融合结果对齐的是"多模态中心"，不会被强制拟合纯文本
                    # 单模态时锚点 ≈ text_feat；多模态时锚点包含所有模态信息
                    anchor_features = [self.fuser.text_encoder(text_emb_i)]
                    if audio_values is not None:
                        anchor_features.append(self.fuser.audio_encoder(audio_values))
                    if gesture_images is not None:
                        anchor_features.append(self.fuser.gesture_encoder(gesture_images))
                    if annotation_images is not None:
                        anchor_features.append(self.fuser.annotation_encoder(annotation_images))
                    anchor = torch.stack(anchor_features, dim=0).mean(dim=0)  # (B, D)
                    cosine_sim = F.cosine_similarity(fused, anchor, dim=-1).mean()
                    recon_loss = 1.0 - cosine_sim

                    # 2. 模态置信度损失（关键：让模型学会区分单模态和多模态）
                    # 监督信号 = 实际模态数 / 最大模态数
                    target_bias = actual_modality_count / max_modalities
                    target_bias = torch.tensor([[target_bias]], device=self.device)
                    bias_loss = F.mse_loss(bias, target_bias)

                    # 总损失
                    loss = recon_loss + 0.5 * bias_loss
                    total_loss = total_loss + loss

                # 平均损失
                avg_loss = total_loss / batch_size
                avg_loss.backward()
                optimizer.step()
                epoch_train_loss += avg_loss.item()

            avg_train = epoch_train_loss / max(len(train_loader), 1)

            # --- PHASE: VALIDATE ---
            self.fuser.eval()
            epoch_val_loss = 0.0

            with torch.no_grad():
                for batch in val_loader:
                    text_emb = batch["text_emb"].to(self.device)
                    n_modalities = batch["n_modalities"]
                    indices = batch["idx"]  # 样本索引

                    batch_size = text_emb.size(0)
                    total_loss = 0.0

                    for i in range(batch_size):
                        idx = indices[i].item() if hasattr(indices[i], 'item') else indices[i]
                        sample = val_samples[idx]  # 通过索引获取样本
                        n_mod = n_modalities[i].item() if hasattr(n_modalities[i], 'item') else n_modalities[i]

                        # 加载多模态数据（验证时不丢弃）
                        audio_values, gesture_images, annotation_images = \
                            self._load_modality_tensors(sample)

                        if audio_values is not None:
                            audio_values = audio_values.to(self.device)
                        if gesture_images is not None:
                            gesture_images = gesture_images.to(self.device)
                        if annotation_images is not None:
                            annotation_images = annotation_images.to(self.device)

                        text_emb_i = text_emb[i:i+1]
                        fused, bias = self.fuser(
                            text_emb=text_emb_i,
                            audio_values=audio_values,
                            gesture_images=gesture_images,
                            annotation_images=annotation_images,
                            return_bias=True,
                        )

                        # 损失计算
                        # 多模态锚点 = 所有可用模态特征的平均
                        v_anchor_features = [self.fuser.text_encoder(text_emb_i)]
                        if audio_values is not None:
                            v_anchor_features.append(self.fuser.audio_encoder(audio_values))
                        if gesture_images is not None:
                            v_anchor_features.append(self.fuser.gesture_encoder(gesture_images))
                        if annotation_images is not None:
                            v_anchor_features.append(self.fuser.annotation_encoder(annotation_images))
                        v_anchor = torch.stack(v_anchor_features, dim=0).mean(dim=0)
                        v_cosine_sim = F.cosine_similarity(fused, v_anchor, dim=-1).mean()
                        v_recon_loss = 1.0 - v_cosine_sim
                        target_bias = n_mod / max_modalities
                        target_bias = torch.tensor([[target_bias]], device=self.device)
                        v_bias_loss = F.mse_loss(bias, target_bias)

                        v_loss = v_recon_loss + 0.5 * v_bias_loss
                        total_loss = total_loss + v_loss

                    avg_val_batch = total_loss / batch_size
                    epoch_val_loss += avg_val_batch.item()

            avg_val = epoch_val_loss / max(len(val_loader), 1)

            # 更新学习率
            scheduler.step()

            history["train_loss"].append(avg_train)
            history["val_loss"].append(avg_val)

            # 日志记录
            if epoch % self.config.log_interval == 0 or epoch == 1:
                logger.info(
                    "Fusion Epoch [%d/%d] | Train Loss: %.4f | Val Loss: %.4f",
                    epoch, self.config.fusion_epochs, avg_train, avg_val
                )
                print(
                    "[Fusion] epoch %d/%d complete | train_loss=%.4f | val_loss=%.4f | elapsed=%.1f min"
                    % (epoch, self.config.fusion_epochs, avg_train, avg_val,
                       (time.monotonic() - started_at) / 60.0),
                    flush=True,
                )

            # Save only when validation loss improves.  The optimizer state is
            # not needed for downstream evaluation, so omitting it reduces the
            # checkpoint size and avoids exhausting shared Slurm storage.
            if avg_val < best_checkpoint_loss:
                best_checkpoint_loss = avg_val
                checkpoint_path = Path(self.ckpt_mgr.save(
                    self.fuser,
                    None,
                    epoch,
                    {"train_loss": avg_train, "val_loss": avg_val},
                    name="best_fusion_model",
                ))
                for stale_path in checkpoint_path.parent.glob("best_fusion_model_epoch*.pt"):
                    if stale_path != checkpoint_path:
                        stale_path.unlink()
                print(
                    "[Fusion] best checkpoint saved | epoch=%d | val_loss=%.4f | path=%s"
                    % (epoch, avg_val, checkpoint_path.name),
                    flush=True,
                )

            # 早停检查
            if stopper(avg_val):
                logger.info("Early stopping triggered at epoch %d.", epoch)
                print("[Fusion] early stopping | epoch=%d" % epoch, flush=True)
                break

        print(
            "[Fusion] complete | epochs=%d | elapsed=%.1f min"
            % (len(history["train_loss"]), (time.monotonic() - started_at) / 60.0),
            flush=True,
        )
        return history
