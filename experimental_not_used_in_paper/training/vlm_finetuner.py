"""
EXPERIMENTAL ONLY -- NOT USED IN THE ASSOCIATED PAPER.

VLM fine-tuner using LoRA (PEFT).

Fine-tunes the Molmo-7B-O vision-language model on the multimodal
scenario dataset to improve coordinate extraction from instructions. The
reported experiments instead use frozen 4-bit Molmo-7B-O inference and do not
invoke this prototype.
"""

import os
import logging
from typing import List, Dict

import torch
from torch.utils.data import Dataset, DataLoader

# Setup offline environment FIRST
from utils.offline_config import setup_offline_environment
setup_offline_environment()

from data.scenario_schema import ScenarioSample
from configs.experiment_config import TrainingConfig, ModelConfig
from training.training_utils import EarlyStopping, CheckpointManager

logger = logging.getLogger("experiment")


class InstructionDataset(Dataset):
    """
    Dataset pairing (text instruction, expected coordinate XML output).

    The target XML mimics the format produced by Molmo VLM:
    <points x1="..." y1="..." alt="building">...</points>
    """

    def __init__(self, samples: List[ScenarioSample]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # Build expected XML from target coordinates
        attrs = []
        types = set()
        for i, t in enumerate(s.targets, 1):
            attrs.append(f'x{i}="{t.coordinates_percent[0]}" y{i}="{t.coordinates_percent[1]}"')
            types.add(t.target_type)

        type_str = ", ".join(sorted(types)) if types else "building"
        xml_output = f'<points {" ".join(attrs)} alt="{type_str}">{type_str}</points>'

        return {
            "input_text": s.text_instruction,
            "target_text": xml_output,
            "image_id": s.image_id,
        }


class VLMFineTuner:
    """
    LoRA fine-tuner for the Molmo-7B-O VLM.

    Uses PEFT for parameter-efficient training, targeting the query
    and value projection layers.
    采用 FP16 模式以避开 bitsandbytes 环境冲突。
    """

    def __init__(
        self,
        model_cfg: ModelConfig,
        train_cfg: TrainingConfig,
        device: str = "cpu",
    ):
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.device = device
        self.ckpt_mgr = CheckpointManager(train_cfg.checkpoint_dir)
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """Load the base VLM and apply LoRA adapters."""
        if self._model is not None:
            return

        try:
            # 这里的导入保持不变
            from transformers import AutoModelForVision2Seq, AutoTokenizer
            from peft import LoraConfig, get_peft_model, TaskType

            logger.info("Loading VLM (FP16 Mode): %s ...", self.model_cfg.vlm_model)

            # 强制本地加载分词器
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_cfg.vlm_model,
                trust_remote_code=True,
                local_files_only=True,   # 关键：禁止联网
            )

            # --- 核心修改点：强制 FP16 加载，不使用任何量化参数，且强制本地文件 ---
            base_model = AutoModelForVision2Seq.from_pretrained(
                self.model_cfg.vlm_model,
                trust_remote_code=True,
                torch_dtype=torch.float16,    # 显式指定半精度
                device_map="auto",            # 自动分配 GPU
                low_cpu_mem_usage=True,       # 优化内存占用
                local_files_only=True,        # 关键：禁止联网
            )

            lora_config = LoraConfig(
                r=self.train_cfg.lora_rank,
                lora_alpha=self.train_cfg.lora_alpha,
                lora_dropout=self.train_cfg.lora_dropout,
                target_modules=self.train_cfg.lora_target_modules,
                task_type=TaskType.CAUSAL_LM,
            )

            self._model = get_peft_model(base_model, lora_config)

            # 确保模型在正确的设备上并开启训练模式
            self._model.to(self.device)

            trainable = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self._model.parameters())
            logger.info(
                "LoRA applied. Trainable: %d / %d (%.2f%%)",
                trainable, total, 100 * trainable / total,
            )
        except Exception as exc:
            logger.error("Failed to load VLM: %s", exc)
            raise

    def train(
        self,
        train_samples: List[ScenarioSample],
        val_samples: List[ScenarioSample],
    ) -> Dict[str, List[float]]:
        """
        Run LoRA fine-tuning on the VLM.
        """
        self._load_model()

        train_ds = InstructionDataset(train_samples)
        val_ds = InstructionDataset(val_samples)
        train_loader = DataLoader(
            train_ds, batch_size=self.train_cfg.vlm_batch_size, shuffle=True
        )
        val_loader = DataLoader(val_ds, batch_size=self.train_cfg.vlm_batch_size)

        optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=self.train_cfg.vlm_lr
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.train_cfg.vlm_epochs
        )
        stopper = EarlyStopping(patience=self.train_cfg.early_stopping_patience)

        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        grad_accum = self.train_cfg.vlm_gradient_accumulation

        for epoch in range(1, self.train_cfg.vlm_epochs + 1):
            self._model.train()
            epoch_loss = 0.0
            optimizer.zero_grad()

            for step, batch in enumerate(train_loader, 1):
                # 文本编码
                inputs = self._tokenizer(
                    batch["input_text"],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )
                labels = self._tokenizer(
                    batch["target_text"],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=256,
                )

                input_ids = inputs.input_ids.to(self.device)
                label_ids = labels.input_ids.to(self.device)

                # 拼接 Input + Target
                combined = torch.cat([input_ids, label_ids], dim=1)
                lm_labels = combined.clone()
                lm_labels[:, : input_ids.size(1)] = -100  # 屏蔽掉 input 的 loss

                # 前向传播
                outputs = self._model(input_ids=combined, labels=lm_labels)
                loss = outputs.loss / grad_accum
                loss.backward()

                epoch_loss += loss.item() * grad_accum

                if step % grad_accum == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            avg_train = epoch_loss / max(len(train_loader), 1)

            # Validation
            self._model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    v_inputs = self._tokenizer(
                        batch["input_text"], return_tensors="pt", padding=True, truncation=True, max_length=512
                    )
                    v_labels = self._tokenizer(
                        batch["target_text"], return_tensors="pt", padding=True, truncation=True, max_length=256
                    )

                    v_input_ids = v_inputs.input_ids.to(self.device)
                    v_label_ids = v_labels.input_ids.to(self.device)

                    combined = torch.cat([v_input_ids, v_label_ids], dim=1)
                    lm_labels = combined.clone()
                    lm_labels[:, : v_input_ids.size(1)] = -100

                    outputs = self._model(input_ids=combined, labels=lm_labels)
                    val_loss += outputs.loss.item()

            avg_val = val_loss / max(len(val_loader), 1)
            scheduler.step()

            history["train_loss"].append(avg_train)
            history["val_loss"].append(avg_val)

            logger.info(
                "VLM epoch %d/%d  train_loss=%.4f  val_loss=%.4f",
                epoch, self.train_cfg.vlm_epochs, avg_train, avg_val,
            )

            if stopper(avg_val):
                logger.info("VLM early stopping at epoch %d.", epoch)
                break

        # 保存结果
        adapter_path = os.path.join(self.train_cfg.checkpoint_dir, "vlm_lora_adapter")
        self._model.save_pretrained(adapter_path)
        logger.info("VLM LoRA adapter saved to %s", adapter_path)

        return history
