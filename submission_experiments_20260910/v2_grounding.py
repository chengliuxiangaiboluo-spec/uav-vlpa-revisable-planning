"""A trainable visual-language coordinate grounder for the V2-OSM protocol.

The previous workflow supplied scenario target coordinates directly to every
planner.  This module makes coordinate prediction an explicit learned stage.
It is intentionally a lightweight visual-language grounder, not a claimed
Molmo/PaLM-E reproduction.  Target count/types remain supplied by the task
parser; only their positions must be inferred from image plus instruction.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms

from data.scenario_schema import ScenarioSample

MAX_TARGETS = 8
VOCAB_SIZE = 4096


def _tokens(text: str):
    return [int(hashlib.sha1(x.encode("utf-8")).hexdigest(), 16) % VOCAB_SIZE
            for x in text.lower().replace(",", " ").replace(".", " ").split()]


class GroundingDataset(Dataset):
    def __init__(self, samples: Sequence[ScenarioSample], benchmark_dir: str):
        self.samples, self.images = list(samples), Path(benchmark_dir) / "images"
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        semantic_path = sample.metadata.get("grounding_image_path")
        if not semantic_path:
            raise ValueError(
                f"{sample.scenario_id} has no audited grounding_image_path; "
                "random-coordinate V1 samples are forbidden in V2-OSM"
            )
        image_path = Path(semantic_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing semantic mission map: {image_path}")
        image = Image.open(image_path).convert("RGB")
        coords = torch.zeros(MAX_TARGETS, 2, dtype=torch.float32)
        mask = torch.zeros(MAX_TARGETS, dtype=torch.bool)
        for i, target in enumerate(sample.targets[:MAX_TARGETS]):
            coords[i] = torch.tensor(target.coordinates_percent, dtype=torch.float32) / 100.0
            mask[i] = True
        token_ids = _tokens(sample.text_instruction)[:64] or [0]
        return self.transform(image), torch.tensor(token_ids), coords, mask


def collate_grounding(batch):
    images, token_lists, coords, masks = zip(*batch)
    longest = max(x.numel() for x in token_lists)
    tokens = torch.zeros(len(batch), longest, dtype=torch.long)
    valid = torch.zeros(len(batch), longest, dtype=torch.float32)
    for row, token_ids in enumerate(token_lists):
        tokens[row, :token_ids.numel()] = token_ids
        valid[row, :token_ids.numel()] = 1.0
    return torch.stack(images), tokens, valid, torch.stack(coords), torch.stack(masks)


class VisualLanguageGrounder(nn.Module):
    def __init__(self, use_image: bool = True):
        super().__init__()
        self.use_image = use_image
        self.text = nn.Embedding(VOCAB_SIZE, 128)
        self.text_proj = nn.Sequential(
            nn.Linear(128, 128), nn.GELU(), nn.LayerNorm(128),
        )
        self.image = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.GELU(),
            nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.MaxPool2d(2), nn.Conv2d(64, 128, 3, padding=1), nn.GELU(),
        )
        self.position = nn.Sequential(nn.Linear(2, 128), nn.GELU(), nn.Linear(128, 128))
        self.slot = nn.Embedding(MAX_TARGETS, 128)
        self.query = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.LayerNorm(128))
        self.text_only_head = nn.Sequential(
            nn.Linear(256, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 2),
        )

    def forward(self, images, tokens, valid):
        text = self.text(tokens)
        text = (text * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True).clamp_min(1.0)
        text = self.text_proj(text)
        batch = images.shape[0]
        slots = self.slot.weight.unsqueeze(0).expand(batch, -1, -1)
        text_slots = text.unsqueeze(1).expand(-1, MAX_TARGETS, -1)
        query = self.query(torch.cat([text_slots, slots], dim=-1))
        if not self.use_image:
            return torch.sigmoid(self.text_only_head(
                torch.cat([text_slots, slots], dim=-1)
            ))

        features = self.image(images)
        _, channels, height, width = features.shape
        ys = torch.linspace(0.0, 1.0, height, device=features.device)
        xs = torch.linspace(0.0, 1.0, width, device=features.device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2)
        spatial = features.flatten(2).transpose(1, 2)
        spatial = spatial + self.position(grid).unsqueeze(0)
        scores = torch.einsum("bqc,bnc->bqn", query, spatial) / channels ** 0.5
        attention = torch.softmax(scores, dim=-1)
        return torch.einsum("bqn,nc->bqc", attention, grid)


def train_epoch(model, loader, optimizer, device):
    model.train(); total = 0.0
    for images, tokens, valid, coords, mask in loader:
        images, tokens, valid, coords, mask = (x.to(device) for x in (images, tokens, valid, coords, mask))
        prediction = model(images, tokens, valid)
        loss = ((prediction - coords).square().sum(-1)[mask]).mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total += float(loss.detach())
    return total / max(len(loader), 1)


def evaluate_loss(model, loader, device):
    """Validation loss used for checkpoint selection; never updates weights."""
    model.eval(); total = 0.0
    with torch.no_grad():
        for images, tokens, valid, coords, mask in loader:
            images, tokens, valid, coords, mask = (
                x.to(device) for x in (images, tokens, valid, coords, mask)
            )
            prediction = model(images, tokens, valid)
            loss = ((prediction - coords).square().sum(-1)[mask]).mean()
            total += float(loss)
    return total / max(len(loader), 1)
