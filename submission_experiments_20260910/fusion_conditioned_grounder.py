"""Fusion-conditioned coordinate grounders for the Exp6 fusion controls.

The original Exp6 fusion controls were evaluated behind a single, fixed V2
grounder.  Consequently, changing AFFNet/MAFTNet/SCAL could not change the
coordinates supplied to the planner.  This module fixes that experimental
boundary: each control is fine-tuned with its own fusion encoder and a shared
map-attention coordinate head on exactly the same V2 train/validation split.

The V2 OSM protocol contains a semantic map and an instruction, but not the
raw voice, gesture, or annotation files.  These experiments therefore test
the *text-conditioned fusion architecture in map grounding*, rather than a
claim that all four raw interaction modalities are available on V2.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch import nn
from torch.utils.data import Dataset

from v2_grounding import GroundingDataset, MAX_TARGETS


class FusionGroundingDataset(Dataset):
    """V2 grounding data plus fixed, local Sentence-BERT instruction vectors."""

    def __init__(self, samples: Sequence, benchmark_dir: str, text_encoder):
        self.base = GroundingDataset(samples, benchmark_dir)
        self.samples = list(samples)
        # Do not permit the project's random-embedding fallback.  A fallback
        # would make an offline run appear successful while invalidating it.
        text_encoder._load_model()
        if getattr(text_encoder, "_sbert", None) == "fallback":
            raise RuntimeError(
                "The local sentence encoder is unavailable; refusing to train "
                "a fusion-conditioned grounder with random embeddings."
            )
        vectors = []
        with torch.no_grad():
            for start in range(0, len(self.samples), 64):
                texts = [s.text_instruction for s in self.samples[start:start + 64]]
                vectors.append(text_encoder.encode_texts(texts).detach().cpu())
        self.text_embeddings = torch.cat(vectors, dim=0)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image, _tokens, coords, mask = self.base[index]
        return image, self.text_embeddings[index], coords, mask


def collate_fusion_grounding(batch):
    images, text, coords, masks = zip(*batch)
    return torch.stack(images), torch.stack(text), torch.stack(coords), torch.stack(masks)


class FusionConditionedGrounder(nn.Module):
    """Predict ordered target coordinates from an OSM map and a fusion feature."""

    def __init__(self, fuser, fusion_dim: int):
        super().__init__()
        self.fuser = fuser
        self.fusion_dim = fusion_dim
        self.image = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.GELU(),
            nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.MaxPool2d(2), nn.Conv2d(64, 128, 3, padding=1), nn.GELU(),
        )
        self.map_projection = nn.Linear(128, fusion_dim)
        self.position = nn.Sequential(
            nn.Linear(2, fusion_dim), nn.GELU(), nn.Linear(fusion_dim, fusion_dim),
        )
        self.slot = nn.Embedding(MAX_TARGETS, fusion_dim)
        self.query = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim), nn.GELU(), nn.LayerNorm(fusion_dim),
        )

    def forward(self, images, text_embeddings):
        fused = self.fuser(text_emb=text_embeddings)
        batch = images.shape[0]
        slots = self.slot.weight.unsqueeze(0).expand(batch, -1, -1)
        query = self.query(torch.cat([
            fused.unsqueeze(1).expand(-1, MAX_TARGETS, -1), slots,
        ], dim=-1))
        features = self.image(images)
        _, _, height, width = features.shape
        ys = torch.linspace(0.0, 1.0, height, device=images.device)
        xs = torch.linspace(0.0, 1.0, width, device=images.device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2)
        spatial = self.map_projection(features.flatten(2).transpose(1, 2))
        spatial = spatial + self.position(grid).unsqueeze(0)
        scores = torch.einsum("bqc,bnc->bqn", query, spatial) / self.fusion_dim ** 0.5
        attention = torch.softmax(scores, dim=-1)
        return torch.einsum("bqn,nc->bqc", attention, grid)

    def predict_sample(self, sample, benchmark_dir: str, device):
        """GroundedPlanner hook; returns percentage coordinates without labels."""
        image, _tokens, _coords, _mask = GroundingDataset([sample], benchmark_dir)[0]
        encoder = self.fuser.text_encoder
        encoder._load_model()
        if getattr(encoder, "_sbert", None) == "fallback":
            raise RuntimeError("Sentence encoder fallback detected during inference")
        with torch.no_grad():
            text = encoder.encode_texts([sample.text_instruction]).to(device)
            prediction = self(image.unsqueeze(0).to(device), text)[0].cpu()
        return [
            (float(x * 100.0), float(y * 100.0))
            for x, y in prediction[:len(sample.targets)]
        ]


def coordinate_loss(prediction, coordinates, mask):
    return ((prediction - coordinates).square().sum(-1)[mask]).mean()


def train_epoch(model, loader, optimizer, device):
    model.train(); total = 0.0
    for images, text, coordinates, mask in loader:
        images, text, coordinates, mask = (
            item.to(device) for item in (images, text, coordinates, mask)
        )
        loss = coordinate_loss(model(images, text), coordinates, mask)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total += float(loss.detach())
    return total / max(len(loader), 1)


def evaluate_loss(model, loader, device):
    model.eval(); total = 0.0
    with torch.no_grad():
        for images, text, coordinates, mask in loader:
            images, text, coordinates, mask = (
                item.to(device) for item in (images, text, coordinates, mask)
            )
            total += float(coordinate_loss(model(images, text), coordinates, mask))
    return total / max(len(loader), 1)
