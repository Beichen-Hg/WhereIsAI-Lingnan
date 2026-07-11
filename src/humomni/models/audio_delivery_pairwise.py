"""Torch pairwise audio-delivery scorer."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class AudioDeliveryPairwiseMLP(nn.Module):
    """MLP that predicts whether candidate A is preferred over candidate B."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dims: Sequence[int] = (256, 128),
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        layers: list[nn.Module] = []
        current = int(input_dim)
        for hidden_dim in hidden_dims:
            hidden = int(hidden_dim)
            if hidden <= 0:
                continue
            layers.extend(
                [
                    nn.Linear(current, hidden),
                    nn.LayerNorm(hidden),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                ]
            )
            current = hidden
        layers.append(nn.Linear(current, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.network(features)
        return logits.squeeze(-1)


def pairwise_bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    pos_weight: float | None = None,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    labels = labels.float()
    pos_weight_tensor = None
    if pos_weight is not None:
        pos_weight_tensor = torch.as_tensor(float(pos_weight), dtype=logits.dtype, device=logits.device)
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits,
        labels,
        pos_weight=pos_weight_tensor,
        reduction="none",
    )
    if sample_weight is not None:
        sample_weight = sample_weight.to(device=logits.device, dtype=logits.dtype)
        loss = loss * sample_weight
        denominator = sample_weight.sum().clamp_min(1e-8)
        return loss.sum() / denominator
    return loss.mean()
