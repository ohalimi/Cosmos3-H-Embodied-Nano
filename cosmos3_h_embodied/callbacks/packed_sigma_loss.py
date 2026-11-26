# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import torch

from cosmos_framework.callbacks.sigma_loss_analysis import SigmaLossAnalysis
from cosmos_framework.model._base import ImaginaireModel


class PackedSigmaLossAnalysis(SigmaLossAnalysis):
    """Skip local vision statistics when a packed batch has no per-item losses."""

    def on_training_step_end(
        self, model: ImaginaireModel, data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor], loss: torch.Tensor, iteration: int = 0,
    ) -> None:
        sigma = output_batch["sigma"]  # [B], [B,1], or [B,T]
        key = "flow_matching_loss_vision_per_instance"
        per_item = output_batch.get(key)
        if per_item is None or (sigma.ndim in (1, 2) and per_item.shape != (sigma.shape[0],)):
            # A fully conditioned stream can return one graph-preserving dummy
            # loss for many vision items. Never repeat it into the statistics.
            # Use the parent's empty-vision path, not an early return: even a
            # rank without pairs has to participate in both scheduled gathers.
            empty = sigma.new_empty(0)
            output_batch = {**output_batch, "sigma": empty, key: empty}
        super().on_training_step_end(model, data_batch, output_batch, loss, iteration)
