# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Validation callback that accepts a packed batch.

The framework logger asserts that ``dataset_name`` is a list of length 1.
This loader packs several samples into one batch, so that list is longer and
the assert stops validation before ``val/loss`` is logged. ``val/loss`` is
logged by the main training callback. This object keeps the callback slot.
"""

from __future__ import annotations

from cosmos_framework.utils.callback import Callback


class PackedBatchValLog(Callback):
    """Drop-in for the per-dataset validation logger on a packed batch."""

    def __init__(self, save_s3: bool = False) -> None:
        super().__init__()
        self.save_s3 = bool(save_s3)
