# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""The packed-batch validation callback must not reject a list of names."""

from __future__ import annotations

import inspect

from cosmos_framework.callbacks.wandb_log_eval import WandbCallback

from cosmos3_h_embodied.callbacks.packed_batch_val import PackedBatchValLog


def test_framework_logger_still_requires_one_name() -> None:
    source = inspect.getsource(WandbCallback.on_validation_step_end)
    assert 'assert len(dataset_name) == 1, "dataset_name should be a list of 1"' in source


def test_packed_batch_callback_accepts_several_names() -> None:
    callback = PackedBatchValLog(save_s3=False)
    callback.on_validation_step_end(
        model=None,  # type: ignore[arg-type]
        data_batch={"dataset_name": ["action_data", "action_data"]},  # type: ignore[arg-type]
        output_batch={},  # type: ignore[arg-type]
        loss=None,  # type: ignore[arg-type]
        iteration=0,
    )
