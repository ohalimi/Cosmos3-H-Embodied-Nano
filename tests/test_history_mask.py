# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import pytest
import torch

from cosmos3_h_embodied.data.history_dataset import _HistoryMixin


class _Source:
    def __init__(self):
        self._pose_convention = "backward_anchored_world_trans"
        self._latent_cache_active = False
        self._dt = 0.1
        self._delta_timestamps = {k: [0.0] for k in ("action", "observation.state")}

    def _fetch_sample(self, idx, **kwargs):
        return "wam", 0, idx, self.raw

    def _build_result(self, **kwargs):
        return kwargs


class _Dataset(_HistoryMixin, _Source):
    def _compute_history(self, sample, h):
        return sample["action"][:h]

    def _normalize_action(self, x):
        return x + 7


@pytest.mark.parametrize("valid_rows", [0, 1, 6, 16])
def test_mask_preserves_valid_suffix_layout_and_future(valid_rows):
    values = torch.arange(40 * 20, dtype=torch.float32).reshape(40, 20)
    raw = {k: values for k in ("action", "observation.state")}
    raw.update({f"{k}_is_pad": torch.arange(40) < 16 - valid_rows for k in tuple(raw)})
    results = []
    for options in ({}, {"mask_older_history": True}):
        ds = _Dataset(num_history_actions=16, **options)
        ds.raw = raw
        _, _, _, future = ds._fetch_sample(0)
        for key in raw:
            assert torch.equal(future[key], raw[key][16:])
        results.append(ds._build_result()["history_action"])
    control, masked = results
    assert masked.shape == control.shape == (valid_rows, 20)
    assert torch.equal(control, values[16 - valid_rows : 16] + 7)
    assert torch.equal(masked[-1:], control[-1:])
    assert torch.count_nonzero(masked[:-1]) == 0
