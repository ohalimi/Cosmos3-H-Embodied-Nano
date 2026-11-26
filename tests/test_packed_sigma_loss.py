# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from unittest.mock import Mock

import pytest
import torch

from cosmos3_h_embodied.callbacks.packed_sigma_loss import PackedSigmaLossAnalysis


@pytest.mark.parametrize("per_item", [None, torch.empty(0), torch.zeros(1)])
def test_unpaired_step_skips_only_local_cache(monkeypatch, per_item):
    cb = PackedSigmaLossAnalysis(every_n=2, every_n_viz=2)
    gather = Mock(return_value={})
    monkeypatch.setattr(cb, "_gather_and_save", gather)
    model = Mock()
    model.is_image_batch.return_value = False
    sigma = torch.full((9, 2), 0.5)
    batch = {"sigma": sigma}
    if per_item is not None:
        batch["flow_matching_loss_vision_per_instance"] = per_item
    total_loss = torch.tensor(2.0)
    cb.on_training_step_end(model, {}, batch, total_loss, iteration=2)
    assert batch["sigma"] is sigma
    assert all(cache.get_arrays()[0].numel() == 0 for cache in (cb.image_cache, cb.video_cache))
    assert [call.args[0] for call in gather.call_args_list] == [cb.image_cache, cb.video_cache]


def test_valid_pairs_still_split_by_modality():
    cb = PackedSigmaLossAnalysis(every_n=2, every_n_viz=2)
    batch = {"sigma": torch.tensor([[0.25, 0.75], [0.25, 0.25]]),
             "vision_is_image": [True, False], "flow_matching_loss_vision_per_instance": torch.tensor([1.0, 2.0])}
    cb.on_training_step_end(Mock(), {}, batch, torch.tensor(1.5), iteration=1)
    assert [x.tolist() for x in cb.image_cache.get_arrays()] == [[0.5], [1.0]]
    assert [x.tolist() for x in cb.video_cache.get_arrays()] == [[0.25], [2.0]]
