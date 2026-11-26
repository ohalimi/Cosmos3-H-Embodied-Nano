# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from unittest.mock import Mock

import pytest

from cosmos3_h_embodied.callbacks.nano_warm_start import InitEmaFromNano
from cosmos3_h_embodied.configs.experiment_h16 import nano_h16_suturebot, nano_h16_suturebot_ref32
from cosmos3_h_embodied.configs.openh_datasets import build_dataset_entries
from cosmos3_h_embodied.data.cosmos3_action_lerobot import split_episode_ids


@pytest.mark.parametrize("cfg,caption,accum", [(nano_h16_suturebot, "fixed", 1), (nano_h16_suturebot_ref32, "legacy", 2)])
def test_two_root_recipe(monkeypatch, cfg, caption, accum):
    monkeypatch.setenv("OPENH_DATA_ROOT", "/DATA")
    expected = ["/DATA/jhu/imerse/wound_closure/point_labeled/fausto_0_1_jesse_0_1_2_labeled", "/DATA/jhu/hf_suturebot"]
    for split, loader in [("train", cfg.dataloader_train), ("val", cfg.dataloader_val)]:
        d = loader.dataloaders.action_data.dataloader.dataset
        assert bool(d.video_augment) == (split == "train")
        entries = d.list_of_datasets
        assert [r for e in entries for r in e.dataset.roots] == expected
        assert [e.dataset.video_key for e in entries] == ["observation.images.endoscope.left", "observation.images.endoscope_left"]
        assert [e.dataset.caption_embodiments[0] for e in entries] == ["dvrk_wound_closure", "suturebot"]
        assert all(e.dataset.split == split and e.dataset.split_seed == 42 and e.dataset.split_val_ratio == .05 for e in entries)
        assert [e.ratio for e in entries] == [2.5928, 0.28775]
        assert all(e.dataset.caption_prefix_mode == caption for e in entries)
    assert cfg.trainer.grad_accum_iter == accum and cfg.trainer.max_iter == 50000 and cfg.trainer.run_validation
    assert cfg.trainer.run_validation_on_start and list(cfg.scheduler.cycle_lengths) == [50000]
    assert cfg.model.config.parallelism.data_parallel_shard_degree == 32
    assert list(cfg.scheduler.f_max) == [.4] and list(cfg.scheduler.f_min) == [.05]
    assert list(cfg.checkpoint.keys_to_skip_loading) == ["net_ema."] and cfg.checkpoint.strict_resume
    train = set(split_episode_ids(1458, 42, .05, "train"))
    assert {298, 579, 323, 628, 350, 675, *range(46, 52)} <= train


@pytest.mark.parametrize('roots', [(), ('unknown',), ('jhu/hf_suturebot',)])
def test_allowlist_cannot_silently_widen_or_drop_roots(roots):
    with pytest.raises(ValueError, match='root_allowlist'):
        build_dataset_entries(include_nonpublic=False, root_allowlist=roots)


def test_only_seed_ema_on_warm_start():
    m = Mock()
    cb = InitEmaFromNano()
    cb.on_train_start(m, iteration=0)
    cb.on_train_start(m, iteration=100)
    m.net_ema_worker.copy_to.assert_called_once_with(src_model=m.net, tgt_model=m.net_ema)
