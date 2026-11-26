# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from pathlib import Path

from omegaconf import OmegaConf

from cosmos3_h_embodied.configs.experiment_h16 import nano_h16_suturebot, nano_h16_suturebot_families
from cosmos3_h_embodied.configs.openh_datasets import FAMILIES


def test_plus_families_differs_only_in_the_mixture(monkeypatch):
    monkeypatch.setenv("OPENH_DATA_ROOT", "/DATA")
    cfg = nano_h16_suturebot_families
    expected_counts = {
        "cmr_clinical": 4, "dvrk_obuda": 11, "dvrk_ucsd": 2, "dvrk_jhu": 6,
        "dvrk_suturebot_mono": 1, "dvrk_stanford": 3, "dvrk_rob_surgical": 1,
        "dvrk_hamlyn": 6, "dvrk_ucberkeley": 1, "turin_mitic": 4,
        "dvrk_tud_grasping_retraction": 1, "dvrk_smarts": 3,
    }
    expected = {f.name: [r.path for r in f.roots if r.public] for f in FAMILIES if f.name in expected_counts}
    expected["dvrk_suturebot_mono"] = ["jhu/hf_suturebot"]
    assert {k: len(v) for k, v in expected.items()} == expected_counts
    for split in ("train", "val"):
        ld = cfg[f"dataloader_{split}"].dataloaders.action_data.dataloader
        ds = ld.dataset
        assert ld.batch_size == 1 and ds.shard_across_workers
        assert bool(ds.video_augment) == (split == "train")
        entries = ds.list_of_datasets
        assert {e.name: [str(Path(p).relative_to("/DATA")) for p in e.dataset.roots] for e in entries} == expected
        assert {e.name: e.ratio for e in entries} == {f.name: f.ratio for f in FAMILIES if f.name in expected}
        assert all(e.dataset.split == split and e.dataset.split_seed == 42 and e.dataset.split_val_ratio == .05 for e in entries)
    repro = cfg.trainer.callbacks.repro
    assert set(repro.data_roots) == {path for paths in expected.values() for path in paths}
    assert len(repro.normalizer) == len(expected)
    for field in ("checkpoint", "model", "scheduler", "optimizer"):
        assert cfg[field] == nano_h16_suturebot[field]
    assert OmegaConf.is_missing(cfg.checkpoint, "load_path")
    assert cfg.trainer.grad_accum_iter == 1 and cfg.trainer.max_iter == 50000
    assert cfg.trainer.run_validation and cfg.trainer.run_validation_on_start
