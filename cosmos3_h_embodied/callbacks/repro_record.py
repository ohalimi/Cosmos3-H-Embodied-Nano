# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Put the launch's reproducibility record into the W&B run.

The launch writes the record into a directory and exports its path as ``REPRO_DIR``: ``commit.txt``,
``framework_commit.txt``, ``branch.txt``, ``remote.txt``, ``upstream.txt`` (lines ``fetch <url>``, ...),
``status.txt``, ``diff.patch``, ``command.txt`` and ``sbatch.sha256`` (the launch file's hash, then its command).
Once the W&B run exists, rank 0 uploads the directory with ``wandb.save``, sets the ``repro/*`` config keys, reads
them back, and raises when ``repro/commit`` is not a 40-character commit hash or ``repro/command`` is empty.
Nothing is recorded when W&B is disabled.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any

import wandb

from cosmos_framework.model._base import ImaginaireModel
from cosmos_framework.utils import distributed, log
from cosmos_framework.utils.callback import Callback

REPRO_DIR_ENV = "REPRO_DIR"
NORMALIZER_DIR = Path(__file__).resolve().parents[1] / "data" / "normalizers"
# A larger diff is only uploaded as a file.
MAX_CONFIG_DIFF_BYTES = 64 * 1024
_COMMIT = re.compile(r"[0-9a-f]{40}")


class ReproRecordCallback(Callback):
    """Record ``REPRO_DIR`` in the W&B run, once, on rank 0.

    Args:
        normalizer: Action-normalizer file name (under ``cosmos3_h_embodied/data/normalizers/``) whose hash is recorded.
        data_roots: Dataset roots of the run, relative to ``OPENH_DATA_ROOT``.
    """

    def __init__(self, normalizer: str | list[str], data_roots: list[str]) -> None:
        super().__init__()
        self.normalizer = normalizer
        self.data_roots = list(data_roots)
        self._done = False

    def on_train_start(self, model: ImaginaireModel, iteration: int = 0) -> None:
        # The W&B callback may start the run after this one; the first training step then records.
        if wandb.run is not None:
            self._record()

    def on_training_step_start(self, model: ImaginaireModel, data: dict[str, Any], iteration: int = 0) -> None:
        if not self._done:
            self._record()

    def _record(self) -> None:
        self._done = True
        if not distributed.is_rank0():
            return
        if wandb.run is None or wandb.run.disabled:
            log.info("repro: no W&B run, nothing recorded")
            return
        repro_dir = os.environ.get(REPRO_DIR_ENV, "")
        if not repro_dir or not Path(repro_dir).is_dir():
            raise RuntimeError(
                f"repro: {REPRO_DIR_ENV} must name the run's reproducibility directory, got {repro_dir!r}"
            )
        repro = Path(repro_dir)

        def read(name: str) -> str:
            path = repro / name
            return path.read_text().strip() if path.is_file() else ""

        upstream = dict(line.split(" ", 1) for line in read("upstream.txt").splitlines() if " " in line)
        diff = read("diff.patch")
        values: dict[str, Any] = {
            "repro/commit": read("commit.txt"),
            "repro/framework_commit": read("framework_commit.txt"),
            "repro/remote": read("remote.txt"),
            "repro/upstream": upstream.get("fetch", "").removesuffix(".git"),
            "repro/branch": read("branch.txt"),
            "repro/dirty": bool(diff),
            "repro/command": read("command.txt"),
            "repro/sbatch_sha256": read("sbatch.sha256").split(" ", 1)[0],
            "repro/normalizer_sha256": (
                hashlib.sha256((NORMALIZER_DIR / self.normalizer).read_bytes()).hexdigest()
                if isinstance(self.normalizer, str) else
                {name: hashlib.sha256((NORMALIZER_DIR / name).read_bytes()).hexdigest() for name in self.normalizer}
            ),
            "repro/data_roots": self.data_roots,
            "repro/seed": self.config.trainer.seed,
            "repro/load_checkpoint": Path(str(self.config.checkpoint.load_path or "")).name,
        }
        if diff and len(diff.encode()) <= MAX_CONFIG_DIFF_BYTES:
            values["repro/diff"] = diff

        wandb.save(str(repro / "*"), base_path=str(repro.parent), policy="now")
        wandb.run.config.update(values, allow_val_change=True)
        stored = dict(wandb.run.config)
        if not _COMMIT.fullmatch(str(stored.get("repro/commit", ""))):
            raise RuntimeError(f"repro: repro/commit is not a 40-character commit hash: {stored.get('repro/commit')!r}")
        if not str(stored.get("repro/command", "")).strip():
            raise RuntimeError("repro: repro/command is empty")
        log.info(
            f"repro: recorded commit {stored['repro/commit']} (dirty={values['repro/dirty']}) "
            f"from {repro} in W&B run {wandb.run.id}"
        )
