# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""In-training checkpoint pruning: what ``tools/prune_checkpoints.py`` keeps, deletes and refuses."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PRUNER = Path(__file__).resolve().parents[1] / "tools/prune_checkpoints.py"


def make_run(root: Path, iterations: list[int], pointer: int | None, without_model: tuple[int, ...] = ()) -> Path:
    """Saves ``checkpoints/iter_*`` under ``root``; ``pointer`` is the iteration named as the latest."""
    checkpoints = root / "checkpoints"
    for iteration in iterations:
        save = checkpoints / f"iter_{iteration:09d}"
        save.mkdir(parents=True)
        if iteration not in without_model:
            (save / "model").mkdir()
    if pointer is not None:
        (checkpoints / "latest_checkpoint.txt").write_text(f"iter_{pointer:09d}\n")
    return checkpoints


def remaining(checkpoints: Path) -> list[int]:
    return sorted(int(path.name.split("_")[1]) for path in checkpoints.glob("iter_*"))


def prune(checkpoint_dir: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(PRUNER), "--checkpoint-dir", str(checkpoint_dir), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_dry_run_deletes_nothing(tmp_path) -> None:
    checkpoints = make_run(tmp_path, [100, 200, 300, 400], pointer=400)
    result = prune(checkpoints, "--dry-run")
    assert result.returncode == 0, result.stderr
    assert remaining(checkpoints) == [100, 200, 300, 400]
    assert "would delete iter_000000100" in result.stdout
    assert "would delete iter_000000200" in result.stdout


def test_keeps_resume_target_and_last_n_and_deletes_the_rest(tmp_path) -> None:
    checkpoints = make_run(tmp_path, [100, 200, 300], pointer=300)
    result = prune(checkpoints, "--keep-last", "2")
    assert result.returncode == 0, result.stderr
    assert remaining(checkpoints) == [200, 300]
    assert "deleting iter_000000100" in result.stdout


def test_keeps_milestones_and_every_kth_iteration(tmp_path) -> None:
    checkpoints = make_run(tmp_path, list(range(100, 1100, 100)), pointer=1000)
    result = prune(checkpoints, "--keep-last", "2", "--keep-every", "400", "--milestones", "300,500")
    assert result.returncode == 0, result.stderr
    assert remaining(checkpoints) == [300, 400, 500, 800, 900, 1000]


def test_defaults_keep_the_last_two_and_no_milestones(tmp_path) -> None:
    checkpoints = make_run(tmp_path, [500, 1000, 1100, 1200], pointer=1200)
    result = prune(checkpoints)
    assert result.returncode == 0, result.stderr
    assert remaining(checkpoints) == [1100, 1200]


def test_accepts_the_run_directory(tmp_path) -> None:
    checkpoints = make_run(tmp_path, [100, 200, 300], pointer=300)
    result = prune(tmp_path, "--keep-last", "1")
    assert result.returncode == 0, result.stderr
    assert remaining(checkpoints) == [300]


def test_pointer_may_be_an_absolute_path(tmp_path) -> None:
    checkpoints = make_run(tmp_path, [100, 200, 300], pointer=None)
    (checkpoints / "latest_checkpoint.txt").write_text(str(checkpoints / "iter_000000300"))
    result = prune(checkpoints, "--keep-last", "1")
    assert result.returncode == 0, result.stderr
    assert remaining(checkpoints) == [300]


@pytest.mark.parametrize(
    ("pointer", "without_model"),
    [
        pytest.param(200, (), id="pointer-names-an-older-save"),
        pytest.param(None, (), id="no-pointer"),
        pytest.param(300, (300,), id="newest-save-without-model"),
    ],
)
def test_refuses_and_deletes_nothing(tmp_path, pointer, without_model) -> None:
    checkpoints = make_run(tmp_path, [100, 200, 300], pointer=pointer, without_model=without_model)
    result = prune(checkpoints, "--keep-last", "1")
    assert result.returncode == 0, result.stderr
    assert "refusing, nothing deleted" in result.stdout
    assert remaining(checkpoints) == [100, 200, 300]


def test_missing_directory_is_not_an_error(tmp_path) -> None:
    result = prune(tmp_path / "not-created-yet")
    assert result.returncode == 0, result.stderr
    assert "does not exist yet" in result.stdout


def test_keep_last_zero_is_rejected(tmp_path) -> None:
    checkpoints = make_run(tmp_path, [100, 200, 300], pointer=300)
    result = prune(checkpoints, "--keep-last", "0")
    assert result.returncode == 2
    assert remaining(checkpoints) == [100, 200, 300]
