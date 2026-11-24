# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""``tools/train_with_prune.sh``: the pruner runs while training is alive and once more after it exits."""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

WRAPPER = Path(__file__).resolve().parents[1] / "tools/train_with_prune.sh"

# Stands in for training: waits until the pruner has deleted the oldest save, saves once more, exits 3.
FAKE_TRAIN = textwrap.dedent(
    """
    import sys, time
    from pathlib import Path

    checkpoints, marker = Path(sys.argv[1]), Path(sys.argv[2])
    oldest = checkpoints / "iter_000000100"
    deadline = time.monotonic() + 60
    while oldest.exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    marker.write_text("timed out" if oldest.exists() else "pruned while alive")
    (checkpoints / "iter_000000400" / "model").mkdir(parents=True)
    (checkpoints / "latest_checkpoint.txt").write_text("iter_000000400\\n")
    sys.exit(3)
    """
)

# Stands in for training that is stopped: reports TERM and exits 7.
FAKE_TRAIN_TERM = textwrap.dedent(
    """
    import signal, sys, time
    from pathlib import Path

    ready, marker = Path(sys.argv[1]), Path(sys.argv[2])

    def stop(signum, frame):
        marker.write_text("got TERM")
        sys.exit(7)

    signal.signal(signal.SIGTERM, stop)
    ready.write_text("ready")
    time.sleep(60)
    sys.exit(0)
    """
)


def make_run(root: Path, iterations: list[int], pointer: int) -> Path:
    checkpoints = root / "checkpoints"
    for iteration in iterations:
        (checkpoints / f"iter_{iteration:09d}" / "model").mkdir(parents=True)
    (checkpoints / "latest_checkpoint.txt").write_text(f"iter_{pointer:09d}\n")
    return checkpoints


def remaining(checkpoints: Path) -> list[int]:
    return sorted(int(path.name.split("_")[1]) for path in checkpoints.glob("iter_*"))


def wrapper_env(node: str | None = None) -> dict[str, str]:
    env = {**os.environ, "PRUNE_PYTHON": sys.executable}
    env.pop("SLURM_NODEID", None)
    if node is not None:
        env["SLURM_NODEID"] = node
    return env


def run_wrapper(*args: str, node: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(WRAPPER), *args], capture_output=True, text=True, env=wrapper_env(node), timeout=120, check=False
    )


def test_prunes_while_training_runs_and_once_after_it_exits(tmp_path) -> None:
    checkpoints = make_run(tmp_path, [100, 200, 300], pointer=300)
    script, marker = tmp_path / "fake_train.py", tmp_path / "marker"
    script.write_text(FAKE_TRAIN)
    result = run_wrapper(
        "--checkpoint-dir", str(checkpoints), "--interval", "1", "--milestones", "",
        "--", sys.executable, str(script), str(checkpoints), str(marker),
    )  # fmt: skip
    assert result.returncode == 3, result.stderr
    assert marker.read_text() == "pruned while alive"
    assert "pruning pass while training runs" in result.stdout
    assert "final pruning pass, training exited with status 3" in result.stdout
    assert remaining(checkpoints) == [300, 400]


def test_other_nodes_only_train(tmp_path) -> None:
    checkpoints = make_run(tmp_path, [100, 200, 300], pointer=300)
    result = run_wrapper(
        "--checkpoint-dir", str(checkpoints), "--interval", "1",
        "--", sys.executable, "-c", "import time; time.sleep(2)",
        node="1",
    )  # fmt: skip
    assert result.returncode == 0, result.stderr
    assert "pruning" not in result.stdout
    assert remaining(checkpoints) == [100, 200, 300]


def test_defaults(tmp_path) -> None:
    result = run_wrapper("--checkpoint-dir", str(tmp_path / "checkpoints"), "true")
    assert result.returncode == 0, result.stderr
    line = next(line for line in result.stdout.splitlines() if line.startswith("train_with_prune: pruning "))
    assert int(re.search(r"keep-every (\d+)", line).group(1)) not in (20, 100)
    assert "every 600 s" in line
    assert "keep-last 2," in line
    assert "milestones 500,1000,2000," in line
    assert "save-iter 100" in line


@pytest.mark.parametrize("keep_every", ["20", "50", "100"])
def test_keep_every_that_keeps_every_save_is_refused(tmp_path, keep_every) -> None:
    started = tmp_path / "started"
    result = run_wrapper("--checkpoint-dir", str(tmp_path), "--keep-every", keep_every, "--", "touch", str(started))
    assert result.returncode == 2
    assert "would keep every save" in result.stderr
    assert not started.exists()


def test_term_reaches_training_and_the_final_pass_runs(tmp_path) -> None:
    checkpoints = make_run(tmp_path, [100, 200, 300], pointer=300)
    script, ready, marker = tmp_path / "fake_train.py", tmp_path / "ready", tmp_path / "marker"
    script.write_text(FAKE_TRAIN_TERM)
    proc = subprocess.Popen(
        ["bash", str(WRAPPER), "--checkpoint-dir", str(checkpoints), "--milestones", "",
         "--", sys.executable, str(script), str(ready), str(marker)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=wrapper_env(),
    )  # fmt: skip
    deadline = time.monotonic() + 30
    while not ready.exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    proc.send_signal(signal.SIGTERM)
    stdout, stderr = proc.communicate(timeout=60)
    assert proc.returncode == 7, stderr
    assert marker.read_text() == "got TERM"
    assert "final pruning pass, training exited with status 7" in stdout
    assert remaining(checkpoints) == [200, 300]
