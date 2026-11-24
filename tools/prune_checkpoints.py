# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Delete the old checkpoints of a training run, also while it trains.

A run saves ``iter_<NNNNNNNNN>`` directories under its ``checkpoints/`` directory and names the newest
complete one in ``checkpoints/latest_checkpoint.txt``. This script keeps

- the directory named by ``latest_checkpoint.txt`` (the resume target),
- the newest ``--keep-last`` directories,
- every ``--keep-every``-th iteration (``0``, the default, keeps none this way),
- the iterations listed in ``--milestones`` (none by default),

and deletes the other ``iter_*`` directories. It deletes nothing when the newest save does not look
finished: when ``latest_checkpoint.txt`` is missing or does not name the newest ``iter_*`` directory
(a save is still being written), or when that directory has no ``model`` subdirectory. A refusal exits
with status 0, so that a periodic caller such as ``tools/train_with_prune.sh`` never stops training;
its next pass tries again.

Example, a dry run that only prints what it would delete::

    python tools/prune_checkpoints.py --checkpoint-dir /path/to/run/checkpoints \\
        --keep-last 2 --milestones 500,1000,2000 --dry-run

``--checkpoint-dir`` may also be the run directory that holds ``checkpoints/``.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

POINTER_FILE = "latest_checkpoint.txt"
_ITER_DIR = re.compile(r"iter_(\d+)")


class Refusal(Exception):
    """The newest save does not look finished, so nothing may be deleted."""


def iter_dirs(checkpoint_dir: Path) -> list[tuple[int, Path]]:
    """The ``iter_*`` directories of ``checkpoint_dir`` (symlinks excluded), oldest first."""
    found = []
    for path in checkpoint_dir.iterdir():
        match = _ITER_DIR.fullmatch(path.name)
        if match and path.is_dir() and not path.is_symlink():
            found.append((int(match.group(1)), path))
    return sorted(found)


def resolve_checkpoint_dir(path: Path) -> Path:
    """``path``, or its ``checkpoints`` subdirectory when ``path`` is the run directory."""
    nested = path / "checkpoints"
    if path.is_dir() and nested.is_dir() and not (path / POINTER_FILE).exists() and not iter_dirs(path):
        return nested
    return path


def select(
    checkpoint_dir: Path, keep_last: int, keep_every: int, milestones: set[int]
) -> tuple[list[Path], list[Path]]:
    """Split the ``iter_*`` directories into (kept, deleted), both oldest first.

    Raises ``Refusal`` when the newest save does not look finished.
    """
    dirs = iter_dirs(checkpoint_dir)
    if not dirs:
        return [], []
    newest = dirs[-1][1]
    pointer = checkpoint_dir / POINTER_FILE
    if not pointer.is_file():
        raise Refusal(f"{POINTER_FILE} is missing")
    named = pointer.read_text().strip()
    target = Path(named) if Path(named).is_absolute() else checkpoint_dir / named
    if target.resolve() != newest.resolve():
        raise Refusal(f"{POINTER_FILE} names {named!r}, but the newest save is {newest.name} (still being written?)")
    if not (newest / "model").is_dir():
        raise Refusal(f"the newest save {newest.name} has no model directory")

    keep = {path for _, path in dirs[-keep_last:]} | {newest}
    for number, path in dirs:
        if number in milestones or (keep_every > 0 and number % keep_every == 0):
            keep.add(path)
    kept = [path for _, path in dirs if path in keep]
    deleted = [path for _, path in dirs if path not in keep]
    return kept, deleted


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def _non_negative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be 0 or more, got {value}")
    return value


def _milestones(text: str) -> set[int]:
    try:
        return {int(item) for item in text.split(",") if item.strip()}
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected comma-separated iterations, got {text!r}") from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="directory holding the iter_* directories and latest_checkpoint.txt, or the run directory above it",
    )
    parser.add_argument("--keep-last", type=_positive_int, default=2, help="keep the newest N saves (default 2)")
    parser.add_argument(
        "--keep-every", type=_non_negative_int, default=0, help="also keep every K-th iteration; 0 keeps none (default)"
    )
    parser.add_argument(
        "--milestones",
        type=_milestones,
        default=set(),
        help="also keep these iterations, comma-separated (default none)",
    )
    parser.add_argument("--dry-run", action="store_true", help="print what would be deleted and delete nothing")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    checkpoint_dir = resolve_checkpoint_dir(args.checkpoint_dir)
    if not checkpoint_dir.is_dir():
        print(f"prune: {checkpoint_dir} does not exist yet; nothing to do")
        return 0
    try:
        kept, deleted = select(checkpoint_dir, args.keep_last, args.keep_every, args.milestones)
    except Refusal as exc:
        print(f"prune: refusing, nothing deleted: {exc}")
        return 0
    if not kept:
        print(f"prune: no iter_* directories in {checkpoint_dir}; nothing to do")
        return 0

    verb = "would delete" if args.dry_run else "deleting"
    failed = 0
    for path in deleted:
        print(f"prune: {verb} {path.name}", flush=True)
        if not args.dry_run:
            try:
                shutil.rmtree(path)
            except OSError as exc:
                failed += 1
                print(f"prune: could not delete {path.name}: {exc}", file=sys.stderr)
    print(f"prune: kept {' '.join(path.name for path in kept)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
