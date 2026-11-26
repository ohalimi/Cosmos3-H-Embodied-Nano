# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Summarize ``cosmos3_h_embodied.scripts.eval`` runs of several seeds per dataset family and mode.

Each ``--runs`` directory is the output of one ``eval.py`` run (one seed, any families); samples are
matched across runs by family, mode and dataset index. Forward dynamics: video PSNR; inverse
dynamics: action MSE; policy: action MSE and video PSNR, as the mean over the runs and the best of
them per sample (lowest MSE, highest PSNR), since the policy's action distribution is multimodal.
Action MSE is in the normalized action space (most channels in [-1, 1]) over the predicted rows.
The baseline predicts the family's mean recorded trajectory for every policy sample, so its MSE is
the variance of the recorded actions.

Example::

    python tools/benchmark/summarize_eval.py --runs outputs/eval_s0 outputs/eval_s1 -o outputs/eval_summary
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_runs(runs: list[Path]) -> dict[tuple[str, str, str], list[dict]]:
    """``{(family, mode, sample index): [metrics of each run]}``; sample dirs are ``<family>/<mode>/<index>``."""
    samples: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for run in runs:
        for path in sorted(run.rglob("metrics.json")):
            family, mode, index = path.parts[-4:-1]
            samples[(family, mode, index)].append(
                {**json.loads(path.read_text()), "dir": path.parent}
            )
    return samples


def baseline_mse(sample_dirs: list[Path]) -> float:
    """MSE of predicting the mean recorded trajectory of these samples (``gt_action.json``, scored rows)."""
    gts = [
        np.asarray(json.loads((d / "gt_action.json").read_text()), dtype=np.float64)
        for d in sample_dirs
    ]
    if not gts or len({g.shape for g in gts}) != 1:
        return float("nan")
    g = np.stack(gts)
    return float(((g - g.mean(axis=0)) ** 2).mean())


def summarize(samples: dict[tuple[str, str, str], list[dict]]) -> dict[str, dict]:
    rows: dict[str, dict] = defaultdict(lambda: defaultdict(list))
    policy_dirs: dict[str, list[Path]] = defaultdict(list)
    for (family, mode, _), runs in sorted(samples.items()):
        for key in (family, "All"):
            row = rows[key]
            if mode == "forward_dynamics":
                row["fd_psnr"].append(np.mean([r["psnr"] for r in runs]))
            elif mode == "inverse_dynamics":
                row["id_mse"].append(np.mean([r["action_mse"] for r in runs]))
            elif mode == "wam":
                mse = [r["action_mse"] for r in runs]
                psnr = [r["psnr"] for r in runs]
                row["policy_runs"].append(len(runs))
                row["policy_mse_mean"].append(np.mean(mse))
                row["policy_mse_best"].append(np.min(mse))
                row["policy_psnr_mean"].append(np.mean(psnr))
                row["policy_psnr_best"].append(np.max(psnr))
                if key != "All":
                    policy_dirs[family].append(runs[0]["dir"])
    out: dict[str, dict] = {}
    for key, row in rows.items():
        out[key] = {
            name: float(np.mean(values))
            for name, values in row.items()
            if name != "policy_runs"
        }
        out[key]["samples"] = max(
            (len(v) for n, v in row.items() if n != "policy_runs"), default=0
        )
        out[key]["policy_runs"] = int(min(row["policy_runs"], default=0))
    families = [k for k in out if k != "All"]
    for family in families:
        out[family]["baseline_mse"] = baseline_mse(policy_dirs[family])
    if "All" in out:
        weights = [len(policy_dirs[f]) for f in families]
        values = [out[f]["baseline_mse"] for f in families]
        out["All"]["baseline_mse"] = (
            float(np.average(values, weights=weights)) if sum(weights) else float("nan")
        )
    return dict(sorted(out.items(), key=lambda kv: (kv[0] == "All", kv[0])))


def seed_spread(
    by_seed: list[dict], families: set[str], clips: int = 64
) -> dict[str, dict]:
    """Policy means per seed (in input order), only for a complete matched all-modes evaluation."""
    if len(by_seed) != 4:
        raise ValueError("expected the joint seed followed by three policy seeds")
    output = {
        f: {"policy_mse": [], "policy_psnr": []} for f in sorted(families | {"All"})
    }
    expected = {(f, i) for f, mode, i in by_seed[0] if mode == "wam"}
    for family in families:
        if sum(f == family for f, _ in expected) != clips:
            raise ValueError(f"seed 0: expected {clips} clips of {family}")
    if {f for f, _ in expected} != families:
        raise ValueError("seed 0: unexpected families")
    truth = {}
    for s, samples in enumerate(by_seed):
        modes = ("wam", "forward_dynamics", "inverse_dynamics") if s == 0 else ("wam",)
        if samples.keys() != {(f, m, i) for f, i in expected for m in modes}:
            raise ValueError(f"seed {s}: wrong clip/mode coverage")
        per_family = defaultdict(list)
        for (f, mode, i), runs in sorted(samples.items()):
            if len(runs) != 1:
                raise ValueError(f"seed {s}: duplicate {f}/{mode}/{i}")
            row = runs[0]
            keys = {
                "wam": ("action_mse", "psnr"),
                "forward_dynamics": ("psnr",),
                "inverse_dynamics": ("action_mse",),
            }
            if not all(np.isfinite(row[k]) for k in keys[mode]):
                raise ValueError(f"seed {s}: nonfinite {f}/{mode}/{i}")
            if mode == "wam":
                gt = np.asarray(json.loads((row["dir"] / "gt_action.json").read_text()))
                if s == 0:
                    truth[f, i] = gt
                if (
                    not gt.size
                    or not np.isfinite(gt).all()
                    or not np.array_equal(gt, truth[f, i])
                ):
                    raise ValueError(
                        f"seed {s}: invalid or changed ground truth for {f}/{i}"
                    )
                for name in (f, "All"):
                    per_family[name].append((row["action_mse"], row["psnr"]))
        for name, values in per_family.items():
            mse, psnr = np.mean(values, axis=0)
            output[name]["policy_mse"].append(float(mse))
            output[name]["policy_psnr"].append(float(psnr))
    return output


def write_table(summary: dict[str, dict], path: Path) -> None:
    def fmt(row: dict, key: str, digits: int) -> str:
        value = row.get(key, float("nan"))
        return "" if np.isnan(value) else f"{value:.{digits}f}"

    runs = max((row["policy_runs"] for row in summary.values()), default=0)
    lines = [
        f"Policy: mean over {runs} seeds / best of {runs} per sample. Action MSE in normalized action space "
        "(lower is better); baseline: the family's mean recorded trajectory.",
        "",
        "| Family | Samples | FD PSNR ↑ | ID action MSE ↓ | Policy action MSE ↓ (mean / best) "
        "| Policy PSNR ↑ (mean / best) | Baseline action MSE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for family, row in summary.items():
        name = f"**{family}**" if family == "All" else family
        lines.append(
            f"| {name} | {row['samples']} | {fmt(row, 'fd_psnr', 2)} | {fmt(row, 'id_mse', 4)} "
            f"| {fmt(row, 'policy_mse_mean', 4)} / {fmt(row, 'policy_mse_best', 4)} "
            f"| {fmt(row, 'policy_psnr_mean', 2)} / {fmt(row, 'policy_psnr_best', 2)} | {fmt(row, 'baseline_mse', 4)} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--runs",
        type=Path,
        nargs="+",
        required=True,
        help="eval.py output directories (one per seed)",
    )
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument(
        "--all-modes-families",
        nargs="+",
        help="Validate 64 clips per listed family and four seeds "
        "(joint first), and write the per-seed policy means",
    )
    args = parser.parse_args()
    spread = None
    if args.all_modes_families:
        spread = seed_spread(
            [load_runs([r]) for r in args.runs], set(args.all_modes_families)
        )
    summary = summarize(load_runs(args.runs))
    args.output.mkdir(parents=True, exist_ok=True)
    if spread is not None:
        # A non-finite baseline (e.g. inconsistent GT shapes) is not a valid release result.
        json.dumps(summary, allow_nan=False)
        (args.output / "policy_seeds.json").write_text(
            json.dumps(spread, indent=2, allow_nan=False) + "\n"
        )
    (args.output / "eval_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_table(summary, args.output / "eval_summary.md")
    print((args.output / "eval_summary.md").read_text())


if __name__ == "__main__":
    main()
