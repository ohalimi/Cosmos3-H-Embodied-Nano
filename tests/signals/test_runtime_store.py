"""Synthetic collector/writer test only; never invoke an evaluator or read a model."""

from hashlib import sha256
import io
import json
from types import SimpleNamespace as NS
import zipfile

import numpy as np
import pytest
import torch

from openh_signals.events import EventWriter, evaluation_reservation
from openh_signals.runtime import FRAMEWORK
from openh_signals.runtime_eval_store import SingleWorkerEvaluation, _identity


def test_private_eval_coordinator(tmp_path):
    ident = lambda x: sha256(x.encode()).hexdigest()
    cfg = (1, 1, 1, 2)
    reserve = evaluation_reservation(*cfg)
    spec = dict(
        protocol="readme-eval",
        eval_counts=cfg,
        reservation=reserve,
        source_commit=FRAMEWORK,
        config_digest="0" * 64,
        probe_digest="1" * 64,
        step=40,
        kernel="test",
    )
    writer = EventWriter(tmp_path, reservation=reserve, kept_bytes=50 * reserve)
    profiles = []
    batches = []
    for i in range(2):
        b = dict(video=[torch.zeros(1, 2, 2, 2) + i], action=[torch.zeros(24, 2) + i])
        batches.append(b)
        profiles.append(
            dict(
                name=f"synthetic{i}",
                seed=3,
                chunk=0,
                ablation="real",
                mode="policy",
                family="mock",
                probe=i == 0,
                target_digest=ident(f"t{i}"),
                root_digest=ident("r"),
                index_digest=ident("i"),
                window_digest=ident(f"w{i}"),
                video_digest=_identity(b["video"][0]),
                action_digest=_identity(b["action"][0]),
                solver_branches=[[i, ["positive"]] for i in (0, 17, 34)],
                branches={},
                required_signals=["mot"],
            )
        )
    coordinator = SingleWorkerEvaluation(
        spec, dict(text_spans={}, instances=profiles), writer
    )
    with pytest.raises(ValueError, match="differs"):
        coordinator.identify(NS(name=profiles[0]["name"], seed=3), batches[1])
    for i, row in enumerate(profiles):
        cell = coordinator.identify(NS(name=row["name"], seed=3), batches[i])
        for step in (0, 17, 34):
            loc = dict(
                cell,
                kind="eval",
                t=[0.5],
                solver_call=step,
                solver_calls=35,
                branch="positive",
                cache_hit=False,
            )
            for layer in range(36):
                coordinator.emit(
                    "mot",
                    dict(
                        cell=loc,
                        sample=0,
                        layer=layer,
                        entropy=torch.tensor([0.25 + i]),
                    ),
                )
        coordinator.emit(
            "eval.error", dict(sample_name=row["name"], mode="policy", psnr=15.0)
        )
    receipt = coordinator.finish()
    assert receipt.created
    with zipfile.ZipFile(receipt.path) as z:
        assert sorted(z.namelist()) == [
            "S/clips",
            "S/probe0",
            "S/stratum0",
            "manifest.json",
        ]
        clips = json.loads(z.read("S/clips"))
        assert [r["probe_archive"] for r in clips] == ["S/probe0", None]
        assert [(r["window_digest"], r["mode"], r["seed"]) for r in clips] == [
            (p["window_digest"], "policy", 3) for p in profiles
        ]
        with np.load(io.BytesIO(z.read("S/stratum0")), allow_pickle=False) as archive:
            assert archive["f0"].item() == 0.75 and archive["f0_count"].item() == 2


def test_repeated_text_canonical_strata_and_map_pointer():
    from openh_signals.runtime_eval_store import _canonical_text

    bank = {}
    first = {
        "mass": np.arange(2 * 12 * 12).reshape(2, 12, 12).astype(np.float32),
        "entropy": np.arange(2 * 12).reshape(2, 12).astype(np.float32),
        "text_rows": np.array([0, 1]),
        "mot_query_groups": np.arange(12),
    }
    ref = "probe/seed/first/positive/layer"
    next_ref = "probe/seed/middle/positive/layer"
    a = _canonical_text(
        first, {"text_reference": ref, "observation_reference": ref}, bank
    )
    second = dict(
        first,
        mass=first["mass"][:, 2:] + 3,
        entropy=first["entropy"][:, 2:] + 2,
        mot_query_groups=np.arange(2, 12),
    )
    b = _canonical_text(
        second, {"text_reference": ref, "observation_reference": next_ref}, bank
    )
    assert a["mass"].shape == b["mass"].shape == (2, 12, 12)
    assert np.array_equal(a["mass"][:, :2], b["mass"][:, :2]) and np.array_equal(
        b["mass"][:, 2:], second["mass"]
    )
    with pytest.raises(ValueError, match="missing retained"):
        _canonical_text(
            second,
            {"text_reference": "absent", "observation_reference": next_ref},
            bank,
        )

    coordinator = object.__new__(SingleWorkerEvaluation)
    coordinator.active = ("synthetic", 3)
    coordinator.current = {}
    coordinator.current_meta = {}
    coordinator.current_bytes = 0
    coordinator.map_info = {}
    coordinator.maps = {}
    cell = dict(
        full_map=True,
        window_digest="3" * 64,
        branch="positive",
        solver_call=17,
        seed=3,
        mode="wam",
    )
    coordinator.emit(
        "mot", dict(cell=cell, layer=35, full_map=torch.eye(2, dtype=torch.float16))
    )
    assert coordinator.map_info["head0"] == dict(
        window="3" * 64,
        layer=35,
        head=31,
        branch="positive",
        solver_call=17,
        seed=3,
        mode="wam",
        shape=[2, 2],
    )
