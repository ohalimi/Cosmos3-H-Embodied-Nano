"""Synthetic, in-memory inputs only. Optional CAMP source roots are code, not artifacts."""

from contextlib import nullcontext
from importlib.util import module_from_spec, spec_from_file_location
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from cosmos_framework.model.generator.mot.domain_aware_linear import DomainAwareLinear
from openh_signals.memory import (
    code_use,
    intervention_scores,
    public_assignments,
    row_summary,
    track_reads,
    writer_gates,
    writer_spatial,
)


def source(variable, name, relative):
    root = os.environ.get(variable)
    if not root:
        pytest.skip("set the pinned CAMP source root for integration tests")
    spec = spec_from_file_location(name, Path(root) / "cosmos3_h_embodied" / relative)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bits(t):
    return t.detach().cpu().contiguous().numpy().tobytes()


def test_frame_exact_real_reader_unchanged_updates(monkeypatch):
    mod = source("OPENH_POLICY_SOURCE", "signal_test_tracks", "data/frozen_tracks.py")
    cls = mod.FrozenTracks
    fetches = []
    fake = np.linspace(-1, 1, 120, dtype=np.float16).reshape(2, 3, 20)

    def in_memory(path):
        fetches.append(path)
        return fake

    monkeypatch.setattr(mod, "_track", in_memory)
    reader = object.__new__(cls)  # do not open a manifest or a track file
    root, ep = "synthetic", len(fake)
    reader.folder = Path("synthetic-writer") / "synthetic-split"
    reader.lengths = {(root, ep): len(fake)}

    def run(enabled):
        torch.manual_seed(17)
        model = DomainAwareLinear(20, 4, 1)
        opt = torch.optim.SGD(model.parameters(), lr=0.125)
        emitted, trace = [], []

        def receive(key, record):
            assert key[:3] == (reader.folder.name, root, ep)
            assert record["code_use"] == "not present in policy input"
            emitted.append((key, record))

        with track_reads(cls, receive) if enabled else nullcontext():
            for f in range(len(fake)):
                rows = reader.at(root, ep, f)
                assert torch.equal(rows, torch.from_numpy(fake[f].astype(np.float32)))
                x = torch.cat([rows, torch.rand(24, 20)])
                y = model(x, torch.zeros(len(x), dtype=torch.long))
                loss = y.square().mean()
                loss.backward()
                opt.step()
                opt.zero_grad()
                trace.append(
                    tuple(
                        bits(t)
                        for t in (y, loss, torch.get_rng_state(), *model.parameters())
                    )
                )
        if enabled:
            assert len(emitted) == len(fake)
            for f, (key, rec) in enumerate(emitted):
                assert key[-1] == f
                torch.testing.assert_close(
                    rec["norm"],
                    torch.from_numpy(fake[f].astype(np.float32)).norm(dim=1),
                )
        return trace

    assert run(False) == run(True)
    assert len(fetches) == 2 * len(fake)


def test_actual_producer_labels_before_ema():
    mod = source(
        "OPENH_WRITER_SOURCE", "signal_test_dictionary", "model/trajectory_codes.py"
    )

    def run(on):
        torch.manual_seed(8)
        dictionary = mod.EpisodeDictionary()
        hidden = torch.nn.Parameter(torch.randn(1, 4, 64))
        opt = torch.optim.SGD([hidden], lr=0.1)
        mask = torch.tensor([[True, True, True, False]])
        received, trace = [], []
        with (
            public_assignments(
                dictionary,
                lambda shape: ("synthetic", mask),
                lambda k, x: received.append(x),
            )
            if on
            else nullcontext()
        ):
            for i in range(2):
                result, penalty = dictionary(hidden, mask)
                loss = result.square().mean() + penalty
                loss.backward()
                opt.step()
                opt.zero_grad()
                if on:
                    assert received[i]["valid"] == 3
                    assert torch.equal(
                        received[i]["histogram"].long(), dictionary.batch_votes.long()
                    )
                trace.append(
                    tuple(
                        bits(t)
                        for t in (
                            result,
                            loss,
                            hidden,
                            torch.get_rng_state(),
                            *dictionary.buffers(),
                        )
                    )
                )
        return trace

    assert run(False) == run(True)


def test_actual_intervention_scope():
    # The native evaluator disables grad at import; isolate that effect from other unit fixtures.
    with torch.no_grad():
        mod = source(
            "OPENH_POLICY_SOURCE",
            "signal_test_score",
            "scripts/eval_trajectory_codes.py",
        )
    action = torch.linspace(-1, 1, 28 * 20).reshape(1, 28, 20)
    batch = dict(
        action=action,
        sequence_plan=[SimpleNamespace(condition_frame_indexes_action=list(range(4)))],
    )
    donor = -action[0, :3]
    receive = []
    calls = []

    def gen(name, x, seed):
        calls.append(name)
        return 0.5 * x["action"][0] + x["action"][0, 0, 0]

    def bind(b, s):
        return (("synthetic-A", 0, 0), ("synthetic-B", 0, 0))

    before = torch.get_rng_state()
    off = mod.score_triplet(gen, batch, donor, 19)
    with intervention_scores(mod.score_triplet, bind, receive.append):
        on = mod.score_triplet(gen, batch, donor, 19)
    assert (
        on == off
        and calls == ["real", "zero", "shuffled"] * 2
        and torch.equal(before, torch.get_rng_state())
    )
    (row,) = receive
    assert row["target_elements"] == 480 and row["zero_changed_fraction"] == 1
    assert row["shuffled_delta_mse"] == on["mse"]["shuffled"] - on["mse"]["real"]


def test_prefix_separates_history_and_targets():
    from openh_signals.memory import memory_prefix, memory_reads

    x = torch.zeros(29, 64)
    projected = torch.ones(29, 8)
    r = memory_prefix(x, projected, 2)
    assert len(r["raw_prefix_norm"]) == 5 and r["history_lags"] == (-1, 0)
    assert (
        r["raw_prefix_norm"].count_nonzero() == 0
        and (r["projected_prefix_norm"] > 0).all()
    )
    mass = torch.zeros(2, 12, 12)
    mass[:, 7, 6:11] = torch.tensor([0.5, 0, 0.25, 0.125, 0.125])
    counts = torch.zeros(12, dtype=torch.long)
    counts[6:11] = torch.tensor([1, 24, 1, 1, 1])
    summary = dict(mass=mass, counts=counts, read_strength=torch.zeros(12, 12))
    assert (
        memory_reads(summary)["relative_to_latest"].tolist() == [[0.5, 0.25, 0.25]] * 2
    )
    counts[6] = 0
    assert memory_reads(summary)["relative_to_latest"] is None


def test_other_writer_reads_returned_ids_only():
    import importlib
    from openh_signals.memory import northstar_assignments

    module = os.environ.get("OPENH_TEST_VQ_MODULE")
    if not module:
        pytest.skip("set the pinned returned-ID producer source module")
    cls = importlib.import_module(module).VectorQuantizer

    def run(on):
        torch.manual_seed(94)
        m = cls(codebook_size=512, dim=8)
        x = torch.nn.Parameter(torch.randn(1, 4, 8))
        opt = torch.optim.SGD([x], lr=0.1)
        select = torch.tensor([[True, True, True, False]])
        valid = torch.tensor([[True, True, False, False]])
        hist = []
        trace = []
        with (
            northstar_assignments(
                m, lambda shape: ("test", select), lambda key, r: hist.append(r)
            )
            if on
            else nullcontext()
        ):
            for _ in range(2):
                y, ids, penalty, perplexity = m(x, valid)
                loss = y.square().mean() + penalty
                loss.backward()
                opt.step()
                opt.zero_grad()
                if on:
                    assert torch.equal(
                        hist[-1]["histogram"].long(),
                        ids[valid & select].bincount(minlength=512),
                    )
                trace.append(
                    tuple(
                        bits(t)
                        for t in (y, ids, loss, x, torch.get_rng_state(), *m.buffers())
                    )
                )
        return trace

    assert run(False) == run(True)


def test_gate_recurrence_matches_true_fused_terminal_states():
    torch.manual_seed(11)
    model = torch.nn.LSTM(5, 4, num_layers=2, batch_first=True)
    x = torch.randn(2, 3, 5)
    init = (torch.randn(2, 2, 4), torch.randn(2, 2, 4))
    y = model(x, init)
    selected = torch.tensor([[True, False, True], [False, True, False]])
    state = torch.get_rng_state()
    record = writer_gates(model, x, init, y, selected)
    assert (
        record["mean"].shape == (2, 4)
        and record["valid"] == 3
        and torch.equal(state, torch.get_rng_state())
    )
    with pytest.raises(AssertionError):
        writer_gates(model, x, init, (y[0] + 1, y[1]), selected)
    summary = writer_spatial(torch.zeros(1, 32, 3, 3), torch.zeros(9, 2), 0)
    assert summary["probabilities"].shape == (32, 9)
    assert (
        code_use(torch.zeros(2, dtype=torch.long), torch.zeros(2, dtype=torch.bool))[
            "perplexity"
        ]
        is None
    )
    with pytest.raises(ValueError):
        row_summary(torch.ones(3, 19))
