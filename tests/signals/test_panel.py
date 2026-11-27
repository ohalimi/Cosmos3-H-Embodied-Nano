"""Panel scheduling and delivery are independent from the optimizer's data stream."""

from copy import deepcopy
import hashlib
import random
import tempfile
import zipfile

import numpy as np
import pytest
import torch
from cosmos_framework.model.generator.mot.domain_aware_linear import DomainAwareLinear
from openh_signals.events import EventWriter, PLAN_SHA256, event_for
from openh_signals.panel import MODES, ProbePanel, Window


def digest(x):
    return hashlib.sha256(x).hexdigest()


def windows():
    return (
        Window(digest(b"x"), digest(b"target"), tuple(sorted(MODES)), True),
        Window(digest(b"y"), digest(b"target2"), ("forward_dynamics",), False),
    )


def evidence():
    return dict(source_commit="0" * 40, config_digest="0" * 64, probe_digest="1" * 64)


def records(panel, event, frozen):
    snapshot = digest(
        b"".join(p.detach().numpy().tobytes() for p in frozen.parameters())
    )
    for cell in panel.select(event):
        with torch.no_grad():
            x = (
                torch.randn(2, 4, generator=cell.generator("action", "cpu"))
                * cell.sigma
            )
            y = frozen(x, torch.zeros(len(x), dtype=torch.long))
        # Bounded synthetic diagnostic, no physical window material saved.
        summary = y.square().mean().numpy().tobytes()
        actual = dict(
            plan_sha256=PLAN_SHA256,
            window=cell.window.identity,
            target=cell.window.target,
            mode=cell.mode,
            nominal_sigma=cell.sigma,
            phase=event.phase,
            step=event.step,
            snapshot=snapshot,
            branch="positive",
            real_tokens=2,
        )
        mats = (
            {(0, 0): torch.eye(2).half(), (35, 31): torch.eye(2).half()}
            if cell.full_map
            else {}
        )
        yield cell, actual, summary, mats


def test_panel_delivery_does_not_advance_optimizer_rng():
    panel = ProbePanel(windows())

    def run(d, enabled):
        torch.manual_seed(16)
        random.seed(512)
        np.random.seed(33)
        model = DomainAwareLinear(4, 4, 1)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        writer = EventWriter(d, reservation=120000, kept_bytes=6000000)
        trace = []
        for step in range(1, 4):
            x = torch.randn(2, 4)
            pred = model(x, torch.zeros(2, dtype=torch.long))
            loss = pred.square().mean()
            loss.backward()
            opt.step()
            opt.zero_grad()
            if enabled and (e := event_for("stage1", step)):
                # For this *unit fixture only*, an independent tiny CPU snapshot; no production snapshot creation.
                frozen = deepcopy(model).requires_grad_(False)
                rows = list(records(panel, e, frozen))
                receipt = panel.complete(
                    e,
                    snapshot_digest=rows[0][1]["snapshot"],
                    cells=reversed(rows),
                    writer=writer,
                    protocol="synthetic",
                    evidence=evidence(),
                    step_envelope=b"update",
                    rank=0,
                    reduced=True,
                )
                with zipfile.ZipFile(receipt.path) as z:
                    assert sum(n.startswith("M/") for n in z.namelist()) == 2 * int(
                        e.deep
                    )
            trace.append(
                tuple(
                    t.detach().numpy().tobytes()
                    for t in (pred, loss, torch.get_rng_state(), *model.parameters())
                )
            )
            trace.append((random.getstate(), np.random.get_state()[1].tobytes()))
        return trace

    with tempfile.TemporaryDirectory() as d:
        assert run(d, False) == run(d, True)


def test_hold_missing_or_changed_cell():
    p = ProbePanel(windows())
    e = event_for("stage1", 1)
    model = DomainAwareLinear(4, 4, 1)
    rows = list(records(p, e, model))
    assert len(rows) == 12 and sum(r[0].full_map for r in rows) == 1
    with tempfile.TemporaryDirectory() as d:
        opts = dict(
            snapshot_digest=rows[0][1]["snapshot"],
            writer=EventWriter(d, reservation=100000, kept_bytes=5000000),
            protocol="fixture",
            evidence=evidence(),
            step_envelope=b"",
            rank=0,
            reduced=True,
        )
        with pytest.raises(ValueError, match="incomplete"):
            p.complete(e, cells=rows[:-1], **opts)
        rows[0][1]["target"] = digest(b"substitution")
        with pytest.raises(ValueError, match="mismatch"):
            p.complete(e, cells=rows, **opts)
