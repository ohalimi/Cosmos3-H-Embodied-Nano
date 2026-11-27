"""CPU-only plan, writer, and non-interference checks. No input artifacts."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import random
import unittest
from tempfile import TemporaryDirectory
import zipfile

from openh_signals.events import (CELLS, EventWriter, event_for, evaluation_reservation,
                                  read_probe_list, reduce_sums, schedule, solver_calls, training_reservation)


def evidence():
    return dict(source_commit='0' * 40, config_digest='1' * 64, probe_digest='2' * 64)


def writer(path, budget=65536):
    return EventWriter(path, reservation=budget, kept_bytes=50 * budget)


def publish(w, **overrides):
    options = dict(evidence=evidence(), summaries={'test': bytes(range(128))}, reduced=True)
    options.update(overrides)
    return w.publish(event_for('stage1', 1), 'smoke', **options)


class ScheduleTest(unittest.TestCase):
    def test_plan_rows(self):
        for phase, cap, p, e, d, b in [('stage1', None, 13, 57, 5, 3375366144),
                                       ('stage2', None, 13, 6, 5, 1931476992),
                                       ('suturebot', 2000, 8, 14, 3, 1054867456),
                                       ('suturebot', 20000, 8, 32, 5, 1912602624),
                                       ('suturebot', 50000, 8, 62, 6, 2715811840)]:
            events = schedule(phase, cap)
            self.assertEqual((len(events), sum(x.deep for x in events)), (e, d))
            self.assertEqual(training_reservation(phase, p, cap), b)
        self.assertEqual(evaluation_reservation(10, 10, 6, 3840), 1660944384)
        self.assertEqual(solver_calls(35), (0, 17, 34))
        self.assertEqual(solver_calls(2), (0, 1))
        self.assertEqual(solver_calls(1), (0,))

    def test_boundary_and_resume(self):
        self.assertEqual(event_for('stage1', 46937).kind, 'update')
        self.assertIsNone(event_for('stage2', 46937))
        self.assertTrue(event_for('stage2', 46937, kind='entry').deep)
        self.assertIsNone(event_for('stage1', 3))
        self.assertIsNone(event_for('stage1', 1, kind='microstep'))
        self.assertTrue(event_for('suturebot', 111, final_step=111).deep)
        self.assertEqual(event_for('stage2', 47000).cells, (CELLS[58 % 9],))

    def test_probe_metadata(self):
        with TemporaryDirectory() as d:
            p = Path(d, 'private'); raw = json.dumps({'synthetic': True}).encode(); p.write_bytes(raw)
            self.assertTrue(read_probe_list(p, expected_sha256=hashlib.sha256(raw).hexdigest())['synthetic'])
            with self.assertRaises(ValueError):
                read_probe_list(p, expected_sha256='0' * 64)


class WriterTest(unittest.TestCase):
    def test_atomic_and_idempotent(self):
        with TemporaryDirectory() as d:
            w = writer(d)
            with concurrent.futures.ThreadPoolExecutor(4) as pool:
                results = list(pool.map(lambda _: publish(w), range(4)))
            self.assertEqual(sum(r.created for r in results), 1)
            r = results[0]
            self.assertEqual(list(Path(d, 'signals').iterdir()), [r.path])
            self.assertEqual(r.path.stat().st_mode & 0o777, 0o600)
            duplicate = publish(writer(d))
            self.assertFalse(duplicate.created)
            self.assertEqual(duplicate.cumulative_bytes, r.path.stat().st_size)
            with self.assertRaisesRegex(ValueError, 'different evidence'):
                publish(w, evidence={**evidence(), 'patch': 'different'})
            with zipfile.ZipFile(r.path) as z:
                manifest = z.read('manifest.json'); payload = z.read('S/test')
                self.assertEqual(hashlib.sha256(manifest).hexdigest(), r.manifest_sha256)
                self.assertEqual(json.loads(manifest)['members']['S/test']['sha256'], hashlib.sha256(payload).hexdigest())
            # Changed content with an otherwise valid container and identical manifest.
            with zipfile.ZipFile(r.path, 'w') as z:
                z.writestr('manifest.json', manifest); z.writestr('S/test', payload[::-1])
            with self.assertRaisesRegex(ValueError, 'checksum'):
                publish(writer(d))

    def test_budget_and_ownership(self):
        with TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, '1 bytes'):
                EventWriter(d, reservation=100, kept_bytes=5000, previous_bytes=1)
            self.assertIsNone(publish(writer(d), rank=1))
            with self.assertRaisesRegex(ValueError, 'unreduced'):
                publish(writer(d), reduced=False)
            self.assertFalse(Path(d, 'signals').exists())
            with self.assertRaisesRegex(ValueError, 'reservation'):
                publish(writer(d, budget=200))
            self.assertEqual(list(Path(d, 'signals').iterdir()), [])
            w = writer(d)
            w.publish(event_for('stage1', 1), 'proto', evidence=evidence(), summaries={}, maps={'map': bytes(32)}, reduced=True)
            with self.assertRaisesRegex(ValueError, 'no full maps'):
                w.publish(event_for('stage1', 2), 'proto', evidence=evidence(), summaries={}, maps={'map': bytes(32)}, reduced=True)

    def test_wandb_spool(self):
        class Run:
            calls = []
            def log(self, fields, **kw):
                self.calls.append((fields, kw))
        with TemporaryDirectory() as d:
            w = writer(d); run = Run()
            with self.assertRaisesRegex(ValueError, 'scalars'):
                publish(w, wandb_run=run, scalars={'wrong': [1]})
            for _ in range(2):
                publish(w, wandb_run=run, scalars={'mass': 0.5})
            self.assertEqual(run.calls, [({'signals/global_step': 1, 'signals/mass': 0.5}, {'commit': False})])

    def test_real_projection_noninterference(self):
        import numpy as np
        import torch
        from cosmos_framework.model.generator.mot.domain_aware_linear import DomainAwareLinear
        def run(d, observed):
            torch.manual_seed(134); random.seed(134); np.random.seed(134)
            model = DomainAwareLinear(4, 3, 2); opt = torch.optim.SGD(model.parameters(), lr=.01)
            trace = []
            for i in (1, 2, 3):
                x = torch.randn(2, 4); y = model(x, torch.tensor([1, 0])); loss = y.square().mean()
                loss.backward(); opt.step(); opt.zero_grad()
                if observed and (event := event_for('stage1', i)):
                    writer(d).publish(event, 'calibration', evidence=evidence(), summaries={},
                                      scalars={'loss': loss.detach().item()}, reduced=True)
                # Compare bytes, not approximate values, for every step and parameter.
                for name, t in [('input', x), ('prediction', y), ('loss', loss),
                                ('rng', torch.get_rng_state()), *model.named_parameters()]:
                    trace.append((f'{i}:{name}', t.detach().numpy().tobytes()))
                trace.append((f'{i}:python_rng', random.getstate()))
                alg, keys, pos, has_gauss, cached = np.random.get_state()
                trace.append((f'{i}:numpy_rng', (alg, keys.tobytes(), pos, has_gauss, cached)))
            return trace
        with TemporaryDirectory() as d:
            off, on = run(d, False), run(d, True)
            for (name, a), (other, b) in zip(off, on, strict=True):
                self.assertEqual(name, other); self.assertEqual(a, b, f'first mismatch: {name}')
            self.assertEqual(len(list(Path(d, 'signals').glob('*.zip'))), 2)


def _collective_test(rank, rendezvous):
    import torch
    import torch.distributed as dist
    dist.init_process_group('gloo', init_method=f'file://{rendezvous}', rank=rank, world_size=3)
    try:
        # Shards 0 and 1 count; rank 2 is a replica of shard 0.
        local = torch.tensor([float(rank + 1 if rank < 2 else 1), 1.0], requires_grad=True)
        before = local.detach().clone()
        got = reduce_sums(local, group=dist.group.WORLD, contributor=rank < 2)
        assert torch.equal(local, before) and not got.requires_grad
        assert torch.equal(got, torch.tensor([3., 2.]))
    finally:
        dist.destroy_process_group()


class ReductionTest(unittest.TestCase):
    def test_disjoint_shards_exclude_replica(self):
        import torch.multiprocessing as mp
        with TemporaryDirectory() as d:
            mp.spawn(_collective_test, args=(os.path.join(d, 'store'),), nprocs=3)
