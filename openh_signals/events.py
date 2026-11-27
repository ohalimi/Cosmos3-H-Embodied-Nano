"""Plan-v1 scheduling and the single-writer boundary. No model or data imports."""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import itertools
import json
import math
import os
import re
import zipfile
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

MIB = 1 << 20
PLAN_SHA256 = '2df5076fbfffd8fa441d21ad805b7d52d91b4822a3301bdfb3669efa902e92ba'
CELLS = tuple(itertools.product(('forward_dynamics', 'inverse_dynamics', 'policy'), ('low', 'mid', 'high')))
_LABEL = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\Z')


@dataclass(frozen=True)
class Event:
    step: int
    phase: str
    kind: str
    deep: bool
    cells: tuple


@lru_cache(maxsize=32)
def schedule(phase, final_step=None):
    """Canonical global order, including events that predate a resumed process."""
    early = {1 << i for i in range(10)}
    if phase == 'stage1' and final_step in (None, 46937):
        deep = {1, 8, 64, 512, 46937}
        steps = early | deep | set(range(1000, 46001, 1000))
    elif phase == 'stage2' and final_step in (None, 50000):
        deep = {46937, 47437, 48000, 49000, 50000}
        steps = deep | {47000}
    elif phase == 'suturebot' and type(final_step) is int and 1 <= final_step <= 50000:
        deep = {s for s in {100, 1000, 5000, 10000, 25000, 50000, final_step} if s <= final_step}
        steps = {s for s in early | {100, 500} if s <= final_step} | set(range(1000, final_step + 1, 1000)) | deep
    else:
        raise ValueError('unrecognized phase or cap outside plan v1')
    offset = len(schedule('stage1')) if phase == 'stage2' else 0
    return tuple(Event(s, phase, 'entry' if (phase, s) == ('stage2', 46937) else 'update', s in deep,
                       CELLS if s in deep else (CELLS[(i + offset) % len(CELLS)],)) for i, s in enumerate(sorted(steps)))


def event_for(phase, completed_step, *, kind='update', final_step=None):
    """Call *after* a successful optimizer update, never on a microstep/scaler skip."""
    return next((e for e in schedule(phase, final_step) if (e.step, e.kind) == (completed_step, kind)), None)


def solver_calls(n):
    if type(n) is not int or n <= 0:
        raise ValueError('invalid actual solver call count')
    return tuple(sorted({0, (n - 1) // 2, n - 1}))


def _counts(*values):
    if any(type(x) is not int or x < 0 for x in values):
        raise ValueError('counts and bytes must be non-negative integers')


def training_reservation(phase, probes, final_step=None):
    _counts(probes)
    events = schedule(phase, final_step)
    d = sum(e.deep for e in events)
    k = probes * (len(events) + 8 * d)
    return MIB * (2 * k + len(events) + 128 * d)


def evaluation_reservation(strata, individual, combinations, instances):
    _counts(strata, individual, combinations, instances)
    return 2 * MIB * (strata + individual) * combinations * 3 * 2 + 4096 * instances + 129 * MIB


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def read_probe_list(path, *, expected_sha256):
    """Only the attested metadata file; never resolve, open or substitute a source window."""
    with Path(path).open('rb') as f:
        raw = f.read(MIB + 1)
    if len(raw) > MIB or _sha(raw) != expected_sha256:
        raise ValueError('private probe manifest size/digest mismatch')
    result = json.loads(raw)
    if not isinstance(result, (dict, list)) or not result:
        raise ValueError('private probe manifest is empty')
    return result


def reduce_sums(local_sums, *, group, contributor):
    """All ranks in the identified group call in identical order; sum disjoint contributions.

    Caller supplies the ownership decision: exactly one contributor per logical
    replica/shard. Counts travel as sums too, not already-normalized means. This
    does NOT make ranks holding partial softmax denominators independently valid.
    """
    import torch
    import torch.distributed as dist
    if type(contributor) is not bool or not dist.is_initialized() or group is None:
        raise ValueError('must bind shard group and explicit replica ownership')
    with torch.no_grad():
        result = local_sums.detach().clone() if contributor else torch.zeros_like(local_sums)
        dist.all_reduce(result, op=dist.ReduceOp.SUM, group=group)
    return result


def _json(obj):
    return json.dumps(obj, allow_nan=False, ensure_ascii=True, sort_keys=True, separators=(',', ':')).encode()


def _name(value):
    if not isinstance(value, str) or not _LABEL.fullmatch(value):
        raise ValueError('unsafe or oversized event/member name')
    return value


def _scalar_series(values):
    if len(values) > 256:
        raise ValueError('too many scalar series')
    for k, v in values.items():
        _name(k)
        if type(v) not in (float, int) or not math.isfinite(v):
            raise ValueError('W&B accepts finite Python numeric scalars only')
    return values


@dataclass(frozen=True)
class Receipt:
    path: Path
    manifest_sha256: str
    bytes: int
    cumulative_bytes: int
    created: bool


def _verify_existing(path, manifest, entries):
    with zipfile.ZipFile(path) as z:
        if sorted(z.namelist()) != sorted(['manifest.json', *entries]) or z.read('manifest.json') != manifest:
            raise ValueError('event key already exists with different evidence or payload')
        for name, expected in entries.items():
            with z.open(name) as f:
                if hashlib.file_digest(f, 'sha256').hexdigest() != _sha(expected):
                    raise ValueError('published member checksum mismatch')


class _BoundedFile:
    """zipfile's seekable target. Enforce reservation even on incompressible framing."""
    def __init__(self, f, limit):
        self.f, self.limit = f, limit

    def write(self, data):
        if self.f.tell() + len(data) > self.limit:
            raise ValueError('serialized event exceeds remaining reservation')
        return self.f.write(data)

    def __getattr__(self, name):
        return getattr(self.f, name)


class EventWriter:
    """Private directory is the ledger: sum actual published container lengths under flock.

    reservation is the TOTAL reserved for this directory, not a new quota on
    every restart. previous_bytes is the separate charge outside this directory;
    kept_bytes counts only unique retained checkpoint bytes of the parent run.
    This class never opens a checkpoint, requests a collective, or starts W&B.
    """
    def __init__(self, run_dir, *, reservation, kept_bytes, previous_bytes=0):
        _counts(reservation, kept_bytes, previous_bytes)
        deficit = reservation + previous_bytes - kept_bytes // 50
        if deficit > 0:
            raise ValueError(f'2% storage ceiling exceeded by {deficit} bytes')
        self.directory = Path(run_dir) / 'signals'
        self.reservation = reservation
        self.previous_bytes = previous_bytes

    def publish(self, event, protocol, *, evidence, summaries, maps=None, scalars=None,
                rank=0, reduced=False, wandb_run=None):
        """Publish once, only after reductions. Duplicate exact events are verified, not overwritten.

        summaries/maps are compact named bytes, never live tensors. W&B receives
        ONLY bounded scalar fields (use aggregate, non-identifying series names);
        they are also durably spooled inside the event if transport later fails.
        """
        _counts(rank, event.step)
        if not reduced:
            raise ValueError('unreduced event; bind reduction and writer ownership first')
        if rank != 0:
            return None
        key = f'{event.step}.{_name(event.phase)}.{_name(protocol)}'
        for field, regex in [('source_commit', '[0-9a-f]{40}'), ('config_digest', '[0-9a-f]{64}'),
                             ('probe_digest', '[0-9a-f]{64}')]:
            if not re.fullmatch(regex, evidence.get(field, '')):
                raise ValueError(f'missing or malformed {field}')
        scalars = _scalar_series(scalars or {})
        entries = {}
        maps = maps or {}
        if maps and not event.deep:
            raise ValueError('no full maps outside deep events')
        if len(maps) > 4 or any(len(v) > 32 * MIB for v in maps.values()):
            raise ValueError('full map exceeds v1 reservation; do not truncate')
        for tier, fields in [('S', summaries), ('M', maps)]:
            for name, b in fields.items():
                if not isinstance(b, bytes):
                    raise TypeError('serialize only compact observer-owned payloads')
                entries[f'{tier}/{_name(name)}'] = b
        if len(entries) > 8192 or sum(map(len, entries.values())) > self.reservation:
            raise ValueError('raw payload exceeds reservation or member count cap')
        manifest = _json({'plan': 'v1', 'plan_sha256': PLAN_SHA256, 'event': asdict(event), 'protocol': protocol,
                          'evidence': evidence, 'scalars': scalars, 'members': {
                              n: {'bytes': len(b), 'sha256': _sha(b)} for n, b in entries.items()}})
        if len(manifest) > MIB:
            raise ValueError('metadata and scalar spool exceed event envelope')
        self.directory.mkdir(mode=0o700, exist_ok=True)
        target = self.directory / (key + '.zip')
        with contextlib.ExitStack() as stack:
            dfd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
            stack.callback(os.close, dfd)
            fcntl.flock(dfd, fcntl.LOCK_EX)
            used = sum(p.stat().st_size for p in self.directory.glob('*.zip'))
            if used > self.reservation:
                raise ValueError('published files already exceed reservation')
            if target.exists():
                _verify_existing(target, manifest, entries)
                return Receipt(target, _sha(manifest), target.stat().st_size, self.previous_bytes + used, False)
            # Only one in-flight private file; a crash before link leaves no event.
            # A stale file is ours exclusively under the directory-wide lock.
            tmp = self.directory / '.pending-v1'
            tmp.unlink(missing_ok=True)
            fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            stack.callback(tmp.unlink, missing_ok=True)
            with os.fdopen(fd, 'w+b') as f:
                bounded = _BoundedFile(f, self.reservation - used)
                with zipfile.ZipFile(bounded, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
                    for n, b in entries.items():
                        z.writestr(n, b)
                    z.writestr('manifest.json', manifest)
                size = f.tell()
                if used + size > self.reservation:
                    raise ValueError(f'event exceeds remaining reservation by {used + size - self.reservation} bytes')
                f.flush()
                os.fsync(f.fileno())
            os.link(tmp, target)  # never overwrite a published record
            os.fsync(dfd)
        if wandb_run is not None and scalars:
            # No step advancement in the trainer's W&B run. No IDs, maps, arrays or media.
            wandb_run.log({'signals/global_step': event.step, **{f'signals/{k}': v for k, v in scalars.items()}}, commit=False)
        return Receipt(target, _sha(manifest), size, self.previous_bytes + used + size, True)
