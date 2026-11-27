"""One-factor pretraining diagnostics. No episode-track writer or policy model."""
import copy
import hashlib
import json
import random

import numpy as np
import torch

from cosmos3_h_embodied.data.trajectory_episodes import kinematics


FACTORS = ('mixed-root-batches', 'no-camera', 'internal-action-scaling', 'fast-learning-rate')
PROBE_STEPS = (600, 800, 1000)


def mixed_positions(n, count, first, step):
    """Permute ranks inside 8-batch-wide length buckets, not roots with replacement.

    A contiguous-window sampler includes boundary ranks less often. Within each
    bucket exchange only slots with identical window multiplicity. For any
    fixed step the permutation is independent of the random start, preserving
    every episode's inclusion probability exactly, with no duplicate draw.
    """
    width = 8 * count
    chosen = []
    for block in range(first // width, (first + count - 1) // width + 1):
        lo, hi = block * width, min((block + 1) * width, n)
        groups = {}
        for i in range(lo, hi):
            ways = min(i, n - count) - max(0, i - count + 1) + 1
            groups.setdefault(ways, []).append(i)
        perm = list(range(lo, hi))
        rng = random.Random(f'{step}:{block}:0')
        for slots in groups.values():
            values = slots.copy()
            rng.shuffle(values)
            for i, v in zip(slots, values):
                perm[i - lo] = v
        chosen.extend(perm[i-lo] for i in range(max(first, lo), min(first+count, hi)))
    return chosen


def action_fit(dataset, train, pose_fit):
    """Fit only executed transitions, in exactly the pose fit's episode order."""
    if dataset._split != 'train':
        raise ValueError('action fit requires training split')
    lookup = {e.key: e for e in train}
    keys = [tuple(k) for k in pose_fit['episodes']]
    if len(set(keys)) != len(keys) or not set(keys) <= lookup.keys():
        raise ValueError('pose-fit keys are not unique registered training episodes')
    rows = np.concatenate([kinematics(lookup[k], dataset, normalize=False)[1][:-1].numpy() for k in keys])
    if len(rows) != pose_fit['frames'] - len(keys) or not np.isfinite(rows).all():
        raise ValueError('invalid action-fit transitions')
    return dict(mean=rows.mean(0).tolist(), std=rows.std(0).clip(1e-4).tolist())


@torch.inference_mode()
def fixed_panel(model, observations, episodes, step, out):
    # Match the frozen public CPU readout without perturbing the training model.
    frozen = copy.deepcopy(model).cpu().eval().requires_grad_(False)
    pool = sorted(episodes, key=lambda e: e.key)
    manifest = [[*e.key, 0, min(512, e.steps)] for e in pool]
    digest = hashlib.sha256((json.dumps(manifest, indent=1) + '\n').encode()).hexdigest()
    if digest != 'b9608d6bd9629a8d6834788af2191155ce6bd537796f82b3c171c94dc50cd8db':
        raise ValueError('panel identity mismatch')
    hist = torch.zeros(512, dtype=torch.long)
    for e, (_, _, _, stop) in zip(pool, manifest):
        rgb, pose, act = (x[:stop][None] for x in observations.sequence(e))
        previous = torch.nn.functional.pad(act[:, :-1], (0, 0, 1, 0))
        hidden, _ = frozen.recurrent(frozen.features(rgb, pose, previous))
        hist += frozen.dictionary.assign(hidden).flatten().bincount(minlength=512)
    measures = frozen.dictionary.summarize_votes(hist)
    if measures['valid'] != 13524:
        raise ValueError('panel vote count mismatch')
    result = dict(step=step, counts=hist.tolist(), panel_sha256=digest, **measures)
    (out / f'probe-{step}.json').write_text(json.dumps(result, allow_nan=False) + '\n')
    return {f'probe/{k}': v for k, v in measures.items()}
