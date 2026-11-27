"""Single-device pretraining and frozen episode export for the H16 trajectory-code arm.

Pass the resolved control config, not a new root list. No policy is instantiated.
This driver does not implement the cluster launcher or authorize a training run.
"""
import argparse
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from cosmos3_h_embodied.data.action_normalization import NORM_CLIP, quantile_affine
from cosmos3_h_embodied.data.trajectory_episodes import registered, left_frames, kinematics, fit_pose
from cosmos3_h_embodied.model.trajectory_codes import HIDDEN, OFFSETS, TrajectoryCodes, fixed_camera_encoder


from .trajectory_probes import FACTORS, PROBE_STEPS, action_fit, fixed_panel, mixed_positions


class Observations:
    def __init__(self, dataset, stats, camera=True, action_stats=None, *, max_steps=None,
                 unclipped_quantiles=False, clip_z=False):
        self.max_steps = max_steps
        self.unclipped_quantiles = unclipped_quantiles
        self.clip_z = clip_z
        self.camera = camera
        self.action_stats = None if action_stats is None else tuple(torch.tensor(action_stats[k], dtype=torch.float32) for k in ('mean', 'std'))
        self.dataset = dataset
        self.mean = torch.tensor(stats['mean'], dtype=torch.float32)
        self.std = torch.tensor(stats['std'], dtype=torch.float32)

    @lru_cache(maxsize=16)
    def images(self, e):
        return left_frames(e)

    @lru_cache(maxsize=64)
    def sequence(self, e, phase=0):
        native = self.action_stats is None and not self.unclipped_quantiles
        pose, action = kinematics(e, self.dataset, phase, normalize=native)
        if self.action_stats is not None:
            mean, std = self.action_stats
            calibrated = (action[:-1] - mean) / std
            action[:-1] = calibrated.clamp(-NORM_CLIP, NORM_CLIP) if self.clip_z else calibrated
        elif self.unclipped_quantiles:
            if self.dataset._action_normalization != 'quantile':
                raise ValueError('unclipped quantiles require the original quantile recipe')
            action[:-1] = quantile_affine(action[:-1], self.dataset._load_norm_stats())
        if not native and not torch.isfinite(action).all():
            raise ValueError('nonfinite diagnostic action inputs')
        # Clone the downsample, so the cache does not retain full-rate video.
        # Truncate the 10 Hz sequence, not native video frames, the fit or the pool used for sampling.
        if self.max_steps is not None:
            pose, action = pose[:self.max_steps], action[:self.max_steps]
        rgb = self.images(e)[phase::e.stride][:self.max_steps].contiguous() if self.camera else pose.new_empty((len(pose), 0))
        return rgb, (pose - self.mean) / self.std, action

    def batch(self, pool, rng, count, device, mix_step=None):
        if len(pool) < count:
            chosen = rng.choices(pool, k=count)
        else:
            first = rng.randrange(len(pool) - count + 1)
            chosen = (pool[first:first+count] if mix_step is None else
                      [pool[i] for i in mixed_positions(len(pool), count, first, mix_step)])
        self.last_root_count = len({e.root for e in chosen})
        rows = [self.sequence(e) for e in chosen]
        lengths = torch.tensor([len(row[0]) for row in rows])
        return (*[pad_sequence(col, batch_first=True).to(device) for col in zip(*rows)], lengths)


def save_atomic(obj, dest):
    tmp = dest.with_suffix('.partial')
    torch.save(obj, tmp)
    tmp.replace(dest)


def make_observations(dataset, stats, factor=None, scaling=None):
    needs_fit = factor in ('internal-action-scaling', 'clipped-z-actions')
    if needs_fit != (scaling is not None):
        raise ValueError('supply the frozen training action fit exactly for z-score factors')
    return Observations(dataset, stats, camera=factor != 'no-camera', action_stats=scaling,
                        max_steps=512 if factor == 'cap-sequences' else None,
                        unclipped_quantiles=factor == 'unclipped-quantile-actions', clip_z=factor == 'clipped-z-actions')


def construct(window, device, camera=True, *, factor=None):
    """Build from the registered horizon or a saved effective window and the recorded factor."""
    camera = camera and factor != 'no-camera'
    if camera and factor == 'independent-camera-init':
        with torch.random.fork_rng(devices=[]):
            encoder, dim = fixed_camera_encoder((3, 96, 96), 20)
    else:
        encoder, dim = fixed_camera_encoder((3, 96, 96), 20) if camera else (None, 20)
    if camera and factor == 'center-training-crop':
        encoder.center_training_crop = True
    return TrajectoryCodes(encoder, dim, 512 if factor == 'short-dct-window' else window,
                           hidden_width=128 if factor == 'wide-hidden' else HIDDEN,
                           gamma=1. if factor == 'frequency-decay' else 0.,
                           offsets=(16,) if factor == 'single-consistency-lag' else OFFSETS,
                           numeric_factor=factor).to(device)


def validation(model, obs, pool, rng, device):
    model.eval()
    with torch.inference_mode():
        images, pos, act, lengths = obs.batch(pool, rng, min(16, len(pool)), device)
        h, code, commit = model(images, pos, act, lengths)
        end = int(lengths.min()) - 1
        anchors = torch.tensor([max(1, end//2), end], device=device)
        return model.losses(h, code, commit, pos, anchors, lengths)['reconstruction'].item()


def make_optimizer(model, factor):
    lr = 1e-3 if factor == 'fast-learning-rate' else 1e-4
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)


def pretrain(config, out, device, log=None, factor=None):
    if factor is not None and factor not in FACTORS:
        raise ValueError('unknown diagnostic factor')
    train, tr = registered(config, 'train')
    val, va = registered(config, 'val')
    if {e.key for e in tr} & {e.key for e in va}:
        raise ValueError('overlapping training and validation episode IDs')
    window = max(e.steps for e in tr)
    stats = fit_pose(train, tr)
    scaling = action_fit(train, tr, stats) if factor in ('internal-action-scaling', 'clipped-z-actions') else None
    train_obs, val_obs = (make_observations(data, stats, factor, scaling) for data in (train, val))
    tr, va = (sorted((e for e in eps if e.steps >= 32), key=lambda e:e.steps) for eps in (tr,va))
    if not tr or not va:
        raise ValueError('no eligible training or validation episodes')
    out.mkdir(parents=True, exist_ok=True)
    if scaling is not None:
        (out / 'action-fit.json').write_text(json.dumps(scaling, sort_keys=True, indent=1) + '\n')
    stats_file = out / 'pose-fit.json'
    stats_text = json.dumps(stats, sort_keys=True, indent=1) + '\n'
    if stats_file.exists() and stats_file.read_text() != stats_text:
        raise ValueError('pose calibration changed since initial run')
    stats_file.write_text(stats_text)
    config_digest = hashlib.sha256(Path(config).read_bytes()).hexdigest()
    torch.manual_seed(0)
    rng = random.Random(0)
    model = construct(window, device, factor=factor)
    window = model.basis.size(0)  # Stored horizon is the actual DCT window, not the sequence cap.
    opt = make_optimizer(model, factor)
    first, best = 1, float('inf')
    import torch.distributed.checkpoint as dcp
    checkpoints = out / 'checkpoints'
    latest = checkpoints / 'latest_checkpoint.txt'
    if latest.exists():
        current = checkpoints / latest.read_text().strip()
        state = torch.load(current/'trainer.pt', map_location='cpu', weights_only=True)
        if state.get('factor') != factor or state['config_sha256'] != config_digest or state['pose'] != stats or state['window'] != window:
            raise ValueError('input contract changed on resume')
        weights = model.state_dict()
        dcp.load({'model': weights}, checkpoint_id=current/'model', no_dist=True)
        model.load_state_dict(weights); opt.load_state_dict(state['optimizer'])
        rng.setstate(state['episode_rng']); torch.set_rng_state(state['cpu_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state(state['gpu_rng'], device)
        first, best = state['step']+1, state['best']
    for step in range(first, 1001 if factor else 20001):
        model.train()
        images, pos, act, lengths = train_obs.batch(tr, rng, 8, device, step if factor == 'mixed-root-batches' else None)
        h, code, commit = model(images, pos, act, lengths)
        end = int(lengths.min()) - 1
        lo = min(16, end - 1)
        anchors = torch.tensor(sorted(rng.sample(range(lo, end+1), min(4, end-lo+1))), device=device)
        terms = model.losses(h, code, commit, pos, anchors, lengths)
        loss = sum(terms.values())
        if not torch.isfinite(loss):
            raise FloatingPointError(f'nonfinite memory objective at step {step}')
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True); opt.step()
        if step == 1 or step % 50 == 0:
            record = {'step':step, 'train/loss':loss.item(), **{k:v.item() for k,v in terms.items()}}
            record.update(model.dictionary.usage())
            if factor:
                record['batch_roots'] = train_obs.last_root_count
                if step in PROBE_STEPS:
                    record.update(fixed_panel(model, val_obs, va, step, out))
            print(json.dumps(record), flush=True)
            if log is not None:
                log(record, step=step)
        if step % 500 == 0:
            score = validation(model, val_obs, va, rng, device)
            if not math.isfinite(score):
                raise FloatingPointError(f'nonfinite validation reconstruction at step {step}')
            payload = dict(step=step, factor=factor, window=window, model=model.state_dict(), pose=stats, config_sha256=config_digest, val_reconstruction=score)
            if score < best:
                best = score
                save_atomic(payload, out/'best.pt')
            print(f'validation step={step} primary={score:.8g} best={best:.8g}', flush=True)
        if step == 100 or step % 500 == 0:
            # Publish the pointer only after the trainer and DCP model are complete.
            current = checkpoints / f'iter_{step:09d}'
            current.mkdir(parents=True)
            save_atomic(dict(step=step, factor=factor, window=window, pose=stats, config_sha256=config_digest,
                             optimizer=opt.state_dict(), best=best, episode_rng=rng.getstate(),
                             cpu_rng=torch.get_rng_state(), gpu_rng=torch.cuda.get_rng_state(device)
                             if device.type=='cuda' else None), current/'trainer.pt')
            dcp.save({'model': model.state_dict()}, checkpoint_id=current/'model', no_dist=True)
            pointer = latest.with_suffix('.partial')
            pointer.write_text(current.name + '\n'); pointer.replace(latest)


@torch.inference_mode()
def export(config, out, device):
    path = out/'best.pt'
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('factor') is not None:
        raise ValueError('diagnostic checkpoints cannot be exported')
    if hashlib.sha256(Path(config).read_bytes()).hexdigest() != payload['config_sha256']:
        raise ValueError('control config differs from training')
    model = construct(payload['window'], device).eval().requires_grad_(False)
    model.load_state_dict(payload['model'])
    identity = hashlib.sha256(path.read_bytes()).hexdigest()
    folder = out/'tracks'/identity
    manifest = {'checkpoint_sha256':identity, 'step':payload['step'], 'episodes':{}}
    for split in ('train', 'val'):
        dataset, episodes = registered(config, split)
        obs = Observations(dataset, payload['pose'])
        written = []
        for e in episodes:
            # Window anchors stride in native frames and need not be on phase 0
            # of the 10 Hz clock. Interleave causal 10 Hz tracks, never round up.
            arr = np.empty((e.size, 3, 20), dtype=np.float16)
            for p in range(e.stride):
                images, pos, act = (x.to(device) for x in obs.sequence(e, p))
                arr[p::e.stride] = model.track(images, pos, act).cpu().numpy()
            if not np.isfinite(arr).all():
                raise FloatingPointError(f'nonfinite exported track at {e.key}')
            dest = folder/split/e.root.name/f'{e.index:06}.npy'
            dest.parent.mkdir(parents=True,exist_ok=True)
            tmp = dest.with_suffix('.partial')
            with tmp.open('wb') as f:
                np.save(f, arr, allow_pickle=False)
            tmp.replace(dest)
            written.append([*e.key, len(arr)])
        manifest['episodes'][split] = written
    tmp = folder/'manifest.partial'
    tmp.write_text(json.dumps(manifest,indent=1)+'\n'); tmp.replace(folder/'manifest.json')


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('mode', choices=['pretrain','export']); cli.add_argument('--control-config',required=True,type=Path)
    cli.add_argument('--out',required=True,type=Path); cli.add_argument('--device',default='cuda')
    group = cli.add_mutually_exclusive_group()
    for name in FACTORS:
        group.add_argument('--' + name, dest='factor', action='store_const', const=name)
    args = cli.parse_args()
    if args.mode == 'pretrain':
        pretrain(args.control_config, args.out, torch.device(args.device), factor=args.factor)
    elif args.factor:
        cli.error('diagnostics are pretrain-only')
    else:
        export(args.control_config, args.out, torch.device(args.device))
