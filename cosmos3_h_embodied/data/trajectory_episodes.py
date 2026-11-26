"""Episode clock and calibration for frozen trajectory codes.

Use the *registered* episode records of the H16 loader, after overlays and
minimum future-window checks; the metadata's nominal split is not enough.
"""
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import random

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml

from .dvrk_schema import parse_arm_specs
from .history_dataset import GeneralizedDvrkHistoryDataset
from .pose_utils import build_abs_pose_from_components, convert_rotation


@dataclass(frozen=True)
class Episode:
    root: Path
    index: int
    size: int
    stride: int
    data: Path
    video: Path
    video_start: float
    source_fps: float

    @property
    def steps(self):
        return (self.size + self.stride - 1) // self.stride

    @property
    def key(self):
        return (self.root.name, self.index)


def _entries(node):
    if isinstance(node, list):
        for child in node:
            yield from _entries(child)
    elif isinstance(node, dict):
        if node.get('_target_') == 'cosmos3_h_embodied.data.history_dataset.GeneralizedDvrkHistoryDataset':
            yield {k: v for k, v in node.items() if k != '_target_'}
        else:
            for child in node.values():
                yield from _entries(child)


def registered(config, split):
    """Take one resolved control config; never resplit, shorten or expand roots."""
    cfg, = (c for c in _entries(yaml.safe_load(Path(config).read_text())) if c['split'] == split)
    dataset = GeneralizedDvrkHistoryDataset(**cfg)
    dataset._register_sources()
    records = {(s, ep) for s, _, _, ep in dataset._episode_records}
    episodes = []
    for s, source in enumerate(dataset._dataset_build_args):
        root = Path(source['root'])
        info = json.loads((root / 'meta/info.json').read_text())
        camera = cfg['video_key']
        if cfg['video_source'] == 'clean' and camera + '_clean' in info['features'] and (root / 'videos' / (camera + '_clean')).is_dir():
            camera += '_clean'
        prefix = 'videos/' + camera + '/'
        stride = info['fps'] / cfg['fps']
        if not stride.is_integer() or stride < 1:
            raise ValueError(f'nonintegral sample clock at {root.name}')
        columns = ['episode_index', 'length', 'data/chunk_index', 'data/file_index', prefix + 'chunk_index', prefix + 'file_index', prefix + 'from_timestamp']
        for part in sorted((root / 'meta/episodes').rglob('*.parquet')):
            for row in pq.read_table(part, columns=columns).to_pylist():
                ep = row['episode_index']
                if (s, ep) not in records:
                    continue
                data = info['data_path'].format(chunk_index=row['data/chunk_index'], file_index=row['data/file_index'])
                video = info['video_path'].format(video_key=camera, chunk_index=row[prefix + 'chunk_index'], file_index=row[prefix + 'file_index'])
                episodes.append(Episode(root, ep, row['length'], int(stride), root/data, root/video, row[prefix+'from_timestamp'], info['fps']))
    if len(episodes) != len(records) or len({e.key for e in episodes}) != len(records):
        raise ValueError('episode metadata does not uniquely cover the registered split')
    return dataset, episodes


@lru_cache(maxsize=4)
def _columns(path):
    table = pq.read_table(path, columns=['episode_index', 'frame_index', 'observation.state', 'action'])
    return (table['episode_index'].to_numpy(), table['frame_index'].to_numpy(),
            *(np.stack(table[name].to_numpy(zero_copy_only=False)).astype(np.float32, copy=False)
              for name in ('observation.state', 'action')))


def raw(ep):
    ids, frames, state, action = _columns(ep.data)
    positions = np.flatnonzero(ids == ep.index)
    positions = positions[np.argsort(frames[positions])]
    if not np.array_equal(frames[positions], np.arange(ep.size)):
        raise ValueError(f'missing or duplicated telemetry at {ep.key}')
    return state[positions], action[positions]


@lru_cache(maxsize=32)
def state_specs(root):
    return parse_arm_specs(json.loads((root / 'meta/modality.json').read_text())['state'], quaternion_order='xyzw')


def measured(state, ep, dataset):
    specs = {s.name: s for s in state_specs(ep.root)}
    fields = []
    for action_spec in dataset._arm_specs_for_root(ep.root):
        s = specs[action_spec.name]
        lo, hi = s.pose_cols
        pose = build_abs_pose_from_components(state[:, lo:lo+3], state[:, lo+3:hi], s.pose_format)
        # The 20D working absolute target flattens the first two rotation columns
        # row-by-row; it is a *separate* stream from the control's action layout.
        fields.extend((pose[:, :3, 3], pose[:, :3, :2].reshape(-1, 6), state[:, [s.gripper_col]]))
    out = np.concatenate(fields, axis=1)
    if out.shape != (len(state), 20) or not np.isfinite(out).all():
        raise ValueError(f'invalid absolute state at {ep.key}')
    return out


def kinematics(ep, dataset, phase=0):
    state, command = (x[phase::ep.stride] for x in raw(ep))
    pose = measured(state, ep, dataset)
    # One executed transition, using the same world-frame delta, relative
    # rotation and step-start jaw setpoint as the control's 1-transition window.
    values = []
    for s in dataset._arm_specs_for_root(ep.root):
        lo, hi = s.pose_cols
        m = build_abs_pose_from_components(state[:, lo:lo+3], state[:, lo+3:hi], s.pose_format)
        delta = np.linalg.inv(m[:-1]) @ m[1:]
        rot = convert_rotation(delta[:, :3, :3], 'matrix', 'rot6d')
        values.extend((np.diff(m[:, :3, 3], axis=0), rot, command[:-1, [s.gripper_col]]))
    action = dataset._normalize_action(torch.from_numpy(np.concatenate(values, axis=1)))
    if not torch.isfinite(action).all():
        raise ValueError(f'invalid action at {ep.key}')
    # Terminal placeholder is never used: the memory consumes action at t-1.
    return torch.from_numpy(pose), torch.cat((action, action.new_zeros(1, 20)))


def left_frames(ep):
    """Decode exactly this episode's native frames (past-only selection downstream)."""
    import av

    first = round(ep.video_start * ep.source_fps)
    frames = []
    with av.open(str(ep.video)) as container:
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        container.seek(int(ep.video_start / stream.time_base), stream=stream, backward=True)
        for frame in container.decode(stream):
            i = round(frame.pts * frame.time_base * ep.source_fps) - first
            if i < 0:
                continue
            if i != len(frames):
                raise ValueError(f'missing video frame at {ep.key}: expected {len(frames)}, got {i}')
            frames.append(frame.reformat(width=96, height=96, format='rgb24').to_ndarray())
            if len(frames) == ep.size:
                break
    if len(frames) != ep.size:
        raise ValueError(f'short video at {ep.key}: {len(frames)} != {ep.size}')
    return torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)


def fit_pose(dataset, episodes, seed=0):
    if dataset._split != 'train':
        raise ValueError('calibration requires registered training episodes')
    registered_keys = {(Path(dataset._dataset_build_args[s]['root']).resolve(), i) for s, _, _, i in dataset._episode_records}
    if not episodes or any((e.root.resolve(), e.index) not in registered_keys for e in episodes):
        raise ValueError('non-training episode in the pose fit')
    pool = sorted((e for e in episodes if e.steps >= 32), key=lambda e:e.steps)
    selected = random.Random(seed).sample(pool, min(len(pool), 200))
    values = np.concatenate([measured(raw(e)[0][::e.stride], e, dataset) for e in selected]).astype(np.float64)
    return {'seed': seed, 'episodes': [e.key for e in selected], 'frames': len(values),
            'mean': values.mean(0).tolist(), 'std': np.maximum(values.std(0), 1e-4).tolist()}
