"""Read the frozen code at the native frame of a registered decision window."""

from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import torch


@lru_cache(maxsize=16)
def _track(path):
    return np.load(path, mmap_mode="r", allow_pickle=False)


class FrozenTracks:
    def __init__(self, manifest, split):
        path = Path(manifest)
        record = json.loads(path.read_text())
        if path.name != "manifest.json" or path.parent.name != record["checkpoint_sha256"]:
            raise ValueError("expected the complete manifest from a frozen checkpoint export")
        self.folder = path.parent / split
        self.lengths = {}
        for root, ep, size in record["episodes"][split]:
            if (root, ep) in self.lengths or size <= 0:
                raise ValueError("invalid or repeated episode in code manifest")
            self.lengths[root, ep] = size

    def at(self, root, episode, frame):
        root, episode, frame = Path(root).name, int(episode), int(frame)
        size = self.lengths[root, episode]
        if not 0 <= frame < size:
            raise IndexError(f"code frame outside episode: {root}/{episode}:{frame}")
        track = _track(self.folder / root / f"{episode:06}.npy")
        if track.shape != (size, 3, 20) or track.dtype != np.float16:
            raise ValueError(f"invalid frozen track for {root}/{episode}")
        rows = torch.from_numpy(track[frame].astype(np.float32))
        if not rows.isfinite().all():
            raise ValueError(f"nonfinite frozen code at {root}/{episode}:{frame}")
        return rows
