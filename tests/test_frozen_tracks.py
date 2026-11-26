import json

import numpy as np
import pytest
import torch

from cosmos3_h_embodied.data.frozen_tracks import FrozenTracks
from cosmos3_h_embodied.data.transforms import ActionTransformPipeline


def test_frozen_track_uses_native_episode_clock(tmp_path):
    folder = tmp_path / ('a' * 64)
    root = folder / 'val' / 'root'
    root.mkdir(parents=True)
    array = np.arange(9 * 3 * 20, dtype=np.float16).reshape(9, 3, 20)
    np.save(root / '000004.npy', array)
    manifest = folder / 'manifest.json'
    manifest.write_text(json.dumps({'checkpoint_sha256': folder.name, 'episodes': {'val': [['root', 4, 9]], 'train': []}}))
    tracks = FrozenTracks(manifest, 'val')
    torch.testing.assert_close(tracks.at('/data/root', 4, 5), torch.from_numpy(array[5].astype(np.float32)))
    with pytest.raises(KeyError):
        FrozenTracks(manifest, 'train').at('/data/root', 4, 5)
    with pytest.raises(IndexError):
        tracks.at('/data/root', 4, -1)


@pytest.mark.parametrize('mode', ['policy', 'inverse_dynamics', 'forward_dynamics'])
@pytest.mark.parametrize('past', [0, 7, 16])
def test_codes_only_extend_clean_prefix(mode, past):
    pipeline = ActionTransformPipeline(pad_keys=[], append_viewpoint_info=False, append_duration_fps_timestamps=False,
                                       append_resolution_info=False, max_action_dim=64)
    sample = dict(mode=mode, video=torch.zeros(3, 25, 16, 16), action=torch.randn(24, 20),
                  history_action=torch.randn(past, 20))
    code = torch.randn(3, 20)
    control = pipeline(dict(sample), resolution='256')
    treatment = pipeline(dict(sample, trajectory_code=code), resolution='256')
    a, c = control['sequence_plan'], treatment['sequence_plan']
    torch.testing.assert_close(treatment['action'][3:], control['action'])
    torch.testing.assert_close(treatment['action'][:3, :20], code)
    assert treatment['action'][:3, 20:].count_nonzero() == 0
    assert c.condition_frame_indexes_action == [0, 1, 2] + [j + 3 for j in a.condition_frame_indexes_action]
    assert c.action_start_frame_offset + 3 == a.action_start_frame_offset
    assert c.condition_frame_indexes_vision == a.condition_frame_indexes_vision
