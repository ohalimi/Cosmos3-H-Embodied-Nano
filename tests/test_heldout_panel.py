import copy

import pytest

from tools.benchmark.heldout_panel import FAMILIES, make_panel


def test_panel_is_seed_independent_and_stratified():
    split = {'roots': {root: {'val': list(range(count))} for (_, (root, _)), count in zip(FAMILIES.items(), (73, 11))}}
    panel = make_panel(split)
    permuted = copy.deepcopy(split)
    for spec in permuted['roots'].values():
        spec['val'].reverse()
    assert make_panel(permuted) == panel
    for family, (root, count) in FAMILIES.items():
        clips = panel[family]['clips']
        assert {r['root'] for r in clips} == {root}
        assert len({r['episode'] for r in clips}) == count
        assert {r['start_frame'] for r in clips} == {48}
        assert panel[family]['rollouts'] == clips[:3]
    split['roots']['extra'] = {'val': [0]}
    with pytest.raises(ValueError, match='roots'):
        make_panel(split)
