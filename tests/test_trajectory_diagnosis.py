"""A diagnostic optimizer changes only the rate, not other AdamW settings or parameters."""
import pytest
import torch
from cosmos3_h_embodied.scripts.pretrain_trajectory_codes import make_optimizer
from cosmos3_h_embodied.scripts.trajectory_probes import FACTORS


def test_fast_rate_is_opt_in_and_otherwise_the_same_optimizer():
    assert 'fast-learning-rate' in FACTORS
    model = torch.nn.Linear(4, 2)
    ordinary = make_optimizer(model, None)
    test = make_optimizer(model, 'fast-learning-rate')
    expected = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    assert type(ordinary) is type(test) is type(expected)
    assert ordinary.defaults == expected.defaults
    assert test.defaults == {**ordinary.defaults, 'lr': 1e-3}
    assert [id(p) for p in test.param_groups[0]['params']] == [id(p) for p in model.parameters()]
    for factor in set(FACTORS) - {'fast-learning-rate'}:
        assert make_optimizer(model, factor).defaults == ordinary.defaults
