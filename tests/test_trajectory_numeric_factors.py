"""Numerical treatments are isolated from frozen readout and preserve genuine training votes."""
import math

import pytest
import torch
from torch.nn.functional import one_hot

from cosmos3_h_embodied.model.trajectory_codes import EpisodeDictionary
from cosmos3_h_embodied.scripts.pretrain_trajectory_codes import construct
from cosmos3_h_embodied.scripts.trajectory_probes import FACTORS


def close_centers():
    g = torch.Generator().manual_seed(119)
    base = torch.randn(1,64,generator=g)
    centers = base + torch.randn(512,64,generator=g)*.00005
    x = (base + torch.randn(64,64,generator=g)*.00005).reshape(2,32,64)
    return centers,x


def test_squared_training_assignments_do_not_replace_frozen_readout():
    centers,x = close_centers(); flat=x.reshape(-1,64)
    raw = flat.square().sum(1)[:,None] - (2*flat) @ centers.T + centers.square().sum(1)
    ids=raw.argmin(-1).view(2,32); euclidean=torch.cdist(flat,centers).argmin(-1).view(2,32)
    assert not torch.equal(ids,euclidean) and not torch.equal(ids,raw.clamp_min(0).argmin(-1).view(2,32))
    valid=torch.arange(32)<torch.tensor([[32],[19]])
    d=EpisodeDictionary(numeric_factor='squared-training-assignment'); d.ready.fill_(True)
    d.centers.copy_(centers); d.mass.fill_(1.); d.moment.copy_(centers)
    assert torch.equal(d.assign(x),euclidean)  # Frozen readout is always cdist, even on a training instance.
    d.eval(); eval_out,_=d(x,valid); torch.testing.assert_close(eval_out,centers[euclidean]); assert d.batch_votes is None
    d.train(); q,loss=d(x,valid)
    torch.testing.assert_close(q,centers[ids]); torch.testing.assert_close(loss,.25*(x[valid]-centers[ids[valid]]).square().mean())
    votes=ids[valid].bincount(minlength=512)
    assert torch.equal(d.batch_votes,votes) and d.usage()['valid']==int(valid.sum())
    assert d.usage()=={**d.summarize_votes(votes), 'ema_used_codes':int((d.mass>.1).sum())}


@pytest.mark.parametrize('factor',[None,'matrix-center-sums','multiply-add-ema','divide-first-smoothing'])
def test_one_ema_operation_changes_at_a_time(factor):
    if factor is not None: assert factor in FACTORS
    g=torch.Generator().manual_seed(123); d=EpisodeDictionary(numeric_factor=factor)
    ids=torch.randint(512,(5000,),generator=g); samples=torch.randn(5000,64,generator=g)
    mass=torch.rand(512,generator=g); moment=torch.randn(512,64,generator=g)
    d.mass.copy_(mass); d.moment.copy_(moment); counts=ids.bincount(minlength=512)
    sm=(one_hot(ids,512).float().T @ samples if factor=='matrix-center-sums' else
        torch.zeros_like(moment).index_add_(0,ids,samples))
    if factor=='multiply-add-ema':
        mass.mul_(.99).add_(counts.float(),alpha=1.-.99); moment.mul_(.99).add_(sm,alpha=1.-.99)
    else:
        mass.lerp_(counts.float(),.01); moment.lerp_(sm,.01)
    total=mass.sum(); numer=mass+1e-5; denom=total+512e-5
    divisor=(numer/denom)*total if factor=='divide-first-smoothing' else numer*(total/denom)
    d._observe(ids,samples)
    assert torch.equal(d.mass,mass) and torch.equal(d.moment,moment) and torch.equal(d.batch_votes,counts)
    assert torch.equal(d.centers,moment/divisor[:,None])


def test_basis_treatment_preserves_parameter_initialization_and_dct_scope():
    assert 'single-precision-basis' in FACTORS
    torch.manual_seed(17); baseline=construct(619,'cpu',camera=False)
    torch.manual_seed(17); variant=construct(619,'cpu',camera=False,factor='single-precision-basis')
    for k,t in baseline.state_dict().items(): assert torch.equal(t,variant.state_dict()[k])
    angles=(math.pi/619)*(torch.arange(619,dtype=torch.float32)[:,None]+.5)
    ref=(angles*torch.arange(32,dtype=torch.float32)).cos(); ref[:,0]*=math.sqrt(1/619); ref[:,1:]*=math.sqrt(2/619)
    assert torch.equal(ref,variant.basis) and not torch.equal(baseline.basis,variant.basis)
    assert baseline.basis.dtype==variant.basis.dtype==torch.float32
