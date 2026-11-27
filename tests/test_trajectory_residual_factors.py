"""Check each arithmetic treatment against the unchanged objective, graph and causal mask."""
import pytest
import torch
from torch import nn

from cosmos3_h_embodied.model.trajectory_codes import OFFSETS, TrajectoryCodes
from cosmos3_h_embodied.scripts.pretrain_trajectory_codes import total_loss
from cosmos3_h_embodied.scripts.trajectory_probes import FACTORS


RESIDUAL = ('repeat-right-padding','stacked-target-windows','anchor-only-projection',
            'full-window-overlap','auxiliary-last-sum','bandwise-error-reduction')


class ToyCamera(nn.Module):
    def forward(self, obs):
        return torch.cat((obs['left'].flatten(1),obs['proprio']),-1)


def case(f):
    torch.manual_seed(401)
    m=TrajectoryCodes(ToyCamera(), 23, 48, numeric_factor=f)
    n,t = 2,56
    rgb=torch.randn(n,t,3,1,1); p=torch.randn(n,t,20); act=torch.randn_like(p)
    rgb[0,37:]=0; p[0,37:]=0; act[0,36:]=0
    return m, (rgb,p,act,torch.tensor([37,56]))


@pytest.mark.parametrize('factor',RESIDUAL)
def test_training_objectives_and_gradients_preserve_the_other_recipe_choices(factor):
    assert factor in FACTORS
    def evaluate(f):
        m, inputs=case(f); p=inputs[1].requires_grad_(); anchors=torch.tensor([1,5,20,36]); calls=[]; pre=[]
        handle=m.to_rows.register_forward_pre_hook(lambda mod,args:calls.append(args[0].shape[1]))
        before=m.recurrent.register_forward_pre_hook(lambda mod,args:pre.append(args[0].detach()))
        h, c, commit=m(*inputs,for_loss=True); h.retain_grad()
        terms=m.losses(h,c,commit,p,anchors,inputs[3]); assert m.gamma==0 and m.offsets==OFFSETS
        # Verify straight-through gradients from only the code head, not a bypass from the primary objective.
        aux_grad=torch.autograd.grad(terms['code_reconstruction'],h,retain_graph=True)[0]
        total_loss(terms,f).backward(); before.remove(); handle.remove()
        grads={n:v.grad.clone() for n,v in m.named_parameters() if v.grad is not None}
        return h.detach(),terms,aux_grad,grads,calls,pre[0],m.dictionary.batch_votes
    base=evaluate(None); alt=evaluate(factor)
    h,terms,a,grads,calls,pre,votes=alt
    mask=torch.arange(56)<torch.tensor([[37],[56]])
    for k in terms: torch.testing.assert_close(terms[k],base[1][k])
    assert torch.equal(votes,base[-1]) and torch.equal(h[mask],base[0][mask]); assert torch.count_nonzero(a)>0
    torch.testing.assert_close(a,base[2]); assert grads.keys()==base[3].keys()
    for k,t in grads.items(): torch.testing.assert_close(t,base[3][k],atol=1.e-7,rtol=2.e-5)
    assert calls==([4] if factor=='anchor-only-projection' else [56])
    assert torch.equal(pre[mask],base[5][mask]); assert torch.equal(pre[1],base[5][1]);
    if factor=='repeat-right-padding':
        assert pre[0,37:, -20:].equal(pre[0,36:37,-20:].expand_as(pre[0,37:, -20:]))
        assert not torch.equal(pre[0,37:],base[5][0,37:])
    else: assert torch.equal(pre,base[5])


def test_lag_weights_and_endpoints_using_unequal_valid_pair_counts():
    m,_=case('full-window-overlap');h=torch.randn(2,40,64);c=torch.randn(2,40,60);p=torch.randn(2,40,20)
    anchors=torch.tensor([1,13,25,29]);lens=torch.tensor([30,40]);by_lag=[];all_pairs=[]
    for lag in OFFSETS:
        active=[]
        for a in anchors.tolist():
            if a+lag>=min(lens) or lag>=48: continue
            ca,cb=(m.basis @ m.hidden_readout(h[:,j]).reshape(-1,32,20) for j in (a,a+lag))
            active.append((ca[:,lag:] - cb[:,:-lag]).square().mean()); all_pairs.append(active[-1])
        if active: by_lag.append(torch.stack(active).mean())
    v=m.losses(h,c,torch.tensor(0.),p,anchors,lens)['overlap']
    torch.testing.assert_close(v,torch.stack(by_lag).mean())
    assert not torch.isclose(v,torch.stack(all_pairs).mean())
    assert m.losses(h,c,torch.tensor(0.),p,torch.tensor([29]),lens)['overlap']==0


def test_scalar_order_and_band_reduction_are_exercised_not_just_close():
    names=['reconstruction','code_reconstruction','overlap','commitment']
    terms=dict(zip(names,[torch.tensor(x,requires_grad=True) for x in (1.,2.**24,1.,1.)]))
    v=total_loss(terms,'auxiliary-last-sum'); ordinary=total_loss(terms,None)
    assert v!=ordinary and all(x==1 for x in torch.autograd.grad(v,tuple(terms.values())))
    base,_=case(None); alt,_=case('bandwise-error-reduction');g=torch.Generator().manual_seed(7)
    for _ in range(24):
        pred=torch.randn(2,4,32,20,generator=g); target=torch.randn(2,4,32,20,generator=g); sq=(pred-target).square()
        got=alt._reconstruction_error(pred,target)
        assert got.equal(sq.mean(-1).sum(-1).mean()/32)
        if not got.equal(base._reconstruction_error(pred,target)): break
    else: pytest.fail('no observable reduction-order difference')
