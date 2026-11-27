import pytest
import torch
from cosmos3_h_embodied.model.trajectory_codes import EndoscopeEncoder, EpisodeDictionary, TrajectoryCodes
from cosmos3_h_embodied.scripts import pretrain_trajectory_codes as train


def test_centered_training_crop_keeps_the_ordinary_rng_advance():
    torch.manual_seed(33); cam=EndoscopeEncoder(); xs={'left':torch.rand(2,3,96,96),'proprio':torch.rand(2,20)}
    cam.eval(); centered=cam(xs); cam.train(); torch.manual_seed(41); random=cam(xs); state=torch.get_rng_state()
    cam.center_training_crop=True; torch.manual_seed(41); got=cam(xs)
    assert torch.equal(centered,got) and not torch.equal(random,got) and torch.equal(state,torch.get_rng_state())


def test_constructor_draw_settings_are_targeted(monkeypatch):
    monkeypatch.setattr(train,'fixed_camera_encoder',lambda *a:(torch.nn.Linear(5,7),84))
    torch.manual_seed(17); base=train.construct(100,'cpu'); torch.manual_seed(17); isolated=train.construct(100,'cpu',factor='independent-camera-init');torch.manual_seed(17)
    with torch.random.fork_rng(devices=[]): enc,dim=train.fixed_camera_encoder()
    check=TrajectoryCodes(enc,dim,100)
    assert all(torch.equal(v,check.state_dict()[k]) for k,v in isolated.state_dict().items()); assert torch.equal(base.encoder.weight,isolated.encoder.weight) and not torch.equal(base.recurrent.weight_ih_l0,isolated.recurrent.weight_ih_l0)
    torch.manual_seed(17); shifted=train.construct(100,'cpu',factor='embedding-init-draws');assert torch.equal(shifted.recurrent.weight_ih_l0,base.recurrent.weight_ih_l0) and torch.equal(shifted.hidden_readout.weight,base.hidden_readout.weight) and not torch.equal(base.to_rows[0].weight,shifted.to_rows[0].weight)


def test_vq_stream_isolated_from_crops_and_checkpointed():
    ds=[EpisodeDictionary(64,numeric_factor='independent-vq-draws') for _ in range(2)]; plain=EpisodeDictionary(64); x=torch.arange(9*64,dtype=torch.float).reshape(9,64); slots=torch.arange(12)
    for i,model in enumerate(ds):
        torch.manual_seed(17); torch.rand(9+i*47); before=torch.get_rng_state();model._populate(slots,x); after=torch.get_rng_state(); torch.set_rng_state(before);plain._populate(slots,x); assert torch.equal(torch.get_rng_state(),after)
    assert torch.equal(ds[0].centers,ds[1].centers) and not torch.equal(ds[0].centers,plain.centers)
    restored=EpisodeDictionary(64,numeric_factor='independent-vq-draws'); restored.load_state_dict(ds[0].state_dict(),strict=True); ds[0]._populate(slots,x); torch.rand(223);restored._populate(slots,x);assert torch.equal(restored.centers,ds[0].centers)
