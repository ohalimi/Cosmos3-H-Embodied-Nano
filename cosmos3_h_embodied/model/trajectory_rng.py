"""Reproducible private draw stream for a trajectory diagnostic (not the default recipe)."""
import torch
from torch import nn


class IsolatedDraws(nn.Module):
    """Keep a VQ-only generator while leaving external draws at the ordinary positions.

    Store the generator state in the model so a checkpoint resume does not reset it.
    CPU and CUDA byte states differ in length, so use a length-tagged fixed-size buffer.
    """
    def __init__(self):
        super().__init__()
        self.register_buffer('state_bytes', torch.zeros(8192, dtype=torch.uint8))
        self.register_buffer('length', torch.zeros((), dtype=torch.long))

    def sample(self, samples, slots):
        g = torch.Generator(device=samples.device)
        n = int(self.length)
        if n:
            g.set_state(self.state_bytes[:n].cpu())
        else:
            g.manual_seed(torch.initial_seed())
        # Advance the shared generator by the usual draws of this call; the crop
        # stream is not rewound, restarted or changed into a constant sequence.
        torch.randint(samples.size(0), slots.shape, device=samples.device)
        ix = torch.randint(samples.size(0), slots.shape, device=samples.device, generator=g)
        seeds = samples.index_select(0, ix)
        if samples.size(0) < slots.numel():
            torch.randn_like(seeds)
            noise = torch.randn(seeds.shape, device=seeds.device, dtype=seeds.dtype, generator=g)
            seeds += noise * (.01 * samples.std() + 1e-6)
        state = g.get_state(); n = state.numel()
        if n > self.state_bytes.numel():
            raise ValueError('generator state exceeds diagnostic checkpoint capacity')
        self.state_bytes[:n].copy_(state); self.length.fill_(n)
        return seeds
