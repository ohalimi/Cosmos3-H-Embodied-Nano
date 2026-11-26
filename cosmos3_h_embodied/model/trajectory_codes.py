"""Frozen trajectory codes for action conditioning (CAMP, arXiv:2606.21188).

This module consumes normalized episode tensors, not raw robot coordinates. The
caller owns the split, calibration, frame clock and episode boundaries. Nothing
here fits a normalizer or loads a diffusion policy. Public RGB factory revision:
KuanchengWang/CAMP@656554ae750fa5d6ff6f06a104d3d0b9e38ffbde.
"""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


N_BANDS, WIDTH, HIDDEN, ENTRIES = 32, 20, 64, 512
OFFSETS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)


class EndoscopeEncoder(nn.Module):
    """CAMP's robomimic image configuration without importing a policy or simulator.

    One 84x84 crop, ResNet18Conv with group normalization, 32 spatial
    keypoints at temperature 1, 64 rectified visual features, identity proprio.
    """

    def __init__(self):
        super().__init__()
        from torchvision.models import resnet18

        net = resnet18(weights=None, norm_layer=lambda c: nn.GroupNorm(c // 16, c))
        self.pixels = nn.Sequential(*list(net.children())[:-2], nn.Conv2d(512, 32, 1))
        self.readout = nn.Linear(64, 64)
        y, x = torch.meshgrid(torch.linspace(-1, 1, 3), torch.linspace(-1, 1, 3), indexing='ij')
        self.register_buffer('xy', torch.stack((x, y), -1).reshape(9, 2))

    def forward(self, obs):
        image = obs['left']
        n = image.shape[0]
        off = (torch.rand(n, 2, device=image.device) * 12).long() if self.training else image.new_full((n, 2), 6, dtype=torch.long)
        axis = torch.arange(84, device=image.device)
        iy = off[:, :1] + axis
        ix = off[:, 1:] + axis
        crop = image.permute(0, 2, 3, 1)[torch.arange(n, device=image.device)[:, None, None], iy[:, :, None], ix[:, None, :]].permute(0, 3, 1, 2)
        heatmaps = self.pixels(crop).flatten(2)
        coords = heatmaps.softmax(dim=-1) @ self.xy
        return torch.cat((self.readout(coords.flatten(1)).relu(), obs['proprio']), dim=-1)


def fixed_camera_encoder(rgb_shape, proprio_dim):
    if tuple(rgb_shape) != (3, 96, 96) or proprio_dim != WIDTH:
        raise ValueError('expected one 96px left camera and a measured 20D pose')
    return EndoscopeEncoder(), 64 + WIDTH


class EpisodeDictionary(nn.Module):
    """Single-trainer EMA dictionary. Padding never votes; eval never updates."""

    def __init__(self):
        super().__init__()
        self.register_buffer('centers', torch.zeros(ENTRIES, HIDDEN))
        self.register_buffer('mass', torch.zeros(ENTRIES))
        self.register_buffer('moment', torch.zeros(ENTRIES, HIDDEN))
        self.register_buffer('age', torch.zeros(ENTRIES, dtype=torch.long))
        self.register_buffer('ready', torch.tensor(False))

    @torch.no_grad()
    def _seed(self, vectors, slots):
        seeds = vectors[torch.randint(len(vectors), (len(slots),), device=vectors.device)]
        if len(vectors) < len(slots):
            seeds = seeds + torch.randn_like(seeds) * (vectors.std() * .01 + 1e-6)
        self.centers[slots] = self.moment[slots] = seeds
        self.mass[slots] = 1
        self.age[slots] = 0

    def usage(self):
        """Summarize the most recent training forward, not padded or reseeded slots."""
        votes = self.batch_votes
        p = votes[votes > 0] / votes.sum()
        return {'used_codes': len(p), 'perplexity': (-(p * p.log()).sum()).exp().item(),
                'ema_used_codes': int((self.mass > .1).sum()), 'dictionary_size': ENTRIES}

    def forward(self, hidden, valid):
        x = hidden[valid].detach().float()
        if not len(x):
            raise ValueError('empty episode batch')
        if not self.ready:
            if not self.training:
                raise RuntimeError('no trained dictionary')
            self._seed(x, torch.arange(ENTRIES, device=x.device))
            self.ready.fill_(True)
        with torch.no_grad():
            labels = torch.cdist(hidden.detach().reshape(-1, HIDDEN).float(), self.centers).argmin(1)
            q = self.centers[labels].reshape_as(hidden).to(hidden.dtype)
            if self.training:
                used = labels.view_as(valid)[valid]
                counts = used.bincount(minlength=ENTRIES).to(self.mass)
                # Pre-update, non-padding votes. EMA mass includes newly seeded
                # centers and alone can hide collapse. Keep both definitions.
                self.batch_votes = counts
                self.mass.lerp_(counts, .01)
                sums = torch.zeros_like(self.moment).index_add_(0, used, x)
                self.moment.lerp_(sums, .01)
                total = self.mass.sum()
                divisor = (self.mass + 1e-5) * (total / (total + ENTRIES * 1e-5))
                self.centers.copy_(self.moment / divisor[:, None])
                self.age.add_(1).masked_fill_(counts > 0, 0)
                dead = torch.where(self.age >= 100)[0]
                if len(dead):
                    self._seed(x, dead)
        return hidden + (q - hidden).detach(), .25 * F.mse_loss(hidden[valid], q[valid])


class TrajectoryCodes(nn.Module):
    """Two DCT readouts: continuous-state primary, projected-code auxiliary."""

    def __init__(self, encoder, feature_dim, window):
        super().__init__()
        if window < N_BANDS:
            raise ValueError('past window cannot be shorter than the basis')
        self.encoder = encoder
        self.recurrent = nn.LSTM(feature_dim + WIDTH, HIDDEN, num_layers=2, batch_first=True)
        self.dictionary = EpisodeDictionary()
        self.hidden_readout = nn.Linear(HIDDEN, N_BANDS * WIDTH)
        self.to_rows = nn.Sequential(nn.Linear(HIDDEN, 3 * WIDTH), nn.Tanh())
        self.code_readout = nn.Linear(3 * WIDTH, N_BANDS * WIDTH)
        n = torch.arange(window, dtype=torch.float64)[:, None] + .5
        basis = torch.cos(n * torch.arange(N_BANDS) * (math.pi / window)) * math.sqrt(2 / window)
        basis[:, 0] /= math.sqrt(2)
        self.register_buffer('basis', basis.float(), persistent=False)

    def features(self, rgb, proprio, previous_action):
        """Observed image/pose at t, normalized action at t-1 (zero at reset)."""
        batch, steps = previous_action.shape[:2]
        # Recompute visual activations on backward, not the recurrent history.
        # RNG preservation gives the same crop during the recomputation.
        from torch.utils.checkpoint import checkpoint
        images, poses = rgb.flatten(0, 1), proprio.flatten(0, 1)
        def encode(x, p):
            return self.encoder({'left': (x.float() / 127.5 - 1) if x.dtype == torch.uint8 else (2*x-1), 'proprio': p})
        parts = []
        for i in range(0, len(poses), 64):
            inputs = images[i:i+64], poses[i:i+64]
            if self.training and torch.is_grad_enabled() and len(poses) > 64:
                parts.append(checkpoint(encode, *inputs, use_reentrant=False))
            else:
                parts.append(encode(*inputs))
        return torch.cat((torch.cat(parts).reshape(batch, steps, -1), previous_action), dim=-1)

    def forward(self, rgb, proprio, action, lengths):
        f = self.features(rgb, proprio, F.pad(action[:, :-1], (0, 0, 1, 0)))
        h, _ = self.recurrent(f)
        mask = torch.arange(h.size(1), device=h.device) < lengths.to(h.device)[:, None]
        q, commitment = self.dictionary(h, mask)
        return h, self.to_rows(q), commitment

    def losses(self, hidden, code, commitment, trajectory, anchors, lengths):
        """The executed pose trajectory ends at observation t, not the command for t+1.
        Repeat the first measured point on the left.

        Anchors are shared across this length bucket and valid in every episode.
        Only anchors are decoded; memory use does not grow as T times window.
        """
        end = int(lengths.min())
        if not anchors.numel() or int(anchors.min()) < 1 or int(anchors.max()) >= end:
            raise ValueError('anchors must follow a real sample and precede padding')
        w = self.basis.size(0)
        slots = anchors[:, None] + torch.arange(1 - w, 1, device=anchors.device)
        targets = torch.einsum('wk,bmwd->bmkd', self.basis, trajectory[:, slots.clamp_min(0)])
        pred = self.hidden_readout(hidden[:, anchors]).unflatten(-1, (N_BANDS, WIDTH))
        aux = self.code_readout(code[:, anchors]).unflatten(-1, (N_BANDS, WIDTH))
        # For chronologically ordered windows the later prefix is the earlier suffix.
        terms = []
        for lag in OFFSETS:
            keep = anchors + lag < end
            if lag < w and keep.any():
                other = self.hidden_readout(hidden[:, anchors[keep] + lag]).unflatten(-1, (N_BANDS, WIDTH))
                early = torch.einsum('wk,bmkd->bmwd', self.basis[lag:], pred[:, keep])
                late = torch.einsum('wk,bmkd->bmwd', self.basis[:-lag], other)
                terms.append(F.mse_loss(early, late))
        return {'reconstruction': F.mse_loss(pred, targets), 'code_reconstruction': F.mse_loss(aux, targets),
                'overlap': torch.stack(terms).mean() if terms else hidden.new_zeros(()), 'commitment': commitment}

    @torch.inference_mode()
    def track(self, rgb: Tensor, proprio: Tensor, action: Tensor, chunk=64):
        """Encode one episode at a time, retaining both recurrent state and the input delay."""
        if self.training or chunk < 1 or not len(action):
            raise ValueError('export requires eval mode, a positive chunk and a nonempty episode')
        state = previous = None
        rows = []
        for i in range(0, len(action), chunk):
            s = slice(i, i + chunk)
            commands = action[None, s]
            if previous is None:
                previous = torch.zeros_like(commands[:, :1])
            f = self.features(rgb[None, s], proprio[None, s], torch.cat((previous, commands[:, :-1]), 1))
            h, state = self.recurrent(f, state)
            previous = commands[:, -1:]
            q, _ = self.dictionary(h, torch.ones_like(h[..., 0], dtype=torch.bool))
            rows.append(self.to_rows(q)[0].reshape(-1, 3, WIDTH))
        return torch.cat(rows)
