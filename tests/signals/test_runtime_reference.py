"""Tiny source-class worker for launch calibration only. No dataset/model files or trainer."""

import hashlib
import json
import random
import sys


def main():
    import numpy as np
    import torch
    from cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16 import (
        Encoder3d,
    )

    torch.set_num_threads(1)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    model = Encoder3d(
        dim=4, z_dim=3, dim_mult=[1], num_res_blocks=1, temperal_downsample=[]
    )
    opt = torch.optim.SGD(model.parameters(), lr=0.01)

    def digest(x):
        return hashlib.sha256(
            x.detach().cpu().contiguous().numpy().tobytes()
        ).hexdigest()

    rows = []
    for step in range(2):
        x = torch.randn(1, 12, 2, 4, 4)
        y = model(x)
        loss = y.square().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        rows.append(
            dict(
                input=digest(x),
                prediction=digest(y),
                loss=digest(loss),
                params=[digest(p) for p in model.parameters()],
                torch=digest(torch.get_rng_state()),
                python=hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
                numpy=hashlib.sha256(repr(np.random.get_state()).encode()).hexdigest(),
            )
        )
    print(
        json.dumps(
            dict(
                states=rows,
                imported=any(m.startswith("openh_signals") for m in sys.modules),
            )
        )
    )


if __name__ == "__main__":
    main()
