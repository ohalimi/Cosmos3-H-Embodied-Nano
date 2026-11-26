"""Paired policy evaluation at a frozen-code checkpoint, using the ordinary eval CLI.

Use model mode wam, held-out split, and the same sampling parameters as the joint
comparison. Choose windows without looking at outcomes or at the value of their
codes, then hold the batch and diffusion seed fixed across each triplet.
"""
import copy
import json
from pathlib import Path

import numpy as np
import torch

from cosmos3_h_embodied.scripts.eval import (
    EVAL_DATASET_OVERRIDES, EvalArgs, create_dataset, extract_action_prediction_mask,
    extract_gt_action, init_output_dir, register_checkpoints, tyro_cli,
)


def score_triplet(generate, batch, donor, seed):
    mask = extract_action_prediction_mask(batch)
    gt = extract_gt_action(batch)
    if gt is None or mask is None or gt.shape[1] != 20 or mask.sum().item() != 24 or mask[:3].any():
        raise ValueError('ablation requires 24 predicted future rows and three clean code rows')
    real = batch['action'][0, :3, :20].clone()
    preds, mse = {}, {}
    for name, code in [('real', real), ('zero', torch.zeros_like(real)), ('shuffled', donor)]:
        candidate = copy.deepcopy(batch)
        candidate['action'][0, :3, :20] = code
        output = generate(name, candidate, seed).detach().cpu().float()
        if output.shape != gt.shape or not torch.isfinite(output).all():
            raise ValueError('invalid policy prediction')
        preds[name] = output[mask]
        mse[name] = (preds[name] - gt.cpu()[mask]).square().mean().item()
    return dict(mse=mse, zero_changed_input=bool(real.count_nonzero()),
                zero_changed_prediction=not torch.equal(preds['real'], preds['zero']),
                zero_changed_mse=mse['real'] != mse['zero'], donor_code_differs=not torch.equal(real, donor))


def main():
    args = tyro_cli(EvalArgs, description=__doc__)
    if args.dataset.model_mode != 'wam' or args.dataset.dataset_split != 'val':
        raise ValueError('only held-out policy windows are admissible')
    args.setup.experiment_overrides = [*args.setup.experiment_overrides, *EVAL_DATASET_OVERRIDES]
    setup = args.setup.build_setup()
    init_output_dir(setup.output_dir)
    register_checkpoints()
    samples = create_dataset(args.dataset.model_copy(update={'num_samples': None}), config_args=setup)
    # Move across the whole registered validation split, not only adjacent windows
    # from its first episode. Select the first eight distinct actual fetched episodes.
    ids = samples._sample_ids
    samples._sample_ids = [ids[k] for k in np.unique(np.linspace(0, len(ids)-1, min(64,len(ids))).round().astype(int))]
    windows, episodes = [], set()
    for sample_args, batch in samples:
        root, ep, frame = samples._dataset._code_key
        key = Path(root).name, ep
        if key not in episodes:
            windows.append((sample_args, batch, (*key, frame)))
            episodes.add(key)
        if len(windows) == 8:
            break
    if len(windows) < 2:
        raise ValueError('a cross-episode shuffle needs distinct held-out episodes')
    pipe = setup.get_inference_cls().create(setup)
    dest = Path(setup.output_dir)
    records = []
    for i, (sa, batch, key) in enumerate(windows):
        _, other, donor_key = windows[(i+1) % len(windows)]
        assert key[:2] != donor_key[:2]
        donor = other['action'][0, :3, :20].clone()
        base = sa.build_sample(model_config=pipe.model_config)
        def generate(label, candidate, seed):
            a = base.model_copy(deep=True, update={'output_dir': dest/str(i)/str(seed)/label, 'seed':seed})
            outputs = pipe.generate_batch([a], candidate)
            if len(outputs) != 1 or outputs[0].status != 'success':
                raise RuntimeError(f'policy generation failed: {key} {seed} {label}')
            return torch.as_tensor(outputs[0].outputs[0].content['action'])
        for seed in range(base.seed, base.seed + 3):
            r = dict(window=key, donor=donor_key, seed=seed, **score_triplet(generate,batch,donor,seed))
            records.append(r)
    result = dict(windows=len(windows), seeds_per_window=3, records=records,
                  policy_mse={k:np.mean([r['mse'][k] for r in records]).item() for k in ('real','zero','shuffled')},
                  zero_changed_predictions=sum(r['zero_changed_prediction'] for r in records),
                  zero_changed_mse=sum(r['zero_changed_mse'] for r in records),
                  shuffled_pairs_different=sum(r['donor_code_differs'] for r in records[::3]))
    (dest/'paired_policy.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
