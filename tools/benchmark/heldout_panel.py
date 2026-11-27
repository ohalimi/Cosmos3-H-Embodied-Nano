"""Score a fixed two-root validation panel using the joint and rollout evaluators."""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

FAMILIES = {
    'dvrk_suturebot_mono': ('jhu/hf_suturebot', 28),
    'dvrk_jhu': ('jhu/imerse/wound_closure/point_labeled/fausto_0_1_jesse_0_1_2_labeled', 4),
}
MODES = ('forward_dynamics', 'inverse_dynamics', 'wam')


def make_panel(manifest):
    """Choose distinct episodes proportionally, without looking at predictions or drawing a seed."""
    if set(manifest['roots']) != {r for r, _ in FAMILIES.values()}:
        raise ValueError('unexpected validation roots')
    panel = {}
    for family, (root, n) in FAMILIES.items():
        episodes = sorted(manifest['roots'][root]['val'])
        if len(set(episodes)) != len(episodes) or len(episodes) < n:
            raise ValueError(f'invalid episode list: {root}')
        chosen = [episodes[round(i * (len(episodes) - 1) / (n - 1))] for i in range(n)]
        # Source frame 48 supplies a full 16-step history at 30 Hz source / 10 Hz model.
        clips = [dict(root=root, episode=e, start_frame=48) for e in chosen]
        panel[family] = dict(eligible=episodes, clips=clips, rollouts=clips[:3])
    return panel


def exact_samples(samples, family, panel):
    from cosmos3_h_embodied.scripts.quant_eval import _fetch_exact, _window_index
    from cosmos_framework.scripts.dataset_utils import set_dataset_mode

    ds = samples._dataset
    windows = _window_index(ds)
    root = os.path.normpath(os.path.join(os.environ['OPENH_DATA_ROOT'], FAMILIES[family][0]))
    spec = panel[family]
    if set(windows) != {(root, e) for e in spec['eligible']}:
        raise ValueError(f'actual validation split differs from frozen {family} split')
    if set(samples._modes) != set(MODES):
        raise ValueError('held-out scoring needs all three joint modes')
    ids = [windows[root, c['episode']][c['start_frame']] for c in spec['clips']]

    class Replay:
        def __len__(self):
            return len(ids) * len(MODES)

        def __iter__(self):
            for mode in samples._modes:
                set_dataset_mode(ds, mode)
                for idx in ids:
                    yield samples._make_sample_from_raw(_fetch_exact(ds, idx), idx, mode)

    return Replay()


def score(args):
    from tools.benchmark.summarize_eval import load_runs, summarize, write_table

    raw = args.manifest.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.manifest_sha256:
        raise ValueError('frozen manifest digest mismatch')
    panel = make_panel(json.loads(raw))
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    panel_path = out / 'panel.json'
    panel_path.write_text(json.dumps(panel, indent=2, sort_keys=True) + '\n')
    plan_path = out / 'rollouts.json'
    plan_path.write_text(json.dumps([r for p in panel.values() for r in p['rollouts']]))
    joint, quant = out / 'eval_joint', out / 'quant_eval'
    for family in FAMILIES:
        common = ['--no-guardrails', '--checkpoint-path', str(args.checkpoint), '--config-file',
                  'cosmos3_h_embodied/configs/config.py', '--experiment', 'nano-h16-suturebot',
                  '--dataset.dataset', family, '--dataset.dataset-split', 'val', '--seed', str(args.seed),
                  '--num-steps', '35', '--guidance', '6.0']
        if args.weights == 'regular':
            common.append('--no-use-ema-weights')
        for module, extra in [('eval', ['--dataset.model-mode', 'joint', '--panel', str(panel_path), '-o', str(joint / family)]),
                              ('quant_eval', ['--plan', str(plan_path), '--num-chunks', '3', '--num-seeds', '1',
                                              '--save-videos', '-o', str(quant / family)])]:
            subprocess.run([sys.executable, '-m', f'cosmos3_h_embodied.scripts.{module}', *common, *extra], check=True)
    data = load_runs([joint])
    for family, (_, n) in FAMILIES.items():
        indices = [{idx for f, m, idx in data if f == family and m == mode} for mode in MODES]
        if len(indices[0]) != n or any(i != indices[0] for i in indices):
            raise ValueError(f'incomplete joint results for {family}')
    if len(data) != 96 or any(len(rows) != 1 for rows in data.values()):
        raise ValueError('duplicate or unexpected joint results')
    table = summarize(data)
    (joint / 'eval_summary.json').write_text(json.dumps(table, indent=2))
    write_table(table, joint / 'eval_summary.md')
    expected = {(r['root'], r['episode'], r['start_frame'], args.seed, 3) for p in panel.values() for r in p['rollouts']}
    found = []
    for path in quant.rglob('rollout.json'):
        meta = json.loads(path.read_text())
        if not path.with_name('metrics.json').is_file():
            raise ValueError(f'missing metrics for {path}')
        found.append(tuple(meta[k] for k in ('root', 'episode', 'start_frame', 'seed', 'num_chunks')))
    if len(found) != len(expected) or set(found) != expected:
        raise ValueError('quant rollouts do not match plan')
    # Quant's installed scorer imports its training environment at module load.
    subprocess.run([sys.executable, '-m', 'cosmos3_h_embodied.scripts.quant_eval', *common,
                    '--summarize-only', '-o', str(quant)], check=True)
    psnr = json.loads((quant / 'summary.json').read_text())['overall']['psnr']
    assert psnr['n'] == len(found)
    row = table['All']
    result = dict(policy_action_mse=row['policy_mse_mean'], fd_psnr_db=row['fd_psnr'],
                  id_action_mse=row['id_mse'], rollout_psnr_db=psnr['mean'])
    if not all(math.isfinite(v) for v in result.values()):
        raise ValueError('nonfinite held-out score')
    result.update(seed=args.seed, panel_sha256=hashlib.sha256(panel_path.read_bytes()).hexdigest())
    (out / 'scores.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'checkpoint', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--weights', choices=('regular', 'ema'), required=True)
    score(parser.parse_args())


if __name__ == '__main__':
    main()
