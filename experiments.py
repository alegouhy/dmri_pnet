# The local part of every paper experiment: derived sets, tissue masks, predictions on test sets, evaluation
# and summaries. The heavy part (datasets, training, NLLS, whole-brain predictions, timing) runs on the
# cluster with cluster/*.sh; its outputs are inputs here. A step is (output, inputs, command): it is skipped
# when its output exists and waits until its inputs do.
#   python experiments.py all --dry-run     what can run, and how many cluster outputs are missing
#   python experiments.py E1 E7             run it

import os
import sys
import argparse
import subprocess

from dmri_pnet import paths

p = argparse.ArgumentParser()
p.add_argument('experiments', nargs='+')
p.add_argument('--dry-run', action='store_true', help='print the steps without running them')
a = p.parse_args()

SNRS = [20, 30, 40]
# training runs (MODELS.md): set networks at 128 and 256 points, MLPs, ablations at 128 points
MAIN = ['pointnet_n128', 'pairwise_n128', 'pointnet_n256', 'pairwise_n256']
MLP = {'ukb': 'mlp_ukb', 'motif': 'mlp_motif'}
ABLATED = ['pointnet_mirror', 'pointnet_tnet', 'pairwise_n64', 'pairwise_v10pc', 'pairwise_v30pc', 'pairwise_narrow']
# copies of a test set with the same voxels and noise (derive.py)
DERIVED = {**{f'{t}_{op}': (f'{t}_test', ['--' + op]) for t in ['ukb', 'hardi'] for op in ['rotate', 'flip', 'permute']},
           **{f'motif_keep{n}': ('motif_test', ['--keep', n]) for n in (24, 48, 96)},
           **{f'dsi_keep{n}': ('dsi_test', ['--keep', n]) for n in (64, 128, 256)},
           'motif_b500_3200': ('motif_test', ['--shells', 500, 3200]),
           'motif_b1700_3200': ('motif_test', ['--shells', 1700, 3200])}


def step(output, inputs, script, *args):
    return output, list(inputs), [sys.executable, os.path.join(paths.REPO, script), *map(str, args)]


def finished(run):
    return os.path.join(paths.run(run), 'losses.npz')


def derive(name):
    source, args = DERIVED[name]
    yield step(paths.synth(name), [paths.synth(source)], 'derive.py', '--set', source, '--name', name, *args)


def estimates(test, snr, methods):
    # predictions of each run, made here; NLLS comes from the cluster, except on derived sets
    files = [paths.preds(test, snr, m) for m in methods]
    for m, f in zip(methods, files):
        if m != 'nlls':
            yield step(f, [paths.synth(test), finished(m)], 'predict.py', '--run', m, '--set', test, '--snr', snr)
        elif test in DERIVED:
            yield step(f, [paths.synth(test)], 'fit.py', '--set', test, '--snr', snr)
    return files


def evaluate(exp, test, snr, methods):
    files = yield from estimates(test, snr, methods)
    out = f'{exp}/{test}_snr{snr}'
    yield step(paths.results(out, 'metrics.csv'), files, 'evaluate.py', '--set', test, '--snr', snr,
               '--methods', *methods, '--out', out)


def summarise(out, inputs, kind, *args):
    yield step(paths.results(out) + '.csv', inputs, 'summarise.py', kind, *args, '--out', out)


def curves(out, dirs, x, xlabel):
    inputs = [paths.results(d, 'metrics.csv') for d in dirs]
    for subset, tag in [('all', 'all'), ('FWF<0.7', 'fwf07')]:
        yield from summarise(f'{out}_{tag}', inputs, 'curves', '--dirs', *dirs, '--x', *x, '--xlabel', xlabel,
                             '--subset', subset)


def subject(name, methods):
    # tissue masks, made here; the NLLS and network maps come from the cluster
    masks = paths.maps(name, 'tissue_csf.nii.gz')
    yield step(masks, [], 'masks.py', '--subject', name)
    return [paths.maps(name, m) + '_ndi.nii.gz' for m in methods] + [masks]


def E1():
    # accuracy against ground truth, and the cost of agnosticism: set networks vs the MLP on its own protocol
    for test in MLP:
        for snr in SNRS:
            yield from evaluate('E1', f'{test}_test', snr, MAIN + [MLP[test], 'nlls'])


def E2():
    # E1 across SNR: the gap between the set networks and the protocol-specific MLP
    yield from E1()
    for test in MLP:
        yield from curves(f'E2/{test}', [f'E1/{test}_test_snr{snr}' for snr in SNRS], SNRS, 'SNR')


def E3():
    # protocols outside the training distribution (HARDI: one shell; DSI: 22 shells), and random in-distribution ones
    for test in ['hardi', 'dsi', 'random']:
        for snr in SNRS:
            yield from evaluate('E3', f'{test}_test', snr, MAIN + ['nlls'])
        yield from curves(f'E3/{test}', [f'E3/{test}_test_snr{snr}' for snr in SNRS], SNRS, 'SNR')


def E4():
    # invariance: rotate, flip or permute the protocol under the same signals (UKB and HARDI fit in max_pts,
    # so no random subsampling blurs the comparison)
    methods = MAIN + ['pointnet_mirror']
    for test in ['ukb', 'hardi']:
        variants = [f'{test}_{op}' for op in ['rotate', 'flip', 'permute']]
        files = []
        for name in [f'{test}_test'] + variants:
            if name in DERIVED:
                yield from derive(name)
            files += yield from estimates(name, 30, methods)
        yield from summarise(f'E4/{test}', files, 'invariance', '--set', f'{test}_test', '--variants', *variants,
                             '--snr', 30, '--methods', *methods)


def E5():
    # graceful degradation: fewer measurements, or fewer shells, on the same voxels and noise
    full = {'motif': 195, 'dsi': 514}
    for test, ns in {'motif': (24, 48, 96), 'dsi': (64, 128, 256)}.items():
        names = [f'{test}_keep{n}' for n in ns] + [f'{test}_test']
        for name in names:
            if name in DERIVED:
                yield from derive(name)
            yield from evaluate('E5', name, 30, MAIN + ['nlls'])
        yield from curves(f'E5/{test}', [f'E5/{n}_snr30' for n in names], [*ns, full[test]], 'measurements')
    for name in ['motif_b500_3200', 'motif_b1700_3200']:
        yield from derive(name)
        yield from evaluate('E5', name, 30, MAIN + ['nlls'])


def E6():
    # identifiability: errors over a regular parameter grid, many noise and orientation draws per point
    for test in MLP:
        methods = MAIN + [MLP[test], 'nlls']
        files = yield from estimates(f'{test}_grid', 30, methods)
        yield step(paths.results(f'E6/{test}') + '_ndi_fwf.png', files, 'summarise.py', 'identifiability',
                   '--set', f'{test}_grid', '--snr', 30, '--methods', *methods, '--out', f'E6/{test}')


def E7():
    # real data: whole-brain maps, compared with NLLS and with the maps shipped with the data, by tissue
    for name in ['sub-01', 'antoine']:
        methods = MAIN + (['mlp_motif'] if name == 'sub-01' else []) + ['nlls']
        files = yield from subject(name, methods)
        for ref in ['nlls', 'ref']:
            others = [m for m in methods + ['ref'] if m != ref]
            yield step(paths.results('E7', f'{name}_vs_{ref}', 'metrics.csv'), files, 'evaluate.py',
                       '--subject', name, '--reference', ref, '--methods', *others, '--out', f'E7/{name}_vs_{ref}')


def E8():
    # cross-protocol consistency on sub-05: run 2 with few vs many directions (same session, aligned),
    # and run 1 vs run 2 (registered)
    methods = MAIN + ['nlls']
    files = {}
    for name in ['sub-05r1', 'sub-05r2fewdir', 'sub-05r2manydir']:
        files[name] = yield from subject(name, methods)
    for sa, sb, out, reg in [('sub-05r2manydir', 'sub-05r2fewdir', 'r2few_vs_r2many', []),
                             ('sub-05r1', 'sub-05r2fewdir', 'r1_vs_r2fewdir', ['--register']),
                             ('sub-05r1', 'sub-05r2manydir', 'r1_vs_r2manydir', ['--register'])]:
        yield from summarise(f'E8/{out}', files[sa] + files[sb], 'consistency', '--a', sa, '--b', sb,
                             '--methods', *methods, 'ref', *reg)


def ablations():
    # mirroring and T-Net (vanilla), max_pts, training-set size, breadth of the protocol prior; reference: 128 points
    for test in MLP:
        yield from evaluate('A', f'{test}_test', 30, ['pointnet_n128', 'pairwise_n128'] + ABLATED)


EXPERIMENTS = dict(E1=E1, E2=E2, E3=E3, E4=E4, E5=E5, E6=E6, E7=E7, E8=E8, A=ablations)
names = list(EXPERIMENTS) if a.experiments == ['all'] else a.experiments
steps = {}
for name in names:
    for output, inputs, cmd in EXPERIMENTS[name]():
        steps.setdefault(output, (inputs, cmd))

planned, cluster = set(), set()
for output, (inputs, cmd) in steps.items():
    if os.path.exists(output):
        continue
    missing = [i for i in inputs if not os.path.exists(i) and i not in planned]
    if missing:
        cluster.update(i for i in missing if i not in steps)
        continue
    print(' '.join(cmd[1:]), flush=True)
    planned.add(output)
    if not a.dry_run:
        subprocess.run(cmd, check=True)

if cluster:
    print(f'waiting for {len(cluster)} cluster outputs, e.g. {os.path.relpath(sorted(cluster)[0], paths.ROOT)}')
