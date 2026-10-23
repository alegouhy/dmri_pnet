import os
import argparse
import tempfile
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dmri_pnet import data, paths, plotting, subjects, tools
from dmri_pnet.sampling import PARAMS

p = argparse.ArgumentParser(description='Summaries across sets, runs or subjects -> results/<out>.{csv,txt,png}')
kinds = p.add_subparsers(dest='kind', required=True)
k = kinds.add_parser('curves', help='a metric across evaluate.py folders, e.g. vs SNR or number of measurements')
k.add_argument('--dirs', nargs='+', required=True)
k.add_argument('--x', type=float, nargs='+', required=True)
k.add_argument('--xlabel', required=True)
k.add_argument('--metric', default='RMSE')
k.add_argument('--subset', default='all')
k = kinds.add_parser('invariance', help='max change of the estimates when the protocol is transformed')
k.add_argument('--set', required=True)
k.add_argument('--variants', nargs='+', required=True, help='derive.py copies of --set')
k.add_argument('--snr', type=int, required=True)
k.add_argument('--methods', nargs='+', required=True)
k = kinds.add_parser('identifiability', help='RMSE over the (NDI, FWF) and (NDI, ODI) planes of a grid set')
k.add_argument('--set', required=True)
k.add_argument('--snr', type=int, required=True)
k.add_argument('--methods', nargs='+', required=True)
k = kinds.add_parser('consistency', help='same subject under two protocols: |estimate A - estimate B|')
k.add_argument('--a', required=True)
k.add_argument('--b', required=True)
k.add_argument('--methods', nargs='+', required=True)
k.add_argument('--register', action='store_true', help='rigidly register B to A first (Anima, on mean b=0)')
for k in kinds.choices.values():
    k.add_argument('--out', required=True)
a = p.parse_args()
out = paths.ensure(paths.results(a.out))


def curves():
    df = pd.concat([pd.read_csv(paths.results(d, 'metrics.csv')).assign(x=x) for d, x in zip(a.dirs, a.x)])
    df = df[(df.subset == a.subset) & (df.metric == a.metric)]
    df.to_csv(out + '.csv', index=False)
    col = plotting.colours(df.method.unique())
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, param in zip(axes, PARAMS):
        for m, g in df[df.param == param].groupby('method', sort=False):
            ax.plot(g.x, g.value, 'o-', color=col[m], label=m)
        ax.set(title=param.upper(), xlabel=a.xlabel, ylabel=f'{a.metric} ({a.subset})', ylim=(0, None))
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out + '.png', dpi=120)


def invariance():
    # identical estimates up to float32 rounding (~1e-6) for an estimator invariant to the transform
    rows = []
    for m in a.methods:
        est = np.load(paths.preds(a.set, a.snr, m))
        for v in a.variants:
            eps = np.abs(np.load(paths.preds(v, a.snr, m)) - est).max(1)
            rows.append(dict(variant=v, method=m, median=np.median(eps), P99=np.percentile(eps, 99), max=eps.max()))
    df = pd.DataFrame(rows)
    df.to_csv(out + '.csv', index=False)
    text = df.set_index(['variant', 'method']).to_string(float_format='{:.2e}'.format)
    open(out + '.txt', 'w').write(text)
    print(text)


def identifiability():
    # where in parameter space each estimator fails: RMSE per grid cell, pooled over the third parameter
    ref = data.load_set(paths.synth(a.set))['params']
    err2 = {m: (np.load(paths.preds(a.set, a.snr, m)) - ref) ** 2 for m in a.methods}
    for i, j in [(0, 2), (0, 1)]:
        ui, uj = np.unique(ref[:, i]), np.unique(ref[:, j])
        fig, axes = plt.subplots(len(err2), 3, figsize=(11, 3.2 * len(err2)), squeeze=False, layout='constrained')
        di, dj = (ui[1] - ui[0]) / 2, (uj[1] - uj[0]) / 2
        for r, (m, e) in enumerate(err2.items()):
            for k, param in enumerate(PARAMS):
                cell = np.array([[np.sqrt(e[(ref[:, i] == x) & (ref[:, j] == y), k].mean()) for x in ui] for y in uj])
                ax = axes[r, k]
                im = ax.imshow(cell, origin='lower', vmin=0, vmax=0.3, cmap='magma',
                               extent=(ui[0] - di, ui[-1] + di, uj[0] - dj, uj[-1] + dj), aspect='auto')
                ax.set(title=f'{m}: RMSE of {param.upper()}', xlabel=PARAMS[i].upper(), ylabel=PARAMS[j].upper())
        fig.colorbar(im, ax=axes, shrink=0.6)
        fig.savefig(f'{out}_{PARAMS[i]}_{PARAMS[j]}.png', dpi=110)


def consistency():
    # B is resampled into A's space when registered: maps linearly, their support nearest-neighbour
    vox = np.argwhere(nib.load(subjects.files(a.a)['mask']).get_fdata() > 0)
    subsets = {'all': np.ones(len(vox), bool), **subjects.tissues(a.a, vox)}
    ref_b0 = paths.maps(a.a, 'b0.nii.gz')
    with tempfile.TemporaryDirectory() as tmp:
        xml = os.path.join(tmp, 'b_to_a.xml')
        if a.register:
            tools.register(paths.maps(a.b, 'b0.nii.gz'), ref_b0, xml)

        def load_b(files):
            if not a.register:
                return data.load_maps(files, vox)
            cols = []
            for f in files:
                im = nib.load(f)
                v = im.get_fdata()
                for name, vol, interp in [('value', np.nan_to_num(v), 'linear'),
                                          ('valid', np.isfinite(v).astype(np.float32), 'nearest')]:
                    nib.save(nib.Nifti1Image(vol, im.affine), os.path.join(tmp, name + '.nii.gz'))
                    tools.resample(os.path.join(tmp, name + '.nii.gz'), ref_b0, xml, os.path.join(tmp, name + '_a.nii.gz'),
                                   interp)
                value, valid = (nib.load(os.path.join(tmp, n + '_a.nii.gz')).get_fdata()[tuple(vox.T)]
                                for n in ['value', 'valid'])
                cols.append(np.where(valid > 0.5, value, np.nan))
            return np.stack(cols, 1)

        rows = []
        for m in a.methods:
            d = np.abs(data.load_maps(subjects.estimates(a.a, m), vox) - load_b(subjects.estimates(a.b, m)))
            for subset, sel in subsets.items():
                x = d[sel & np.isfinite(d).all(1)]
                if len(x) == 0:
                    continue
                rows += [dict(subset=subset, param=param, method=m, median=np.median(x[:, k]), mean=x[:, k].mean(),
                              P90=np.percentile(x[:, k], 90)) for k, param in enumerate(PARAMS)]
    df = pd.DataFrame(rows)
    df.to_csv(out + '.csv', index=False)
    text = f'|{a.a} - {a.b}|' + (' (B registered to A)' if a.register else '') + '\n' + \
        df.set_index(['subset', 'param', 'method']).to_string(float_format='{:.3f}'.format)
    open(out + '.txt', 'w').write(text)
    print(text)


dict(curves=curves, invariance=invariance, identifiability=identifiability, consistency=consistency)[a.kind]()
