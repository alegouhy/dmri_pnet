import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

from dmri_pnet import data, metrics, paths, plotting, subjects

p = argparse.ArgumentParser(description='Compare estimates already on disk -> results/<out>/ '
                                        '(metrics.csv, metrics.txt, figures)')
p.add_argument('--set', help='synthetic test set (reference: ground truth)')
p.add_argument('--snr', type=int)
p.add_argument('--subject', help='real subject, instead of --set')
p.add_argument('--reference', default='nlls', help='real: method used as reference (nlls | ref)')
p.add_argument('--methods', nargs='+', required=True, help='runs, nlls, ref')
p.add_argument('--out', required=True)
p.add_argument('--fwf_max', type=float, default=0.7, help='subset: reference FWF below this '
                                                               '(real data: also WM, GM and deep GM, if masks.py ran)')
p.add_argument('--z', type=int, help='real: axial slice for the maps (default: middle)')
a = p.parse_args()


if a.subject:
    s, vox, img = subjects.load(a.subject)
    files = lambda m: subjects.estimates(a.subject, m)
    load = lambda f: data.load_maps(f, vox)
    ref, label = load(files(a.reference)), a.subject
else:
    s = data.load_set(paths.synth(a.set))
    files = lambda m: paths.preds(a.set, a.snr, m)
    load = np.load
    ref, label = s['params'], f'{a.set} SNR {a.snr}'
ref_name = a.reference if a.subject else 'ground truth'

preds = {}
for m in a.methods:
    f = files(m)
    if os.path.exists(f[0] if isinstance(f, list) else f):
        preds[m] = load(f)
    else:
        print(f'{m}: no estimates for {label}, skipped')

# real data: compare only where every estimate exists (a fit may cover a single slice)
keep = np.isfinite(ref).all(1) & np.all([np.isfinite(e).all(1) for e in preds.values()], 0)
ref, preds = ref[keep], {m: e[keep] for m, e in preds.items()}

subsets = {'all': np.ones(len(ref), bool), f'FWF<{a.fwf_max}': ref[:, 2] < a.fwf_max}
if a.subject:
    subsets |= {t: m[keep] for t, m in subjects.tissues(a.subject, vox).items()}
df = pd.concat([metrics.frame({m: e[sel] for m, e in preds.items()}, ref[sel], name) for name, sel in subsets.items()])
out = paths.results(a.out)
os.makedirs(out, exist_ok=True)
df.to_csv(os.path.join(out, 'metrics.csv'), index=False)
text = f'{label}: {keep.sum()} voxels, vs {ref_name}\n' + metrics.table(df)
open(os.path.join(out, 'metrics.txt'), 'w').write(text)
print(text)

for name, fig in [('signed_error', plotting.signed_error(preds, ref, ref_name, label)),
                  ('scatter', plotting.scatter(preds, ref, ref_name, label)),
                  ('error_dependence', plotting.error_dependence(preds, ref, ref_name, label))]:
    fig.savefig(os.path.join(out, name + '.png'), dpi=120)

if a.subject:
    vox = vox[keep]
    z = int(np.median(vox[:, 2])) if a.z is None else a.z
    plotting.maps({ref_name: ref, **preds}, vox, img.shape, z, label='value', title=label) \
        .savefig(os.path.join(out, 'maps.png'), dpi=100)
    plotting.maps({m: e - ref for m, e in preds.items()}, vox, img.shape, z, 'coolwarm', -0.5, 0.5,
                  f'difference from {ref_name}', label).savefig(os.path.join(out, 'maps_diff.png'), dpi=100)
