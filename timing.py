import os
import time
import argparse
import numpy as np
import pandas as pd

from dmri_pnet import data, forward, models, paths, subjects

p = argparse.ArgumentParser(description='Whole-brain run time of each estimator -> results/<out>.{csv,txt}')
p.add_argument('--subject', required=True)
p.add_argument('--runs', nargs='+', required=True)
p.add_argument('--n_nlls', type=int, default=2000, help='NLLS is timed on this many voxels and extrapolated')
p.add_argument('--z', type=int, help='one axial slice only, for a quick test')
p.add_argument('--out', required=True)
a = p.parse_args()

dev = models.device()
s = subjects.load(a.subject, a.z)[0]
nvox = len(s['pid'])
rows = []
for run in a.runs:
    model, cfg = models.load_run(paths.run(run), dev)
    if not models.accepts(cfg, s):
        continue
    for n_sub in [1] if cfg['model'] == 'mlp' else [1, 10]:
        t = time.time()
        models.predict(model, cfg, s, n_sub=n_sub, dev=dev)
        rows.append(dict(method=run, n_sub=n_sub, device=dev.type, seconds=time.time() - t))

bvals, bvecs, signals = data.fixed(s)
idx = np.random.default_rng(0).choice(nvox, min(a.n_nlls, nvox), replace=False)
t = time.time()
forward.fit(bvals, bvecs, signals[idx])
rows.append(dict(method='nlls', n_sub=None, device=f'{os.cpu_count()} cpu', seconds=(time.time() - t) * nvox / len(idx)))

df = pd.DataFrame(rows).assign(voxels=nvox)
df['ms_per_voxel'] = df.seconds / nvox * 1e3
df.to_csv(paths.ensure(paths.results(a.out)) + '.csv', index=False)
text = f'{a.subject}, {nvox} voxels (nlls extrapolated from {len(idx)})\n' + df.to_string(index=False, float_format='{:.3g}'.format)
open(paths.results(a.out) + '.txt', 'w').write(text)
print(text)
