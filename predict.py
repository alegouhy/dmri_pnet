import sys
import time
import argparse
import numpy as np

from dmri_pnet import data, models, paths, subjects

p = argparse.ArgumentParser(description='Network predictions -> data/preds/<set>_snr<snr>/<run>.npy '
                                        'or data/maps/<subject>/<run>_*.nii.gz')
p.add_argument('--run', required=True)
p.add_argument('--set', help='synthetic test set')
p.add_argument('--snr', type=int)
p.add_argument('--subject', help='real subject, instead of --set')
p.add_argument('--z', type=int, help='real: one axial slice only')
p.add_argument('--n_sub', type=int, default=10, help='average over random max_pts-subsets of long protocols')
a = p.parse_args()

dev = models.device()
model, cfg = models.load_run(paths.run(a.run), dev)
s, vox, img = subjects.load(a.subject, a.z) if a.subject else (data.load_set(paths.synth(a.set)), None, None)
if not models.accepts(cfg, s):
    sys.exit(f'{a.run} was trained on another protocol')

t = time.time()
est = models.predict(model, cfg, s, 'signals' if a.subject else f'snr{a.snr}', a.n_sub, dev=dev)
print(f'{a.run}: {len(est)} voxels in {time.time() - t:.1f} s')
if a.subject:
    data.save_maps(est, vox, img, paths.ensure(paths.maps(a.subject, a.run)))
else:
    np.save(paths.ensure(paths.preds(a.set, a.snr, a.run)), est)
