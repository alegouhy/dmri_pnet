import time
import argparse
import numpy as np

from dmri_pnet import data, forward, paths, subjects

p = argparse.ArgumentParser(description='Conventional NLLS fit -> data/preds/<set>_snr<snr>/nlls.npy '
                                        'or data/maps/<subject>/nlls_*.nii.gz')
p.add_argument('--set', help='synthetic test set')
p.add_argument('--snr', type=int)
p.add_argument('--subject', help='real subject, instead of --set')
p.add_argument('--z', type=int, help='real: one axial slice only')
a = p.parse_args()

t = time.time()
if a.subject:
    s, vox, img = subjects.load(a.subject, a.z)
    est = forward.fit(*data.fixed(s))
    data.save_maps(est, vox, img, paths.ensure(paths.maps(a.subject, 'nlls')))
else:
    s = data.load_set(paths.synth(a.set))
    key = f'snr{a.snr}'
    if len(s['poff']) == 2:
        est = forward.fit(*data.fixed(s, key))
    else:
        est = np.concatenate([forward.fit(*data.protocol(s, i), data.signal(s, i, key), parallel=False)
                              for i in range(len(s['pid']))])
    np.save(paths.ensure(paths.preds(a.set, a.snr, 'nlls')), est)
print(f'nlls: {len(est)} voxels in {time.time() - t:.0f} s')
