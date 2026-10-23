import json
import argparse
import numpy as np
from scipy.spatial.transform import Rotation

from dmri_pnet import data, paths

p = argparse.ArgumentParser(description='Transformed or reduced copy of a fixed-protocol set, same voxels and noise '
                                        '-> data/synth/<name>.npz')
p.add_argument('--set', required=True)
p.add_argument('--name', required=True)
g = p.add_mutually_exclusive_group(required=True)
g.add_argument('--rotate', action='store_true', help='rotate all directions jointly')
g.add_argument('--flip', action='store_true', help='flip the sign of random directions')
g.add_argument('--permute', action='store_true', help='reorder the measurements')
g.add_argument('--keep', type=int, help='keep this many measurements, drawn per shell in proportion')
g.add_argument('--shells', type=float, nargs='+', help='keep only these shells (b in s/mm², rounded to 100)')
p.add_argument('--seed', type=int, default=0)
a = p.parse_args()

rng = np.random.default_rng(a.seed)
s = data.load_set(paths.synth(a.set))
source = data.set_record(s)
bvals, bvecs, _ = data.fixed(s)
shell = np.round(bvals, -2)

# the signals are not resimulated: rotating or flipping the protocol under a fixed signal is the
# same as rotating the fibres, which an invariant estimator must not see
if a.rotate:
    s['bvecs'] = (bvecs @ Rotation.random(random_state=a.seed).as_matrix().T).astype(np.float32)
elif a.flip:
    s['bvecs'] = bvecs * rng.choice([-1, 1], (len(bvecs), 1)).astype(np.float32)
elif a.permute:
    s = data.select(s, rng.permutation(len(bvals)))
elif a.keep:
    keep = [rng.choice(np.flatnonzero(shell == b), max(1, round(a.keep * (shell == b).mean())), replace=False)
            for b in np.unique(shell)]
    s = data.select(s, np.sort(np.concatenate(keep)))
else:
    s = data.select(s, np.flatnonzero(np.isin(shell, np.round(a.shells, -2))))

s['args'] = json.dumps(data.record(a, source=source))
np.savez(paths.ensure(paths.synth(a.name)), **s)
print(f'{a.name}: {len(s["bvals"])} measurements, shells {np.unique(np.round(s["bvals"], -2)).astype(int)}')
