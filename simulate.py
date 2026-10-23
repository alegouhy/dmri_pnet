import json
import argparse
import numpy as np

from dmri_pnet import data, forward, paths, protocols, sampling, subjects

p = argparse.ArgumentParser(description='Simulate a NODDI dataset -> data/synth/<name>.npz')
p.add_argument('--name', required=True)
p.add_argument('--protocol', required=True, help='random | ukb | dsi | hardi | a subject, to copy its acquisition')
p.add_argument('--nvox', type=int, default=10000)
p.add_argument('--nprotoc', type=int, default=1, help='random: protocols drawn per voxel')
p.add_argument('--grid', type=int, nargs=2, metavar=('N', 'NREP'), help='N³ parameter grid, NREP orientations each')
p.add_argument('--snrs', type=int, nargs='*', default=[], help='noisy copies for test sets')
p.add_argument('--n_b0', type=int, nargs=2, metavar=('MIN', 'MAX'),
               help='noisy copies normalised by the mean of MIN-MAX noisy b=0s (random per sample)')
p.add_argument('--nshell', type=int, nargs=2, default=(2, 5), help='random: shell count range')
p.add_argument('--b', type=float, nargs=2, default=(250, 5000), help='random: b-value range (s/mm²)')
p.add_argument('--ndir', type=int, nargs=2, default=(12, 128), help='random: directions per shell range')
p.add_argument('--seed', type=int, default=0)
a = p.parse_args()

rng = np.random.default_rng(a.seed)
params, mu = sampling.noddi_params(a.nvox, rng) if a.grid is None else sampling.noddi_grid(*a.grid, rng)

if a.protocol == 'random':
    voxel = np.repeat(np.arange(len(params)), a.nprotoc)
    protos = [sampling.random_protocol(rng, a.nshell, a.b, a.ndir) for _ in voxel]
    pid = np.arange(len(voxel))
    signals = [forward.simulate(*protos[i], params[v], mu[v])[0] for i, v in enumerate(voxel)]
else:
    fixed = {'ukb': protocols.ukb, 'dsi': protocols.dsi, 'hardi': protocols.hardi}
    if a.protocol in fixed:
        protos = [fixed[a.protocol]()]
    else:
        f = subjects.files(a.protocol)
        protos = [protocols.read(f['bval'], f['bvec'])[:2]]
    voxel = np.arange(len(params))
    pid = np.zeros(len(voxel), int)
    signals = forward.simulate(*protos[0], params, mu)

noisy = {f'snr{snr}': data.flat(sampling.noisy(x, 1 / snr, rng, a.n_b0) for x in signals) for snr in a.snrs}
s = data.make_set(protos, pid, signals=data.flat(signals), params=params[voxel], mu=mu[voxel],
                  voxel=voxel, seed=a.seed, args=json.dumps(data.record(a)), **noisy)
np.savez(paths.ensure(paths.synth(a.name)), **s)
print(f'{a.name}: {len(voxel)} samples, {len(s["poff"]) - 1} protocols, {len(s["signals"])} measurements')
