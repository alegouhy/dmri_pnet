# A "set" is a dict of flat arrays (saved as one .npz):
#   bvals (P,), bvecs (P, 3), poff (K+1,)   K protocols stored end to end
#   pid (N,)                                protocol of each sample
#   off (N+1,), signals (S,)                per-sample signals end to end; noisy copies under 'snr<k>'
#   params (N, 3) [ndi, odi, fwf], mu (N, 3), voxel (N,)   (synthetic sets only)
#   args                                    JSON record of how the set was made (record())

import json
import subprocess
from datetime import datetime
import numpy as np
import torch
import nibabel as nib
from torch.utils.data import Dataset

from . import paths, sampling, protocols

BMAX = 6000


def record(args, **extra):
    # what made a file: the script's arguments, the code version (-dirty: uncommitted changes) and the date
    git = subprocess.run(['git', '-C', paths.REPO, 'describe', '--always', '--dirty'], capture_output=True, text=True)
    return dict(args=vars(args), commit=git.stdout.strip() or 'unknown',
                date=datetime.now().isoformat(timespec='seconds'), **extra)


def set_record(s):
    return json.loads(str(s['args'])) if 'args' in s else None


def flat(rows):
    return np.concatenate([np.ravel(r) for r in rows]).astype(np.float32)


def make_set(protocol_list, pid, **arrays):
    lengths = np.array([len(b) for b, _ in protocol_list])
    pid = np.asarray(pid)
    return dict(bvals=flat(b for b, _ in protocol_list),
                bvecs=np.concatenate([g for _, g in protocol_list]).astype(np.float32),
                poff=np.r_[0, np.cumsum(lengths)], pid=pid,
                off=np.r_[0, np.cumsum(lengths[pid])], **arrays)


def load_set(path):
    return dict(np.load(path))


def protocol(s, i):
    a, b = s['poff'][s['pid'][i]], s['poff'][s['pid'][i] + 1]
    return s['bvals'][a:b], s['bvecs'][a:b]


def signal(s, i, key='signals'):
    return s[key][s['off'][i]:s['off'][i + 1]]


def fixed(s, key='signals'):
    assert len(s['poff']) == 2, 'set has more than one protocol'
    return s['bvals'], s['bvecs'], s[key].reshape(len(s['pid']), -1)


def select(s, keep):
    # same samples, measurements keep of the single protocol (subset or reordering), noisy copies included
    bvals, bvecs, _ = fixed(s)
    n = len(s['pid'])
    out = make_set([(bvals[keep], bvecs[keep])], s['pid'], **{k: s[k] for k in ('params', 'mu', 'voxel', 'seed') if k in s})
    for k in s:
        if k == 'signals' or k.startswith('snr'):
            out[k] = s[k].reshape(n, -1)[:, keep].ravel()
    return out


def make_cloud(bvals, bvecs, signal, bmax=BMAX):
    return np.column_stack([bvecs, np.sqrt(bvals / bmax), signal]).astype(np.float32)


def split(s, n_val, seed=0):
    # by voxel: a voxel simulated under several protocols must not sit on both sides
    voxels = np.random.default_rng(seed).permutation(np.unique(s['voxel']))
    val = np.isin(s['voxel'], voxels[:n_val])
    return np.flatnonzero(~val), np.flatnonzero(val)


class CloudDataset(Dataset):
    # randomness is drawn from (seed, epoch, sample): reproducible, and workers never share a stream
    # sigma: range of the noise level 1/SNR, drawn per sample
    def __init__(self, s, idx=None, key='signals', sigma=None, p_drop=0., fdrop=(0, 0.5),
                 max_pts=None, bmax=BMAX, seed=0, n_b0=None):
        self.s, self.key, self.sigma, self.p_drop, self.fdrop, self.n_b0 = s, key, sigma, p_drop, fdrop, n_b0
        self.max_pts, self.bmax, self.seed, self.epoch = max_pts, bmax, seed, 0
        n = len(s['pid'])
        self.idx = np.arange(n) if idx is None else idx
        self.targets = s.get('params', np.zeros((n, 3))).astype(np.float32)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        i = self.idx[k]
        rng = np.random.default_rng((self.seed, self.epoch, i))
        bvals, bvecs = protocol(self.s, i)
        x = signal(self.s, i, self.key)
        if self.sigma:
            x = sampling.noisy(x, rng.uniform(*self.sigma), rng, self.n_b0)
        keep = sampling.keep_points(len(x), rng, self.p_drop, self.fdrop, self.max_pts)
        return (torch.from_numpy(make_cloud(bvals[keep], bvecs[keep], x[keep], self.bmax)),
                torch.from_numpy(self.targets[i]))


def pad_collate(batch):
    # pad by repeating points: max pooling ignores repeats, so the padding is exact
    clouds, targets = zip(*batch)
    n = max(len(c) for c in clouds)
    return torch.stack([c[torch.arange(n) % len(c)] for c in clouds]), torch.stack(targets)


class SignalDataset(Dataset):
    # fixed protocol: a sample is the bare signal vector (MLP input)
    def __init__(self, s, idx=None, key='signals', sigma=None, seed=0, n_b0=None):
        self.signals = fixed(s, key)[2]
        self.sigma, self.seed, self.epoch, self.n_b0 = sigma, seed, 0, n_b0
        self.idx = np.arange(len(self.signals)) if idx is None else idx
        self.targets = s.get('params', np.zeros((len(self.signals), 3))).astype(np.float32)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        i = self.idx[k]
        x = self.signals[i]
        if self.sigma:
            rng = np.random.default_rng((self.seed, self.epoch, i))
            x = sampling.noisy(x, rng.uniform(*self.sigma), rng, self.n_b0)
        return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(self.targets[i])


# ── real data ─────────────────────────────────────────────────────────────────

def load_dwi(dwi, bval, bvec, mask, z=None):
    # normalise by the mean b=0, drop b=0, clip to [0, 2]; optionally keep one axial slice
    bvals, bvecs, is_b0 = protocols.read(bval, bvec)
    img = nib.load(dwi)
    m = nib.load(mask).get_fdata() > 0
    if z is not None:
        m[..., np.arange(m.shape[2]) != z] = False
    raw = np.asarray(img.dataobj, dtype=np.float32)[m]
    S0 = raw[:, is_b0].mean(1)
    ok = S0 > 0
    S = np.clip(raw[ok][:, ~is_b0] / S0[ok, None], 0, 2)
    return make_set([(bvals, bvecs)], np.zeros(len(S), int), signals=S.ravel()), np.argwhere(m)[ok], img


def map_paths(prefix):
    return [f'{prefix}_{p}.nii.gz' for p in sampling.PARAMS]


def save_maps(values, vox, img, prefix):
    for k, path in enumerate(map_paths(prefix)):
        vol = np.full(img.shape[:3], np.nan, np.float32)
        vol[tuple(vox.T)] = values[:, k]
        nib.save(nib.Nifti1Image(vol, img.affine), path)


def load_maps(paths, vox):
    return np.stack([nib.load(p).get_fdata()[tuple(vox.T)] for p in paths], 1)
