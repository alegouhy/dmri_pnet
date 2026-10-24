# All models output [ndi, odi, fwf] through a sigmoid and have ~0.80M weights.
# No BatchNorm: batch statistics would depend on the duplicated padding. A LayerNorm (per sample) on each
# max-pooled feature keeps the layers after the pooling from blowing up and saturating the sigmoid for good.

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import data


def mlp(*sizes):
    # nn.Linear acts on the last axis, so the same block serves points (b, n, c) and pairs (b, n, n, c)
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), nn.SiLU()]
    return nn.Sequential(*layers)


def pooled(*sizes):
    # the MLP applied after a max over points
    return nn.Sequential(nn.LayerNorm(sizes[0]), mlp(*sizes))


def head(n_in, n_out=3):
    return nn.Sequential(nn.Linear(n_in, n_out), nn.Sigmoid())


class TNet(nn.Module):
    # predicts a k×k transform of the points, initialised to identity
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.net = mlp(k, 64, 128, 1024)
        self.glob = pooled(1024, 512, 256)
        self.out = nn.Linear(256, k * k)
        nn.init.zeros_(self.out.weight)
        self.out.bias.data = torch.eye(k).flatten()

    def forward(self, x):
        T = self.out(self.glob(self.net(x).amax(1))).view(-1, self.k, self.k)
        return x @ T.transpose(1, 2)


class PointNet(nn.Module):
    # permutation invariant; antipodal invariant only with mirror (adds -g copies); not rotation invariant
    def __init__(self, tnet=False, mirror=False):
        super().__init__()
        self.mirror = mirror
        self.tnet_in = TNet(3) if tnet else None
        self.local1 = mlp(5, 64, 64)
        self.tnet_feat = TNet(64) if tnet else None
        self.local2 = mlp(64, 128, 1024)
        self.glob = pooled(1024, 512, 256)
        self.head = head(256)

    def forward(self, x):
        if self.mirror:
            x = torch.cat([x, x * x.new_tensor([-1, -1, -1, 1, 1])], 1)
        if self.tnet_in is not None:
            x = torch.cat([self.tnet_in(x[..., :3]), x[..., 3:]], -1)
        x = self.local1(x)
        if self.tnet_feat is not None:
            x = self.tnet_feat(x)
        return self.head(self.glob(self.local2(x).amax(1)))


class PairwisePointNet(nn.Module):
    # pair features [(gi·gj)², bi+bj, bi·bj, Si+Sj, Si·Sj] are invariant to rotation, antipodal flips
    # and i<->j; max over j then over i adds permutation invariance. Cost O(n²).
    def __init__(self):
        super().__init__()
        self.pair = mlp(5, 64, 64)
        self.local = mlp(64, 128, 1024)
        self.glob = pooled(1024, 512, 256)
        self.head = head(256)

    def forward(self, x):
        g, b, s = x[..., :3], x[..., 3], x[..., 4]
        pairs = torch.stack([(g @ g.transpose(1, 2)) ** 2,
                             b[:, :, None] + b[:, None], b[:, :, None] * b[:, None],
                             s[:, :, None] + s[:, None], s[:, :, None] * s[:, None]], -1)
        x = self.pair(pairs).amax(2)
        return self.head(self.glob(self.local(x).amax(1)))


class SignalMLP(nn.Module):
    # fixed-protocol baseline: input is the signal vector, measurement k must always be the same acquisition
    def __init__(self, nmeas, hidden):
        super().__init__()
        self.net = nn.Sequential(mlp(nmeas, *hidden), head(hidden[-1]))

    def forward(self, x):
        return self.net(x)


def mlp_width(nmeas, depth, target):
    # hidden width whose parameter count is closest to target
    def count(w):
        sizes = [nmeas] + [w] * depth + [3]
        return sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:]))
    return min(range(8, 4096), key=lambda w: abs(count(w) - target))


def build(cfg):
    if cfg['model'] == 'pointnet':
        return PointNet(cfg['tnet'], cfg['mirror'])
    if cfg['model'] == 'pairwise':
        return PairwisePointNet()
    if cfg['model'] == 'mlp':
        return SignalMLP(cfg['nmeas'], cfg['hidden'])
    raise ValueError(cfg['model'])


def device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_run(rundir, dev='cpu'):
    with open(os.path.join(rundir, 'config.json')) as f:
        cfg = json.load(f)
    model = build(cfg)
    model.load_state_dict(torch.load(os.path.join(rundir, 'model.pt'), map_location=dev))
    return model.to(dev).eval(), cfg


def accepts(cfg, s):
    # an MLP only runs on the exact acquisition it was trained on: same b-values *and* directions, same order
    if cfg['model'] != 'mlp':
        return True
    b, g = np.array(cfg['protocol']['bvals']), np.array(cfg['protocol']['bvecs'])
    return (len(s['poff']) == 2 and len(s['bvals']) == len(b)
            and np.allclose(s['bvals'], b, atol=1e-3) and np.allclose(s['bvecs'], g, atol=1e-4))


@torch.no_grad()
def predict(model, cfg, s, key='signals', n_sub=1, batch=256, dev='cpu'):
    # n_sub > 1 averages over random max_pts-subsets when the protocol is longer than max_pts
    if cfg['model'] == 'mlp':
        loader = DataLoader(data.SignalDataset(s, key=key), batch)
        return torch.cat([model(x.to(dev)).cpu() for x, _ in loader]).numpy()

    if cfg['max_pts'] is None or np.diff(s['poff']).max() <= cfg['max_pts']:
        n_sub = 1
    preds = []
    for r in range(n_sub):
        ds = data.CloudDataset(s, key=key, max_pts=cfg['max_pts'], bmax=cfg['bmax'], seed=r)
        loader = DataLoader(ds, batch, collate_fn=data.pad_collate)
        preds.append(torch.cat([model(x.to(dev)).cpu() for x, _ in loader]).numpy())
    return np.mean(preds, 0)
