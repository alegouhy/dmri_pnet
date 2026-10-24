import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dmri_pnet import data, models, paths, plotting

p = argparse.ArgumentParser(description='Train a model -> runs/<run>/ (resumes from its checkpoint)')
p.add_argument('--model', required=True, choices=['pointnet', 'pairwise', 'mlp'])
p.add_argument('--trainset', default='train', help='data/synth/<trainset>.npz; mlp needs a fixed-protocol set')
p.add_argument('--run', required=True)
p.add_argument('--tnet', action='store_true', help='pointnet only')
p.add_argument('--mirror', action='store_true', help='pointnet only: antipodal invariance by adding -g copies')
p.add_argument('--max_pts', type=int, default=128, help='same for every model compared; pairwise cost grows as n²')
p.add_argument('--nvox', type=int, help='train on this many voxels only (training-set size ablation)')
p.add_argument('--n_b0', type=int, nargs=2, metavar=('MIN', 'MAX'), default=[1, 10],
               help='normalise by the mean of MIN-MAX noisy b=0s (random per sample), as real S/S0')
p.add_argument('--epochs', type=int, default=1000)
p.add_argument('--batch_size', type=int, default=64)
p.add_argument('--workers', type=int, default=0)
p.add_argument('--seed', type=int, default=0)
a = p.parse_args()

sigma = (0, 0.1)                   # noise level 1/SNR ~ U: from noise-free to SNR 10
p_drop, fdrop = 0.5, (0, 0.5)      # point dropout, not for mlp
n_weights = 803907                 # mlp width is solved to match the PointNets
lr, lr_warmup, lr_decay, lr_step, grad_clip = 1e-3, 5, 0.99, 5, 1.0
stall = 20                         # epochs of an unchanged train loss: a dead network
save_every, keep_every = 10, 100   # checkpoint to resume from; weights kept for good

torch.manual_seed(a.seed)
s = data.load_set(paths.synth(a.trainset))
n_val = min(5000, len(np.unique(s['voxel'])) // 10)   # held-out voxels
train_idx, val_idx = data.split(s, n_val, a.seed)
if a.nvox:
    keep = np.random.default_rng(a.seed).permutation(np.unique(s['voxel'][train_idx]))[:a.nvox]
    train_idx = train_idx[np.isin(s['voxel'][train_idx], keep)]

if a.model == 'mlp':
    nmeas = len(s['bvals'])
    hidden = [models.mlp_width(nmeas, 3, n_weights)] * 3
    train_ds = data.SignalDataset(s, train_idx, sigma=sigma, seed=a.seed, n_b0=a.n_b0)
    val_ds = data.SignalDataset(s, val_idx, sigma=sigma, seed=a.seed + 1, n_b0=a.n_b0)
    collate = None
else:
    nmeas, hidden = None, None
    kw = dict(sigma=sigma, p_drop=p_drop, fdrop=fdrop, max_pts=a.max_pts, n_b0=a.n_b0)
    train_ds = data.CloudDataset(s, train_idx, seed=a.seed, **kw)
    val_ds = data.CloudDataset(s, val_idx, seed=a.seed + 1, **kw)
    collate = data.pad_collate

train_loader = DataLoader(train_ds, a.batch_size, shuffle=True, collate_fn=collate, num_workers=a.workers)
val_loader = DataLoader(val_ds, a.batch_size, collate_fn=collate, num_workers=a.workers)

cfg = dict(model=a.model, tnet=a.tnet, mirror=a.mirror, nmeas=nmeas, hidden=hidden, max_pts=a.max_pts,
           bmax=data.BMAX, trainset=a.trainset, nvox=a.nvox, n_train=len(train_ds), sigma=sigma,
           p_drop=p_drop, fdrop=fdrop, n_b0=a.n_b0, n_val=n_val, epochs=a.epochs, batch_size=a.batch_size,
           lr=lr, lr_warmup=lr_warmup, lr_decay=lr_decay, lr_step=lr_step, grad_clip=grad_clip, seed=a.seed)
if a.model == 'mlp':
    cfg['protocol'] = dict(bvals=s['bvals'].tolist(), bvecs=s['bvecs'].tolist())
dev = models.device()
model = models.build(cfg).to(dev)
cfg['n_weights'] = sum(p.numel() for p in model.parameters())
cfg = json.loads(json.dumps(cfg))     # as it reads back from disk (tuples become lists)

# a resumed run must be the same run: refuse rather than mix two settings in one checkpoint
rundir = paths.run(a.run)
cfgfile = os.path.join(rundir, 'config.json')
if os.path.exists(cfgfile):
    with open(cfgfile) as f:
        saved = json.load(f)
    changed = [k for k in cfg if saved.get(k) != cfg[k]]
    if changed:
        sys.exit(f'{a.run} exists with different {changed}: use another --run, or delete {rundir}')
else:
    os.makedirs(rundir, exist_ok=True)
    with open(cfgfile, 'w') as f:
        json.dump(dict(cfg, **data.record(a, trainset_record=data.set_record(s))), f, indent=2)
print(f'{a.run}: {len(train_ds)} train / {len(val_ds)} val samples, {cfg["n_weights"]} weights, {dev}')

criterion = nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=lr)
# linear warm-up over lr_warmup epochs, then x lr_decay every lr_step epochs
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimiser, lambda e: min(1, (e + 1) / lr_warmup) * lr_decay ** (e // lr_step))

ckptfile = os.path.join(rundir, 'checkpoint.pt')
start, train_losses, val_losses = 0, [], []
if os.path.exists(ckptfile):
    ckpt = torch.load(ckptfile, map_location=dev)
    model.load_state_dict(ckpt['model'])
    optimiser.load_state_dict(ckpt['optimiser'])
    scheduler.load_state_dict(ckpt['scheduler'])
    start, train_losses, val_losses = ckpt['epoch'], ckpt['train_losses'], ckpt['val_losses']
    print(f'resuming at epoch {start}')


def run_epoch(loader, train):
    model.train(train)
    total = 0.
    with torch.set_grad_enabled(train):
        for x, y in loader:
            loss = criterion(model(x.to(dev)), y.to(dev))
            if train:
                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimiser.step()
            total += loss.item()
    return total / len(loader)


for epoch in range(start, a.epochs):
    train_ds.epoch = epoch
    train_losses.append(run_epoch(train_loader, True))
    val_losses.append(run_epoch(val_loader, False))
    scheduler.step()
    print(f'epoch {epoch + 1:4d} | train {train_losses[-1]:.4f} | val {val_losses[-1]:.4f} | '
          f'lr {scheduler.get_last_lr()[0]:.2e}', flush=True)

    if len(train_losses) >= stall and np.ptp(train_losses[-stall:]) < 1e-6:
        sys.exit(f'{a.run}: train loss unchanged for {stall} epochs, the network is dead')

    # the evaluated weights are the ones the validation split chose
    if val_losses[-1] == min(val_losses):
        torch.save(model.state_dict(), os.path.join(rundir, 'model.pt'))
    if (epoch + 1) % keep_every == 0:
        torch.save(model.state_dict(), os.path.join(rundir, f'model_e{epoch + 1:04d}.pt'))
    if (epoch + 1) % save_every == 0 or epoch + 1 == a.epochs:
        torch.save(dict(epoch=epoch + 1, model=model.state_dict(), optimiser=optimiser.state_dict(),
                        scheduler=scheduler.state_dict(), train_losses=train_losses, val_losses=val_losses),
                   ckptfile + '.tmp')
        os.replace(ckptfile + '.tmp', ckptfile)   # a crash mid-write cannot corrupt the checkpoint

# losses.npz marks the run as finished
np.savez(os.path.join(rundir, 'losses.npz'), train=train_losses, val=val_losses)
plotting.losses({a.run: (train_losses, val_losses)}).savefig(os.path.join(rundir, 'loss.png'), dpi=150)
print(f'best val {min(val_losses):.4f} at epoch {np.argmin(val_losses) + 1}')
