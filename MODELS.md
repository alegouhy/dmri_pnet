# Models

The training runs of the paper. Each one is one `train.py` command, run from the repository root once its
training set exists (see `DATASETS.md`), and writes `runs/<run>/` (under `$DMRI_PNET_ROOT` if set):
`config.json` (settings, code version, date, and the training set's record), `model.pt` (weights of the
best validation epoch), `checkpoint.pt` (every 10 epochs), `losses.npz` (written at the end: it marks the
run as finished) and `loss.png`.

## Common to all

- **Output**: `[ndi, odi, fwf]` through a sigmoid; MSE loss.
- **Weights**: ~0.80 M for every model compared. The set networks' count does not depend on the number of
  points (the same small networks are applied to each point or pair); the MLP width is solved to match.
- **Optimisation**: Adam, learning rate 1e-3 reached by a linear warm-up over 5 epochs then multiplied by
  0.99 every 5 epochs, gradient clipping at 1, batch 64, 1000 epochs. A run whose train loss stays unchanged
  for 20 epochs stops with an error (a dead network).
- **Augmentation, drawn afresh each epoch**: Rician noise of level σ = 1/SNR ~ U(0, 0.1), from noise-free
  to SNR 10; for the set networks, point dropout (probability 0.5, a fraction U(0, 0.5) of the points
  removed) and a random subset of `max_pts` points. The noisy signal is then divided, as S/S0 is in real
  data, by the mean of 1–10 noisy b=0s (number drawn per sample) rather than by the true S0 = 1
  (`--n_b0 MIN MAX` to change the range).
- **Validation**: 5,000 voxels held out of the training set (a tenth of its voxels if fewer than 50,000), with
  all their protocols; the saved model is the epoch with the lowest validation loss. With 500, the loss varies
  enough between voxels for the validation set to be easier or harder than the training set by chance.
- **Resuming**: rerunning the same command resumes from `checkpoint.pt`. A run refuses to resume with
  different settings (it says which), so use a new `--run` name to change anything.

## Architectures

- **`pointnet`**: per-point MLP 5 → 64 → 64 → 128 → 1024, max over points, LayerNorm, MLP 1024 → 512 → 256
  → 3. Input per point `[gx, gy, gz, sqrt(b/6000), S/S0]`. Permutation invariant only. Options: `--mirror`
  (adds the -g copy of each point: antipodal invariance), `--tnet` (input and feature transforms, as in the
  original PointNet, with the same LayerNorm after their max).
- **`pairwise`**: pair features `[(gi·gj)², bi+bj, bi·bj, Si+Sj, Si·Sj]` → MLP 5 → 64 → 64, max over j, MLP
  64 → 128 → 1024, max over i, LayerNorm, MLP 1024 → 512 → 256 → 3. Invariant to permutation, rotation and antipodal
  flips by construction. Memory and time grow as max_pts².
- **`mlp`**: the signal vector of one fixed protocol → 3 hidden layers of equal width → 3. Runs only on the
  exact protocol it was trained on (stored in its `config.json`).

## Main runs

| Run | Model | max_pts | Training set | Weights |
|---|---|---|---|---|
| `pointnet_n128` | pointnet | 128 | `train` | 803,907 |
| `pointnet_n256` | pointnet | 256 | `train` | 803,907 |
| `pairwise_n128` | pairwise | 128 | `train` | 803,907 |
| `pairwise_n256` | pairwise | 256 | `train` | 803,907 |
| `mlp_ukb` | mlp, 100 inputs, width 608 | – | `ukb_train` | 803,779 |
| `mlp_motif` | mlp, 195 inputs, width 586 | – | `motif_train` | 804,581 |

`max_pts` caps the points seen per voxel: 87% of the training samples have more than 128 points, 43% more
than 256. MOTIF (195) fits whole at 256; DSI (514) never does. At test time, long protocols are averaged over
10 random subsets.

```bash
python train.py --run pointnet_n128 --model pointnet --max_pts 128
```

```bash
python train.py --run pointnet_n256 --model pointnet --max_pts 256
```

```bash
python train.py --run pairwise_n128 --model pairwise --max_pts 128
```

```bash
python train.py --run pairwise_n256 --model pairwise --max_pts 256
```

```bash
python train.py --run mlp_ukb --model mlp --trainset ukb_train
```

```bash
python train.py --run mlp_motif --model mlp --trainset motif_train
```

## Ablations

At 128 points, compared with `pointnet_n128` and `pairwise_n128`.

| Run | Changes | Question |
|---|---|---|
| `pointnet_mirror` | `--mirror` | does built-in antipodal symmetry help the vanilla model? |
| `pointnet_tnet` | `--tnet` (3,460,492 weights) | do the T-Nets of the original PointNet (and the abstract) help? |
| `pairwise_n64` | `--max_pts 64` | how much does seeing fewer points cost? |
| `pairwise_v10pc` | `--nvox 19500` | training on 10% of the voxels |
| `pairwise_v30pc` | `--nvox 58500` | training on 30% of the voxels |
| `pairwise_narrow` | `--trainset train_narrow` | training on a narrower protocol distribution |

```bash
python train.py --run pointnet_mirror --model pointnet --mirror
```

```bash
python train.py --run pointnet_tnet --model pointnet --tnet
```

```bash
python train.py --run pairwise_n64 --model pairwise --max_pts 64
```

```bash
python train.py --run pairwise_v10pc --model pairwise --nvox 19500
```

```bash
python train.py --run pairwise_v30pc --model pairwise --nvox 58500
```

```bash
python train.py --run pairwise_narrow --model pairwise --trainset train_narrow
```

## Cost

One GPU per run. An epoch is ~3,050 steps (195,000 training samples, batch 64). Measured on a laptop GPU
(Apple M-series, batch 64): one step takes 11 ms for `pointnet` at 128 points, 20 ms at 256; 175 ms for
`pairwise` at 128 and ~1 s at 256. A datacentre GPU is several times faster, but `pairwise_n256` will likely
exceed a 24 h job: rerun the same command to resume. A run with `--epochs 1` under a separate `--run` name
gives the real time per epoch first. The MLPs take minutes.

GPU memory (pairwise, activations kept for the backward pass): 1.1 GB at 128 points, 4.2 GB at 256, so
peaks of ~2–3 and ~8–10 GB; `pointnet` and the MLPs need far less. Host memory: the training set (~1 GB)
plus PyTorch, 16 GB is plenty. Data loading runs in the main process by default (`--workers` to change).

For a quick test of the pipeline, add `--epochs 2` and use a separate `DMRI_PNET_ROOT`.
