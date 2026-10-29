# Models

The training runs of the paper. Each one is one `train.py` command, run from the repository root once its
training set exists (see `DATASETS.md`), and writes `runs/<run>/` (under `$DMRI_PNET_ROOT` if set):
`config.json` (settings, code version, date, and the training set's record), `model.pt` (weights of the
best validation epoch), `model_e0100.pt` … `model_e1000.pt` (weights every 100 epochs), `checkpoint.pt`
(every 10 epochs, to resume from), `losses.npz` (written at the end: it marks the
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
  original PointNet, with the same LayerNorm after their max), `--width` (scales every hidden size, T-Nets
  included: 0.521 brings `--tnet` down to the weight budget).
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
| `pointnet_tnet_small` | `--tnet --width 0.521` (804,549 weights) | do they still help within the weight budget, or did `pointnet_tnet` gain from its size? |
| `pairwise_n64` | `--max_pts 64` | how much does seeing fewer points cost? |
| `pairwise_v10pc` | `--nvox 99500` | training on 10% of the voxels |
| `pairwise_v30pc` | `--nvox 298500` | training on 30% of the voxels |
| `pairwise_narrow` | `--trainset train_narrow` | training on a narrower protocol distribution |
| `pointnet_n256_multi` | `--max_pts 256 --trainset train_multi`, against `pointnet_n256` | does seeing each voxel through 10 protocols (10× fewer voxels) teach the physics better? Tested on unseen protocols too |

```bash
python train.py --run pointnet_mirror --model pointnet --mirror
```

```bash
python train.py --run pointnet_tnet --model pointnet --tnet
```

```bash
python train.py --run pointnet_tnet_small --model pointnet --tnet --width 0.521
```

```bash
python train.py --run pairwise_n64 --model pairwise --max_pts 64
```

```bash
python train.py --run pairwise_v10pc --model pairwise --nvox 99500
```

```bash
python train.py --run pairwise_v30pc --model pairwise --nvox 298500
```

```bash
python train.py --run pairwise_narrow --model pairwise --trainset train_narrow
```

```bash
python train.py --run pointnet_n256_multi --model pointnet --max_pts 256 --trainset train_multi
```

## Cost

One GPU per run. An epoch is ~15,550 steps (995,000 training samples, batch 64). On a Blackwell RTX PRO
6000, 1000 epochs take ~8 h for `pointnet` (~12 h with `--tnet`), ~5 h for the MLPs, ~10 h for `pairwise` at
64 points, ~29 h at 128 and ~4.5 days at 256: past the 3-day job limit, rerun the same command to resume. A
run with `--epochs 1` under a separate `--run` name gives the real time per epoch first.

GPU memory (pairwise, activations kept for the backward pass): 1.1 GB at 128 points, 4.2 GB at 256, so
peaks of ~2–3 and ~8–10 GB; `pointnet` and the MLPs need far less. Host memory: the training set (~5 GB)
plus PyTorch and the loader workers, 48 GB is plenty. Data loading runs in the main process by default (`--workers` to change).

For a quick test of the pipeline, add `--epochs 2` and use a separate `DMRI_PNET_ROOT`.
