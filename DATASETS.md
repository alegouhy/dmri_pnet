# Datasets

The simulated datasets of the paper. Each one is made by one `simulate.py` command, run from the
repository root, and saved to `data/synth/<name>.npz` (under `$DMRI_PNET_ROOT` if set). The file keeps a
record of the command, the code version and the date (`np.load(f)['args']`).

## Common to all

- **Parameters**: NDI, ODI and FWF drawn independently from U(0.025, 0.975); fibre orientation uniform on
  the sphere. On a grid set, (NDI, ODI, FWF) take the values of an N³ grid over the same range, each repeated
  NREP times with a fresh orientation.
- **Signal**: NODDI-Watson in the NODDI toolbox formulation (dmipy compartments, extra-neurite zeppelin with
  Watson-averaged diffusivities), d∥ = 1.7 and d_iso = 3.0 µm²/ms, normalised so that S(b=0) = 1. b=0 is
  not stored.
- **Noise**: the stored signals are noise-free. Test sets also store a Rician-noise copy per SNR
  (σ = 1/SNR), seeded, divided as S/S0 is in real data by the mean of 1–10 noisy b=0s (number drawn per
  voxel, `--n_b0 1 10`). Training adds noise on the fly, the same way.
- **Random protocols**: per sample, a number of shells, a b-value per shell drawn uniformly, and a number of
  directions per shell, with directions uniformly random on the sphere. Default ranges: 2–5 shells,
  b in 250–5000 s/mm², 12–128 directions per shell (mean ~245 measurements, at most 640).
- **Fixed protocols**: `ukb` b = 1000/2000 s/mm² × 50 electrostatically spread directions; `hardi` b = 2000
  × 60; `dsi` Cartesian q-space grid of radius 5 (514 measurements, 22 shells, b 250–5000); `sub-01` copies
  the acquisition of the MOTIF-STROKE subject (b = 500/1000/1700/3200, 195 measurements), read from
  `data/real/philippe/sub-01/sub-01_desc-MSEddyCorrected_dwi.{bval,bvec}`, which must be present.

## Training sets

| Name | Samples | Protocol | Used by |
|---|---|---|---|
| `train` | 1,000,000 voxels × 1 random protocol | random, default ranges | set networks, most ablations |
| `train_multi` | 100,000 voxels × 10 random protocols = 1,000,000 | random, default ranges | ablation: several protocols per voxel |
| `train_narrow` | 1,000,000 voxels × 1 random protocol | random, 2–3 shells, b 500–3000, 30–64 directions | ablation: narrower protocol prior |
| `ukb_train` | 1,000,000 voxels | `ukb` | `mlp_ukb` |
| `motif_train` | 1,000,000 voxels | `sub-01` | `mlp_motif` |

Every sample has its own random protocol, so the network learns what the signal means for the tissue rather
than for one acquisition. `train_multi` tests whether seeing the same voxel through several protocols pushes
it further towards the physics: with the per-sample loss, nothing ties a voxel's samples together, and the
repetition gives 10× fewer distinct targets to learn from (with 20,000 voxels × 10 protocols the set networks
memorised the training voxels). It has as many samples as `train`. So do the MLP sets. No noise is stored.

```bash
python simulate.py --name train --protocol random --nvox 1000000 --seed 0
```

```bash
python simulate.py --name train_multi --protocol random --nvox 100000 --nprotoc 10 --seed 3
```

```bash
python simulate.py --name train_narrow --protocol random --nvox 1000000 --nshell 2 3 --b 500 3000 --ndir 30 64 --seed 100
```

```bash
python simulate.py --name ukb_train --protocol ukb --nvox 1000000 --seed 1
```

```bash
python simulate.py --name motif_train --protocol sub-01 --nvox 1000000 --seed 2
```

## Test sets

10,000 voxels each, noise-free signals plus noisy copies at SNR 20, 30 and 40.

| Name | Protocol | Inside the training protocols? | Used by |
|---|---|---|---|
| `ukb_test` | `ukb` (100 measurements, 2 shells) | yes | E1, E2, E4, A |
| `motif_test` | `sub-01` (195 measurements, 4 shells) | yes | E1, E2, E5, A |
| `dsi_test` | `dsi` (514 measurements, 22 shells) | no: too many shells | E3, E5 |
| `hardi_test` | `hardi` (60 measurements, 1 shell) | no: one shell | E3, E4 |
| `random_test` | one random protocol per voxel, default ranges | yes | E3 |

```bash
python simulate.py --name ukb_test --protocol ukb --nvox 10000 --snrs 20 30 40 --n_b0 1 10 --seed 10
```

```bash
python simulate.py --name motif_test --protocol sub-01 --nvox 10000 --snrs 20 30 40 --n_b0 1 10 --seed 11
```

```bash
python simulate.py --name dsi_test --protocol dsi --nvox 10000 --snrs 20 30 40 --n_b0 1 10 --seed 12
```

```bash
python simulate.py --name hardi_test --protocol hardi --nvox 10000 --snrs 20 30 40 --n_b0 1 10 --seed 13
```

```bash
python simulate.py --name random_test --protocol random --nvox 10000 --snrs 20 30 40 --n_b0 1 10 --seed 14
```

## Grid sets (E6)

5³ = 125 parameter combinations × 100 orientations = 12,500 samples, noisy copy at SNR 30.

```bash
python simulate.py --name ukb_grid --protocol ukb --grid 5 100 --snrs 30 --n_b0 1 10 --seed 20
```

```bash
python simulate.py --name motif_grid --protocol sub-01 --grid 5 100 --snrs 30 --n_b0 1 10 --seed 21
```

## Cost

Simulation is single-threaded. `train`, `train_multi` and `train_narrow` take ~2 h each (one dmipy
acquisition scheme per sample, ~25 min per 200,000 samples) and give ~5 GB files, with a peak of ~12 GB of
memory; `random_test` ~5 min; the others a few minutes. They are independent and can run in parallel.

For a quick test of the pipeline, use the same commands with `--nvox 300` (and `--grid 3 2`) and a
separate `DMRI_PNET_ROOT`.

## Derived sets (made locally, not on the cluster)

`experiments.py` makes copies of test sets with the same voxels and noise (`derive.py`): `ukb_test` and
`hardi_test` rotated, flipped and permuted (E4); `motif_test` cut to 24/48/96 measurements, `dsi_test` to
64/128/256, and `motif_test` reduced to shells b 500 + 3200 or b 1700 + 3200 (E5).
