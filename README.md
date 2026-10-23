# dMRI PointNet

Protocol-agnostic NODDI estimation. A voxel is a set of measurements `[gx, gy, gz, sqrt(b/6000), S/S0]`; a set
network trained once on simulations over random protocols predicts `[ndi, odi, fwf]` for any acquisition.

## Package (`dmri_pnet/`)

```
forward.py     NODDI forward model (dmipy compartments, toolbox formulation) and the NLLS fit
sampling.py    parameter prior, random protocols, Rician noise, point dropout
protocols.py   fixed protocols (UKB, DSI, HARDI) and .bval/.bvec reading
data.py        dataset format, point clouds, torch datasets, NIfTI read/write, provenance records
models.py      PointNet, PairwisePointNet, SignalMLP; loading a run, prediction
metrics.py     error metrics and tables
plotting.py    figures
subjects.py    the real datasets under data/real, tissue masks
paths.py       where every step reads and writes (outputs under $DMRI_PNET_ROOT if set)
tools.py       external tools: Anima (registration), mri_synthseg (segmentation)
```

## Scripts

One step each, configured by arguments (`--help`), reading and writing files only:

```
simulate.py    simulate a dataset                  data/synth/<set>.npz
derive.py      transformed or reduced copy of a test set
train.py       train a model                       runs/<run>/
fit.py         NLLS on a set or a subject          data/preds/ or data/maps/
predict.py     a model on a set or a subject       data/preds/ or data/maps/
masks.py       tissue masks of a subject           data/maps/<subject>/
evaluate.py    compare estimates                   results/
summarise.py   summaries across sets or subjects   results/
timing.py      whole-brain run times               results/
```

## What to run

- `DATASETS.md`: the datasets to simulate, their characteristics and the exact `simulate.py` commands.
- `MODELS.md`: the runs to train, their settings and the exact `train.py` commands.
- `cluster/`: NLLS fits and whole-brain predictions as SLURM arrays (`submit.sh` queues both).
- `experiments.py`: everything else, run locally once the cluster outputs are there. Each step is skipped
  when its output exists and waits until its inputs do; `--dry-run` shows what can run.

Datasets and runs keep a record of how they were made (arguments, code version, date).

## Tests

```bash
python -m pytest
```
