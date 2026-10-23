#!/bin/bash
# Submit the NLLS fits and the whole-brain predictions, once the datasets of DATASETS.md exist and the runs of
# MODELS.md are trained. Run from the repository root, in the Python environment the jobs should use (they
# inherit it). Edit partitions, QoS and time limits in each script.
set -e
mkdir -p logs
f=$(sbatch --parsable cluster/fit.sh)
p=$(sbatch --parsable cluster/predict.sh)
echo "fit $f, predict $p"
