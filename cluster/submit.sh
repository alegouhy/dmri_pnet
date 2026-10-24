#!/bin/bash
# Submit the NLLS fits and the whole-brain predictions, once the datasets of DATASETS.md exist and the runs of
# MODELS.md are trained. Run from the repository root. The jobs use the repository's .venv if there is one,
# else the Python environment they inherit from this shell. Edit partitions, QoS and time limits in each script.
set -e
mkdir -p logs
f=$(sbatch --parsable cluster/fit.sh)
p=$(sbatch --parsable cluster/predict.sh)
echo "fit $f, predict $p"
