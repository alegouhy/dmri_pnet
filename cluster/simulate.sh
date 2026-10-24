#!/bin/bash
# The simulated datasets of DATASETS.md, one per array task. The last three use the sub-01 protocol and need
# data/real/philippe/sub-01/sub-01_desc-MSEddyCorrected_dwi.{bval,bvec}; submit them with --array=9-11. The
# million-sample training sets take ~2 h and ~12 GB each.
#SBATCH --job-name=pnet-simulate
#SBATCH --partition=common --qos=normal
#SBATCH --cpus-per-task=1 --mem=32G --time=08:00:00
#SBATCH --array=0-11
#SBATCH --output=logs/simulate_%A_%a.out

cd "${SLURM_SUBMIT_DIR:-.}"
# the repository's .venv if there is one, else the environment inherited from the submitting shell
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi
TASKS=(
  "--name train --protocol random --nvox 1000000 --seed 0"
  "--name train_multi --protocol random --nvox 100000 --nprotoc 10 --seed 3"
  "--name train_narrow --protocol random --nvox 1000000 --nshell 2 3 --b 500 3000 --ndir 30 64 --seed 100"
  "--name ukb_train --protocol ukb --nvox 1000000 --seed 1"
  "--name ukb_test --protocol ukb --nvox 10000 --snrs 20 30 40 --n_b0 1 10 --seed 10"
  "--name dsi_test --protocol dsi --nvox 10000 --snrs 20 30 40 --n_b0 1 10 --seed 12"
  "--name hardi_test --protocol hardi --nvox 10000 --snrs 20 30 40 --n_b0 1 10 --seed 13"
  "--name random_test --protocol random --nvox 10000 --snrs 20 30 40 --n_b0 1 10 --seed 14"
  "--name ukb_grid --protocol ukb --grid 5 100 --snrs 30 --n_b0 1 10 --seed 20"
  "--name motif_train --protocol sub-01 --nvox 1000000 --seed 2"
  "--name motif_test --protocol sub-01 --nvox 10000 --snrs 20 30 40 --n_b0 1 10 --seed 11"
  "--name motif_grid --protocol sub-01 --grid 5 100 --snrs 30 --n_b0 1 10 --seed 21"
)

for i in ${SLURM_ARRAY_TASK_ID:-${!TASKS[@]}}; do
  python simulate.py ${TASKS[$i]} || exit 1
done
