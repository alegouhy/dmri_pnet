#!/bin/bash
# The training runs of MODELS.md, one per array task, once the training sets of DATASETS.md exist. On the
# Blackwell RTX PRO 6000 the longest (pairwise_n256) takes ~4.5 days, more than the 3-day limit: a run that
# hits it resumes from its checkpoint when its task is resubmitted, e.g. right away with
# sbatch --array=3 --dependency=afterany:<jobid>_3 cluster/train.sh. The last one needs motif_train.
#SBATCH --job-name=pnet-train
#SBATCH --partition=gpu --qos=gpu --gres=gpu:1 --constraint=RTX6000
#SBATCH --cpus-per-task=5 --mem=48G --time=3-00:00:00
#SBATCH --array=0-12
#SBATCH --output=logs/train_%A_%a.out

cd "${SLURM_SUBMIT_DIR:-.}"
# the repository's .venv if there is one, else the environment inherited from the submitting shell
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi
TASKS=(
  "--run pointnet_n128 --model pointnet --max_pts 128"
  "--run pointnet_n256 --model pointnet --max_pts 256"
  "--run pairwise_n128 --model pairwise --max_pts 128"
  "--run pairwise_n256 --model pairwise --max_pts 256"
  "--run mlp_ukb --model mlp --trainset ukb_train"
  "--run pointnet_mirror --model pointnet --mirror"
  "--run pointnet_tnet --model pointnet --tnet"
  "--run pairwise_n64 --model pairwise --max_pts 64"
  "--run pairwise_v10pc --model pairwise --nvox 99500"
  "--run pairwise_v30pc --model pairwise --nvox 298500"
  "--run pairwise_narrow --model pairwise --trainset train_narrow"
  "--run pointnet_n256_multi --model pointnet --max_pts 256 --trainset train_multi"
  "--run mlp_motif --model mlp --trainset motif_train"
)

for i in ${SLURM_ARRAY_TASK_ID:-${!TASKS[@]}}; do
  # loading in the main process would starve the GPU; workers do not change the results
  python train.py ${TASKS[$i]} --workers 4 || exit 1
done
