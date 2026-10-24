#!/bin/bash
# The training runs of MODELS.md, one per array task, once the training sets of DATASETS.md exist. On the
# Blackwell RTX PRO 6000 the longest (pairwise_n256) takes ~19 h. A run that hits the time limit resumes from
# its checkpoint when its task is resubmitted (e.g. --array=3). The last one needs motif_train.
#SBATCH --job-name=pnet-train
#SBATCH --partition=gpu --qos=gpu --gres=gpu:1 --constraint=RTX6000
#SBATCH --cpus-per-task=5 --mem=16G --time=1-00:00:00
#SBATCH --array=0-11
#SBATCH --output=logs/train_%A_%a.out

cd "${SLURM_SUBMIT_DIR:-.}"
TASKS=(
  "--run pointnet_n128 --model pointnet --max_pts 128"
  "--run pointnet_n256 --model pointnet --max_pts 256"
  "--run pairwise_n128 --model pairwise --max_pts 128"
  "--run pairwise_n256 --model pairwise --max_pts 256"
  "--run mlp_ukb --model mlp --trainset ukb_train"
  "--run pointnet_mirror --model pointnet --mirror"
  "--run pointnet_tnet --model pointnet --tnet"
  "--run pairwise_n64 --model pairwise --max_pts 64"
  "--run pairwise_v10pc --model pairwise --nvox 19500"
  "--run pairwise_v30pc --model pairwise --nvox 58500"
  "--run pairwise_narrow --model pairwise --trainset train_narrow"
  "--run mlp_motif --model mlp --trainset motif_train"
)

for i in ${SLURM_ARRAY_TASK_ID:-${!TASKS[@]}}; do
  # loading in the main process would starve the GPU; workers do not change the results
  python train.py ${TASKS[$i]} --workers 4 || exit 1
done
