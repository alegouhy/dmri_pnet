#!/bin/bash
# Whole-brain predictions of every run on the real subjects, and the E9 timing. Predictions on the
# synthetic test sets are cheap and run locally. QUICK=1 predicts on one slice.
#SBATCH --job-name=pnet-predict
#SBATCH --partition=gpu --qos=gpu --gres=gpu:1
#SBATCH --cpus-per-task=4 --mem=32G --time=04:00:00
#SBATCH --array=0-21
#SBATCH --output=logs/predict_%A_%a.out

cd "${SLURM_SUBMIT_DIR:-.}"
TASKS=()
for subject in sub-01 antoine sub-05r1 sub-05r2fewdir sub-05r2manydir; do
  runs="pointnet_n128 pointnet_n256 pairwise_n128 pairwise_n256"
  [ $subject = sub-01 ] && runs="$runs mlp_motif"    # the MLP only runs on its own protocol
  for run in $runs; do TASKS+=("--run $run --subject $subject"); done
done
TASKS+=("timing")

for i in ${SLURM_ARRAY_TASK_ID:-${!TASKS[@]}}; do
  task=${TASKS[$i]}
  z=""
  if [ -n "$QUICK" ]; then case $task in *antoine) z="--z 35" ;; *) z="--z 45" ;; esac; fi
  if [ "$task" = timing ]; then
    python timing.py --subject sub-01 --runs pointnet_n128 pointnet_n256 pairwise_n128 pairwise_n256 mlp_motif \
      $z --out E9/sub-01 || exit 1
  else
    python predict.py $task $z || exit 1
  fi
done
