#!/bin/bash
# Conventional NLLS fits: test sets at each SNR, the E6 grids, and the real subjects (whole brain).
# Fits on derived sets (E5) are small and run locally. QUICK=1 fits subjects on one slice.
#SBATCH --job-name=pnet-fit
#SBATCH --partition=common --qos=normal
#SBATCH --cpus-per-task=16 --mem=32G --time=24:00:00
#SBATCH --array=0-21
#SBATCH --output=logs/fit_%A_%a.out

cd "${SLURM_SUBMIT_DIR:-.}"
TASKS=()
for test in ukb motif dsi hardi random; do
  for snr in 20 30 40; do TASKS+=("--set ${test}_test --snr $snr"); done
done
TASKS+=("--set ukb_grid --snr 30" "--set motif_grid --snr 30")
for subject in sub-01 antoine sub-05r1 sub-05r2fewdir sub-05r2manydir; do TASKS+=("--subject $subject"); done

for i in ${SLURM_ARRAY_TASK_ID:-${!TASKS[@]}}; do
  task=${TASKS[$i]}
  if [ -n "$QUICK" ] && [[ $task == --subject* ]]; then
    case $task in *antoine) task="$task --z 35" ;; *) task="$task --z 45" ;; esac
  fi
  python fit.py $task || exit 1
done
