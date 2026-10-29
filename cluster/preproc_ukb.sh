#!/bin/bash
# UK Biobank raw dMRI -> TOPUP + EDDY, one subject per array task, subjects already done skipped:
#   data/real/ukb/raw/<id>_3T_DWI_dir105_{AP,PA}.{nii.gz,bval,bvec}
#   -> data/real/ukb/preproc/<id>/<id>_dwi.{nii.gz,bval,bvec}, <id>_mask.nii.gz, work/ (TOPUP field, EDDY QC)
# TOPUP on the 5 AP and 3 PA b=0s; EDDY on the AP series (5 b=0 + 50 b=1000 + 50 b=2000), bvecs rotated by
# EDDY. Total readout time = dwell (NIfTI descrip, ms) x (phase-encode lines - 1).
#   n=$(find $DMRI_PNET_ROOT/data/real/ukb/raw -name '*_PA.nii.gz' -size +0 | wc -l); sbatch --array=0-$((n - 1)) cluster/preproc_ukb.sh
#SBATCH --job-name=pnet-ukbprep
#SBATCH --partition=common --qos=normal
#SBATCH --cpus-per-task=8 --mem=16G --time=12:00:00
#SBATCH --output=logs/preproc_ukb_%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source /opt/gensoft/adm/Modules/5.6.1/init/bash
module load FSL/6.0.5
export FSLOUTPUTTYPE=NIFTI_GZ OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

RAW=${DMRI_PNET_ROOT:?}/data/real/ukb/raw
PRE=$DMRI_PNET_ROOT/data/real/ukb/preproc
# subjects whose AP and PA series are both there and not empty; UKB_SUBJECTS overrides (a fixed list keeps the
# task numbering stable while files are still arriving)
ids=(${UKB_SUBJECTS:-$(for f in "$RAW"/*_3T_DWI_dir105_PA.nii.gz; do
  id=$(basename "$f" _3T_DWI_dir105_PA.nii.gz)
  [ -s "$f" ] && [ -s "$RAW/${id}_3T_DWI_dir105_AP.nii.gz" ] && [ -s "$RAW/${id}_3T_DWI_dir105_AP.bvec" ] && echo "$id"
done)})

b0s() { awk '{for (i = 1; i <= NF; i++) if ($i < 50) printf "%s%d", (n++ ? "," : ""), i - 1}' "$1"; }

for i in ${SLURM_ARRAY_TASK_ID:-${!ids[@]}}; do
  id=${ids[$i]}
  out=$PRE/$id
  if [ -f "$out/${id}_dwi.nii.gz" ]; then echo "$id: done already"; continue; fi
  ap=$RAW/${id}_3T_DWI_dir105_AP
  pa=$RAW/${id}_3T_DWI_dir105_PA
  w=$out/work
  mkdir -p "$w"

  dwell=$(fslhd "$ap.nii.gz" | sed -n 's/^descrip.*dwell=\([0-9.]*\).*/\1/p')
  ro=$(awk -v d="$dwell" -v n="$(fslval "$ap.nii.gz" dim2)" 'BEGIN {printf "%.6f", d * (n - 1) / 1000}')
  fslselectvols -i "$ap" -o "$w/b0_AP" --vols="$(b0s "$ap.bval")"
  fslselectvols -i "$pa" -o "$w/b0_PA" --vols="$(b0s "$pa.bval")"
  fslmerge -t "$w/b0" "$w/b0_AP" "$w/b0_PA"
  # phase encoding along the second voxel axis: AP -y, PA +y
  { b0s "$ap.bval" | tr , '\n' | sed "s/.*/0 -1 0 $ro/"; echo
    b0s "$pa.bval" | tr , '\n' | sed "s/.*/0 1 0 $ro/"; echo; } | grep . > "$w/acqparams.txt"
  echo "$id: readout $ro s, $(wc -l < "$w/acqparams.txt") b=0s"

  topup --imain="$w/b0" --datain="$w/acqparams.txt" --config=b02b0.cnf --out="$w/topup" \
        --iout="$w/b0_corrected" --fout="$w/field_hz"
  fslmaths "$w/b0_corrected" -Tmean "$w/b0_mean"
  bet "$w/b0_mean" "$w/b0_brain" -m -f 0.3

  # every AP volume reads the first line of acqparams
  awk '{for (i = 1; i <= NF; i++) printf "1 "; print ""}' "$ap.bval" > "$w/index.txt"
  eddy_openmp --imain="$ap" --mask="$w/b0_brain_mask" --acqp="$w/acqparams.txt" --index="$w/index.txt" \
              --bvecs="$ap.bvec" --bvals="$ap.bval" --topup="$w/topup" --out="$w/eddy" --repol --data_is_shelled

  fslmaths "$w/eddy" -thr 0 "$w/dwi" -odt float
  cp "$ap.bval" "$out/${id}_dwi.bval"
  cp "$w/eddy.eddy_rotated_bvecs" "$out/${id}_dwi.bvec"
  mv "$w/b0_brain_mask.nii.gz" "$out/${id}_mask.nii.gz"
  rm "$w/eddy.nii.gz"
  mv "$w/dwi.nii.gz" "$out/${id}_dwi.nii.gz"    # last: its presence marks the subject as done
done
