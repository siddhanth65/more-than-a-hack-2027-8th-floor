#!/bin/bash
# Plan 8 T2 driver (laptop, one heavy process at a time, OMP_NUM_THREADS=4, every step skipped when its output exists).
cd "$(dirname "$0")/.."
PY="${MTAH_PY:-python}"; PCPY="${MTAH_PATHCRF_PY:-python}"
E=runs/select/exp_seq3
export OMP_NUM_THREADS=4
declare -A NORM=([base]=none [v1]=v1 [touch]=none [v1touch]=v1); declare -A TGT=([base]=ctrl [v1]=ctrl [touch]=touch12 [v1touch]=touch12)
for v in base v1 touch v1touch; do   # final models: all six matches, epochs = median fold choice (reuses fold logs)
  [ -f $E/$v/model.pt ] || $PCPY scripts/exp_seq3_gru.py --name $v --norm ${NORM[$v]} --targets ${TGT[$v]} --final
done
for v in base v1 touch v1touch; do   # sim decode, merge off (th1) + merge th0.7 rows from the honest-LOMO spell model
  [ -f $E/spell_${v}_haz0_summary.csv ] || $PY scripts/exp_seq3_spell.py $v 0
done
$PY scripts/exp_seq3_adv.py none v1
$PY scripts/exp_seq3_robust_real.py base v1 touch v1touch
echo ALLDONE
