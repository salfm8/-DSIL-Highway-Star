#!/bin/zsh
cd '/Users/shinmireu/Desktop/CAU/Project/[DSIL]Highway Star/Python_Workspace/Smart DOF' || exit 1
export MPLCONFIGDIR=/tmp/matplotlib-smartdof
exec > /tmp/smartdof_v9_engine.log 2>&1
exec /Users/shinmireu/miniconda3/bin/python -u cinematic_v9_cinematic_pull.py \
  --input '../0720 Pitch/sample_38_short.mp4' \
  --output /tmp/smartdof_v9_engine.mp4 \
  --point 225,125 \
  --bbox 70,30,255,235 \
  --positive-points '145,155;205,100;130,205' \
  --negative-points '285,125;230,270' \
  --sam-interval 2
