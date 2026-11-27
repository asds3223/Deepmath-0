#!/usr/bin/env bash
# Keep the single-lineage loop running (resumes from the last saved generation after any stop).
# Usage: az/run_forever.sh data/az_run1 [extra loop args...]
set -u
cd "$(dirname "$0")/.."
DIR=${1:-data/az_run1}; shift || true
while true; do
  build/az loop --dir "$DIR" --gens 20 --games 400 --threads 4 --sims 1000 --pcr-full 0.25 --fast-sims 200 --reuse-tree 1 --window 20 --reuse 4 --lr 1e-3 \
    --arena-every 10 --arena-pairs 50 --grow 256x64x64,512x128x128 --stage-every 20 --stage-gap 40 \
    --stage-pairs 100 --stage-elo 35 --stage-min 80 "$@" >> "$DIR.out" 2>&1
  code=$?
  [ -f "$DIR/STOP" ] && { echo "STOP file found, exiting"; exit 0; }
  [ $code -ne 0 ] && sleep 30
done
