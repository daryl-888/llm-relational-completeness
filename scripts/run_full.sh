#!/usr/bin/env bash
# The full experiment plan for one model, in priority order, resuming after usage caps.
#   E1  all 500 questions, direct prompt
#   E3  stratified 100-question subset, direct, independent second sample (run-to-run variation)
#   E2  the same 100 questions with the exhaustive and rowwise prompts
# A step that cannot finish (errors, usage cap) makes the script wait and retry it before moving on.
# usage: scripts/run_full.sh MODEL [WORKERS] [OUT]
set -u
model=$1; workers=${2:-4}; out=${3:-results/full_v0}
cd "$(dirname "$0")/.."
steps=("--variant direct"
       "--variant direct --subset 100 --repeat 1"
       "--variant exhaustive rowwise --subset 100")
for pass in $(seq 1 60); do
  complete=1
  for s in "${steps[@]}"; do
    # shellcheck disable=SC2086
    if ! .venv/bin/python scripts/run.py --model "$model" --workers "$workers" --out "$out" $s; then
      complete=0; break
    fi
  done
  if [ $complete -eq 1 ]; then echo "DONE $model $(date -u +%FT%TZ)"; exit 0; fi
  echo "WAITING $model after pass $pass: will retry in 20 min ($(date -u +%FT%TZ))"
  sleep 1200
done
echo "GAVE UP $model after 60 passes $(date -u +%FT%TZ)"; exit 1
