#!/bin/bash
# Each job waits for gpu.lock to be free, takes it, releases it on exit (also on failure).
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 COURIER_DATA=delivery_public
run_job () {  # $1 = description, rest = command
  local desc="$1"; shift
  until [ ! -f work/gpu.lock ]; do sleep 30; done
  echo "cv $desc" > work/gpu.lock
  "$@"; local rc=$?
  rm -f work/gpu.lock
  return $rc
}
run_job "train crops_v3 (finetune old crops_final, 10 ep, ~25min)" bash -c '/z/anaconda3/python.exe scripts/train_crops.py --init work/old/crops_final.pt --epochs 10 --lr 2.5e-4 --pct-start 0.1 --aug "{\"rot\": 6.0, \"blur\": 2.0, \"noise\": 10.0}" --out work/crops_v3.pt > work/crops_v3.stdout 2>&1; r=$?; echo "exit $r" >> work/crops_v3.stdout; exit $r'
sleep 90   # give other agents a chance to take the GPU
run_job "train heat_v3 (finetune old heat_strong, 8 ep, ~35min)" bash -c '/z/anaconda3/python.exe scripts/train_heatmap.py --init work/old/heat_strong.pt --strong --epochs 8 --lr 2e-4 --pct-start 0.1 --bs 6 --out work/heat_v3.pt > work/heat_v3.stdout 2>&1; r=$?; echo "exit $r" >> work/heat_v3.stdout; exit $r'
