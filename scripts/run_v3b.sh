#!/bin/bash
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 COURIER_DATA=delivery_public
run_job () {
  local desc="$1"; shift
  until [ ! -f work/gpu.lock ]; do sleep 30; done
  echo "cv $desc" > work/gpu.lock
  "$@"; local rc=$?
  rm -f work/gpu.lock
  return $rc
}
until grep -q "^exit" work/heat_v3.stdout 2>/dev/null; do sleep 30; done
sleep 90
run_job "train crops_v3b (strong aug, from crops_v3, 8 ep, ~25min)" bash -c '/z/anaconda3/python.exe scripts/train_crops.py --init work/crops_v3.pt --epochs 8 --lr 2e-4 --pct-start 0.1 --aug "{\"rot\": 7.0, \"blur\": 2.6, \"noise\": 15.0, \"scale_lo\": 0.55, \"p_blur\": 0.6}" --out work/crops_v3b.pt > work/crops_v3b.stdout 2>&1; r=$?; echo "exit $r" >> work/crops_v3b.stdout; exit $r'
