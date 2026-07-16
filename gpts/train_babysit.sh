#!/bin/bash
# Babysit the DDP training run: launch across both GPUs with torchrun, and if it
# crashes, restart it. Because train-gpt-2.py resumes from the latest checkpoint,
# each restart continues instead of starting over. Guards against an infinite
# loop when the failure is a real (immediately-reproducing) bug.
cd /workspace/makemore/gpts || exit 1

LOG=/workspace/makemore/data/train.out
PY=/workspace/makemore/.venv/bin
MAX_RESTARTS=30

attempt=0
consec_quickfail=0
while true; do
  attempt=$((attempt + 1))
  t0=$(date +%s)
  echo "=== [babysit] attempt $attempt starting $(date -u +%H:%M:%S) ===" >> "$LOG"

  # expandable_segments reduces fragmentation-driven OOM over long runs.
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY/torchrun" --standalone --nproc_per_node=2 train-gpt-2.py >> "$LOG" 2>&1
  code=$?

  dur=$(( $(date +%s) - t0 ))
  echo "=== [babysit] attempt $attempt exited code=$code after ${dur}s ===" >> "$LOG"

  if [ $code -eq 0 ]; then
    echo "=== [babysit] DONE: training completed cleanly ===" >> "$LOG"
    break
  fi
  if [ $attempt -ge $MAX_RESTARTS ]; then
    echo "=== [babysit] GIVING UP: hit MAX_RESTARTS=$MAX_RESTARTS ===" >> "$LOG"
    break
  fi
  # a crash within 90s is almost always a real bug, not a transient fault;
  # bail after 3 in a row instead of hammering.
  if [ $dur -lt 90 ]; then
    consec_quickfail=$((consec_quickfail + 1))
  else
    consec_quickfail=0
  fi
  if [ $consec_quickfail -ge 3 ]; then
    echo "=== [babysit] GIVING UP: 3 consecutive fast failures (<90s) — real error, not transient ===" >> "$LOG"
    break
  fi
  echo "=== [babysit] restarting in 10s (resumes from latest checkpoint) ===" >> "$LOG"
  sleep 10
done
echo "=== [babysit] loop ended $(date -u +%H:%M:%S) ===" >> "$LOG"
