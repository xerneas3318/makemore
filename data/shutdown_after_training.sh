#!/bin/bash
# DELAYED BACKSTOP for pod shutdown.
#
# Primary path: the live Claude session detects training completion and spawns a
# Claude agent that verifies the checkpoint and STOPS (not terminates) this pod.
# This script only acts if that never happens (e.g. the session ended) so the
# pod still shuts down and idle GPUs aren't billed.
#
# The RunPod API key is read from RP_STOP_KEY (env, never written to disk).
# Fail-safe: stops the pod ONLY on clean completion + a checkpoint that loads.

KEY="${RP_STOP_KEY}"
POD="${RUNPOD_POD_ID:-whm8232fel5hbr}"
TRAINOUT=/workspace/makemore/data/train.out
CKPT_DIR=/workspace/makemore/data/checkpoints
TRAINLOG=/workspace/makemore/gpts/logs/train.log
STATUS=/workspace/makemore/data/FINAL_STATUS.txt
SELFLOG=/workspace/makemore/data/shutdown_watch.log
PY=/workspace/makemore/.venv/bin/python
GRACE=1200   # 20 min head start for the primary Claude agent

log(){ echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >> "$SELFLOG"; }

if [ -z "$KEY" ]; then log "ERROR: RP_STOP_KEY not set; aborting"; exit 3; fi
log "backstop started; pod=$POD (waiting for training completion)"

# 1) Wait for a terminal state from the babysitter.
RESULT=""
while true; do
  if grep -q "DONE: training completed cleanly" "$TRAINOUT" 2>/dev/null; then RESULT=DONE; break; fi
  if grep -q "GIVING UP" "$TRAINOUT" 2>/dev/null; then RESULT=FAILED; break; fi
  sleep 30
done
log "training terminal state: $RESULT"

# 2) On failure: record and DO NOT stop the pod.
if [ "$RESULT" != "DONE" ]; then
  {
    echo "FINAL STATUS: TRAINING DID NOT COMPLETE ($RESULT)"
    echo "generated: $(date -u)"
    echo "Pod left RUNNING for inspection. Tail of train.out:"
    tail -30 "$TRAINOUT" 2>/dev/null
  } > "$STATUS"
  log "not stopping pod (training incomplete)."
  exit 1
fi

# 3) Give the primary Claude agent a head start. If it stops the pod during this
#    window, the pod (and this script) are killed here and the backstop never
#    fires. If we're still alive after GRACE, the agent path did not run.
log "clean completion seen; yielding to primary agent for ${GRACE}s"
sleep "$GRACE"
log "grace elapsed and pod still running -> backstop taking over"

# 4) Deterministic verify + stop (agent path did not fire).
FINAL=$(ls "$CKPT_DIR"/ckpt_*.pt 2>/dev/null | sort | tail -1)
CKPT_OK=$($PY - "$FINAL" <<'PYEOF'
import sys, torch
p = sys.argv[1]
try:
    c = torch.load(p, map_location='cpu', weights_only=False)
    assert 'model' in c and len(c['model']) > 0
    assert 'optimizer' in c and 'scheduler' in c and 'step' in c
    print(f"OK step={c['step']} tensors={len(c['model'])} vocab={c.get('vocab_size')}")
except Exception as e:
    print(f"FAIL {type(e).__name__}: {e}")
PYEOF
)
log "checkpoint verify: $CKPT_OK"

{
  echo "=== FINAL STATUS: TRAINING COMPLETE (stopped by deterministic backstop) ==="
  echo "generated:        $(date -u)"
  echo "pod id:           $POD"
  echo "final checkpoint: $FINAL"
  echo "checkpoint check: $CKPT_OK"
  echo
  echo "--- last val losses ---"; grep ' val ' "$TRAINLOG" 2>/dev/null | tail -8
  echo
  echo "--- persistent files on /workspace ---"
  echo "shards:      $(ls /workspace/makemore/data/edu_fineweb10B 2>/dev/null | wc -l) files, $(du -sh /workspace/makemore/data/edu_fineweb10B 2>/dev/null | cut -f1)"
  echo "checkpoints: $(ls "$CKPT_DIR" 2>/dev/null | tr '\n' ' ')"
  echo "loss.png:    $(ls -la /workspace/makemore/gpts/loss.png 2>/dev/null | awk '{print $5" bytes"}')"
  echo "disk /workspace: $(df -h /workspace | awk 'NR==2{print $3" used, "$4" free"}')"
} > "$STATUS"

case "$CKPT_OK" in
  OK*) : ;;
  *) log "checkpoint verify FAILED; leaving pod RUNNING"; echo "CHECKPOINT VERIFY FAILED -- POD LEFT RUNNING" >> "$STATUS"; exit 2 ;;
esac

log "issuing STOP for pod $POD"
CODE=$(curl -s -o /workspace/makemore/data/stop_response.json -w "%{http_code}" \
  -X POST "https://rest.runpod.io/v1/pods/$POD/stop" \
  -H "Authorization: Bearer $KEY" --max-time 30)
log "stop REST http=$CODE"
if [ "$CODE" != "200" ]; then
  curl -s -o /workspace/makemore/data/stop_response_gql.json \
    -X POST "https://api.runpod.io/graphql?api_key=$KEY" \
    -H "Content-Type: application/json" \
    -d "{\"query\":\"mutation { podStop(input:{podId:\\\"$POD\\\"}) { id desiredStatus } }\"}" --max-time 30
  log "graphql stop fallback attempted"
fi
log "backstop done"
