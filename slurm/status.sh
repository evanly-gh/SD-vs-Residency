#!/usr/bin/env bash
# Quick overnight status: queue + which result JSONs are complete.
GS=/mmfs1/gscratch/intelligentsystems/evanly
R=$GS/sd-qwen35/results
echo "=== queue $(date -u +%H:%M:%S) ==="
squeue -u "$USER" --format='%.10i %.14j %.8T %.7M %R' 2>/dev/null | grep -E 'sd-|JOBID'
echo "=== results (batch sizes present per condition) ==="
for c in baseline K1 K2 K3 K4 K6 baseline_moe M3 M3_think; do
  f="$R/$c.json"
  if [ -f "$f" ]; then
    python3 -c "
import json
d=json.load(open('$f'))
bs=sorted(int(x) for x in d['aggregate']['by_batch_size'])
ts=d['meta'].get('timestamp_utc')
done='DONE' if ts else 'partial'
print(f'  $c: {done} batch_sizes={bs}')
" 2>/dev/null || echo "  $c: (unreadable/partial-write)"
  else
    echo "  $c: -- no file yet"
  fi
done
