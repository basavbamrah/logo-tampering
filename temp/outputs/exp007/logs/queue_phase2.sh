#!/usr/bin/env bash
# Start phase 2 only after phase 1 has finished (one training on the GPU at a time).
cd /home/valere/Desktop/temp
until grep -q "PHASE1 DONE" outputs/exp007/logs/sweep_phase1.log; do sleep 15; done
python3 src/run_sweep.py phase2
echo "PHASE2 DONE"
