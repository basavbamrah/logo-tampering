#!/usr/bin/env bash
# Train the other two architectures after phase 2 (one training on the GPU at a time).
cd /home/valere/Desktop/temp
until grep -q "PHASE2 DONE" outputs/exp007/logs/sweep_phase2.log 2>/dev/null; do sleep 15; done
python3 src/run_sweep.py phase3
echo "PHASE3 DONE"
