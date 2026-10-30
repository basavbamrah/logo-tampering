#!/usr/bin/env bash
# Phase 5 after phase 4 (one training at a time).
cd /home/valere/Desktop/temp
until grep -q "PHASE4 DONE" outputs/exp007/logs/sweep_phase4.log 2>/dev/null; do sleep 15; done
python3 src/run_sweep.py phase5
echo "PHASE5 DONE"
