#!/usr/bin/env bash
# Wait for the cached baseline check to finish, then run the phase-1 sweep.
# Runs one at a time so two trainings never share the 4 GB GPU.
cd /home/valere/Desktop/temp
until [ -f outputs/exp007/sd_base_cache/metrics.json ] || grep -q Traceback outputs/exp007/logs/sd_base_cache.log; do sleep 10; done
python3 src/run_sweep.py phase1
echo "PHASE1 DONE"
