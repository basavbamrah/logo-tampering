#!/usr/bin/env bash
# Run phase 4 only after the current sweep process has exited (one training at a time).
cd /home/valere/Desktop/temp
while pgrep -f "src/run_sweep.py all" > /dev/null; do sleep 15; done
python3 src/run_sweep.py phase4
echo "PHASE4 DONE"
