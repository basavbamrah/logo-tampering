#!/usr/bin/env bash
# Full-resolution (920x260) crops for the NORVIA synthetic photos, for the 896x256 experiment.
cd /home/valere/Desktop/temp
nice -n 10 python3 src/exp003_register_synthetic.py --data data/synth --full --workers 10
nice -n 10 python3 src/exp003_register_synthetic.py --data data/synth_more --full --workers 10
echo "FULLRES DONE"
