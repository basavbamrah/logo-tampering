#!/usr/bin/env bash
# Extra training data for EXP-007: new random seeds, everything in the train split,
# then the F12 registration step for the NORVIA photos.
cd /home/valere/Desktop/temp
python3 src/exp003_make_data.py --n-d2 3000 --n-d1 3000 --out data/synth_more --seed-offset 1 --all-train --workers 10
python3 src/exp003_register_synthetic.py --data data/synth_more --workers 10
echo "ALL DONE"
