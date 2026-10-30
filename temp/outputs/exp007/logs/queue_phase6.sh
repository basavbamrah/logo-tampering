#!/usr/bin/env bash
# Full-resolution run: wait until (1) the full-size crops exist and (2) the 448 control has finished,
# so there is only one training on the GPU. Then dry-run (memory check) and train.
cd /home/valere/Desktop/temp
until grep -q "FULLRES DONE" outputs/exp007/logs/make_full_res.log; do sleep 20; done
until [ -f outputs/exp007/norvia_448/metrics.json ] || grep -q Traceback outputs/exp007/logs/norvia_448.log; do sleep 20; done
python3 src/train_pairs.py --arch siamese_diff --sets D2 --extra-data data/synth_more --depth 1 --batch 16 \
    --full-res --input-w 896 --input-h 256 --dry-run > outputs/exp007/logs/norvia_896_dryrun.log 2>&1 || { echo "DRYRUN FAILED"; exit 1; }
python3 src/run_sweep.py phase6
echo "PHASE6 DONE"
