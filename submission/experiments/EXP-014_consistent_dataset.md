# EXP-014 — Consistent dataset: every crop registered
- **Status:** Done 2026-10-11. **Result: positive.** Same model and recipe, consistent crops: better on the like-for-like NORVIA test (AUC 0.959 vs 0.943, half the false alarms), and much more consistent across fonts.
- **Scripts:** `src/exp014_rectify.py` (registered crops for data/synth_v2, 15 cores), `train_pairs.py` (run `e14_consistent`), `src/exp014_evaluate.py`.
- **Architecture fit:** every input is registered and therefore aligned, which is what the location-by-location model needs.

## Visual results (click to open)
- [Caught per tamper, both models on the new all-registered test sets](../outputs/exp014/caught_per_tamper.png)
- [Training curve](../outputs/exp014/evidence/training_curve.png) (log: [e14_consistent.log](../outputs/exp007/logs/e14_consistent.log)), [ROC on the three test sets](../outputs/exp014/evidence/roc_test_sets.png), [score histograms](../outputs/exp014/evidence/score_histograms.png), ["which tamper?" confusion](../outputs/exp014/evidence/confusion_kind.png)
- [Same model, different training data](../outputs/exp014/evidence/data_progression.png) (`src/exp014_data_chart.py`)
- [Missed tampers](../outputs/exp014/evidence/examples_missed.jpg), [false alarms](../outputs/exp014/evidence/examples_false_alarms.jpg), [correct detections](../outputs/exp014/evidence/examples_correct.jpg) (`src/exp014_evidence.py`)
- [Dataset examples (NORVIA)](../outputs/exp013/dataset/examples_D2.jpg), [multi-font](../outputs/exp013/dataset/examples_D1.jpg). The photos are the same as EXP-013; here the tampered input is the registered crop.

## Data
`data/synth_v2`: 10 000 samples (5000 NORVIA + 5000 multi-font, half genuine). The 9 885 whose logo was found were registered (35 min on 15 cores). Training 6 770 pairs, validation 1 427, test NORVIA 710 / multi-font 653 / unseen font 325.

## Training
**All settings** (data generator, alignment, input, architecture, loss, optimiser, schedule, thresholds, metrics): [docs/04_final_model_parameters.md](../docs/04_final_model_parameters.md). In short: BCE + 0.3 × CE (8 classes), AdamW lr 3e-4, weight decay 1e-4, cosine decay over 15 epochs (3 180 steps), batch 32, 448×128, no extra augmentation, dropout 0, mixed precision, seed 0; metrics ROC AUC, false-alarm rate and catch rate per tamper type at the validation threshold.

Best epoch 11, validation AUC 0.933 (on the synth_v2 validation set; not comparable with `sd_depth1_more`'s 0.950 on the old mixed validation set). Training AUC 0.975 at that epoch, so the gap is 0.04; validation loss flat after epoch 6: no harmful overfitting. Thresholds: tampered ≥ 0.493, genuine ≤ 0.053.

## Results (full tables: `outputs/exp014/results.md`)
**1. Old NORVIA test set** (the same 206 pairs, like-for-like):

| | **e14_consistent** | sd_depth1_more |
|---|---|---|
| AUC | **0.959** | 0.943 |
| false alarms | **3/105** | 6/105 |
| caught | 84/101 | 86/101 |
| respace / recolour / glyph swap | 19/22, 10/13, 8/9 | 18/22, 11/13, 8/9 |
| stretch | 5/15 | 7/15 |

**2. New test sets, every crop registered** (both models scored on the same pairs):

| | NORVIA | multi-font | unseen font |
|---|---|---|---|
| e14_consistent AUC | 0.945 | **0.932** | **0.925** |
| sd_depth1_more AUC | 0.952 | 0.880 | 0.884 |

- **Multi-font re-spacing:** e14 43/55, sd_depth1_more **20/55**. The old model learned multi-font from known-pose crops; with real registration its multi-font skill drops.
- **Whole-logo stretch** on registered crops: ~0-1 caught for both models, in every test set. This confirms that registration hides it, and that the old model's high multi-font stretch scores came from the known-pose crops.

**3. Real photos:**
- e14_consistent: 15 tampered / 32 don't know / 2 genuine;
- sd_depth1_more: 18 / 25 / 6.
- e14's 15 = the 14 that all four earlier models agreed on + 041. It drops the borderline 015, 019 and 047.
- 021 is still flagged (the black crop-edge artefact).

## What in the data raised accuracy
Same model and settings, four training sets, all scored on the same 206 NORVIA test pairs ([chart](../outputs/exp014/evidence/data_progression.png)):

| Training data | Pairs | AUC | False alarms |
|---|---|---|---|
| original (NORVIA + other fonts) | 2 020 | 0.854 | 3/105 |
| NORVIA only + extra (batch 16) | 3 982 | 0.918 | 14/105 |
| mixed + extra (other fonts with known-camera crops) | 7 789 | 0.943 | 6/105 |
| **consistent (every crop registered)** | **6 770** | **0.959** | **3/105** |

- More data gave the biggest jump (0.854 → 0.943); other fonts help even for NORVIA (0.918 NORVIA-only, though size also differs).
- Making every crop the way real crops are made gave the last step, with fewer pairs, and the largest gain on other fonts (0.880 → 0.932; re-spacing 20/55 → 43/55).
- Resolution did not help (896 px: 0.921 vs 0.918); un-aligned input failed (EXP-012/013).

## ONNX and reproducibility (2026-10-11)
- ONNX export (`submission/scripts/export_onnx.py`) checked with `submission/scripts/evaluate_onnx.py` on all 1 894 test pairs (the three new test sets + the shared 206): same AUCs, same counts per tamper type, largest |p ONNX − p PyTorch| 4e-6, no decision changed (report: `submission/model/onnx_check_full.md`).
- Seeds fixed in registration and inference (`common.fix_seeds`). A re-run of `exp001_register.py` and two runs of `infer_real.py` gave byte-identical crops, transforms and predictions.

## Reading
- With identical model and recipe, **making every training crop the way real crops are made gives a better and more honest model.** It is better on the like-for-like test, has fewer false alarms, and is steadier across fonts. It is also trained on fewer pairs (6 770 vs 7 789).
- The multi-font data are now genuinely useful for generalisation (unseen font 0.925 with registered crops), not just easy samples.
- Whole-logo stretch remains the blind spot of every rectified model (registration absorbs it).

## Limits
- One seed. EXP-007 measured ~±0.01 run-to-run noise; the old-test gain (+0.016) is just above it.
- The data differ in two ways at once (consistent crops AND a new, cleaner generator without mirrored logos), so the gain cannot be split between the two.
- The validation sets differ, so validation AUCs cannot be compared directly; compare on the shared test sets.

