# EXP-007 evidence: what each step did, in pictures

Made by `python src/exp007_evidence.py` (about 45 s on the 4 GB GPU). **Inference only. No model was trained**, and no existing file was changed. Everything here comes from the saved weights (`best.pt`), `metrics.json`, `predictions.csv` and the TensorBoard logs of each run.

Recommended model: **`sd_depth1_more`** (siamese_diff, encoder depth 1, NORVIA + multi-font + extra data).
"Threshold" always means the run's own `threshold_5pct_false_alarm`, which flags 5 % of the genuine **validation** pairs (`train_pairs.py`). For sd_depth1_more it is 0.333.

## Sanity checks (the evidence matches the original results)
| Check | Result |
|---|---|
| Real photos: this script re-ran `sd_base` and compared it with the saved `outputs/exp007/sd_base/real_predictions.csv` | max p_mean difference 0.0002, same decision on 50/50 photos |
| Test AUCs recomputed from `predictions.csv` vs `metrics.json` | largest difference 0.00005 (rounding) |
| Synthetic test pairs re-scored with sd_depth1_more vs the saved `predictions.csv` | 503 pairs, max difference 0.0012 (fp16) |
| Best epoch from TensorBoard vs `metrics.json`, all 23 runs | all identical |

## architecture/
| File | What it shows | Takeaway |
|---|---|---|
| `architecture_diagram.png` | Block diagram of siamese_diff depth 1 and early_fusion. The tensor shapes come from a real forward pass. | Siamese: both images go through the **same** 5-conv encoder → 64 x 32 x 112 maps → \|a−b\| and a·b → 2 convs → max+mean pool → 2 heads. Early fusion: 6-channel stack → full ResNet-18 → 512 → 2 heads. It has no explicit comparison step. |
| `parameter_counts.txt` | Parameter counts per architecture (encoder / compare / heads) and trainable parameters per run, counted from `models.py` | siamese_diff d1 **455,497**, d2 1,128,521, d3 3,523,145; early_fusion **11,190,537** (24x more); dino_frozen 22.7 M total, of which 1,035,273 are trainable. sd_freeze trains fewer weights because its stem and layer1 are frozen. |

## training_curves/
| File | What it shows | Takeaway |
|---|---|---|
| `<run>.png` (23 files, one per run) | Left: training loss per epoch. Right: train AUC (1000-pair subset), **val AUC** (chooses the epoch), test NORVIA AUC (logged for monitoring only). Star = best epoch. | sd_depth1_more: loss 0.79 → 0.14; val AUC peaks at 0.950 (epoch 12) and stays flat (0.947 at epoch 15). sd_base: val peaks at 0.896 (epoch 6) and falls to 0.867 while train AUC reaches 0.99. That is the over-fitting the extra data fixed. |
| `overview_val_auc.png` | Val AUC vs epoch for the 7 key runs (the NORVIA-only runs have their own panel because their validation set is different) | Order on the mixed val set: sd_depth1_more 0.950 > sd_more 0.940 > ef_base 0.902 ≈ sd_base 0.896 >> dino_base 0.783. dino_base peaks at epoch 2 and then stays flat. ef_base is still rising at epoch 14. |
| `best_epochs.csv` | Best epoch, best and final val AUC, final train AUC and first/last loss per run | dino_base: final train AUC 1.000 vs val 0.751, so it memorises. |

## test_results/ (synthetic test sets, from `predictions.csv`)
| File | What it shows | Takeaway |
|---|---|---|
| `roc_test_sets.png` | ROC on NORVIA test (206 pairs), multi-font test (198) and unseen font Lato (99). sd_depth1_more's operating point is marked. | sd_depth1_more AUC: NORVIA 0.943, multi-font 0.898, unseen font 0.951. At the threshold it flags **5.7 %** of genuine NORVIA test pairs and catches 85.1 % of tampers. On the unseen font it flags **14.9 %** of genuine pairs (7/47), so the false-alarm rate does not transfer to new fonts. sd_more is level with it on NORVIA (0.949). |
| `caught_per_tamper.png` | Share caught per tamper type at each run's threshold, for 5 runs × 3 test sets. The whiskers are 95 % Wilson intervals; n per bar is under each label. | sd_depth1_more on NORVIA test: stretch **7/15 (47 %)**, recolour 11/13 (85 %), respace 18/22 (82 %), glyph swap 8/9 (89 %), font swap / overlap / cut 100 %. sd_base: stretch 20 %, recolour 54 %, respace 18 %. The intervals are wide (n = 9-22), so only large differences mean anything. |
| `confusion_kind_sd_depth1_more.png` | The 8-class "which tamper?" head, true vs predicted, on the 3 test sets | The head is right on 87 % / 84 % / 81 % of all pairs. Almost all mistakes are **"none"**: the head misses the same subtle tampers as the yes/no head (NORVIA: 10 of 15 stretches and 8 of 22 respaces predicted "none"). Tampers it does see are rarely given the wrong kind. |
| `score_histograms.png` | Probability histograms, genuine vs tampered, for sd_depth1_more, sd_base and ef_base | sd_depth1_more puts most genuine pairs below 0.1 and most tampers at about 1.0. The overlap is a thin tail of subtle tampers. ef_base's threshold (0.62) sits much higher. |
| `test_summary.csv` | Per run and split: n, threshold, AUC, false-alarm rate, kind accuracy, and caught k/n per tamper type | Exact numbers behind the figures |

## examples/ (sd_depth1_more, synthetic test pairs; columns: candidate | reference | feature difference)
On all 503 test pairs at threshold 0.333: 196 caught, **41 missed**, **24 false alarms**, 242 correct passes.
| File | What it shows | Takeaway |
|---|---|---|
| `correct.jpg` | 2 caught pairs per tamper type plus 2 genuine passes (NORVIA test) | Glyph swap, overlap and cut light up exactly where the change is (e.g. the R→P in "NOPVIA", the cut-off block). Recolour and stretch light up the whole logo only faintly. |
| `missed_1of3.jpg` … `missed_3of3.jpg` | All 41 missed tampers | Misses by type: respace 14, stretch 10, recolour 9, glyph swap 7, font swap 1. These are mostly small changes: 1-2.6 px shifts, 1-7 % stretches, about 4-5° hue shifts. The one missed font swap (D1_00793, NotoSans-Bold → NotoSans-Bold-system, magnitude 0.00) looks like a **label artefact**: the same font drawn by a second copy (see the note below). |
| `false_alarms_1of2.jpg`, `false_alarms_2of2.jpg` | All 24 false alarms (NORVIA 6, multi-font 11, unseen font 7) | Many have a strong colour cast, blur or a background-edge shading on the candidate. The map lights up text edges in general rather than one letter. |
| `examples_index.csv` | Id, split, true tamper, magnitude, detail, p and predicted kind for every tile | Find any tile in `data/synth/labels.csv` |

## real_photos/ (49 registered real photos; 029 cannot be registered)
Same procedure as `infer_real.py`: the rectified crop is scored against all 3 references, p_mean decides, and the run's validation thresholds give tampered / don't know / genuine. ECC < 0.9 adds the "logo partly outside" flag. **There is no ground truth for the real photos.** The probabilities are not calibrated there, so read rankings and agreement between models, not absolute values.
| File | What it shows | Takeaway |
|---|---|---|
| `real_predictions_<run>.csv` | Per photo p_ref1/2/3, mean/min/max, majority kind and decision, for sd_depth1_more, sd_more, ef_base and sd_base (the sd_base file is the reproduction check) | sd_depth1_more: **18 tampered**, 25 don't know, 6 genuine, 1 can't register. sd_more: 21 / 20 / 8. ef_base: 17 / 30 / 2. sd_base (saved): 15 / 23 / 11. |
| `real_comparison.csv` | One row per photo: the saved sd_base result next to the three runs, with a vote count | **14 photos are called tampered by all 4 models**: 009 011 013 016 020 021 025 027 028 030 033 037 038 043. sd_depth1_more adds 015, 019, 041, 047 compared with sd_base. 015, 019 and 047 are borderline (p 0.35-0.50, kind "none"). sd_base alone called 022. |
| `real_scores_by_run.png` | p_mean per photo for every run, sorted by sd_depth1_more, with its two thresholds | There is a clear gap: 14 photos sit at about 1.0, and most others are below 0.35. Borderline photos are where the runs disagree (e.g. 023: ef_base 0.74, others < 0.15). |
| `heatmaps_sd_depth1_more/<id>.jpg`, `heatmaps_sd_more/<id>.jpg` | Per photo: rectified crop \| reference ref_1 \| feature-difference map (mean over the 3 refs, **one colour scale for all photos of that run**) | 009 lights up at R/V/I/A, 013/016/037 at the cut edge, 020/025/030 at V/I/A, 021 at the icon. Recolour photos (027, 028, 033, 038, 043) show a weak spread-out map. |
| `contact_sheet_sd_depth1_more.jpg`, `contact_sheet_sd_more.jpg` | All 49 difference maps on one page. Frame colour = decision (red tampered, amber don't know, green genuine). | Shape tampers show local hot spots. Genuine-looking photos stay mostly pale, apart from an overall pink tint from paper/lighting colour differences. |

## Things worth knowing (measured unless stated otherwise)
- **False-alarm rate in the EXP-007 doc:** the doc says sd_depth1_more catches its per-tamper rates "at ~4 % false alarms" on NORVIA test. Measured from `predictions.csv` and `metrics.json` it is **5.7 % (6 of 105 genuine)**. The per-tamper rates themselves match.
- **The reference choice matters more for sd_depth1_more than the doc says for sd_base** (where it reports ≤ 0.03). For sd_depth1_more the spread between the 3 references reaches 0.22 (photo 022: 0.18 / 0.39 / 0.18). Photo 019 is "tampered" by p_mean (0.346), but with ref_2 alone (0.273) it would be "don't know". This only matters for borderline photos. The 14 clear ones are about 1.0 with every reference.
- **Label artefact in the synthetic data:** 3 font_swap rows swap NotoSans-Bold for NotoSans-Bold-system (D1_00625, D1_00793, D1_01017). They are probably near-identical renderings. One of them (D1_00793, magnitude 0.00) is in the test set and is the only font swap sd_depth1_more misses on the multi-font set. This is inferred from the label and the picture. Not fixed (data left untouched).
- **The heatmap is the raw |a−b| of the encoder features** before the learned comparison. It is a guide to where the two images differ, not an exact explanation of the decision.
