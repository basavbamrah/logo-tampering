# EXP-007 — Two-head (Siamese) model trained on synthetic pairs
- **Status:** Done. Scripts: `src/models.py`, `src/train_pairs.py`, `src/run_sweep.py`, `src/infer_real.py`. Tuning log: `experiments/EXP-007_tuning_log.md`.

- **Licences:** PyTorch / torchvision (BSD-3), ResNet-18 ImageNet weights (BSD-3, torchvision), timm (Apache 2.0), DINOv2 ViT-S/14 weights (Apache 2.0).


## Visual results (click to open)
- [Architecture diagram (real tensor shapes)](../outputs/exp007/evidence/architecture/architecture_diagram.png)
- [Training curves, all runs](../outputs/exp007/evidence/training_curves/overview_val_auc.png)
- [Training curve of the best model](../outputs/exp007/evidence/training_curves/sd_depth1_more.png)
- [ROC per test set](../outputs/exp007/evidence/test_results/roc_test_sets.png)
- [Share caught per tamper (with error bars)](../outputs/exp007/evidence/test_results/caught_per_tamper.png)
- ['Which tamper?' confusion matrix](../outputs/exp007/evidence/test_results/confusion_kind_sd_depth1_more.png)
- [Score histograms genuine vs tampered](../outputs/exp007/evidence/test_results/score_histograms.png)
- [Correct detections (examples)](../outputs/exp007/evidence/examples/correct.jpg)
- [All missed tampers (1 of 3)](../outputs/exp007/evidence/examples/missed_1of3.jpg)
- [All false alarms (1 of 2)](../outputs/exp007/evidence/examples/false_alarms_1of2.jpg)
- [Real photos: heatmaps, all 49](../outputs/exp007/evidence/real_photos/contact_sheet_sd_depth1_more.jpg)
- [Real photos: scores of 4 models](../outputs/exp007/evidence/real_photos/real_scores_by_run.png)
- [Real photo heatmap example (021, black-edge artefact)](../outputs/exp007/evidence/real_photos/heatmaps_sd_depth1_more/021.jpg)

## READ FIRST — summary for review
- **Run name → configuration:** see [`EXP-007_model_registry.md`](EXP-007_model_registry.md) (all 23 trained models, exact settings, results, file paths). Why each tuning run was made: [`EXP-007_tuning_log.md`](EXP-007_tuning_log.md).
- **FINAL MODEL (2026-10-11): `e14_consistent`** (EXP-014): the same siamese_diff depth 1 and recipe, trained on the consistent dataset (every crop registered, `data/synth_v2`). Shared NORVIA test AUC 0.959 (vs 0.943), false alarms ~4 % on every test set. See `experiments/EXP-014_consistent_dataset.md`; every setting (loss, optimiser, schedule, thresholds, metrics) in `docs/04_final_model_parameters.md`.
- **Previous best model:** `siamese_diff`, encoder depth 1, trained on NORVIA + multi-font data with the extra 6000 pairs (`sd_depth1_more`): val AUC 0.950, NORVIA test AUC 0.943, unseen-font AUC 0.951. Caught on NORVIA test at the 5 %-on-validation threshold (measured false alarms on NORVIA test: 5.7 %, 6/105; multi-font 9.6 %; unseen font 14.9 %, so the false-alarm rate does not carry over to new fonts): respace 82 %, recolour 85 %, glyph swap 89 %, font swap / overlap / cut 100 %, **stretch 47 % (weak spot)**.
- **What mattered most:** (1) more synthetic data (+0.03-0.08 AUC), (2) a shallow encoder with fine feature maps (depth 1). Everything else (augmentation, dropout, focal loss, the helper loss weight, pure yes/no training) made no measurable difference; freezing layers, lower LR and the frozen DINO model were worse.
- **Architectures compared:** two-input `siamese_diff` (best) vs 6-channel `early_fusion` (on par with the old baseline, weaker on the unseen font) vs `dino_frozen` (memorises; dropped).
- **Real photos:** the model agrees with the training-free methods (flow, difference map, hue) on the same flagged photos; its probabilities are not calibrated on real photos (use rankings).
- **Full resolution** (`norvia_896` vs control `norvia_448`): test 0.921 vs 0.918 → no measurable gain; data, not resolution, was the bottleneck.
- **Open:** whole-logo stretch (every method is weak on it).

## The two architectures in detail (both fully implemented in `src/models.py`)

Both take two pictures of the same size, the clean reference and the rectified candidate (448x128 by default; 896x256 in the full-resolution run), normalised with ImageNet statistics, and output the same two heads: **tampered?** (1 logit, BCE) and **which tamper?** (8 classes: none, stretch, recolour, respace, glyph_swap, font_swap, overlap, cut; cross-entropy, weight 0.3).

### A. Two-input Siamese (`siamese_diff`) — the one we use
```
reference ─┐  ResNet-18 encoder, ImageNet-pretrained, SHARED weights   ┌─ |a − b|  ┐
           ├─ (stem: 7x7 conv + maxpool, then stage 1 [+ stage 2 ...]) ┤           ├─ conv3x3(128)+BN+ReLU ─ conv3x3(128, stride 2)+BN+ReLU ─ max-pool ⊕ mean-pool ─┬─ tampered? (1)
candidate ─┘                                                            └─  a · b   ┘                                                                          └─ which tamper? (8)
```
- **Encoder:** the first part of ResNet-18 (torchvision `IMAGENET1K_V1` weights). `depth` = number of stages kept:

| depth | layers kept | conv layers | feature map (448x128 input) | channels | encoder params |
|---|---|---|---|---|---|
| **1 (final)** | stem + stage 1 | 5 | 112x32 (1/4) | 64 | 0.16 M |
| 2 (first default) | + stage 2 | 10 | 56x16 (1/8) | 128 | 0.68 M |
| 3 | + stage 3 | 15 | 28x8 (1/16) | 256 | 2.78 M |

- The same encoder (shared weights) runs on both pictures, so they are described the same way.
- **Comparison:** at every map location, |a-b| (how different) and a·b (what both contain) are concatenated, then 2 conv layers learn which differences matter. Max-pooling catches one changed letter; mean-pooling catches whole-logo changes (recolour).
- **Trainable:** everything (encoder fine-tuned + comparison + heads): 0.46 M at depth 1, 1.13 M at depth 2.
- **Why it suits the task:** the inputs are aligned, so comparing location by location keeps WHERE they differ; a shallow encoder keeps fine position detail (1-6 px tampers). Heatmap of |a-b| available for explanation (`infer_real.py --heatmaps`).


### B. Six-channel early fusion (`early_fusion`) — comparison baseline
```
reference ┐
          ├─ stack → 6-channel image → full ResNet-18 (all 4 stages, ImageNet-pretrained; first conv widened 3→6 channels) → global average pool (512) ─┬─ tampered? (1)
candidate ┘                                                                                                                                        └─ which tamper? (8)
```
- **Encoder:** the whole ResNet-18 (stem + stages 1-4). The first 7x7 conv now takes 6 channels; its weights are the pretrained RGB filters copied for both pictures and halved (so the output scale stays the same). The final classifier is removed.
- The network sees both pictures at once from the very first layer; there is no explicit "difference" step, it has to learn the comparison itself.
- **Trainable:** all 11.19 M weights.
- **Result:** val AUC 0.902 (on par with the depth-2 Siamese), NORVIA test 0.858, unseen font 0.878 (weaker); still improving at epoch 14 of 15. More weights, less built-in structure → needs more data.
- **Why it lost (summary for the write-up, 2026-10-11):**
  - (1) It memorised: training AUC 0.997 vs validation 0.90, with 11.2 M weights (24× the depth-1 siamese) for 2 020 training pairs.
  - (2) The full ResNet shrinks the image 32× (4×14 map) before averaging, so 1-6 px tampers are hard to keep.
  - (3) No built-in comparison: ImageNet filters are for one natural image, not two stacked logos.
  - (4) No difference heatmap.
  - Not retrained on the larger datasets. It is the natural candidate for un-aligned input, because it has no location-by-location difference layer.

### C. (Dropped) `dino_frozen`
Frozen DINOv2 ViT-S/14 (timm `vit_small_patch14_dinov2.lvd142m`) gives a 9x32 grid of patch features per picture; the same comparison head as A on top; only the head trains (1.04 M). Memorised the training set (train AUC 1.00, val 0.78). See "Why dino_frozen failed" below.

### Training data per configuration (from `experiments/EXP-003_dataset_stats.md`)
| Configuration | Runs | Training pairs | Genuine / tampered | Validation | Test |
|---|---|---|---|---|---|
| D1+D2, original data | sd_base…, sd_depth1…, ef_base, dino_base | 2020 (D2 1028 + D1 992) | ≈ 50 / 50 | 454 mixed | NORVIA 206, multi-font 198, unseen font 99 |
| D1+D2 + extra data | sd_more, **sd_depth1_more** | **7789** (+ D2 2954 + D1 2815) | 3905 / 3884 | 454 mixed | same |
| NORVIA only + extra data | norvia_448, norvia_896 | 3982 | 2008 / 1974 | 243 NORVIA | NORVIA 206 |

### Shared training recipe (unless a run says otherwise)
AdamW lr 3e-4, weight decay 1e-4, cosine decay per step, 15 epochs, batch 32 (16 in the resolution runs), mixed precision (fp16), best epoch by validation AUC, thresholds on validation (5 % false alarms). Data: D2 = NORVIA with registration-made crops (F12), D1 = multi-font words. Images cached in memory (GPU or pinned CPU).

## Question
Can a two-input model trained on synthetic (reference, candidate) pairs tell genuine from tampered under camera-like nuisance, and does it transfer to the real 50?


## Variants (doc §8.2)
- D1: generic words × many fonts
- D2: tamper the NORVIA reference itself; Google Fonts only as donors for look-alike glyphs
- D1 → D2: pretrain then fine-tune.


## Data
From EXP-003. D1 = generic words × fonts; D2 = NORVIA tampers. Mix or fine-tune (doc §8.2b).

## Metrics
Synthetic held-out: accuracy, F1, ROC-AUC, per-tamper-type and per-magnitude sensitivity, false-alarm rate. Real 50: agreement with blind manual labels (doc §13). Reported separately.

## Risks
Synthetic→real gap; learning generator artefacts; 4 GB VRAM; time.

## The three architectures (candidate: "create all the 3 architectures in parallel, train sequentially")
All take (ref, cand) at 448x128 and output "tampered?" (1 logit) + "which tamper?" (8 classes: none + 7 types).

| Name | How the two pictures are compared | Trainable params | Peak GPU (dry-run, batch 32) |
|---|---|---|---|
| `siamese_diff` (recommended) | Same pretrained ResNet-18 (up to layer2, 1/8 resolution) on each picture; compare the feature maps location by location (\|a-b\|, a*b); 2 conv layers; max+mean pool | 1.13 M | 853 MB |
| `early_fusion` | Both pictures stacked as 6 channels into one pretrained ResNet-18 (first layer's RGB filters copied for both) | 11.19 M | 606 MB |
| `dino_frozen` | Frozen DINOv2 ViT-S/14 gives a 9x32 grid of patch features per picture; same comparison head as siamese_diff | 1.04 M (ViT frozen) | 293 MB |

Training recipe (`train_pairs.py`): D2 uses F12 crops (`_cand_reg.png`, unregistered ones skipped), D1 uses `_cand.png`; loss = BCE + 0.3 x CE(type); AdamW lr 3e-4, cosine, 15 epochs, batch 32, mixed precision; keeps the best epoch by validation AUC; reports per split (val, test D2, test D1, test_unseen_font) AUC, accuracy, false-alarm rate, and "caught" per tamper type at the threshold that flags 5 % of genuine validation samples. `infer_real.py` scores the real crops against all 3 references (mean probability) into tampered / genuine / don't know (+ "can't register", + cut flag from low ECC), for one or several runs, with optional heatmaps. (Replaced `predict_real.py`, which used ref_1 only.)

## Runs
| Date | Command | Params | Result | Notes |
|---|---|---|---|---|
| 2026-10-10 | `train_pairs.py --arch <each> --dry-run` | 2 batches forward+backward, no optimiser step, nothing saved | all three OK | 2020 train pairs (20 unregistered D2 skipped), 454 val |
| 2026-10-10 | `train_pairs.py --arch siamese_diff --run sd_base` | 448x128, batch 32, AdamW 3e-4, cosine, 15 epochs, BCE + 0.3 CE | best epoch 6, val AUC 0.896 | 7 s/epoch, 862 MB GPU. Log `outputs/exp007/logs/sd_base.log`, TensorBoard `outputs/tensorboard/sd_base` |

## Baseline results (sd_base, best epoch 6)
Learning curve: train AUC rises to 0.99 by epoch 15 while val AUC peaks at 0.896 (epoch 6) and drifts down to 0.87 → **over-fitting** after ~6 epochs (2k training pairs). The best epoch is kept by val AUC.

| Split | n | AUC | acc @0.5 | false alarms @thr | kind accuracy |
|---|---|---|---|---|---|
| val (D1+D2) | 454 | 0.896 | 0.815 | 5.4 % (by construction) | 0.61 |
| test D2 (NORVIA) | 206 | 0.827 | 0.801 | 4.8 % | 0.57 |
| test D1 (generic) | 198 | 0.844 | 0.808 | 10.5 % | 0.50 |
| test unseen font (Lato) | 99 | 0.928 | 0.859 | 8.5 % | 0.63 |

Share caught at the 5 % false-alarm threshold (threshold 0.41, set on val):
| Tamper | val | test D2 | test D1 | unseen font |
|---|---|---|---|---|
| stretch | 0.62 | 0.20 | 0.67 | 1.00 |
| recolour | 0.35 | 0.54 | 0.30 | 0.64 |
| respace | 0.17 | 0.18 | 0.29 | 0.00 |
| glyph_swap | 0.59 | 0.78 | 0.67 | 0.60 |
| font_swap | 1.00 | 1.00 | 0.92 | 1.00 |
| overlap | 1.00 | 1.00 | 0.93 | 1.00 |
| cut | 1.00 | 1.00 | 1.00 | 1.00 |

**Observations**
- Obvious tampers (font swap, overlap, cut) are solved. Subtle ones are not: re-spacing is almost missed (17-29 %) although EXP-008 flow catches 97-100 % of the same shifts on the same kind of crops. Recolour is weak (30-54 %); the generator's random colour cast (±6 % per channel) probably competes with 3-15° hue changes.
- Per-tamper test numbers rest on ~15 samples each (D2 test has 206 samples over 8 classes): differences of ±0.15 between runs can be noise. The seed repeats in the sweep measure this.
- Unseen-font AUC (0.93) is not worse than seen fonts: no sign that the model memorised fonts.
- A harmless PyTorch warning appears once per run ("lr_scheduler.step() before optimizer.step()"): mixed-precision skips the very first optimiser step while it calibrates its scale.

## Results
See "READ FIRST" above, the phase sections below, and the generated tuning log (`experiments/EXP-007_tuning_log.md`).



## Evidence level
Synthetic NORVIA/multi-font test sets (≈200 samples each, ≈15 per tamper type → per-tamper numbers are noisy, ±0.15-0.25); run-to-run noise measured (val AUC ±0.006). Real photos: no ground truth; agreement with independent methods only.

## Decision / next step
Use `siamese_diff` depth 1 trained with the extra data as the model component (or the full-resolution model if `norvia_896` clearly beats its control). Next: combined scorer with flow + hue + difference map (proposed EXP-011, awaiting the candidate).


## Inference on the real photos — `src/infer_real.py`
- Each registered real crop is compared with **all 3 references** (same layout, so one crop lines up with all three) → p_ref1/2/3, p_mean (used for the decision), p_min/max. Majority vote of the 3 "which tamper" answers.
- Decision on p_mean with the run's validation thresholds: tampered / genuine / don't know; 029 → "can't register"; ECC < 0.9 → "logo partly outside the photo" flag.
- Several runs at once (`--runs a,b` or `--best N`) → `outputs/exp007/real_summary.csv` with a vote count.
- `--heatmaps` (siamese_diff): crop + difference map per photo and a sheet. First version scaled each tile to its own maximum, which made genuine photos look as "hot" as tampered ones (misleading).
- Replaced `predict_real.py` (ref_1 only, one run).

### First real-photo results (sd_base and sd_base_cache; preliminary, before tuning)
| | sd_base | sd_base_cache |
|---|---|---|
| tampered | 15 | 18 |
| don't know | 23 | 31 |
| genuine | 11 | 0 |
| can't register | 1 (029) | 1 (029) |

- **The 3 references give almost identical probabilities** (differences ≤ 0.03): the choice of reference doesn't matter here.
- Both runs call tampered: 009, 011, 013, 016, 020, 021, 025, 030, 037 (= the shape-difference set of EXP-008/EXP-010) **plus** 027, 028, 033, 038, labelled "recolour". The shape-only methods cannot see colour, so these are new candidates, **or a synthetic→real colour gap** (the model may react to real paper/ink colours it never saw). Needs the colour check (EXP-005) before believing it.
- **Calibration warning:** sd_base_cache gives no "genuine" at all (its low threshold 0.053 from synthetic val). Real photos score higher overall than synthetic genuine ones. Probabilities from synthetic training don't transfer 1:1 to real photos. Ranking is more trustworthy than the absolute thresholds.
- Heatmaps (shared scale, `outputs/exp007/sd_base/real_heatmaps_sheet.jpg`): genuine-looking photos stay blue; 009 lights up at R/V, 013/016 at the cut edge, 011 at N/R, 020 at V/A, consistent with flow and the difference map.

## Run-to-run noise (same settings, different seed / code path) — measured before judging any tuning
| run | best epoch | val AUC | test D2 AUC | test D1 AUC | unseen font AUC | D2 respace caught |
|---|---|---|---|---|---|---|
| sd_base (seed 0, DataLoader) | 6 | 0.896 | 0.827 | 0.844 | 0.928 | 0.18 |
| sd_base_cache (seed 0, GPU cache) | 9 | 0.887 | 0.881 | 0.858 | 0.920 | 0.68 |
| sd_base_s1 (seed 1) | 11 | 0.884 | 0.880 | 0.856 | 0.925 | 0.50 |
| sd_base_s2 (seed 2) | 5 | 0.887 | 0.830 | 0.869 | 0.912 | 0.41 |

- Val AUC varies by about ±0.006 (0.884-0.896); test D2 AUC by 0.83-0.88; per-tamper "caught" on test D2 by up to ±0.25 (≈15 samples per tamper type).
- **Rule adopted for the sweep:** a setting is "better" only if its val AUC clearly exceeds the baseline band (> ~0.905), and per-tamper changes count only if large and repeated. Smaller differences are reported as "no measurable difference".
- The GPU-cache refactor (sd_base_cache) is within this band → behaves like the original code.

## Tuning log
Candidate: "keep logs of all the hyper parameter tuning that you do and reason for them". Every run now carries its reason in `src/run_sweep.py`, and the sweep writes **`experiments/EXP-007_tuning_log.md`** (change, why, val/test AUC, verdict against the noise band) after each run.

## Phase 1-2 conclusions (siamese_diff tuning)
| Change | val AUC | Verdict | Reading |
|---|---|---|---|
| augmentation | 0.896 | no difference | the generator already varies camera effects |
| dropout 0.3 + WD 1e-3 | 0.893 | no difference | over-fitting is not fixed by standard regularisation |
| frozen stem + layer1 | 0.845 | worse | early filters must adapt to this task |
| **encoder depth 1 (3 seeds)** | **0.905 / 0.913 / 0.920 (mean 0.913)** | **better (confirmed)** | finer feature maps keep position detail; test D2 mean 0.868 vs 0.855 (smaller gain) |
| encoder depth 3 | 0.877 | slightly worse | coarser maps lose detail |
| focal loss | 0.875 | slightly worse | — |
| kind-loss weight 0 / 1.0 | 0.898 / 0.888 | no difference | the auxiliary task is neutral; kept at 0.3 for its explanatory value |
| learning rate 1e-4 | 0.854 | worse | learns too slowly in 15 epochs (best epoch 12) |

**Decision D41: encoder depth 1 becomes the default for siamese_diff.** Phase 4 (more data) includes depth 1.

## Architecture comparison (baseline settings, 15 epochs)
| Arch | trainable params | val AUC | test D2 AUC | unseen font AUC | best epoch | note |
|---|---|---|---|---|---|---|
| siamese_diff depth 1 (3 seeds) | 0.46 M | 0.905-0.920 (mean 0.913) | 0.854-0.892 | 0.917-0.934 | 6-10 | **best** |
| siamese_diff depth 2 (4 runs) | 1.13 M | 0.884-0.896 | 0.827-0.881 | 0.912-0.928 | 5-11 | baseline |
| early_fusion | 11.19 M | 0.902 | 0.858 | 0.878 | 14 | on par with baseline; weaker on the unseen font; may still have been improving |
| dino_frozen | 1.04 M | 0.783 | 0.752 | 0.815 | 2 | **memorises**: train AUC 1.00 by epoch 10, val stuck at 0.75-0.78 from epoch 1 |

### Why dino_frozen failed 
1. DINOv2's self-supervised training uses colour jitter, crops and small shifts as augmentations, i.e. it learns to IGNORE exactly the cues of subtle tampers (small recolours, 1-6 px shifts).
2. 14x14 px patches (9x32 grid) are coarse compared with siamese_diff depth 1 (4x4 px cells).
3. Frozen backbone: only the head trains, so it cannot recover that information; it memorises the 2k training pairs instead.
Per tamper (test D2): font swap / overlap / cut 100 %, but stretch 13 %, recolour 0 %, respace 9 %, glyph swap 11 %.
Possible rescues (not pursued): unfreeze the last ViT blocks, full 920x260 input, intermediate-layer features. Reported as an informative negative result.

## Phase 4-5 results
| Run | val AUC | test D2 AUC | unseen font AUC | note |
|---|---|---|---|---|
| sd_more (depth 2 + 4x data) | 0.940 | **0.949** | 0.928 | respace 86 %, recolour 85 %, glyph 89 %, stretch 40 % caught (test D2) |
| sd_depth1_more (depth 1 + 4x data) | **0.950** | 0.943 | **0.951** | respace 82 %, recolour 85 %, glyph 89 %, stretch 47 % |
| sd_depth1_bin_s0 (depth 1, yes/no loss only) | 0.907 | 0.878 | 0.926 | |
| sd_depth1_bin_s1 (same, seed 1) | 0.907 | 0.843 | — | |

- **More data is the biggest lever** (+0.03-0.08 AUC): the model was data-limited.


## Resolution experiment (phase 6) — agreed with the candidate (D44)
- Candidate: "i agree with your recommendation since depth 1 is good so far, and sure train on 448 as well". Asked about all choices before training (D43).
- Settings: NORVIA only (original train 1048 + synth_more 3000 → ~4000 pairs after registration), depth 1, batch 16, AdamW 3e-4, cosine, 15 epochs, best val epoch. Only the input differs: `norvia_448` (448x128, control) vs `norvia_896` (896x256, full-res crops `<id>_cand_reg_full.png`, references `<id>_ref_full.png`).
- Code: `train_pairs.py --full-res`; with `--sets D2` all evaluation splits are NORVIA-only (val selection consistent with training). `infer_real.py` feeds full-res models the full 920x260 real crops.
- Full-res crops are ADDED next to the existing files; nothing deleted (candidate: "do not delete the existing dataset").
- Dry-run 448/b16 NORVIA-only: 1.73 GB peak GPU. The 896 run does its own dry-run first (queue script) before training.


## Resolution experiment — result (phase 6, 2026-10-10)
| Run | input | best epoch | val AUC (NORVIA) | **test D2 AUC** | test false alarms | stretch | recolour | respace | glyph | font/overlap/cut |
|---|---|---|---|---|---|---|---|---|---|---|
| norvia_448 (control) | 448x128 | 6 | 0.950 | **0.918** | 13.3 % | 0.47 | 0.77 | 0.91 | 1.00 | 1.00 |
| norvia_896 | 896x256 | 9 | 0.962 | **0.921** | 8.6 % | 0.47 | 0.69 | 0.82 | 0.89 | 1.00 |

- **No measurable gain from full resolution on the test set** (0.921 vs 0.918, single runs; per-tamper differences both ways, within ±0.15 noise). Val is slightly higher (+0.012).
- **Stretch unchanged (47 %)** → stretch is limited by registration absorbing it, not by resolution.
- Cost: ~46 s per training epoch + slow full-size evaluation, vs ~10 s.
- **Conclusion: data, not resolution, was the bottleneck.** Best overall model stays `sd_depth1_more` (448x128, NORVIA + multi-font + extra data: test D2 0.943, unseen font 0.951).
- Side result (control vs `sd_depth1_more`): without the multi-font data, NORVIA test AUC 0.918 vs 0.943 and higher false alarms → the font set probably helps (single runs; suggestive).

## Intermediate results / evidence (added 2026-10-10, night)
`python src/exp007_evidence.py` (inference only, ~45 s on GPU) → `outputs/exp007/evidence/` (index with takeaways: `outputs/exp007/evidence/README.md`):
- `architecture/`: diagram of siamese_diff depth 1 and early_fusion (shapes from a real forward pass); parameter counts (depth 1: 455,497; depth 2: 1,128,521; early_fusion: 11,190,537; dino_frozen: 1,035,273 trainable of 22.7 M).
- `training_curves/`: loss and AUC per epoch for all 23 runs, best epoch marked; overview of val AUC.
- `test_results/`: ROC per test set, share caught per tamper with 95 % intervals, "which tamper?" confusion matrix, score histograms.
- `examples/`: correct detections, all 41 missed tampers and all 24 false alarms of `sd_depth1_more` on the synthetic test pairs (candidate | reference | difference).
- `real_photos/`: `sd_depth1_more`, `sd_more`, `ef_base` and `sd_base` on the 49 real photos, heatmaps for the two siamese models, comparison CSV.

Checks: re-running `sd_base` reproduces the saved real-photo predictions (max difference 0.0002, same decision 50/50); AUCs recomputed from predictions.csv match metrics.json; best epochs match.

**Real photos, `sd_depth1_more`:** 18 tampered / 25 don't know / 6 genuine / 1 can't register. All 4 models call 14 photos tampered: 009 011 013 016 020 021 025 027 028 030 033 037 038 043.

**New observations from the evidence:**
- False alarms rise on fonts not seen in training (14.9 % on Lato vs the 5 % target).
- The choice of reference changes `sd_depth1_more`'s probability by up to 0.22 (photo 022); this only affects borderline photos.
- Likely label artefact: 3 font_swap samples swap NotoSans-Bold for "NotoSans-Bold-system", which is almost the same font (D1_00625, D1_00793, D1_01017). The data were not changed.


