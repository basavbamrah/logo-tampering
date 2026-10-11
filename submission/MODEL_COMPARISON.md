# Model comparison: the three models

All three output the same two things: a "tampered?" score and a "which tamper?" class (8 classes). All take the reference logo and the aligned photo (448×128 each), and all start from ImageNet-pre-trained ResNet-18 weights.

## Results (synthetic test data, labels known)
"Shared NORVIA test" = the same 206 test pairs for all three, scored at each model's own threshold (5 % of genuine validation pairs flagged).

| | **Final: `e14_consistent`** | Previous best: `sd_depth1_more` | Six-channel: `ef_base` |
|---|---|---|---|
| Architecture | two-input (siamese), depth 1 | two-input (siamese), depth 1 | six-channel (early fusion) |
| Trainable weights | 455 497 | 455 497 | 11 190 537 |
| Training data | variant 2: consistent, every crop aligned like a real photo | variant 1: mixed (NORVIA aligned, other fonts known-camera) + extra | variant 1, original size |
| Training pairs | 6 770 | 7 789 | 2 020 |
| **Shared NORVIA test AUC** | **0.959** | 0.943 | 0.858 |
| False alarms (shared test) | **3/105** | 6/105 | 6/105 |
| Tampers caught (shared test) | 84/101 | 86/101 | 70/101 |
| stretch / recolour / re-spacing | 5/15, 10/13, 19/22 | 7/15, 11/13, 18/22 | 3/15, 7/13, 11/22 |
| letter swap / font swap / overlap / cut | 8/9, 14/14, 15/15, 13/13 | 8/9, 14/14, 15/15, 13/13 | 7/9, 14/14, 15/15, 13/13 |
| Other fonts, all crops aligned (same 653 pairs) | **0.932** | 0.880 | – |
| Unseen font (Lato), all crops aligned (same 325 pairs) | **0.925** | 0.884 | – |
| False alarms on its own test sets | 4.2-4.4 % everywhere | 5.7 % NORVIA, 14.9 % unseen font | 5.7 % NORVIA |
| Train vs validation AUC at the chosen epoch | 0.975 vs 0.933 | 0.991 vs 0.950 | **0.997 vs 0.902 (memorised)** |
| Best epoch (by validation AUC) | 11 | 12 | 14 |
| "Tampered" threshold (5 % of genuine validation pairs flagged) | 0.493 | 0.333 | 0.622 |

**Training settings, identical for all three** (from each run's `config.json`):
- loss: binary cross-entropy ("tampered?") + 0.3 × cross-entropy ("which tamper?", 8 classes);
- optimiser: AdamW, learning rate 3e-4, weight decay 1e-4, cosine decay to 0, no warm-up;
- batch 32, 15 epochs, input 448×128, ImageNet normalisation, no extra augmentation, dropout 0, mixed precision, seed 0;
- model selection: best validation AUC.

**Evaluation metrics:** ROC AUC (needs no threshold); false-alarm rate and share of tampers caught, overall and per tamper type, at each model's own validation threshold. Every value of the final model: [docs/04_final_model_parameters.md](docs/04_final_model_parameters.md).

Figures:
- training curves: [final](outputs/exp014/evidence/training_curve.png), [previous best](outputs/exp007/evidence/training_curves/sd_depth1_more.png), [six-channel](outputs/exp007/evidence/training_curves/ef_base.png);
- [final vs previous, per tamper](outputs/exp014/caught_per_tamper.png);
- [architecture diagram](outputs/exp007/evidence/architecture/architecture_diagram.png);
- [same model, different training data](outputs/exp014/evidence/data_progression.png).

![Training curve of the final model](outputs/exp014/evidence/training_curve.png)

**ONNX:** the final model's ONNX export gives the same AUC, false alarms and catches as PyTorch on all 1 894 test pairs (largest difference in p 4e-6; `model/onnx_check_*.md`).

## The architectures, briefly

**Two-input / siamese (final and previous best).**
- The same small encoder (the first block of ResNet-18) turns each image into a map of 32×112 positions with 64 features each. The two images share the same weights.
- At every position the model computes |reference − tampered| (what differs) and reference × tampered (what both contain).
- Two small convolution layers learn which differences matter. Max + mean pooling then feeds the two outputs.
- Because it compares position by position, it **needs aligned input**, which is what the alignment step provides.
- It can show *where* it sees a difference as a heatmap.

**Six-channel (early fusion).**
- The two images are stacked into one 6-channel image and fed to the full ResNet-18 (four stages, global average pooling), then the same two outputs.
- There is no built-in comparison; the network has to learn how to compare by itself.

## Why the final model is the one to use
- **Same architecture as the previous best, better data.** Every training crop is made the way a real photo is processed, so the model learns exactly what it will see in use.
  - Result on the shared test: 0.959 vs 0.943, with half the false alarms.
  - It holds up on other fonts (0.93 vs 0.88), even with fewer training pairs.
  - The previous model's multi-font crops were made with the known camera, which hid real-world difficulty.
- **The six-channel model lost, for four reasons:**
  - it memorised (11 million weights for 2 020 pairs);
  - it shrinks the image 32× before deciding, while tampers are 1-6 px;
  - it must learn the comparison from scratch;
  - it gives no heatmap.

  It was not retrained on the larger datasets. It remains a candidate for un-aligned input, because it has no position-by-position comparison.

## Common weak spot and open item
- **Whole-logo stretch** is almost never caught by any of them: the alignment absorbs a stretch of the whole logo.
- **No calibrated confidence score yet.** The score from 0 to 1 and the "which tamper?" probabilities exist, but they have not been calibrated.
