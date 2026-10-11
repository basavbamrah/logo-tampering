# Logo tampering detection: submission folder

A self-contained copy of the final work: the write-up, the three key experiments, the final model (PyTorch + ONNX), a small app to try it, and the data needed to check it. **This folder runs on its own**; it does not need the main repository.

## Contents
| Path | What it is |
|---|---|
| `MODEL_COMPARISON.md` | results and short architecture explanation of 3 models: final siamese, previous siamese, six-channel |
| `docs/04_final_model_parameters.md` | **every setting of the final model**: data generator, alignment, input, architecture, loss, optimiser, schedule, thresholds, evaluation metrics, results, graphs |
| `experiments/EXP-001_registration_feasibility.md` | step one: aligning each photo with the reference (SIFT + RANSAC + ECC) |
| `experiments/EXP-007_two_head_model.md`, `EXP-007_model_registry.md` | the two-input model: architectures, all training runs, tuning |
| `experiments/EXP-014_consistent_dataset.md` | the final model: trained on the consistent dataset (every crop aligned like a real photo) |
| `outputs/` | the figures linked from the documents (step pictures, training curve, ROC, examples, heatmaps) |
| `model/` | final model `e14_consistent`: `best.pt` (PyTorch), `e14_consistent.onnx`, `config.json` (training settings), `metrics.json`, `model_info.json` (thresholds, classes, heatmap scale), `onnx_check_*.md` (ONNX vs PyTorch) |
| `app/` | Streamlit app (`app.py`), its processing code (`pipeline.py`) and the 3 reference images |
| `samples/` | the 50 photos of the assignment, to try the app, and `other_methods.csv` (what the independent reference comparisons found per photo) |
| `outputs/real_photos/` | **the final model's output for each of the 50 photos**: one picture per photo (`001.jpg` … `050.jpg`), `results.csv` and `overview.jpg` |
| `test_data/norvia_test/` | the shared synthetic NORVIA test set (206 pairs with labels) used to check the ONNX model |
| `scripts/run_on_samples.py` | runs the final model on the 50 photos and makes `outputs/real_photos/` |
| `scripts/evaluate_onnx.py` | accuracy check of the ONNX model (AUC, false alarms, catches per tamper type, ONNX vs PyTorch) |
| `scripts/export_onnx.py` | converts `best.pt` to ONNX and checks it against PyTorch |
| `scripts/package_docs.py`, `scripts/package_data.py` | rebuild this folder's documents and data from the main repository (only needed by the author) |

## Steps to run
Tested with Python 3.12 on Linux: a copy of this folder, placed outside the main repository, was installed in a fresh virtual environment from `requirements.txt` only (no PyTorch). Steps 2 and 3 ran there: same accuracy, the app server started, and the app gave the expected result for 009, 013, 007 and 029.

**1. Install** (from this folder):
```
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
This installs OpenCV, NumPy, ONNX Runtime and Streamlit. PyTorch is not needed.
If `venv` fails with "ensurepip is not available" (Debian/Ubuntu system Python), install `python3.12-venv`, or use another Python 3.12 (pyenv, conda, uv).

**2. Check the model** (about 5 s):
```
python scripts/evaluate_onnx.py
```
Expected output: AUC 0.959, false alarms 3/105, caught 84/101, largest |p ONNX − p PyTorch| about 3e-6, and 0/206 decisions that differ.

**3. Start the app:**
```
streamlit run app/app.py
```
Then open the address it prints (normally http://localhost:8501).
1. Choose the reference: the 3 built-in references (scores are averaged, as in the experiments), one of them, or your own upload.
2. Choose a photo: upload one, or pick one of the 50 sample photos.
3. Press **Check this photo**.

The app aligns the photo with the reference (the same code as EXP-001) and shows:
- where the logo was found;
- the aligned photo with the reference outline in red;
- the decision TAMPERED / GENUINE / DON'T KNOW, with p(tampered);
- the "which tamper?" probabilities;
- optionally, the heatmap of where the model sees a difference.

It also warns when part of the logo is outside the photo, and says "can't verify" when the logo cannot be aligned. Try for example 009 (tampered), 013 (cut off), 007 (don't know) and 029 (can't verify).

**4. Results on the 50 photos (optional, ~1 min):**
```
python scripts/run_on_samples.py
```
This rewrites `outputs/real_photos/` (already included). Each picture `outputs/real_photos/<photo>.jpg` shows:
- the photo with the found logo outlined in yellow;
- the decision, p(tampered) and the confidence;
- p for each of the 3 references, how much of the logo is outside the photo, and the alignment quality;
- the "which tamper?" probabilities and what the other methods found;
- the reference logo, the aligned photo with the reference outline in red, the aligned photo alone, and the heatmap.

`overview.jpg` puts all 50 heatmaps on one sheet. Result: 15 tampered (3 of them cut off), 32 don't know, 2 genuine, 1 can't verify (029). These are the same decisions as the main pipeline.

## How confidence is shown
Every decision comes with how often that decision was right on the labelled synthetic test sets (1 688 pairs, about half tampered; `model/confidence.json`):

| Decision | Rule | Share of test pairs | How often right |
|---|---|---|---|
| TAMPERED | p ≥ 0.493 | 43 % | 95 % really tampered |
| GENUINE | p ≤ 0.053 | 25 % | 92 % really genuine |
| DON'T KNOW | in between | 32 % | no call (20 % of these were tampered) |


Two more signs are shown for each photo: whether the 3 references agree on p, and how many independent methods (difference map, colour check, cut-off) agree with the model. These rates come from synthetic data. On real photos, where the share of tampered logos is unknown, they are a guide, not a guarantee.

## ONNX vs PyTorch: no accuracy lost
| Test set | Pairs | AUC ONNX | AUC PyTorch | Largest \|p ONNX − p PyTorch\| | Decisions that differ |
|---|---|---|---|---|---|
| NORVIA, shared with earlier models (bundled) | 206 | 0.959 | 0.959 | 2.8e-6 | 0 |
| NORVIA | 710 | 0.945 | 0.945 | 2.7e-6 | 0 |
| Other fonts | 653 | 0.932 | 0.932 | 3.9e-6 | 0 |
| Unseen font (Lato) | 325 | 0.925 | 0.925 | 3.8e-6 | 0 |

The false alarms and the catches per tamper type are also identical (`model/onnx_check_bundled.md`, `model/onnx_check_full.md`). The last three sets are too large to bundle (≈170 MB). With the main repository next to this folder, they are checked with `python scripts/evaluate_onnx.py --data ../data/synth_v2 --pytorch`. On the 50 real photos the app gives the same decisions as the main pipeline (scores within 0.0013).

The app and the scripts fix their random seeds, so the same input always gives the same result.

## Re-export the model to ONNX (optional, needs PyTorch)
```
pip install -r requirements-export.txt
python scripts/export_onnx.py
```
This writes `model/e14_consistent.onnx` (inputs: reference, tampered; outputs: tampered logit, 8 class logits, heatmap) and `model/model_info.json`. It checks ONNX against PyTorch on the 49 sample photos that can be aligned and on the 206 bundled test pairs (largest difference on the raw outputs 1.8e-5). The export gives a byte-identical file. PyTorch prints a deprecation notice for the classic exporter it uses; this is expected.

## Results for the 50 given images

Per-photo pictures (`001.jpg` … `050.jpg`), the sheet of all heatmaps (`overview.jpg`) and the full table (`results.csv`) are in [`outputs/real_photos/`](outputs/real_photos). Counts from `results.csv`:

| Decision | Count | Photos |
|---|---|---|
| TAMPERED | 12 | 009, 011, 020, 021, 025, 027, 028, 030, 033, 038, 041, 043 |
| TAMPERED + CUT OFF | 3 | 013, 016, 037 |
| DON'T KNOW | 32 | 001–004, 006–008, 010, 012, 015, 017–019, 022–024, 026, 031, 032, 034–036, 039, 040, 042, 044–050 |
| GENUINE | 2 | 005, 014 |
| CAN'T VERIFY | 1 | 029 |
| **Total** | **50** | |

**15** photos were called tampered (the 12 above plus the 3 cut-off cases), **32** don't know, **2** genuine, **1** could not be verified.
These outputs are from exp014 experiment the final best models is saved at models/

## Important limits
- The model was trained on synthetic data. On real photos its score is evidence, not proof. The confidence shown is measured on synthetic test data, and p itself is not calibrated: p between 0.8 and 0.95 was right 85 % of the time on the test sets.
- A stretch of the *whole* logo is not detected: the alignment absorbs it.
- The per-photo step pictures of EXP-001 (200 MB) and all other experiments are only in the main repository.

## Licences
OpenCV (Apache 2.0), ONNX (Apache 2.0), ONNX Runtime (MIT), Streamlit (Apache 2.0), PyTorch/torchvision (BSD). The reference images and sample photos come from the assignment package.
