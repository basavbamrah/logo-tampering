# EXP-001 registration accuracy against ground truth

Made by `src/exp001_accuracy.py` (CPU only, about 20 min with 8 workers). Nothing under `data/` was changed.

**Question:** how do we know RANSAC found the correct match? On the 50 real photos we can only check by eye. On the synthetic NORVIA photos (EXP-003) we know the true camera, so we know exactly where the logo is.

**Method**
- Samples: the first 300 D2 rows of `data/synth/labels.csv` (150 genuine, 20-22 of each of the 7 tamper types). We also added the 19 other photos that failed registration in `data/synth/registration_D2.csv`, so all 23 failures of the 1500 are included.
- The true reference-to-photo homography is rebuilt from the stored seed (`exp003_steps.rebuild`). **Every rebuilt photo matched the stored JPEG pixel for pixel (319/319)**, so the ground truth belongs to the photo that was tested. The 3 reference images agree within 0.03 px, so it does not matter which one supplied the ink.
- Registration: the real EXP-001 functions `fit_homography` and `refine_with_ecc`, with the same settings, run on the stored photo. A copy of the matching step without the final checks gives the homography that was rejected for failed photos. For the 296 photos that passed, it gave the same H as `fit_homography`.
- Error measures:
  - **corner error**: the largest distance, in photo px, between the 4 corners of the 920x260 logo box under the true H and under the found H. The corners lie on the white margin, outside the ink, so this is the strictest measure.
  - **ink error**: the mean misalignment of the logo's ink pixels, in reference px. This is the error that the pixel-comparison methods (EXP-008, EXP-010, EXP-007) actually see.
- "Wrong" means a corner error above 10 px.

## Results (main 300)

**Success rate:** 296/300 passed the checks (98.7 %). These failed: cut 3/21 and font swap 1/21. All genuine, stretch, recolour, respace, glyph-swap and overlap samples passed.

**Error of passed registrations** (n=287; the 9 whole-logo stretches are listed separately):

| | median | 95th pct | max |
|---|---|---|---|
| max corner error, RANSAC only (photo px) | 3.51 | 19.79 | 110.3 |
| max corner error, after ECC (photo px) | **0.30** | 26.50 | 108.4 |
| ink error, RANSAC only (reference px) | 1.32 | 4.99 | 60.1 |
| ink error, after ECC (reference px) | **0.10** | 8.85 | 60.3 |

The tail comes from two tamper types only. Per group, after ECC:

| group | n | RANSAC median (px) | ECC median (px) | ECC 95th (px) | ECC max (px) | ink max (ref px) |
|---|---|---|---|---|---|---|
| genuine + recolour, respace, glyph swap, overlap | 236 | 3.18 | **0.26** | **1.45** | **2.70** | **1.40** |
| font swap | 20 | 5.31 | 22.6 | 98.6 | 108.4 | 60.3 |
| cut (part of the logo outside the photo) | 18 | 5.57 | 23.2 | 87.5 | 107.8 | 17.8 |
| stretch of one part (icon / word / tagline) | 13 | 11.67 | 1.56 | 14.0 | 14.7 | 2.45 |

### Does "passed" mean "correct"?
- **Genuine logos and local tampers: yes.** After ECC, 0 of 236 are wrong. The worst corner is 2.7 px and the worst ink error is 1.4 reference px.
- **RANSAC alone is not enough.** 44/287 passed registrations had a corner error above 10 px before ECC, and 15 of those were genuine. ECC fixed all of the genuine ones. On the real photos, ECC moved the corners by 2.4 px (median) and up to 24.7 px (photo 004), which fits this picture.
- **Font swap and cut: no.** After ECC, 34/287 passed registrations are wrong: 18 font swaps, 14 cuts and 2 tagline stretches. On these, ECC makes the fit worse than RANSAC alone. With a font swap, ECC bends the frame to fit the new wordmark. With a cut, the black area outside the photo pulls it off. The crop still looks lined up to the eye (`05_worst_passed.jpg`).
- **The ECC score flags these cases.** 32 of the 34 wrong ones have ECC < 0.9 (or ECC failed). The only two wrong ones with ECC ≥ 0.9 are tagline stretches (D2_00183, D2_00225: corner 13-15 px, ink 2.4 px). Among correct registrations, 14/253 have ECC < 0.9. A gate at "ECC ≥ 0.9" would therefore catch almost every wrong crop, at the cost of rejecting about 5.5 % of good ones. (This is a candidate rule and has not been adopted.)
- **Whole-logo stretches are absorbed, as expected** (9 samples). Against the camera truth the error is 6-49 px. Against the stretched truth it is 0.08-0.88 px, so registration fitted the stretched logo exactly, and the stretch disappears from the crop. This is the known blind spot, now measured.

### Inliers and the 15-inlier minimum
- Among passed registrations, the number of inliers does **not** predict the error. For genuine and local tampers, the worst corner error per inlier band is 2.1 px (15-30), 2.7 (30-50), 2.3 (50-80) and 2.1 (80+). The big errors (font swap, cut) occur at any inlier count, even 120.
- All 23 failures had 6-13 inliers, and none of the passed ones had fewer than 16. 19/23 of the rejected homographies were indeed wrong (> 10 px, some by thousands of px). 4/23 would have been acceptable (4-8 px).
- So the minimum of 15 sits in an empty gap between the failures (≤ 13) and the passes (≥ 16). It rejects mostly garbage, and no passed photo near the threshold was wrong because of having few inliers. This is evidence that 15 is a safe choice. It is not evidence that 15 is the best choice, since 12-14 might also work.

### Pose
For genuine and local tampers, the error does not grow with yaw, pitch or total tilt up to 35° / 28° / 41° (rank correlation +0.04 / +0.07 / +0.06, `03_error_vs_pose.png`).

### RANSAC settings actually used
`cv2.findHomography(src, dst, cv2.RANSAC, 4.0)`. maxIters and confidence are not passed, so OpenCV's defaults apply: **maxIters = 2000, confidence = 0.995**. These come from the C++ signature in `opencv2/calib3d.hpp`, `findHomography(src, dst, method=0, ransacReprojThreshold=3, mask=noArray(), maxIters=2000, confidence=0.995)`. EXP-001 overrides the threshold to 4 px. OpenCV stops early once the confidence is reached, so 2000 is only a ceiling. The run used OpenCV 5.0.0. The Python docstring does not print the default values; they are taken from the C++ header documentation.

## Files
| File | What it shows |
|---|---|
| `accuracy.csv` | One row per photo: tamper, pose, inliers, passed or failed and why, ECC score, corner and ink errors before and after ECC, error of the rejected H for failures, error against the stretched truth for whole-logo stretches |
| `homographies.npy` | True and found (after ECC) H per photo, as a pickled dict, `np.load(..., allow_pickle=True).item()` |
| `summary.txt` | All the numbers above, plus the IDs of every passed-but-wrong case |
| `01_error_before_after_ecc.png` | Error distribution (CDF), RANSAC only vs after ECC, for corner error and ink error |
| `02_error_by_tamper.png` | Box plots per tamper type, before and after ECC |
| `03_error_vs_pose.png` | Error vs yaw, pitch and tilt, coloured by group |
| `04_error_vs_inliers.png` | Error vs inliers, including the 23 failed photos (error of the rejected H), with the 15-inlier line |
| `05_worst_passed.jpg` | The 6 worst passed registrations: true box (yellow) vs found box (red), and the crop |
| `06_ecc_score_vs_error.png` | The ECC score separates the wrong cases (font swap, cut) from the correct ones |

## Caveats
- The photos are synthetic. Real photos may break registration in ways the generator does not produce, such as real bands or glare.
- The generator puts mirrored or 180°-rotated copies of the logo into 19 % of the D2 photos (see `outputs/exp003/steps/README.md`). No passed registration locked onto a copy: the largest error is 110 px, while a copy sits hundreds of px away. Copies are as common among the passed-but-wrong cases (7/34) as among the correct ones (48/253). They are over-represented among the failures (16/23 vs 19 % overall), so they probably cause some of the failures. This means the 98.7 % success rate may be slightly pessimistic.
- The truth uses pixel-centre resizing, (x + 0.5)·s − 0.5. `synth.rectify_with_known_pose` uses plain scaling, which is about 0.1 px different and does not matter here.
