# EXP-001 — Registration feasibility
- **Scripts:** `src/exp001_register.py` (the actual registration; its outputs are used by all later experiments), `src/exp001_steps.py` (re-runs the same steps and saves every intermediate result; changes nothing).


## Visual results (click to open)
- [Step-by-step for one photo (all stages on one page)](../outputs/exp001/steps/combined/041.jpg)
- [What a keypoint is: matched pairs side by side](../outputs/exp001/steps/keypoints_explained_041.jpg)
- [Template keypoints](../outputs/exp001/steps/template_keypoints.jpg)
- [SIFT keypoints on a photo (before matching)](../outputs/exp001/steps/02_keypoints/041.jpg)
- [Pairs after the ratio test](../outputs/exp001/steps/03_matches_ratio_test/041.jpg)
- [What one-partner-per-point removed (photo 011: 135 pairs on one point)](../outputs/exp001/steps/04b_unique_removed/011.jpg)
- [After RANSAC: green kept, red rejected](../outputs/exp001/steps/05_ransac/041.jpg)
- [The failure (029): RANSAC locked onto background text](../outputs/exp001/steps/05_ransac/029.jpg)
- [Found logo + angle, all 50 photos](../outputs/exp001/steps/found_logo_sheet_1.jpg)
- [RANSAC only vs after ECC (4x zoom)](../outputs/exp001/steps/10_ransac_vs_ecc_zoom/041.jpg)
- [Red overlay, all photos (sheet 1)](../outputs/exp001/overlay_sheet_1.jpg)
- [Red overlay, all photos (sheet 2)](../outputs/exp001/overlay_sheet_2.jpg)
- [Whole-photo resize: photo vs flattened (004)](../outputs/exp001/steps/09_whole_photo_flat/004.jpg)
- [Accuracy vs ground truth: error before/after ECC](../outputs/exp001/accuracy/01_error_before_after_ecc.png)
- [Accuracy by tamper type](../outputs/exp001/accuracy/02_error_by_tamper.png)
- [Error vs inlier count (why ≥ 15)](../outputs/exp001/accuracy/04_error_vs_inliers.png)
- [All stage folders (one image per photo each)](../outputs/exp001/steps/)

## For the write-up and slides (closing summary, 2026-10-11)
**Role:** step 1 of the pipeline, alignment. **It is not a detector:** it gives no tampered/genuine decision and no confidence. Every comparison method (EXP-005 colour, EXP-008 flow, EXP-010 difference map, EXP-007 model) works on its output.

**Flow** (code: `src/exp001_register.py`):
```
reference image → logo crop (template, 920×260) → grey + CLAHE → SIFT keypoints + descriptors ─┐
photo           →                                  grey + CLAHE → SIFT keypoints + descriptors ─┤
                                                                                                ▼
 match descriptors (ratio test 0.75) → one partner per photo keypoint
 → RANSAC: fit the homography H that most pairs agree with; wrong pairs are thrown out   (line 75)
 → check: ≥ 15 agreeing pairs, outline not folded, plausible size   (else "can't register" → can't verify)
 → ECC: pixel-level fine-tuning of H, ignoring the area outside the photo   (line 103)
 → warp the photo with H⁻¹ → rectified crop, same 920×260 frame as the reference   (line 230)
 → (a) red overlay for checking by eye    (b) input to EXP-005 / 008 / 010 / model
```
- **Angles** are read off H afterwards for the record (−7.5° to +9.3°). They are not used for any decision.
- **The red lines:** the reference logo's ink outline (green-ink mask → Canny edges) drawn in red on top of the rectified crop (`edge_overlay`, line 178). Both are in the same frame, so a good alignment puts the red outline exactly on the letter edges.

**Evidence that this step matters:**
- 49/50 photos aligned.
- Ground-truth test: genuine and local tampers within 2.7 px.
- **EXP-012:** without this step, the same model drops from test AUC 0.918 to 0.754.

**Outputs that do feed the verdict:**
- "can't register" → "can't verify" (029);
- the cut-off measure (EXP-010 `cut_pct`) works on its crops.

## In one sentence
For each photo, find where the NORVIA logo is and how it is turned, then "un-turn" it so it lies exactly on top of the reference logo. **This step does not decide whether a logo is genuine.** It only makes the photo and the reference comparable, pixel for pixel, for the methods that come after (EXP-005, 007, 008, 010).

## Which method is used?
**Feature-based image registration with a homography**, followed by a pixel-based fine-tuning:

| Part | Method (name to look up) | OpenCV function | Code |
|---|---|---|---|
| Find distinctive points | **SIFT** keypoints + descriptors (Lowe 2004) | `cv2.SIFT_create`, `detectAndCompute` | `exp001_register.py:50-51` |
| Pair them up | **Brute-force nearest-neighbour matching + Lowe's ratio test**, then each photo point used once | `cv2.BFMatcher(...).knnMatch` | `exp001_register.py:57-68` |
| Find the transform | **Homography estimated with RANSAC** | `cv2.findHomography(..., cv2.RANSAC, 4.0)` | `exp001_register.py:73-75` |
| Reject impossible results | our sanity check (no folding, plausible size, ≥ 15 inliers) | `looks_sane` | `exp001_register.py:84-100` |
| Fine-tune | **ECC image alignment** (Evangelidis & Psarakis 2008), homography mode | `cv2.findTransformECC(..., MOTION_HOMOGRAPHY)` | `exp001_register.py:103-121` |
| Apply the transform | **perspective warp** | `cv2.warpPerspective` | `exp001_register.py:224` |

**The transform is a homography:** one 3x3 matrix H that says, for every point of the reference logo, where it is in the photo. It covers moving, resizing, rotating and perspective, where the near side of the label looks bigger than the far side. That is everything that can happen to a flat label photographed from any angle. The rectified crop is made by applying H backwards (`H⁻¹`), which turns the photo so the logo sits on the reference.

### And the angles?
H contains the angles, but not as readable numbers. `exp001_steps.py` reads them off by pushing the reference's corners and edge midpoints through H and measuring the resulting shape in the photo. The numbers are in `outputs/exp001/steps/steps.csv`; across the 49 found photos:

| Number | What it means | Median | Range (photo) |
|---|---|---|---|
| `angle_deg` | rotation in the photo (+ = clockwise, right end lower) | +0.2° | -7.5° (011) to +9.3° (009) |
| `width_x` / `height_x` | logo size compared with the reference crop | x0.76 / x0.86 | x0.49 (018) to x1.21 (013) wide |
| `right_vs_left` | right edge length / left edge length; 1 = facing the camera, < 1 = right side further away | 1.00 | 0.91 (004) to 1.06 (011) |
| `top_vs_bottom` | the same for top / bottom (label tipped forward or back) | 1.00 | 0.93 (030) to 1.16 (004) |
| `centre_x`, `centre_y` | where the logo centre is in the photo (px) | | |

- **Why no "tilted 25° towards the camera" number?** Turning edge ratios into 3D angles needs the camera's focal length, which we don't know. We tried estimating it by splitting H into a camera model (the "implied stretch" below). It was too noisy, so we report the edge ratios, which need no assumptions.
- **`width_x` vs `height_x`.** The median logo is x0.76 wide but x0.86 tall. This is the whole-photo resize again: most photos are 960x1280, squeezed sideways.

## The whole process, end to end, in plain words
Think of it as finding a sticker on a messy desk using a clean picture of that sticker.

**0. The clean picture (template).** The three reference images are clean 1200x700 renders. A green-colour threshold (`green_ink_mask`) finds the ink at x = 168-1030, y = 240-458 in all three; adding a margin gives a 920x260 crop (`REF_LOGO_BOX` in `src/common.py:24`). **This is the only box anywhere in the process, and it is in the reference, not in the photo.** There is no logo detector and no bounding box in the photo.

**1. Prepare the photo.** The photo is turned grey, and its contrast is evened out locally (CLAHE), so light and shadow matter less (`to_gray`, `exp001_register.py:42-45`).
*Picture:* `steps/01_grey_clahe/`.

**2. Find keypoints: spots that are easy to recognise again.** A keypoint is a small spot with a clear pattern, such as a sharp corner, the tip of a letter or the end of a stroke. A blank area or the middle of a straight line is not a keypoint, because it looks the same wherever you are along it. SIFT looks over the whole image, at several zoom levels, and marks such spots (`detectAndCompute`, `exp001_register.py:50-51`). For each keypoint it stores:
- *where* it is (x, y);
- *how big* the patch around it is, so it doesn't matter if the logo is photographed small or large;
- *which way* its pattern points, so it doesn't matter if the photo is rotated;
- a **descriptor**: 128 numbers summarising the pattern of light and dark directions in that patch, like a fingerprint of the spot.

The template gets 521 keypoints. The photos get 190-2540 (median 539) spread over the logo, the background text, bands and noise. SIFT doesn't know which ones belong to the logo.
*Pictures:* `steps/template_keypoints.jpg` and `steps/02_keypoints/<id>.jpg`. Circles show the size of each patch; the line inside shows its direction.

**3. Pair template keypoints with photo keypoints** (`exp001_register.py:57-68`, the "step 2" you asked about). For each of the 521 template keypoints:
- **a.** compare its fingerprint with the fingerprint of *every* photo keypoint (`knnMatch`, line 57) and take the two most similar;
- **b. ratio test** (line 58): keep the pair only if the most similar is clearly more similar than the runner-up (distance ratio < 0.75). If two photo spots look almost equally good, the match is ambiguous and dropped. Example: one corner of an "N" looks much like another corner of the same "N".
- **c. one partner per photo point** (lines 64-68): if several template keypoints picked the *same* photo keypoint, keep only the best one. Without this, 6 photos fitted garbage in the first run.

For photo 041: 521 template points → 124 after the ratio test → 66 after one-partner-per-point.
*Pictures:* `steps/03_matches_ratio_test/` (yellow) and `steps/04_matches_unique/` (blue).
*Best picture for understanding:* `steps/keypoints_explained_041.jpg` shows matched pairs as little squares side by side. The template patch is on the left and its partner in the photo is on the right: the same piece of the same letter, even though the photo is blurred, rotated and coloured differently.

**4. Throw out wrong pairs with RANSAC** (`exp001_register.py:73-75`). Some of the 66 pairs are wrong, such as a letter of "BOTANICALS" paired with a letter of "BATCH 0921". RANSAC works like this:
- take 4 random pairs and compute the one homography that fits them;
- count how many of the *other* pairs also land within 4 px of where that homography says they should;
- repeat many times and keep the homography with the biggest crowd.

Correct pairs all agree with one geometry; wrong pairs point in random directions and never form a big crowd. For 041, 37 of 66 agree (the **inliers**, green) and 29 are rejected (red), including those on "BATCH 0921" and "NET WT 250g".
*Picture:* `steps/05_ransac/`.

**5. Check the result is possible** (`looks_sane`, `exp001_register.py:84-100`). We need at least 15 inliers, and the template outline pushed through H must not fold over itself and must have a plausible size. Otherwise the photo is "can't register" and gets no crop at all, rather than a wrong one.

**6. Where is the logo?** Push the template's 4 corners through H. The result is a turned, slightly trapezoid outline in the photo, not an upright box. This is the "found logo"; the angle numbers above come from it.
*Pictures:* `steps/06_found_logo/` and `steps/found_logo_sheet_1.jpg`, `_2.jpg`, all 50 photos on two sheets.

**7. Rectify, then fine-tune with ECC** (`exp001_register.py:103-121`). Warp the photo with H⁻¹ so the logo lands on the template. Keypoint positions are a little imprecise, so ECC then slides the whole image by tiny amounts until it best overlaps the template, pixel by pixel. ECC moved the logo corners by a median of 2.4 px.
*Pictures:* `steps/07_rectified_ransac/` (before ECC) and `steps/08_rectified_ecc/` (after ECC). The red outline is the reference; on a good result it sits exactly on the letter edges. Step 8 is what all later experiments use.

**8. Bonus check: flatten the whole photo.** `steps/09_whole_photo_flat/` applies the same H⁻¹ to the *entire* photo. In 004, the logo and the "STORE IN A COOL DRY PLACE" text look squeezed in the photo, and once flattened both look normal. This is the evidence that the whole photo was resized, not the logo alone (see the last section).

```
photo ─► grey + CLAHE ─► SIFT keypoints ─► pair with template (ratio test, one partner each) ─► RANSAC → H
                                                                                               │
     rectified 920x260 crop ◄── warp with H⁻¹ ◄── ECC fine-tuning ◄── sanity check ◄───────────┘
```

So "where is the logo?" is answered by "where do most template keypoints find partners that agree with each other?"

**Limits of this way of finding the logo:**
- It finds one logo per photo, the best-supported one.
- It needs enough of the logo to be sharp and intact. Heavy blur or a mostly cut-off logo leaves too few pairs.
- A tampered logo still gets found, as long as the icon and the untouched letters (the majority) pair up. That is wanted: the tampered part then shows up as a *difference* after rectification.
- **Blind spot:** if the *whole* logo was stretched evenly, H simply includes the stretch, and the rectified crop looks normal.

## Intermediate results (all in `outputs/exp001/steps/`, made by `python src/exp001_steps.py`)
| Folder / file | Step | What to look at |
|---|---|---|
| `01_grey_clahe/<id>.jpg` | 1 | the image SIFT actually sees |
| `template_keypoints.jpg` | 2 | the 521 template keypoints (circle = patch, line = direction) |
| `02_keypoints/<id>.jpg` | 2 | all photo keypoints **before matching** |
| `03_matches_ratio_test/<id>.jpg` | 3b | pairs left after the ratio test |
| `04_matches_unique/<id>.jpg` | 3c | pairs left after one-partner-per-point |
| `keypoints_explained_041.jpg` | 3 | matched pairs as side-by-side squares: kept ones and rejected ones |
| `05_ransac/<id>.jpg` | 4 | **after RANSAC**: green kept, red rejected, yellow found logo |
| `06_found_logo/<id>.jpg`, `found_logo_sheet_*.jpg` | 6 | found outline + rotation arrow + angle |
| `07_rectified_ransac/<id>.png` | 7 | rectified before ECC, red reference outline |
| `08_rectified_ecc/<id>.png` | 7 | rectified after ECC: the same crop as `outputs/exp001/rectified/<id>.png`, with the red outline added |
| `09_whole_photo_flat/<id>.jpg` | 8 | photo as given vs whole photo flattened |
| `combined/<id>.jpg` | all | steps 1-7 for one photo on one page |
| `steps.csv` | all | counts after each step, angle, size, edge ratios, ECC shift, check against the saved H |

**Check that nothing changed:** `exp001_steps.py` reproduces the saved homography (`outputs/exp001/rectified/<id>_H.npy`) for 49/49 photos (column `same_as_saved_H`). Checksums of `registration.csv`, `rectified/` and `whole_photo_resize/` were identical before and after running it.

**Suggested photos for the panel:**
- **041:** a normal case; background text pairs rejected by RANSAC.
- **029:** the failure.
- **004:** the whole-photo resize (`09_whole_photo_flat/004.jpg`).
- **013 / 016 / 037:** cut-off logos (black areas in step 7/8, low ECC score).

## Further reading (plain, with pictures)
- OpenCV tutorial "Introduction to SIFT": https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html
- OpenCV tutorial "Feature Matching" (brute-force matcher + ratio test): https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
- OpenCV tutorial "Feature Matching + Homography to find Objects": https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html. **This is almost exactly our steps 2-6**: find a known object in a cluttered photo and draw its outline.
- PyImageSearch, "Image alignment and registration with OpenCV" (2020): https://pyimagesearch.com/2020/08/31/image-alignment-and-registration-with-opencv/. It aligns a photographed form to a template, the same idea with ORB instead of SIFT.



## Key terms (short reference)

| Term | Plain meaning | Example in this project |
|---|---|---|
| **Reference / template** | The clean, known-genuine logo we compare against. | The 920 x 260 crop of `reference/ref_1.jpg`. |
| **Logo frame** | Coordinates inside that crop: (0,0) is its top-left corner. | "The R starts at x = 471" = 471 px from the left of the crop. |
| **Registration** (= alignment) | Finding the transform that says where and how the logo appears in the photo. | Photo 041: logo x0.88 the reference size, turned +5.3°, centre at (416, 294). |
| **Rectification** | Applying that transform backwards, so the logo sits exactly on the reference. | `outputs/exp001/rectified/041.png`. |
| **Homography (H)** | A 3x3 matrix that maps a flat surface from one viewpoint to another: move, resize, rotate, perspective. | Maps each template pixel to its place in the photo. |
| **Keypoint** | A small spot with a clear pattern that can be found again. | Tips and corners of letters, the icon's stem. |
| **Descriptor** | 128 numbers summarising the pattern around a keypoint: its "fingerprint". | Compared between template and photo to find partners. |
| **SIFT** | The method that finds keypoints and descriptors that survive zoom, rotation and lighting changes. | 5000 keypoints max per image. |
| **Ratio test** | Keep a pair only if the best partner is clearly better than the second best (< 0.75). | Removes ambiguous pairs. |
| **One partner per point** (our fix) | Each photo keypoint may be used by only one template keypoint. | Fixed 6 garbage fits in the first run. |
| **RANSAC** | Try many small random sets of pairs, keep the geometry most pairs agree with. | 041: 37 of 66 pairs agree. |
| **Inliers / outliers** | Pairs that agree with the final H (within 4 px) / those that don't. | Median 61 inliers; < 15 = failed. |
| **Reprojection error** | How far the inliers land from where H predicts. | Median 1.1 px. |
| **ECC** | Pixel-level fine-tuning; its score (0-1) says how well the crop overlaps the reference. | Median 0.977; cut-off logos 0.53-0.69. |
| **CLAHE** | Local contrast evening-out before SIFT and ECC. | Helps with light gradients and bands. |
| **Warping** | Resampling an image through H. | Areas outside the photo become black. |
| **Overlay** | The rectified crop with the reference outline in red. | Red on letter edges = well aligned. |
| **Implied stretch** (dropped) | Splitting H into a camera model to estimate the logo's own width/height ratio. | Noise ±3-9 % vs tampers of 1-8 % (EXP-003) → not usable. |
| **Whole-photo resize** | Photos rendered square, then resized to 3:4 / 4:3, so logo and background are squeezed together. | Removed automatically by registration. |

## Question
Can each of the 50 images be mapped onto the reference logo reliably enough for the methods that compare against the reference?

## Hypothesis
Most images register with many inliers and low reprojection error. Cut-off (013, 016, 037), heavily blurred or banded images will fail or degrade. If many clean-looking images fail, the rectify-first plan must change.

## Results (49 registered photos)
| Measure | Median | Min | Max |
|---|---|---|---|
| Photo keypoints | 539 | 190 (016) | 2540 (031) |
| Pairs after ratio test | 124 | 68 | 274 |
| Pairs after one-partner-per-point | 86 | 55 | 150 |
| RANSAC inliers | 61 | 34 (034) | 124 (008) |
| Mean reprojection error (px) | 1.1 | 0.5 | 1.5 |
| ECC score after fine-tuning | 0.980 | 0.921 | 0.992 |
| Overlap with reference, RANSAC only → after ECC | 0.939 → 0.954 | | ECC better on 49/49 |
| ECC corner shift (px) | 2.2 | 0.4 | 24.7 (004) |
| Rotation (deg) | +0.2 | -7.5 | +9.3 |

uses the whole crop including its margin. EXP-010's `cut_pct` is the better measure.
- **No photo is rotated more than 10°;** perspective is mild (edge ratios 0.91-1.16).
- **Open observation:** ECC moved 004's outline corners by 24.7 px, far more than the median 2.4 px, while its final score is high (0.986). The corners include the white margin, so a small perspective correction can move them a lot. Not investigated further.
- **The photos look rendered square, then resized.** The raw "implied stretch" was 0.72-0.77 for portrait (960x1280), 1.27-1.38 for landscape (1280x960) and ~1.0 for square, i.e. width/height of the photo. After undoing that resize: mean 0.997, std 0.031. EXP-003 later showed this estimate is too noisy to detect real stretches.

## Evidence level
Measured on all 49 registrable real photos. Alignment was judged by eye on the overlays and step pictures, plus the ECC and inlier numbers. Every intermediate step is saved and reproducible. Registration has no ground truth for tampering: it only says "lined up / not lined up".

## Decision / next step
- Rectification works for 49/50, so methods that compare against the reference are possible (D1 stands).
- Inlier count (≥ 15) and the sanity check are the quality gate; 029 → "can't register". A cut-off logo is detected with EXP-010's `cut_pct`, not with the ECC score.
- Open for the candidate: try to recover 029; is the whole-photo resize tampering (Q13 / F14)?
- Outputs: `outputs/exp001/` (registration.csv, rectified crops + H, overlays, sheets) and `outputs/exp001/steps/` (intermediate results).

## Evidence: the "stretched look" belongs to the whole photo
`outputs/exp001/steps/09_whole_photo_flat/` (all 49 photos, reproducible) and the earlier hand-made `outputs/exp001/whole_photo_resize/flat_002/004/005.jpg` (kept unchanged). In each image, the left is the photo as given and the right is the whole photo flattened with the EXP-001 homography.
- In 004 (portrait 960x1280), the logo AND "STORE IN A COOL DRY PLACE" look narrow; flattened, both look normal and the photo becomes nearly square.
- In 002 (landscape), both the logo and "100% NATURAL" look wide.

So the distortion is a whole-photo resize (rendered square, then resized to 3:4 / 4:3), and registration removes it. Open question for the candidate (F14 / Q13): count it as tampering or not.

**Effect on the real photos:**

| photo | outside the photo | ECC score before → after | logo outline moved |
|---|---|---|---|
| 013 | 8.6 % | 0.61 → 0.95 | 3.9 px |
| 016 | 14.7 % | 0.53 → 0.99 | 6.8 px |
| 021 | 2.2 % | 0.90 → 0.98 | 9.8 px |
| 037 | 7.0 % | 0.69 → 0.99 | 3.5 px |

The other 45 photos moved < 0.5 px.

**Effect on the synthetic ground-truth test (below):** for cut-off samples, the error after ECC went from median 23 px (worst 108 px) to median 0.26 px (worst 3.7 px).


## Registration accuracy against ground truth 
This measures how accurate the same registration code is on synthetic NORVIA photos with a known true homography (rebuilt from the seed, photos verified identical, 319/319): 300 samples (150 genuine, ~21 per tamper type) plus all 23 failures of the 1500.
- **Success:** 296/300 (98.7 %). The failures are 3 cut and 1 font swap.
- **Error of passed registrations** (max corner error of the 920x260 box, photo px; n=287, whole-logo stretches excluded):
  - RANSAC only: median **3.5**, 95th 19.8
  - after ECC: median **0.28**, 95th 14.2 (before the fix: 0.30 / 26.5)
  - ink misalignment after ECC: median 0.10 reference px

- **Inliers:**
  - Among passed registrations, the inlier count does not predict the error. The big errors occur at any count, up to 120 inliers.
  - All failures had 6-13 inliers, and every pass had at least 16. 19/23 of the rejected homographies were indeed wrong (> 10 px).
  - The 15-inlier minimum is therefore safe here, though not shown to be optimal.
- **Pose:** the error does not grow with yaw, pitch or tilt (rank correlation ≤ 0.07).
- **RANSAC settings:** threshold 4 px (set by EXP-001); maxIters 2000 and confidence 0.995 are OpenCV defaults, not passed explicitly.
