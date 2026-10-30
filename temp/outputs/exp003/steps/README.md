# EXP-003 intermediate results: how a synthetic sample is made

Made by `src/exp003_steps.py` (CPU only, about 12 min with 8 workers). The script reads `data/synth` and `outputs/exp001` and writes only to this folder.

## Can samples be reproduced from their seed? Yes, exactly.
The generator is deterministic. The script re-runs `exp003_make_data.make_sample` step by step from the seed in `labels.csv`.
- **The rebuilt photo is pixel-identical to the stored `<id>_photo.jpg` for all 8 examples** (max difference 0). The EXP-001 accuracy study confirmed the same for **319/319** photos.
- The EXP-001 registration crop we recompute is identical to the stored F12 crop `<id>_cand_reg_full.png` for all 8 examples.
- `synth.photograph()` does lighting, blur, noise and colour cast in one call. To show these stages separately, the script splits it with the same random draws in the same order. It asserts that the split version gives exactly the same image, so the stage pictures are the real intermediate states, not look-alikes.

## Stage figures (one per tamper type, plus one overview)
Each 3x3 figure shows: (1) the clean logo, (2) the tampered logo, with a difference image and the stored magnitude, (3) the flat artwork, (4) the camera only, with the true logo box, (5) + lighting, blur, noise and colour cast, (6) the final resized JPEG with the true logo box, (7a) the known-pose crop (old `_cand.png`) and (7b) the registration crop (F12, used for training). Both crops show the reference ink edges in red.

| File | Sample | Tamper (stored values) | Registration (F12) |
|---|---|---|---|
| `00_overview_genuine_D2_00000.jpg` | D2_00000 | genuine; stages zoomed on the logo, plus the JPEG difference | 108 inliers, corner error 0.25 px |
| `01_tamper_stretch_D2_00001.jpg` | D2_00001 | stretch icon, y −2.6 % | 43 inl, 0.74 px |
| `02_tamper_recolour_D2_00003.jpg` | D2_00003 | recolour whole, −3.7° hue (ΔE 3.2) | 79 inl, 0.14 px |
| `03_tamper_respace_D2_00005.jpg` | D2_00005 | respace tagline "B", +4.1 px | 37 inl, 0.28 px |
| `04_tamper_glyph_swap_D2_00007.jpg` | D2_00007 | glyph swap word "A" → R in Ubuntu-Bold (1−IoU 0.56) | 48 inl, 0.84 px |
| `05_tamper_font_swap_D2_00009.jpg` | D2_00009 | font swap DejaVu → Lato (width 14 %) | 53 inl, **19.9 px** (ECC 0.52) |
| `06_tamper_overlap_D2_00011.jpg` | D2_00011 | overlap "NET WT 250g", 25 px text | 128 inl, 0.34 px |
| `07_tamper_cut_D2_00013.jpg` | D2_00013 | cut, 19 % outside the frame (left) | 120 inl, **15.3 px** (ECC 0.47) |

`examples.csv` holds these numbers. The two large errors are not chance. Font swap and cut are the two tamper types where the registration passes but is wrong; see `outputs/exp001/accuracy/README.md`.

## Realism: real vs synthetic genuine
`realism_side_by_side.jpg` shows 8 real photos (every 6th) and the first 8 synthetic genuine photos. `realism_stats.png` / `realism_stats.csv` compare all 50 real photos with 150 synthetic genuine ones. The crop-based numbers use the registered 920x260 crop, so the same logo area is compared.

| measure | real median [10-90 %] | synthetic median [10-90 %] |
|---|---|---|
| whole-photo brightness (0-255) | 199.7 [170-223] | 190.4 [146-229] |
| paper brightness near the logo | 203 [171-227] | 193 [152-236] |
| sharpness (Laplacian variance of the crop) | 30.7 [15.6-66.6] | **19.4 [5.0-52.7]** |
| ink hue (deg) | 154 [150-158] | 152 [150-156] |
| ink saturation (0-255) | 191 [161-234] | **192 [185-196]** |
| RANSAC inliers | 61 [41-93] | 63 [32-110] |

Takeaways:
- Brightness, ink hue and inlier counts overlap well.
- **The synthetic photos are blurrier.** The median sharpness is 19 vs 31, and the lowest decile goes down to 5. Either the blur is too frequent or strong, or the logos are rendered smaller.
- **The synthetic ink saturation is much narrower** (185-196 vs 161-234). The real photos vary the ink colour much more than the generator's colour cast does.
- **The synthetic paper colours are more varied and more saturated** (pink, lilac) than the real ones (cream, pale green, grey). This shows clearly in the side-by-side figure and is also visible in the wider brightness spread.
- This is the first pixel-statistics comparison. The EXP-003 write-up had noted that it was still missing.

## Generator side effect found: mirrored logo copies (reported, not fixed)
`synth.photograph()` warps the artwork with `BORDER_REFLECT`. Where the camera sees past the edge of the 1800 px artwork, the scene, including the logo, is reflected back in, so the photo shows mirrored or 180°-rotated copies of the logo (visible in `03_tamper_respace_D2_00005.jpg` at the bottom and in `07_tamper_cut_D2_00013.jpg` at the top right). `mirrored_logo_check.csv/.txt` counts this from the known pose for all 1500 D2 samples:
- 1022/1500 renders see past the artwork edge.
- **286/1500 (19 %) show reflected logo ink**, and 213 of those show more than 2000 px of it.
- **16 of the 23 registration failures** (in `registration_D2.csv`) show a copy, against 19 % overall.
- For "cut" samples this matters most: the "missing" part of the logo can reappear as a copy elsewhere in the photo. A 180°-rotated copy can be matched by SIFT, because SIFT handles rotation.
- None of the 50 real photos shows such a copy (checked by eye on `outputs/exp001/steps/found_logo_sheet_1/2.jpg`). A fix would be `BORDER_REPLICATE` or a larger artwork, but that is the candidate's decision. Changing it would also change every sample generated from a seed.

## Stretch diagnostic as a plot
`stretch_diagnostic.png`, data in `stretch_pairs.csv`. The script re-runs `exp003_check_realism.register_photo` on the same samples as before: 150 genuine and all 36 whole-logo x-stretches.
- Genuine photos (true value 1.0): std 0.092, 5-95 % range 0.91-1.20.
- Whole-logo x-stretches: corr(true, estimate) = 0.54. The tamper range is 1-8 %, the noise is ±9 %.
- These numbers equal the "solved focal" line of `outputs/exp003/stretch_diagnostic.txt` (std 0.092, corr 0.536), so the earlier result is reproduced.
- **Why it was recomputed:** the saved `outputs/exp003/stretch_pairs.npy` is empty (shape 2x0). `outputs/exp003/realism.txt` reports only 4 genuine samples and 0 stretches. Both files were overwritten at 15:28 by a small test run, so they no longer back the 150-sample table in `experiments/EXP-003_synthetic_data_generator.md`. The `realism_stats.csv` here does back that table: synthetic median inliers 63 over 150, taken from `registration_D2.csv`. The old files were left untouched.

## Small notes
- `synth.rectify_with_known_pose` scales by w/1024 without the half-pixel offset of `cv2.resize`. This is about a 0.1 px bias in the known-pose crops and does not matter.
- The step figures cover the NORVIA set (D2) only. D1 (made-up words) goes through the same tamper and camera code.
