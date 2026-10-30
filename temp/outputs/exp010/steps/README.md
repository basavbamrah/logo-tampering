# EXP-010 intermediate results (golden-template difference map)

Made by `python src/exp010_steps.py` (CPU, about 1.5 minutes). The script reuses `ink_mask()`,
`region_masks()` and `SLACK` from `src/exp010_template_diff.py`; `ink_mask()` is split into its stages
and every photo is checked to give exactly the same ink mask as the original function.
Nothing in `outputs/exp010/` outside this folder was changed.

## Check against the saved results
- Per-photo scores equal to `outputs/exp010/diff_scores.csv`: **49/49**.
- `exp010_template_diff.main()` re-run into `rerun_original/`: `diff_scores.csv`, `diff_sheet_1.jpg`,
  `diff_sheet_2.jpg` are **byte-identical** to the saved files.

## The steps, in order
| Path | What it shows |
|---|---|
| `01_reference_ink.png` | Reference ink (saturated dark green on the clean reference, 55 897 px) and the three scored part boxes (icon / word / tagline, 8 px margin). |
| `02_grey_filled/<id>.png` | Rectified photo in grey. Pixels outside the original photo (pure black after warping) are filled with the median grey, so the black area is not mistaken for ink. |
| `03_paper_estimate/<id>.png` | Morphological **closing with a 161 px disc** on the grey image. A closing removes every dark shape the disc does not fit inside, so only the paper and its slow shading remain. 161 px is larger than the solid icon halves. |
| `04_darker/<id>.png` | Paper estimate minus grey = "how much darker than the paper". The Otsu threshold (in the title, 42-87 on these photos) splits ink from paper automatically, so no fixed number is needed per photo. |
| `05_photo_ink/<id>.png` | Photo ink = darker than the Otsu threshold, then a 3x3 opening to remove single-pixel specks. |
| `06_raw_difference/<id>.png` | Photo ink vs reference ink with **no slack**: red = reference ink not found in the photo, blue = photo ink where the reference has none, grey = both. Even genuine photos have a thin outline here (001: 56 missing, 1 504 extra px) from blur and sub-pixel alignment error. |
| `07_after_slack/<id>.png` | The same with **2 px slack**: both masks are grown by a 5x5 disc before comparing. The thin outlines disappear; what is left is the final map (identical colours to `diff_sheet_*.jpg`; black = outside the photo). |
| `08_scores/<id>.png` | Missing and extra pixels per part, as % of that part's reference ink, plus cut_pct (share of the reference ink that falls outside the photo). |
| `combined/<id>.png` | All of the above for one photo. |
| `kernel_45_vs_161_001.png` | Why 161 px: with the first-run 45 px kernel the disc fits inside the solid icon halves, the closing keeps them as "paper", and the icon comes out 88.6 % missing on a clean photo. With 161 px: 0 %. |
| `kernel_45_vs_161.csv` | Missing % per part with both kernels for all 49 photos. 45 px: icon missing 86.4-88.6 % (median 88.4 %) on **every** photo, word and tagline unaffected; 161 px fixes the icon without changing the others. |
| `steps.csv` | Per photo: the saved scores, Otsu threshold, ink pixels, specks removed by the opening, raw and after-slack missing/extra pixel counts, same_as_saved. |
| `rerun_original/` | Output of the unchanged `exp010_template_diff.main()`, only for the byte-for-byte check. |

There is no synthetic evaluation in the original script (the EXP-010 file says so), so none is drawn here.

## Takeaways (measured)
- **39/49 photos: no difference at all after the 2 px slack** (every score 0.00). Their raw maps
  (stage 06) show only a 1-2 px outline, which is what the slack is for.
- **10 photos differ:** 016, 021, 013, 041, 009, 011, 030, 025, 020, 037 (worst part 51.0 % down to 1.7 %).
  The same 10 that EXP-008 flags.

| Photo | Worst part | What the stage pictures show |
|---|---|---|
| 016 | 51.0 % (tagline 21 % missing / 30 % extra) | Cut 11.7 % (A off the frame). Red on one side and blue on the other of **every** letter: the whole logo is offset by a few px, i.e. a registration error on a cut photo. The thin tagline letters score highest because a few px offset is a large share of a thin stroke. |
| 021 | 21.9 % (icon 6 % / 15 %) | Icon, N, the first tagline letters and A show blue-left / red-right edges (offset to the left) by different amounts, the middle letters are clean: uneven offsets, not one moved letter. EXP-008 agrees (N -1.83, A -1.32, R +1.17 half px). |
| 013 | 17.6 % (tagline 9 % / 9 %) | Cut 5.3 %. |
| 041 | 16.1 % (tagline 16 % extra, 0 % missing) | Blue all around every tagline letter and nothing red: the tagline is **thicker/bolder** than the reference; icon and wordmark only slightly (2.7 %, 1.8 %). Visible by eye in the input crop. |
| 009 | 5.2 % (word 2.5 % / 2.7 %) | R: red on the right, blue on the left = R moved left; N the opposite. Agrees with EXP-008 (R -3.2 px, N +2.7 px). |
| 011 | 5.1 % | Small offsets across all parts. |
| 030 | 5.0 % | Mostly extra ink on wordmark/tagline. |
| 025 | 3.3 % | Small edges at V and A (EXP-008: V +4.6 px, A -2.4 px). |
| 020 | 2.1 % | Small. |
| 037 | 1.7 % | Cut 1.9 %; tagline only. |

What this method cannot see: recolour (ink shape only), a uniform whole-logo stretch (absorbed by
registration), anything smaller than the 2 px slack. "No difference" does not mean "genuine".

Note on the EXP-010 write-up: it says "38/49 no difference" and "11 with differences" but lists 10 photos;
the data give **39 and 10**.
