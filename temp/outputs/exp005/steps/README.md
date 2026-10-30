# EXP-005 intermediate results (colour check, hue only)

Made by `python src/exp005_steps.py` (about 1 min, CPU only). It imports the functions of
`src/exp005_colour.py`, so the settings are the same, and only writes into this folder.

**Check against the saved results:** all 49 real photos give exactly the same numbers as
`outputs/exp005/real_colour.csv` (raw, paper, hue, the three flags and the three part hues;
column `same_as_saved` in `steps.csv`). The synthetic numbers are also identical to
`outputs/exp005/summary.txt` (745 genuine / 107 recolour, AUC 0.654 / 0.536 / 0.813,
thresholds 18.43 / 32.84 / 9.64).

## What the check does, in one line
Rectified crop (EXP-001) -> take the ink pixels of icon, wordmark and tagline (mask from the
reference, the crop is already lined up) -> hue of the median ink colour per part -> difference
to the reference hue -> image score = worst part -> flag if above 9.64 deg (the value that
flags 5 % of synthetic genuine crops).

## Per real photo (49 photos; 029 is missing because EXP-001 could not register it)

| Folder | What it shows |
|---|---|
| `01_rectified/<id>.png` | The rectified crop at 460x130, the size it is measured at (shown 2x). |
| `02_ink_mask/<id>.png` | Which pixels are measured: icon blue, wordmark orange, tagline purple, paper ring grey. The ink mask is eroded by 1 px so blurry edges are not used. |
| `03_hue_histograms/<id>.png` | Hue of every ink pixel per part (colour) vs the reference (grey). Solid line = hue of the photo's median colour (this is what is scored), dashed = reference. |
| `04_hue_difference/<id>.png` | Absolute hue difference per part (signed value on top) vs the threshold. |
| `05_tagline_mask_zoom/<id>.png` | Tagline at 5x with the measured pixels outlined in red. Shows whether the mask sits on the letters. |
| `combined/<id>.png` | All steps in one figure, plus the numbers and a one-line reading ("all parts shift the same way" / "only the tagline is off"). |

`steps.csv`: per photo the number of ink pixels used per part, hue per part, signed difference
per part, worst part, raw / paper / hue scores and flags, paper difference to the reference
(`paper_dE`), `mask_offset_<part>` (see below) and `same_as_saved`.

`real_hue_by_part.png`: signed hue difference of all three parts for all 49 photos on one chart.
Takeaway: for 028 (-12), 033 (-11) and 038 (+15) the three parts move together, i.e. the whole
logo has a different hue. 027 (+8.5, all parts) is the same pattern but just under the
threshold. 022 is only its tagline (-9.7, threshold 9.64). 016 is only its tagline (-28).

## Synthetic evaluation (how AUC 0.81 and "paper normalisation hurts" were measured)
Data: the D2 (NORVIA) crops of `data/synth`, registered the same way as the real photos:
745 genuine and 107 recolours with a known hue shift of 3-15 deg (whole logo or one part),
under random paper colour, light and a camera colour cast of +-6 % per channel.

| File | What it shows | Takeaway (measured) |
|---|---|---|
| `synth_01_hue_error_vs_recolour_size.png` | Measured hue difference vs injected shift, genuine at 0, threshold line | Genuine crops already reach up to ~10 deg (95th pct 9.64), so shifts below ~6 deg sit inside the genuine spread. Above ~10 deg most recolours are clearly above the threshold. |
| `synth_02_detection_rate_by_size.png` | Share of recolours caught at 5 % false alarms, per shift bin and method | hue: 12 % (3-6 deg, n=57), 39 % (6-10, n=23), 85 % (10-15, n=27). raw: 7 / 0 / 7 %. paper: 4 / 4 / 7 %. |
| `synth_03_score_distributions.png` | Genuine vs recolour score histograms per method, and ROC curves | AUC raw 0.654, paper-normalised 0.536 (close to a coin), hue 0.813. |
| `synth_04_why_paper_normalisation_hurts.png` | Genuine crops only: score vs how different the paper is from the reference paper (black crosses = real photos) | Paper-normalised error follows the paper difference (r = 0.78); hue does not (r = 0.13). The correction assumes the paper is the reference paper; when the paper is tinted, it "corrects" genuine ink into a wrong colour. |
| `synth_05_paper_examples.png` | The 6 genuine crops the paper method scores worst | All on tinted paper (purple, grey, pink), ink looks normal, hue score mostly low. |

`synthetic.csv`: per synthetic crop the label, target, injected shift, the scores, `hue_paper`
and `paper_dE`.

Extra measurement (not in the original script): **hue after paper normalisation** (`hue_paper`,
dotted line in the ROC). AUC 0.744, lower than plain hue (0.813). It catches a few more small
shifts (16 % / 48 % / 74 %) but fewer large ones, so normalisation still does not pay off
overall, also for hue.

Caveat for the review: the synthetic papers are more varied than the real ones
(paper difference to the reference: median 21.5 dE synthetic vs 15.2 dE real), so the size
of the paper-normalisation penalty on real photos may be smaller than on synthetic data. The
direction (it adds error from the paper) is the same.

## Things these pictures revealed
- **016's flag is a measurement artefact, not a colour.** Its tagline is measured at 125 deg
  (-28), but the zoom (`05_tagline_mask_zoom/016.png`) shows the mask about 2 px off the
  letters, so the "ink" median contains paper. `mask_offset` (the shift of the mask that lands
  it on the darkest pixels) is about 2 px for all three parts of 016; it is 0,0 for 40 (icon),
  45 (wordmark) and 43 (tagline) of the 49 photos. 016 is a photo where the logo is cut by the frame, which probably explains the weaker
  registration. The EXP-005 file said "partial parts make its colour less reliable"; the
  actual reason is the 2 px misregistration.
- Other photos with a non-zero mask offset (1-3 px): 009, 011, 013, 021, 025, 037, 041, 042.
  Most of them are photos the shape methods (EXP-008/010) flag, or cut-off photos. None is
  hue-flagged.
- The tagline is the most fragile part: its strokes are only ~3 px wide at 460x130 and ~1 px
  after the erosion, so any blur or 1 px offset mixes in paper. In 40 of 49 photos the tagline
  hue is lower (more yellow, like the paper) than the icon/wordmark average (median -1.0 deg).
  022's flag is a tagline-only shift of -9.7 deg, just over the threshold, with the mask on the
  letters: borderline, could be this effect.
- The flags for 028, 033 and 038 have the mask on the letters (offset 0,0 for all parts) and the
  same shift in all three parts.
