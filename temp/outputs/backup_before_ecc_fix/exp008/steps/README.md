# EXP-008 intermediate results (letter shifts by optical flow)

Made by `python src/exp008_steps.py` (CPU, about 2 minutes). The script reuses the functions of
`src/exp008_flow.py` (same masks, same DIS preset, same half-size 460x130 crops), so the numbers are the
same ones the experiment reported. Nothing in `outputs/exp008/` outside this folder was changed.

All shifts are in **half-size px** (the 460x130 frame). Double them for the 920x260 logo frame.

## Check against the saved results
- Per-photo numbers (max_dev, slope, 6 letter shifts) equal to `outputs/exp008/real_letter_shifts.csv`: **49/49**.
- `exp008_flow.main()` re-run into `rerun_original/` (both `--crops reg` and `--crops pose`):
  `real_letter_shifts.csv`, `summary_reg.txt`, `summary_pose.txt` are **byte-identical** to the saved files.

## What each file / folder shows
| Path | What it shows |
|---|---|
| `01_reference_letters.png` | The reference at half size with the 6 wordmark letter masks (coloured) and the 14 tagline masks (grey). The shift of a letter is the median flow inside its mask. Tagline shifts are computed but not scored. |
| `02_prepared/<id>.png` | What the flow algorithm receives: reference (top) and photo (bottom) in grey with the slow lighting removed (minus a 15 px blur). This is why paper colour and shading do not matter. |
| `03_flow_colour/<id>.png` | Horizontal flow for every reference pixel, -2..+2 half px. Red = that pixel is further right in the photo, blue = further left. Black outlines = letter masks. Only the colour **inside** the outlines is used: the plain paper has nothing to track, so the flow there wanders (see the red blob in 001) and is ignored. |
| `04_flow_arrows/<id>.png` | The same flow as arrows on the photo, only on letters, drawn 10x longer than the real shift. |
| `05_letter_shifts/<id>.png` | One bar per letter (median horizontal flow). Black line = median of the 6 letters. Grey band = median +/- 0.17, the 95th percentile of max_dev on genuine synthetic crops (reg). A bar outside the band is orange. Title: max_dev, the letter that gives it, and its shift in full-frame px. |
| `06_slope/<id>.png` | Shift vs letter position with the fitted line. A wordmark stretch would give a clear slope. The genuine 95th pct of \|slope\| (0.00036) is shown for context only: the original script reports slope but sets no threshold on it for real photos. |
| `combined/<id>.png` | Steps 1-6 for one photo in one figure. |
| `overview_max_dev.png` | max_dev of all 49 photos, sorted, with the threshold. |
| `synthetic_reg.png`, `synthetic_pose.png` | Known-truth check on synthetic NORVIA crops: (a) measured vs injected shift of the moved letter, (b) detection rate by shift size at 5 % false alarms, (c) max_dev of genuine vs re-spaced, (d) \|slope\| of genuine vs wordmark x-stretch. |
| `synthetic_reg.csv`, `synthetic_pose.csv` | One row per synthetic sample used (id, kind, max_dev, slope, true and measured shift of the moved letter). |
| `steps.csv` | Per photo: 6 letter shifts, median, max_dev, worst letter and its shift in full px, slope, flagged, same_as_saved. |
| `rerun_original/` | Output of the unchanged `exp008_flow.main()`, used only for the byte-for-byte check. |

## Takeaways (measured)
**Synthetic, reg crops (made by the real registration code):** genuine max_dev median 0.09, 95th pct 0.17;
re-spaced vs genuine AUC 0.993; detected 97 % of 1-2 px, 100 % of 2-4 px and 4-6 px shifts. Wordmark
x-stretch vs genuine on |slope| AUC 0.896 (only n=10).

**New from the scatter (figure a):** the measured shift of the moved letter is on average only
**0.63x the injected shift** on reg crops (0.91x on pose crops). The summary's "error mean +0.06" hides this
because positive and negative shifts cancel; the mean |error| is 0.52 half px. So detection works, but the
size of a shift on real photos is probably **under-estimated by about a third**. Likely cause: registration
fits partly to the moved letter and pulls it back (the pose crops, which skip registration, show almost no shrink).

**Real photos:** 39/49 have max_dev <= 0.155 (most 0.02-0.12, like genuine synthetic). 10 are above 0.17,
with a clear gap (lowest flagged 0.447 for 011, highest unflagged 0.155 for 042):

| Photo | max_dev | Worst letter, shift vs median (full px) | What the pictures show |
|---|---|---|---|
| 025 | 2.31 | V +4.6 | V moved right **and** A moved left (-2.4 px): two letters, not one. |
| 021 | 1.92 | N -3.8 | Irregular: N left, R right, A left. Difference map also shows the icon shifted. |
| 009 | 1.61 | R -3.2 | R moved left; N moved right (+2.7 px). Matches the red/blue edges in EXP-010. |
| 016 | 1.45 | (N,O,R about -2.1 together) | Frame cuts the A (11.7 % cut). Flow next to the black area is garbage, the left half is offset as a block: misregistration of a cut photo, not a letter move. Slope 0.0089 (25x the genuine 95th pct). |
| 037 | 1.02 | A +2.0 | Cut photo (1.9 %); slope 0.0053, a ramp rather than one letter. |
| 030 | 0.85 | N +1.7 | R/V/I left of N. |
| 041 | 0.61 | I -1.2 | Small; EXP-010 shows the tagline is bolder. |
| 013 | 0.53 | A -1.1 | Cut photo (5.3 %); all letters shifted ~+0.8 together. |
| 020 | 0.50 | N -1.0 | Small. |
| 011 | 0.45 | I -0.9 | Small; slope -0.0024 (a ramp). |

Caveats seen in the pictures: (1) the method does not exclude pixels outside the original photo (black
area of a cut crop), so on 013/016/037 the flow near the cut is meaningless; (2) the threshold comes from
synthetic data; (3) shape/position only, blind to recolour. Flags are candidates, not verdicts.

Note on the EXP-008 write-up: "009 (R about -3.9 px)" and "025 (V about +3.9 px)" are the absolute letter
shifts x2. Relative to the other letters (which is what max_dev measures) they are -3.2 px and +4.6 px.
