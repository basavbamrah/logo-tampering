# EXP-009 intermediate results (which font is the logo)

Made by `python src/exp009_steps.py` (a few seconds, CPU only). It imports `render_letter` and
`compare` from `src/exp009_font_match.py`, so the measurement is the same, and only writes into
this folder.

**Check against the saved results:** all 9 fonts give exactly the same wordmark IoU, tagline IoU
and width error as `outputs/exp009/font_ranking.csv` (9/9).

## How the test works
Each of the 20 letters (6 wordmark, 14 tagline) is cut out of ref_1 with the green-ink mask.
The same letter is drawn in each candidate font, scaled to the reference letter's height and
centred on it. IoU = overlapping pixels / pixels in either shape (1 = identical).
Spacing is ignored in this test (each letter on its own), see file 05 for spacing.

| File | What it shows | Takeaway (measured) |
|---|---|---|
| `01_reference_letters.png` | The 20 letters found on ref_1 (red boxes) and each letter's mask | The split works: 6 + 14 letters, wordmark 113 px high, tagline 30-32 px. |
| `02_overlays/<font>.png` | Every reference letter with that font's letter on top. yellow = both, red = reference only, green = font only | DejaVu Sans Bold is almost all yellow; the others leave red/green fringes (narrower letters, different R leg, A crossbar higher). |
| `03_iou_by_font.png` | Mean IoU per font, wordmark and tagline | DejaVu Sans Bold 0.980 / 0.937; next is Open Sans Bold 0.813 / 0.811. Gap 0.17. |
| `04_iou_per_letter.png` | IoU of every letter in every font | DejaVu is the best font on 19 of 20 letters (the exception is the small tagline O: 0.85 vs 0.91 for Open/Noto Sans). |
| `07_zoom_top3_fonts.png` | R and A of the wordmark, large, in the three best fonts | DejaVu R 0.977 (width +0.0 %), A 0.960. Open Sans / Noto Sans: 12 % narrower R with a different leg, A crossbar in a different place. |
| `05_whole_word_dejavu.png` | "NORVIA" and "PURE BOTANICALS" typed in DejaVu with default spacing, left-aligned on the reference | Tagline: IoU 0.828, width +0.5 %, letters stay on top of each other -> default spacing. Wordmark: IoU 0.467, typed word 4.6 % narrower, letters drift apart to the right -> **the wordmark uses wider letter spacing than the font's default** (the letter shapes are DejaVu, the spacing was set by the designer). |
| `06_noise_floor.png` | The same letters cut from ref_2 and ref_3 compared with ref_1 | Same artwork, different JPEG: 0.986 (wordmark), 0.987 (tagline). DejaVu reaches 0.980 on the wordmark, i.e. at the noise floor. On the tagline it reaches 0.937, lower than 0.987: at 30 px the font renders slightly differently (anti-aliasing / hinting at small size), still far above the second font (0.811). |

`steps.csv`: per font and letter, IoU and signed width error. `noise_floor.csv`: per letter,
ref_2 / ref_3 vs ref_1.

## Why DejaVu Sans Bold was chosen
1. Highest IoU by a wide margin (0.980 vs 0.813 for the next font).
2. Best on 19 of 20 letters, not just on average.
3. Width matches (mean width error 0.8 % vs 7.8-16.8 % for the others).
4. On the wordmark it scores as well as the same artwork cut from another reference image
   (0.980 vs 0.986), so the remaining difference is JPEG / edge noise, not font.

## Notes for the review
- The comparison uses 9 fonts (5 Google Fonts + 4 installed on this machine).
  `NotoSans-Bold-system` (0.793 / 0.806) is in `font_ranking.csv` but missing from the results
  table in the EXP-009 experiment file.
- The wider wordmark spacing does not affect the NORVIA synthetic data: `src/synth.py` builds
  the NORVIA logo from the reference artwork itself (`logo_from_reference`), not by typing it.
  It only matters if someone re-types the wordmark in DejaVu.
