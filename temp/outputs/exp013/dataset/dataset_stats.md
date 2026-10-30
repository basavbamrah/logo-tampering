# EXP-013 dataset (data/synth_v2)

Generated: 10000 samples. Usable (logo found in the photo, so a tampered crop exists): 9885.

| set | split | genuine | tampered | usable |
|---|---|---|---|---|
| D2 | train | 1769 | 1772 | 3519 |
| D2 | val | 357 | 382 | 733 |
| D2 | test | 374 | 346 | 710 |
| D1 | train | 1660 | 1639 | 3251 |
| D1 | val | 352 | 353 | 694 |
| D1 | test | 341 | 321 | 653 |
| D1 | test_unseen_font | 147 | 187 | 325 |

| set | none | stretch | recolour | respace | glyph_swap | font_swap | overlap | cut |
|---|---|---|---|---|---|---|---|---|
| D2 | 2500 | 358 | 357 | 357 | 357 | 357 | 357 | 357 |
| D1 | 2500 | 358 | 357 | 357 | 357 | 357 | 357 | 357 |

Registration success per tamper (the tampered crop needs the logo to be found):

| set | none | stretch | recolour | respace | glyph_swap | font_swap | overlap | cut |
|---|---|---|---|---|---|---|---|---|
| D2 | 2491/2500 | 357/358 | 357/357 | 356/357 | 356/357 | 349/357 | 354/357 | 342/357 |
| D1 | 2487/2500 | 355/358 | 354/357 | 356/357 | 356/357 | 340/357 | 346/357 | 329/357 |

Multi-font samples per font: Ubuntu-Bold 766, LiberationSans-Bold 747, NotoSans-Bold 720, DejaVuSans-Bold 709, Roboto-Bold 698, NotoSans-Bold-system 539, Lato-Bold 334, OpenSans-Bold 304, PTSans-Bold 183
