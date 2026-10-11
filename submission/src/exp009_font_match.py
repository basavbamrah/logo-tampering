"""
EXP-009 (step 1): which font is the NORVIA logo set in?

Idea: cut every letter out of the reference logo, render the same letter in
each candidate font, scale it to the same height and see how well the two
shapes overlap. The font with the highest overlap is our best guess.

Why we care: if we find the font, the synthetic generator (EXP-003) can make
"genuine" and "tampered" logos that look like the real thing, and we can pick
look-alike fonts for the glyph-swap tamper.

Output: outputs/exp009/font_ranking.csv and a picture comparing the letters.
Run:    python src/exp009_font_match.py
"""
import csv

import cv2
import numpy as np
from PIL import Image, ImageDraw

from common import OUTPUTS, available_fonts, load_font, load_reference, reference_parts

OUT = OUTPUTS / "exp009"


def render_letter(font_path, char, size=400):
    """Draw one character in black on white and return its tight binary mask."""
    font = load_font(font_path, size)
    canvas = Image.new("L", (size * 2, size * 2), 0)
    ImageDraw.Draw(canvas).text((size // 2, size // 2), char, font=font, fill=255)
    mask = np.array(canvas) > 128
    ys, xs = np.nonzero(mask)
    return mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8) * 255


def compare(ref_mask, font_mask):
    """
    Scale the font letter to the reference letter's height, centre both on
    the same canvas and measure:
      - iou:        shape overlap (1.0 = identical)
      - width_err:  how much wider/narrower the font letter is, in %
    """
    h_ref, w_ref = ref_mask.shape
    h_f, w_f = font_mask.shape
    new_w = max(1, round(w_f * h_ref / h_f))
    font_scaled = cv2.resize(font_mask, (new_w, h_ref), interpolation=cv2.INTER_AREA)

    width = max(w_ref, new_w)
    a = np.zeros((h_ref, width), np.uint8)
    b = np.zeros((h_ref, width), np.uint8)
    a[:, (width - w_ref) // 2:(width - w_ref) // 2 + w_ref] = ref_mask
    b[:, (width - new_w) // 2:(width - new_w) // 2 + new_w] = font_scaled
    a, b = a > 127, b > 127

    iou = (a & b).sum() / max(1, (a | b).sum())
    width_err = 100.0 * (new_w - w_ref) / w_ref
    return iou, width_err, a, b


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ref = load_reference(1)
    letters = [p for p in reference_parts(ref) if p["char"]]
    fonts = available_fonts()

    rows, overlays = [], {}
    for name, path in fonts.items():
        word_iou, tag_iou, width_errs, tiles = [], [], [], []
        for p in letters:
            iou, werr, a, b = compare(p["mask"], render_letter(path, p["char"]))
            (word_iou if p["group"] == "word" else tag_iou).append(iou)
            width_errs.append(abs(werr))
            if p["group"] == "word":
                # red = reference only, green = font only, yellow = both
                tile = np.zeros(a.shape + (3,), np.uint8)
                tile[a] = (0, 0, 255)
                tile[b] = (0, 255, 0)
                tile[a & b] = (0, 255, 255)
                tiles.append(cv2.resize(tile, (int(tile.shape[1] * 100 / tile.shape[0]), 100)))
        rows.append({
            "font": name,
            "mean_iou_wordmark": round(float(np.mean(word_iou)), 4),
            "mean_iou_tagline": round(float(np.mean(tag_iou)), 4),
            "mean_abs_width_err_pct": round(float(np.mean(width_errs)), 2),
            "path": str(path),
        })
        overlays[name] = tiles

    rows.sort(key=lambda r: -r["mean_iou_wordmark"])
    with open(OUT / "font_ranking.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # One strip per font: N O R V I A overlays, best font on top.
    strips = []
    for r in rows:
        strip = np.hstack([np.pad(t, ((0, 0), (4, 4), (0, 0))) for t in overlays[r["font"]]])
        label = np.zeros((100, 330, 3), np.uint8)
        cv2.putText(label, f'{r["font"]}', (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(label, f'IoU {r["mean_iou_wordmark"]:.3f}', (5, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        strips.append((label, strip))
    width = max(s.shape[1] for _, s in strips)
    sheet = np.vstack([np.hstack([lab, np.pad(s, ((0, 0), (0, width - s.shape[1]), (0, 0)))])
                       for lab, s in strips])
    cv2.imwrite(str(OUT / "letter_overlays.png"), sheet)

    print(f"{'font':24s} {'IoU word':>9s} {'IoU tag':>8s} {'|width err| %':>14s}")
    for r in rows:
        print(f'{r["font"]:24s} {r["mean_iou_wordmark"]:9.3f} {r["mean_iou_tagline"]:8.3f} '
              f'{r["mean_abs_width_err_pct"]:14.2f}')


if __name__ == "__main__":
    main()
