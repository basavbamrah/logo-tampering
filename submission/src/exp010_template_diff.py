"""
EXP-010 (approach A, "golden template"): where does each rectified photo differ
from the reference logo?

EXP-001 only lines the photo up with the reference. It does NOT decide whether
the logo is genuine. This script does the simplest possible comparison on the
lined-up crops:

  1. find the ink (logo pixels) in the rectified photo. The ink is darker than
     the paper around it, so we estimate the paper by "closing" the image
     (which wipes out dark shapes smaller than the kernel; 161 px > the icon)
     and keep pixels clearly darker than it
  2. compare with the reference ink, allowing 2 px of slack for alignment error
       red  = ink in the reference but missing in the photo
       blue = ink in the photo that the reference doesn't have
  3. count those pixels per part of the logo (icon / wordmark / tagline)
  4. black pixels in the rectified crop are outside the original photo: that
     tells us how much of the logo was cut off by the frame

Output: outputs/exp010/diff_scores.csv and two picture sheets.
Run:    python src/exp010_template_diff.py
"""
import csv

import cv2
import numpy as np

from common import OUTPUTS, green_ink_mask, load_reference, reference_parts

OUT = OUTPUTS / "exp010"
SLACK = 2   # px of alignment tolerance


def ink_mask(bgr, outside):
    """Logo ink in a rectified photo crop (True = ink)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = np.where(outside, np.median(gray[~outside]), gray).astype(np.uint8)
    paper = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (161, 161)))
    darker = cv2.subtract(paper, gray)
    _, mask = cv2.threshold(darker, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return (mask > 0) & ~outside


def region_masks(ref):
    """One rectangle per logo part (icon / word / tagline), a bit enlarged."""
    regions = {}
    for group in ("icon", "word", "tagline"):
        boxes = [p["box"] for p in reference_parts(ref) if p["group"] == group]
        x0 = min(b[0] for b in boxes) - 8
        y0 = min(b[1] for b in boxes) - 8
        x1 = max(b[0] + b[2] for b in boxes) + 8
        y1 = max(b[1] + b[3] for b in boxes) + 8
        m = np.zeros(ref.shape[:2], bool)
        m[max(y0, 0):y1, max(x0, 0):x1] = True
        regions[group] = m
    return regions


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ref = load_reference(1)
    ref_ink = green_ink_mask(ref) > 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * SLACK + 1, 2 * SLACK + 1))
    ref_grown = cv2.dilate(ref_ink.astype(np.uint8), kernel) > 0
    regions = region_masks(ref)

    rows, tiles = [], []
    for path in sorted((OUTPUTS / "exp001" / "rectified").glob("*.png")):
        crop = cv2.imread(str(path))
        outside = crop.max(axis=2) == 0
        ink = ink_mask(crop, outside)
        ink_grown = cv2.dilate(ink.astype(np.uint8), kernel) > 0

        missing = ref_ink & ~ink_grown & ~outside   # reference ink not found in the photo
        extra = ink & ~ref_grown                     # photo ink where the reference has none

        row = {"image": path.stem,
               "cut_pct": round(100 * (ref_ink & outside).sum() / ref_ink.sum(), 1)}
        for group, reg in regions.items():
            n = max(1, (ref_ink & reg).sum())
            row[f"{group}_missing_pct"] = round(100 * (missing & reg).sum() / n, 2)
            row[f"{group}_extra_pct"] = round(100 * (extra & reg).sum() / n, 2)
        row["worst_part_pct"] = max(row[f"{g}_missing_pct"] + row[f"{g}_extra_pct"] for g in regions)
        rows.append(row)

        # picture: photo crop on top, difference map below
        diff = np.full(crop.shape, 255, np.uint8)
        diff[ref_ink & ink_grown] = (200, 200, 200)
        diff[missing] = (0, 0, 255)
        diff[extra] = (255, 0, 0)
        diff[outside] = (0, 0, 0)
        tile = np.vstack([crop, diff])
        tile = cv2.resize(tile, (460, 260))
        cv2.rectangle(tile, (0, 0), (190, 24), (0, 255, 255), -1)
        cv2.putText(tile, f"{path.stem}  {row['worst_part_pct']:.1f}%", (4, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        tiles.append(np.pad(tile, ((3, 3), (3, 3), (0, 0)), constant_values=120))

    with open(OUT / "diff_scores.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    for i, part in enumerate((tiles[:25], tiles[25:]), 1):
        while len(part) % 5:
            part.append(np.full_like(part[0], 255))
        sheet = np.vstack([np.hstack(part[j:j + 5]) for j in range(0, len(part), 5)])
        cv2.imwrite(str(OUT / f"diff_sheet_{i}.jpg"), sheet)

    print(f"{'image':6s} {'cut%':>5s} {'icon':>11s} {'word':>11s} {'tagline':>11s}   (missing/extra % of that part's ink)")
    for r in sorted(rows, key=lambda r: -r["worst_part_pct"]):
        print(f"{r['image']:6s} {r['cut_pct']:5.1f} "
              f"{r['icon_missing_pct']:5.1f}/{r['icon_extra_pct']:<5.1f} "
              f"{r['word_missing_pct']:5.1f}/{r['word_extra_pct']:<5.1f} "
              f"{r['tagline_missing_pct']:5.1f}/{r['tagline_extra_pct']:<5.1f}")


if __name__ == "__main__":
    main()
