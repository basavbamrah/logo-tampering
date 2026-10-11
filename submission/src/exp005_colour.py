"""
EXP-005: is the logo ink the right colour?

After rectification (EXP-001) the logo sits exactly on the reference, so we
know which pixels are ink and which are paper without guessing. For each part
of the logo (icon / wordmark / tagline) we take the median colour of the ink
and of a thin ring of paper around it, and compare with the reference in three
ways:

  raw     ink colour as photographed (CIELAB difference, "delta E")
  paper   ink colour corrected by the paper: divide by the paper colour and
          multiply by the reference paper colour (simple white balance). This
          removes a colour cast of the light, but assumes the paper colour
          itself is the same as in the reference, which may not hold.
  hue     only the hue angle of the ink (ignores brightness and saturation)


A recolour may touch only one part, so an image's score is the worst part.

Run:    python src/exp005_colour.py
Output: outputs/exp005/
"""
import csv
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from common import OUTPUTS, ROOT, auc, green_ink_mask, load_reference, reference_parts

OUT = OUTPUTS / "exp005"
W, H = 460, 130                      # work at the size the synthetic crops are saved at
METHODS = ("raw", "paper", "hue")
GROUPS = ("icon", "word", "tagline")


def part_masks():
    """Ink and paper-ring masks per logo part, at half size, from the reference."""
    ref = load_reference(1)
    masks = {}
    for group in GROUPS:
        full = np.zeros(ref.shape[:2], np.uint8)
        for p in reference_parts(ref):
            if p["group"] == group:
                x, y, w, h = p["box"]
                full[y:y + h, x:x + w] |= p["mask"]
        small = cv2.resize(full, (W, H), interpolation=cv2.INTER_NEAREST)
        ink = cv2.erode(small, np.ones((3, 3), np.uint8)) > 0              # stay off the blurry edges
        near = cv2.dilate(small, np.ones((7, 7), np.uint8)) > 0
        far = cv2.dilate(small, np.ones((15, 15), np.uint8)) > 0
        masks[group] = (ink, far & ~near)                                   # paper ring: 3-7 px away
    return masks


def lab(rgb01):
    return cv2.cvtColor(np.float32(rgb01).reshape(1, 1, 3), cv2.COLOR_RGB2LAB)[0, 0]


def hue_deg(rgb01):
    return float(cv2.cvtColor(np.float32(rgb01).reshape(1, 1, 3), cv2.COLOR_RGB2HSV)[0, 0, 0])


def colours(img_bgr, masks):
    """Median ink and paper colour (RGB 0..1) per part. None where the part is outside the photo."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    inside = img_bgr.max(axis=2) > 5                  # black = outside the original photo
    out = {}
    for group, (ink, paper) in masks.items():
        ink_px, paper_px = rgb[ink & inside], rgb[paper & inside]
        if len(ink_px) < 0.5 * ink.sum() or len(paper_px) < 50:
            out[group] = None
        else:
            out[group] = (np.median(ink_px, 0), np.median(paper_px, 0))
    return out


def scores(cols, ref_cols):
    """Colour difference to the reference per method = worst part."""
    result = {}
    for method in METHODS:
        worst = 0.0
        for group in GROUPS:
            if cols[group] is None:
                continue
            ink, paper = cols[group]
            ref_ink, ref_paper = ref_cols[group]
            if method == "raw":
                d = float(np.linalg.norm(lab(ink) - lab(ref_ink)))
            elif method == "paper":
                balanced = np.clip(ink * ref_paper / np.maximum(paper, 1e-3), 0, 1)
                d = float(np.linalg.norm(lab(balanced) - lab(ref_ink)))
            else:
                d = abs((hue_deg(ink) - hue_deg(ref_ink) + 180) % 360 - 180)
            worst = max(worst, d)
        result[method] = worst
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    masks = part_masks()
    ref_cols = colours(cv2.resize(load_reference(1), (W, H), interpolation=cv2.INTER_AREA), masks)

    # ---- synthetic NORVIA: genuine vs recolour (known hue shift) ----------------
    data = ROOT / "data" / "synth"
    labels = [r for r in csv.DictReader(open(data / "labels.csv"))
              if r["set"] == "D2" and (r["label"] == "0" or r["tamper"] == "recolour")]
    labels = [r for r in labels if (data / "D2" / f"{r['id']}_cand_reg.png").exists()]

    def score_synth(r):
        img = cv2.imread(str(data / "D2" / f"{r['id']}_cand_reg.png"))
        return r, scores(colours(img, masks), ref_cols)

    with ThreadPoolExecutor(8) as pool:
        synth = list(pool.map(score_synth, labels))

    genuine = [s for r, s in synth if r["label"] == "0"]
    recol = [(r, s) for r, s in synth if r["label"] == "1"]
    thresholds = {m: float(np.percentile([s[m] for s in genuine], 95)) for m in METHODS}

    lines = [f"== synthetic NORVIA (F12 crops): {len(genuine)} genuine, {len(recol)} recolour ==",
             f"{'method':7s} {'AUC':>6s} {'thr(95% genuine)':>17s}   caught by hue shift: 3-6 / 6-10 / 10-15 deg   whole-logo / one-part"]
    for m in METHODS:
        a = auc([s[m] for _, s in recol], [s[m] for s in genuine])
        bins = []
        for lo, hi in ((3, 6), (6, 10), (10, 15.01)):
            sel = [s[m] for r, s in recol if lo <= float(r["magnitude"]) < hi]
            bins.append(f"{np.mean(np.array(sel) > thresholds[m]):.0%} (n={len(sel)})" if sel else "-")
        whole = [s[m] > thresholds[m] for r, s in recol if r["target"] == "whole"]
        part = [s[m] > thresholds[m] for r, s in recol if r["target"] != "whole"]
        lines.append(f"{m:7s} {a:6.3f} {thresholds[m]:17.2f}   {' / '.join(bins)}   "
                     f"{np.mean(whole):.0%} / {np.mean(part):.0%}")

    # ---- real photos -------------------------------------------------------------
    rows = []
    for path in sorted((OUTPUTS / "exp001" / "rectified").glob("*.png")):
        img = cv2.resize(cv2.imread(str(path)), (W, H), interpolation=cv2.INTER_AREA)
        cols = colours(img, masks)
        s = scores(cols, ref_cols)
        row = {"image": path.stem, **{m: round(s[m], 2) for m in METHODS}}
        row.update({f"flag_{m}": s[m] > thresholds[m] for m in METHODS})
        for g in GROUPS:   # the raw ink hue per part, for looking at by hand
            row[f"hue_{g}"] = round(hue_deg(cols[g][0]), 1) if cols[g] else ""
        rows.append(row)
    with open(OUT / "real_colour.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    ref_hues = {g: round(hue_deg(ref_cols[g][0]), 1) for g in GROUPS}
    lines += ["", f"== real photos (n={len(rows)}); reference ink hue per part: {ref_hues} =="]
    for m in METHODS:
        flagged = [r["image"] for r in rows if r[f"flag_{m}"]]
        lines.append(f"{m:7s} flags {len(flagged):2d}: {flagged}")
    watch = ["027", "028", "033", "038"]
    lines.append("model's 'recolour' candidates: " + "; ".join(
        f"{r['image']} raw {r['raw']} paper {r['paper']} hue {r['hue']} (hues {r['hue_icon']}/{r['hue_word']}/{r['hue_tagline']})"
        for r in rows if r["image"] in watch))
    real_hue = np.array([r["hue"] for r in rows])
    lines.append(f"real hue difference: median {np.median(real_hue):.2f}, 90th pct {np.percentile(real_hue, 90):.2f}; "
                 f"synthetic genuine: median {np.median([s['hue'] for s in genuine]):.2f}, "
                 f"90th pct {np.percentile([s['hue'] for s in genuine], 90):.2f}")
    text = "\n".join(lines)
    print(text)
    (OUT / "summary.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
