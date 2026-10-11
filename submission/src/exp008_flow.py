"""
EXP-008: how far did each letter move compared with the reference?

After a photo is rectified (EXP-001) it should sit on top of the reference
logo. If one letter has been slid sideways (re-spacing), or the wordmark has
been stretched, the letters are no longer where the reference says. We measure
that with optical flow: for every pixel of the reference, where did it end up
in the other image? Averaging the flow inside each reference letter gives one
sideways shift per letter.

Two numbers per image come out of that:
  max_dev  the biggest gap between one letter's shift and the median shift of
           the wordmark letters (px). A single moved letter makes this large.
  slope    how the shift changes from the left to the right of the wordmark
           (shift per px). A stretch of the wordmark makes this clearly non-zero.

We work at half size (460 x 130) because that is how the synthetic crops are
saved. Shifts are reported in half-size px; double them for the 920 x 260 frame.

Two tests:
  real        the 49 rectified real photos from EXP-001
  synthetic   NORVIA samples from EXP-003: genuine vs 'respace' vs wordmark 'stretch'.
              We know the true shift, so we can say how well the numbers
              separate them (AUC) and how close the measured shift is.
              By default the synthetic crops are the ones made with the REAL
              registration code (<id>_cand_reg.png, the F12 fix), so they carry
              the same alignment error as the real photos. --crops pose uses the
              older known-pose crops (an optimistic best case).

Run:   python src/exp008_flow.py [--data data/synth] [--crops reg|pose]
Output: outputs/exp008/
"""
import argparse
import csv

import cv2
import numpy as np

from common import OUTPUTS, ROOT, auc, load_reference, reference_parts

OUT = OUTPUTS / "exp008"
W, H = 460, 130


def prepare(img_bgr):
    """Grey image with the slow lighting/paper changes removed, as uint8 for DIS."""
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # Black = outside the photo (cut-off logo). Fill it with the typical grey
    # first, otherwise the blur below spreads that black ~40 px into the
    # neighbouring letters and fakes a shift (016's "I"). Same idea as EXP-010.
    outside = img_bgr.max(axis=2) == 0
    if outside.any() and not outside.all():
        g[outside] = np.median(g[~outside])
    g = g - cv2.GaussianBlur(g, (0, 0), 15)
    g = g / (g.std() + 1e-6)
    return np.clip(g * 40 + 128, 0, 255).astype(np.uint8)


def letter_masks():
    """Half-size masks of the six wordmark letters (and the tagline), from the reference."""
    parts = reference_parts(load_reference(1))
    masks = []
    for p in parts:
        if p["group"] not in ("word", "tagline"):
            continue
        full = np.zeros((260, 920), np.uint8)
        x, y, w, h = p["box"]
        full[y:y + h, x:x + w] = p["mask"]
        small = cv2.resize(full, (W, H), interpolation=cv2.INTER_NEAREST)
        masks.append((p["group"], p["char"], cv2.erode(small, np.ones((3, 3), np.uint8)) > 0))
    return masks


def letter_shifts(ref_bgr, cand_bgr, masks, dis):
    """
    Median horizontal flow (px) inside each letter (median, so a few bad flow
    vectors at the edges don't drag it). Returns ({group: shifts}, {group: letter x-centres}).

    A letter that touches the black area outside the photo (a cut-off logo)
    gets NaN: the flow there is meaningless, and it used to produce large fake
    shifts Same "black = outside" rule as EXP-010.
    """
    flow = dis.calc(prepare(ref_bgr), prepare(cand_bgr), None)
    # a small margin around the black area, where the flow is still unreliable
    outside = cv2.dilate((cand_bgr.max(axis=2) == 0).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    out = {"word": [], "tagline": []}
    centres = {"word": [], "tagline": []}
    for group, _, m in masks:
        near_edge = (m & outside).sum() > 0.02 * m.sum()
        out[group].append(np.nan if near_edge else float(np.median(flow[..., 0][m])))
        centres[group].append(float(np.nonzero(m)[1].mean()))
    return {g: np.array(v) for g, v in out.items()}, {g: np.array(v) for g, v in centres.items()}


def summarise(shifts, centres):
    """max_dev and slope over the wordmark letters that could be measured (NaN if fewer than 3)."""
    ok = ~np.isnan(shifts["word"])
    dx, x = shifts["word"][ok], centres["word"][ok]
    if len(dx) < 3:
        return float("nan"), float("nan")
    max_dev = float(np.abs(dx - np.median(dx)).max())
    slope = float(np.polyfit(x, dx, 1)[0])
    return max_dev, slope


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synth")
    ap.add_argument("--crops", default="reg", choices=["reg", "pose"],
                    help="reg = crops from real registration (F12), pose = crops from the known pose")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    masks = letter_masks()
    ref_small = cv2.resize(load_reference(1), (W, H), interpolation=cv2.INTER_AREA)

    # ---- real photos ----------------------------------------------------------
    rows = []
    for path in sorted((OUTPUTS / "exp001" / "rectified").glob("*.png")):
        crop = cv2.resize(cv2.imread(str(path)), (W, H), interpolation=cv2.INTER_AREA)
        shifts, centres = letter_shifts(ref_small, crop, masks, dis)
        max_dev, slope = summarise(shifts, centres)
        rows.append({"image": path.stem, "max_dev_px": round(max_dev, 3), "slope": round(slope, 5),
                     **{f"dx_{c}": round(float(v), 2) for (g, c, _), v in zip(
                         [m for m in masks if m[0] == "word"], shifts["word"])}})
    with open(OUT / "real_letter_shifts.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    real_dev = np.array([r["max_dev_px"] for r in rows])

    # ---- synthetic NORVIA ----------------------------------------------------
    data = ROOT / args.data
    labels = [r for r in csv.DictReader(open(data / "labels.csv")) if r["set"] == "D2"]
    groups = {"genuine": [], "respace_word": [], "stretch_word_x": []}
    errors = []   # measured shift of the moved letter minus the true shift
    word_letters = [m for m in masks if m[0] == "word"]
    for r in labels:
        kind = None
        if r["label"] == "0":
            kind = "genuine"
        elif r["tamper"] == "respace" and r["target"].startswith("word:"):
            kind = "respace_word"
        elif r["tamper"] == "stretch" and r["target"] == "word" and r["detail"].startswith("x "):
            kind = "stretch_word_x"
        if kind is None:
            continue
        ref = cv2.imread(str(data / "D2" / f"{r['id']}_ref.png"))
        suffix = "_cand_reg.png" if args.crops == "reg" else "_cand.png"
        cand_path = data / "D2" / f"{r['id']}{suffix}"
        if not cand_path.exists():        # registration failed for this synthetic photo
            continue
        cand = cv2.imread(str(cand_path))
        shifts, centres = letter_shifts(ref, cand, masks, dis)
        max_dev, slope = summarise(shifts, centres)
        groups[kind].append((max_dev, slope, r))
        if kind == "respace_word":
            letter = r["target"].split(":")[1]
            idx = [c for _, c, _ in word_letters].index(letter)
            true_dx = float(r["detail"].replace("px", "")) / 2          # half-size px
            errors.append(shifts["word"][idx] - np.median(shifts["word"]) - true_dx)

    lines = ["== real photos (rectified by EXP-001) ==",
             f"max letter deviation px (half size): median {np.median(real_dev):.2f}, "
             f"90th pct {np.percentile(real_dev, 90):.2f}, max {real_dev.max():.2f}  (n={len(real_dev)})",
             "largest deviations: " + ", ".join(f"{r['image']}={r['max_dev_px']}" for r in sorted(rows, key=lambda r: -r['max_dev_px'])[:6]),
             "", f"== synthetic NORVIA (known truth), crops: {args.crops} =="]
    for kind, items in groups.items():
        if items:
            devs = [d for d, _, _ in items]
            slopes = [s for _, s, _ in items]
            lines.append(f"{kind:16s} n={len(items):3d}  max_dev median {np.median(devs):.2f}  slope median {np.median(slopes):+.4f}")
    g = groups["genuine"]
    if g:
        thr = np.percentile([d for d, _, _ in g], 95)
        flagged = [r["image"] for r in rows if r["max_dev_px"] > thr]
        lines.append(f"genuine synthetic 95th pct of max_dev = {thr:.2f} px; real photos above it: {flagged}")
    if g and groups["respace_word"]:
        lines.append(f"AUC respace vs genuine on max_dev: {auc([d for d,_,_ in groups['respace_word']], [d for d,_,_ in g]):.3f}")
        lines.append(f"error of the measured shift of the moved letter (half-size px): "
                     f"mean {np.mean(errors):+.2f}, std {np.std(errors):.2f}")
        for lo, hi in ((1, 2), (2, 4), (4, 6.01)):
            sel = [d for d, _, r in groups["respace_word"] if lo <= float(r["magnitude"]) < hi]
            if sel:
                lines.append(f"  true shift {lo}-{hi:.0f} px: n={len(sel)}, "
                             f"detected (max_dev above genuine 95th pct): "
                             f"{np.mean(np.array(sel) > np.percentile([d for d,_,_ in g], 95)):.0%}")
    if g and groups["stretch_word_x"]:
        lines.append(f"AUC wordmark x-stretch vs genuine on |slope|: "
                     f"{auc([abs(s) for _, s, _ in groups['stretch_word_x']], [abs(s) for _, s, _ in g]):.3f}")
    text = "\n".join(lines)
    print(text)
    (OUT / f"summary_{args.crops}.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
