"""
EXP-005 intermediate results: what the colour check does, step by step, as pictures and numbers.

exp005_colour.py only saves the end result (one score per photo + summary).
This script re-runs the same functions with the same settings (it imports
them) and saves what happens in between:

Per real photo (one image per photo in each stage folder):
  01_rectified/        the rectified crop from EXP-001, resized to 460x130 (the size it is measured at)
  02_ink_mask/         which pixels are measured: ink per part (icon blue, wordmark orange,
                       tagline purple) and the paper ring (grey). Masks come from the reference,
                       not from the photo, because the crop is already lined up.
  03_hue_histograms/   hue of every ink pixel per part, photo vs reference
  04_hue_difference/   hue difference per part vs the threshold (the image score is the worst part)
  05_tagline_mask_zoom/  tagline at 5x with the measured pixels outlined in red: does the mask
                       sit on the letters? (the thin tagline is where a 1-2 px misalignment shows)
  combined/            all of the above in one figure

Synthetic evaluation (same 745 genuine / 107 recolour NORVIA crops as exp005_colour.py):
  synth_01_hue_error_vs_recolour_size.png   measured hue difference vs the injected hue shift
  synth_02_detection_rate_by_size.png       share of recolours caught, per shift size and method
  synth_03_score_distributions.png          genuine vs recolour scores for raw / paper / hue (+ ROC)
  synth_04_why_paper_normalisation_hurts.png  paper-normalised error vs how different the paper is
  synth_05_paper_examples.png               the genuine crops that paper normalisation scores worst

Real photos overview:
  real_hue_by_part.png   signed hue difference of icon / wordmark / tagline for all 49 photos

  steps.csv       per photo: pixel counts, hue per part, difference per part, flags, check vs saved
  synthetic.csv   per synthetic crop: label, shift, scores, paper difference

It does NOT change anything exp005_colour.py saved. It reads
outputs/exp005/real_colour.csv only to check that it got the same numbers.

Run: python src/exp005_steps.py
"""
import csv
from concurrent.futures import ThreadPoolExecutor

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import OUTPUTS, ROOT, auc, load_reference
from exp005_colour import GROUPS, H, METHODS, W, colours, hue_deg, lab, part_masks, scores

OUT = OUTPUTS / "exp005" / "steps"
SAVED = OUTPUTS / "exp005" / "real_colour.csv"   # read only
DATA = ROOT / "data" / "synth"
NAMES = {"icon": "icon", "word": "wordmark", "tagline": "tagline"}
# colour-blind safe (Okabe-Ito); BGR versions for the OpenCV mask picture
PART_RGB = {"icon": "#0072B2", "word": "#E69F00", "tagline": "#CC79A7"}
PART_BGR = {"icon": (178, 114, 0), "word": (0, 159, 230), "tagline": (167, 121, 204)}
METHOD_COL = {"raw": "#0072B2", "paper": "#D55E00", "hue": "#009E73", "hue_paper": "#56B4E9"}
BINS = ((3, 6), (6, 10), (10, 15.01))            # same shift bins as exp005_colour.py


def signed_hue_diff(a, b):
    return (a - b + 180) % 360 - 180


def pixel_hues(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 0]


def mask_picture(img, masks):
    """Photo dimmed, measured pixels painted in their part colour; paper ring in grey."""
    out = (img * 0.35 + 255 * 0.65).astype(np.uint8)
    inside = img.max(axis=2) > 5
    for g in GROUPS:
        ink, paper = masks[g]
        out[paper & inside] = (120, 120, 120)
        out[ink & inside] = PART_BGR[g]
    return out


def mask_offset(img, ink, reach=3):
    """
    Shift (dx, dy) of the ink mask that puts it on the darkest pixels. (0, 0) = the
    mask sits on the letters; anything else means the crop is slightly off there
    and the "ink" median also contains paper.
    """
    dark = 255 - cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    best = max(((dx, dy) for dx in range(-reach, reach + 1) for dy in range(-reach, reach + 1)),
               key=lambda d: dark[np.roll(ink, d, axis=(1, 0))].mean())
    return best


def tagline_zoom(img, ink):
    """Tagline area at 5x with the measured-pixel outline in red: shows whether the mask sits on the letters."""
    out = img.copy()
    contours, _ = cv2.findContours(ink.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, contours, -1, (0, 0, 255), 1)
    ys, xs = np.nonzero(ink)
    crop = out[max(0, ys.min() - 8):ys.max() + 9, max(0, xs.min() - 8):xs.max() + 9]
    return cv2.resize(crop, None, fx=5, fy=5, interpolation=cv2.INTER_NEAREST)


def bigger(img, f=2):
    return cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST)


def save_img(stage, name, img):
    (OUT / stage).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / stage / f"{name}.png"), img)


def save_fig(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------- per photo drawing ----------

def draw_histograms(axes, hues, ref_hues, masks, inside, cols, ref_cols):
    for ax, g in zip(axes, GROUPS):
        ink = masks[g][0]
        ax.hist(ref_hues[ink], bins=np.arange(120, 190, 1), color="#bbbbbb", label="reference pixels")
        if cols[g] is not None:
            ax.hist(hues[ink & inside], bins=np.arange(120, 190, 1), color=PART_RGB[g], alpha=0.75,
                    label="photo pixels")
            ax.axvline(hue_deg(cols[g][0]), color="black", lw=2, label="photo (hue of median colour)")
        ax.axvline(hue_deg(ref_cols[g][0]), color="black", lw=2, ls="--", label="reference")
        ax.set_title(f"3. {NAMES[g]}: ink hue", fontsize=10)
        ax.set_xlabel("hue (deg)")
        ax.set_yticks([])
    axes[0].legend(fontsize=7, loc="upper left")


def draw_differences(ax, diffs, thr):
    names = [NAMES[g] for g in GROUPS]
    vals = [abs(diffs[g]) if diffs[g] is not None else 0 for g in GROUPS]
    bars = ax.bar(names, vals, color=[PART_RGB[g] for g in GROUPS], width=0.6)
    for b, g in zip(bars, GROUPS):
        txt = f"{diffs[g]:+.1f}" if diffs[g] is not None else "part outside photo"
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3, txt, ha="center", fontsize=9)
    ax.axhline(thr, color="#D55E00", lw=2, ls="--")
    ax.set_ylim(0, max(thr * 1.6, max(vals) * 1.4))
    ax.text(0.02, 0.97, f"- - threshold {thr:.2f} deg", color="#D55E00", transform=ax.transAxes,
            va="top", fontsize=9)
    ax.set_ylabel("|hue difference| (deg)")
    ax.spines[["top", "right"]].set_visible(False)


def pattern(diffs, thr):
    """One line saying whether the shift looks like the whole logo or one part."""
    known = {g: d for g, d in diffs.items() if d is not None}
    over = [g for g, d in known.items() if abs(d) > thr]
    if not over:
        return "No part is over the threshold."
    if len(known) == 3 and all(abs(d) > thr / 2 for d in known.values()) and len({np.sign(d) for d in known.values()}) == 1:
        return "All parts shift the same way: whole-logo recolour, or a colour cast of the light."
    return f"Only {', '.join(NAMES[g] for g in over)} over the threshold: one-part recolour or a local problem."


def one_photo(path, masks, ref_small, ref_cols, ref_hues, thresholds):
    thr = thresholds["hue"]
    name = path.stem
    img = cv2.resize(cv2.imread(str(path)), (W, H), interpolation=cv2.INTER_AREA)   # same as exp005_colour.py
    cols = colours(img, masks)
    s = scores(cols, ref_cols)
    inside = img.max(axis=2) > 5
    hues = pixel_hues(img)
    diffs = {g: signed_hue_diff(hue_deg(cols[g][0]), hue_deg(ref_cols[g][0])) if cols[g] else None
             for g in GROUPS}
    worst = max((g for g in GROUPS if diffs[g] is not None), key=lambda g: abs(diffs[g]))

    save_img("01_rectified", name, bigger(img))
    save_img("02_ink_mask", name, bigger(mask_picture(img, masks)))
    save_img("05_tagline_mask_zoom", name, tagline_zoom(img, masks["tagline"][0]))

    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    draw_histograms(axes, hues, ref_hues, masks, inside, cols, ref_cols)
    save_fig(fig, OUT / "03_hue_histograms" / f"{name}.png")

    fig, ax = plt.subplots(figsize=(5, 3.2))
    draw_differences(ax, diffs, thr)
    ax.set_title(f"{name}: worst part {NAMES[worst]} {diffs[worst]:+.1f} deg -> "
                 f"{'FLAG' if s['hue'] > thr else 'ok'}", fontsize=10)
    save_fig(fig, OUT / "04_hue_difference" / f"{name}.png")

    # combined figure: 3 rows
    fig = plt.figure(figsize=(13, 8.5))
    grid = fig.add_gridspec(3, 3, height_ratios=[0.5, 1, 1.1], hspace=0.5)
    pictures = ((img, "1. Rectified crop (460x130)"), (ref_small, "Reference"),
                (mask_picture(img, masks), "2. Measured pixels (ink per part, paper ring grey)"))
    for i, (pic, title) in enumerate(pictures):
        a = fig.add_subplot(grid[0, i])
        a.imshow(cv2.cvtColor(pic, cv2.COLOR_BGR2RGB))
        a.set_title(title, fontsize=10)
        a.axis("off")
    draw_histograms([fig.add_subplot(grid[1, i]) for i in range(3)], hues, ref_hues, masks, inside, cols, ref_cols)
    a = fig.add_subplot(grid[2, 0])
    draw_differences(a, diffs, thr)
    a.set_title("4. Hue difference per part", fontsize=10)
    a = fig.add_subplot(grid[2, 1:])
    a.axis("off")
    lines = [f"Photo {name}",
             f"ink hue icon / wordmark / tagline: " + " / ".join(
                 f"{hue_deg(cols[g][0]):.1f}" if cols[g] else "-" for g in GROUPS),
             f"reference ink hue: " + " / ".join(f"{hue_deg(ref_cols[g][0]):.1f}" for g in GROUPS),
             f"image score = worst part = {s['hue']:.2f} deg ({NAMES[worst]})",
             f"threshold (95th pct of synthetic genuine) = {thr:.2f} deg",
             f"verdict (hue): {'FLAGGED as recolour' if s['hue'] > thr else 'not flagged'}",
             f"for comparison: raw dE {s['raw']:.2f}, paper-normalised dE {s['paper']:.2f}",
             pattern(diffs, thr)]
    a.text(0, 0.95, "\n".join(lines), va="top", fontsize=10, family="monospace")
    fig.suptitle(f"EXP-005 colour check, photo {name}", fontsize=13)
    save_fig(fig, OUT / "combined" / f"{name}.png")

    row = {"image": name}
    for g in GROUPS:
        row[f"ink_px_{g}"] = int((masks[g][0] & inside).sum())
    for g in GROUPS:
        row[f"hue_{g}"] = round(hue_deg(cols[g][0]), 1) if cols[g] else ""
    for g in GROUPS:
        row[f"diff_{g}"] = round(diffs[g], 2) if diffs[g] is not None else ""
    row["paper_dE"] = round(paper_de(cols, ref_cols), 2)
    for g in GROUPS:
        row[f"mask_offset_{g}"] = "%+d,%+d" % mask_offset(img, masks[g][0])
    row.update({"worst_part": worst, **{m: round(s[m], 2) for m in METHODS},
                **{f"flag_{m}": s[m] > thresholds[m] for m in METHODS}})
    return row, diffs


# ---------- synthetic evaluation ----------

def paper_de(cols, ref_cols):
    """How different the paper is from the reference paper (worst part, delta E)."""
    return max(float(np.linalg.norm(lab(cols[g][1]) - lab(ref_cols[g][1]))) for g in GROUPS if cols[g])


def hue_after_paper(cols, ref_cols):
    """Hue-only score but on the paper-normalised ink: tests whether normalisation also hurts hue."""
    worst = 0.0
    for g in GROUPS:
        if cols[g] is None:
            continue
        ink, paper = cols[g]
        balanced = np.clip(ink * ref_cols[g][1] / np.maximum(paper, 1e-3), 0, 1)
        worst = max(worst, abs(signed_hue_diff(hue_deg(balanced), hue_deg(ref_cols[g][0]))))
    return worst


def roc(pos, neg):
    thr = np.sort(np.concatenate([pos, neg]))[::-1]
    return [np.mean(neg >= t) for t in thr], [np.mean(pos >= t) for t in thr]


def synthetic(masks, ref_cols):
    labels = [r for r in csv.DictReader(open(DATA / "labels.csv"))
              if r["set"] == "D2" and (r["label"] == "0" or r["tamper"] == "recolour")]
    labels = [r for r in labels if (DATA / "D2" / f"{r['id']}_cand_reg.png").exists()]

    def one(r):
        img = cv2.imread(str(DATA / "D2" / f"{r['id']}_cand_reg.png"))
        cols = colours(img, masks)
        s = scores(cols, ref_cols)
        s["hue_paper"] = hue_after_paper(cols, ref_cols)
        return {"id": r["id"], "label": int(r["label"]), "target": r["target"],
                "shift": float(r["magnitude"]) if r["label"] == "1" else 0.0,
                "detail": r["detail"], **{k: round(v, 3) for k, v in s.items()},
                "paper_dE": round(paper_de(cols, ref_cols), 3)}

    with ThreadPoolExecutor(8) as pool:
        return list(pool.map(one, labels))


def synthetic_figures(rows):
    gen = [r for r in rows if r["label"] == 0]
    rec = [r for r in rows if r["label"] == 1]
    methods = list(METHODS) + ["hue_paper"]
    thr = {m: float(np.percentile([r[m] for r in gen], 95)) for m in methods}
    aucs = {m: auc([r[m] for r in rec], [r[m] for r in gen]) for m in methods}
    rates = {m: [float(np.mean([r[m] > thr[m] for r in rec if lo <= r["shift"] < hi])) for lo, hi in BINS]
             for m in methods}
    n_bin = [sum(lo <= r["shift"] < hi for r in rec) for lo, hi in BINS]

    # 1. measured hue error vs injected shift
    fig, ax = plt.subplots(figsize=(8, 5))
    rng = np.random.default_rng(0)
    ax.scatter(rng.uniform(-0.6, 0.6, len(gen)), [r["hue"] for r in gen], s=8, color="#999999", alpha=0.5,
               label=f"genuine (n={len(gen)}), at 0")
    for target, marker, lab_ in (("whole", "o", "recolour, whole logo"), (None, "^", "recolour, one part")):
        sel = [r for r in rec if (r["target"] == "whole") == (target == "whole")]
        ax.scatter([r["shift"] for r in sel], [r["hue"] for r in sel], s=26, marker=marker,
                   color="#009E73" if target else "#CC79A7", edgecolor="white", lw=0.5,
                   label=f"{lab_} (n={len(sel)})")
    ax.plot([0, 15], [0, 15], color="black", lw=1, ls=":", label="measured = injected")
    ax.axhline(thr["hue"], color="#D55E00", ls="--", lw=2, label=f"threshold {thr['hue']:.2f} deg (5 % false alarms)")
    ax.set_xlabel("injected hue shift (deg)")
    ax.set_ylabel("measured hue difference, worst part (deg)")
    ax.set_title(f"Hue check on synthetic NORVIA crops (AUC {aucs['hue']:.3f})")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    save_fig(fig, OUT / "synth_01_hue_error_vs_recolour_size.png")

    # 2. detection rate per shift size
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(BINS))
    for i, m in enumerate(METHODS):
        bars = ax.bar(x + (i - 1) * 0.27, rates[m], 0.25, color=METHOD_COL[m],
                      label=f"{m} (AUC {aucs[m]:.3f}, thr {thr[m]:.2f})")
        for b, v in zip(bars, rates[m]):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.0%}", ha="center", fontsize=8)
    ax.set_xticks(x, [f"{lo}-{int(hi)} deg\n(n={n})" for (lo, hi), n in zip(BINS, n_bin)])
    ax.set_ylabel("recolours caught at 5 % false alarms")
    ax.set_ylim(0, 1)
    ax.set_title("Detection rate by injected hue shift")
    ax.legend(fontsize=8, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    save_fig(fig, OUT / "synth_02_detection_rate_by_size.png")

    # 3. score distributions + ROC
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8))
    units = {"raw": "delta E", "paper": "delta E (paper-normalised)", "hue": "hue difference (deg)"}
    for ax, m in zip(axes, METHODS):
        g, t = np.array([r[m] for r in gen]), np.array([r[m] for r in rec])
        bins = np.linspace(0, np.percentile(np.concatenate([g, t]), 99), 40)
        ax.hist(g, bins, density=True, color="#999999", alpha=0.7, label="genuine")
        ax.hist(t, bins, density=True, color=METHOD_COL[m], alpha=0.7, label="recolour")
        ax.axvline(thr[m], color="black", ls="--", lw=1.5, label="threshold")
        ax.set_title(f"{m}: AUC {aucs[m]:.3f}", fontsize=10)
        ax.set_xlabel(units[m])
        ax.set_yticks([])
        ax.legend(fontsize=7)
    for m in methods:
        fpr, tpr = roc(np.array([r[m] for r in rec]), np.array([r[m] for r in gen]))
        axes[3].plot(fpr, tpr, color=METHOD_COL[m], lw=2, ls=":" if m == "hue_paper" else "-",
                     label=f"{m} {aucs[m]:.3f}")
    axes[3].plot([0, 1], [0, 1], color="black", lw=0.8)
    axes[3].set_xlabel("false alarm rate (genuine flagged)")
    axes[3].set_ylabel("recolours caught")
    axes[3].set_title("ROC (hue_paper = hue after paper normalisation)", fontsize=9)
    axes[3].legend(fontsize=7)
    save_fig(fig, OUT / "synth_03_score_distributions.png")

    # 5. the genuine crops the paper method scores worst, next to the reference
    worst = sorted(gen, key=lambda r: -r["paper"])[:6]
    fig, axes = plt.subplots(3, 2, figsize=(12, 6))
    for ax, r in zip(axes.ravel(), worst):
        img = cv2.imread(str(DATA / "D2" / f"{r['id']}_cand_reg.png"))
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{r['id']} (genuine): paper dE {r['paper']:.1f} (thr {thr['paper']:.1f}), "
                     f"hue {r['hue']:.1f} (thr {thr['hue']:.1f}), paper diff {r['paper_dE']:.1f}", fontsize=8)
        ax.axis("off")
    fig.suptitle("Genuine synthetic crops with the highest paper-normalised score: tinted paper, normal ink",
                 fontsize=11)
    save_fig(fig, OUT / "synth_05_paper_examples.png")
    return thr, aucs, rates, n_bin


def paper_figure(synth, real_rows, thr):
    """Why the paper normalisation hurts: on GENUINE crops its error follows how different the paper is."""
    gen = [r for r in synth if r["label"] == 0]
    pde = np.array([r["paper_dE"] for r in gen])
    corr = {m: float(np.corrcoef(pde, [r[m] for r in gen])[0, 1]) for m in ("paper", "raw", "hue")}
    units = {"paper": "delta E (paper-normalised)", "hue": "hue difference (deg)"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, m in zip(axes, ("paper", "hue")):
        ax.scatter(pde, [r[m] for r in gen], s=8, color=METHOD_COL[m], alpha=0.5, label="synthetic genuine")
        ax.scatter([r["paper_dE"] for r in real_rows], [r[m] for r in real_rows], s=30, marker="x",
                   color="black", label="real photos (no ground truth)")
        ax.axhline(thr[m], color="black", ls="--", lw=1.5, label=f"threshold {thr[m]:.2f}")
        ax.set_xlabel("paper colour difference to the reference paper (delta E)")
        ax.set_ylabel(units[m])
        ax.set_title(f"{m} score vs paper difference (synthetic genuine r = {corr[m]:.2f})", fontsize=10)
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Paper normalisation assumes the paper is the reference paper; when it is not, "
                 "genuine ink gets 'corrected' into the wrong colour", fontsize=11)
    save_fig(fig, OUT / "synth_04_why_paper_normalisation_hurts.png")
    return corr


def real_overview(all_diffs, thr):
    names = sorted(all_diffs)
    fig, ax = plt.subplots(figsize=(15, 4.8))
    ax.axhspan(-thr, thr, color="#eeeeee", label=f"within +-{thr:.2f} deg (not flagged)")
    for i, g in enumerate(GROUPS):
        xs = [k + (i - 1) * 0.22 for k, n in enumerate(names) if all_diffs[n][g] is not None]
        ys = [all_diffs[n][g] for n in names if all_diffs[n][g] is not None]
        ax.scatter(xs, ys, s=22, color=PART_RGB[g], label=NAMES[g], zorder=3)
    ax.set_xticks(range(len(names)), names, rotation=90, fontsize=8)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("ink hue - reference hue (deg)")
    ax.set_title("Real photos: signed hue difference per part. Flagged when any part leaves the grey band.")
    ax.legend(fontsize=8, ncol=4, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    save_fig(fig, OUT / "real_hue_by_part.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    masks = part_masks()
    ref_small = cv2.resize(load_reference(1), (W, H), interpolation=cv2.INTER_AREA)
    ref_cols = colours(ref_small, masks)
    ref_hues = pixel_hues(ref_small)

    print("scoring synthetic crops ...")
    synth = synthetic(masks, ref_cols)
    thr, aucs, rates, n_bin = synthetic_figures(synth)
    with open(OUT / "synthetic.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=synth[0].keys())
        w.writeheader()
        w.writerows(synth)

    saved = {r["image"]: r for r in csv.DictReader(open(SAVED))}
    check = ["raw", "paper", "hue", "flag_raw", "flag_paper", "flag_hue", "hue_icon", "hue_word", "hue_tagline"]
    rows, all_diffs = [], {}
    for path in sorted((OUTPUTS / "exp001" / "rectified").glob("*.png")):
        row, diffs = one_photo(path, masks, ref_small, ref_cols, ref_hues, thr)
        old = saved.get(row["image"])
        row["same_as_saved"] = old is not None and all(str(row[k]) == old[k] for k in check)
        rows.append(row)
        all_diffs[row["image"]] = diffs
        print(f"{row['image']}: hue {row['hue']:5.2f} worst {row['worst_part']:7s} "
              f"flag {row['flag_hue']!s:5s} same_as_saved={row['same_as_saved']}")
    real_overview(all_diffs, thr["hue"])
    corr = paper_figure(synth, rows, thr)
    with open(OUT / "steps.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    print(f"\nsynthetic: {sum(r['label'] == 0 for r in synth)} genuine, {sum(r['label'] == 1 for r in synth)} recolour")
    for m in list(METHODS) + ["hue_paper"]:
        print(f"  {m:9s} AUC {aucs[m]:.3f} thr {thr[m]:6.2f} caught " +
              " / ".join(f"{v:.0%} (n={n})" for v, n in zip(rates[m], n_bin)))
    print("  genuine crops, correlation with paper difference: " +
          ", ".join(f"{m} r={v:.2f}" for m, v in corr.items()))
    real_pde = [r["paper_dE"] for r in rows]
    gen_pde = [r["paper_dE"] for r in synth if r["label"] == 0]
    print(f"  paper difference: synthetic genuine median {np.median(gen_pde):.1f}, real median {np.median(real_pde):.1f}")
    print(f"real photos: same numbers as saved for {sum(r['same_as_saved'] for r in rows)}/{len(rows)}; "
          f"hue flags {[r['image'] for r in rows if r['flag_hue']]}; outputs in {OUT}")


if __name__ == "__main__":
    main()
