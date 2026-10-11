"""
EXP-010 intermediate results: every step of the golden-template difference map.

exp010_template_diff.py only saves the scores and two picture sheets.
This script re-runs the same steps with the same functions and settings and
saves what happens in between, so each step can be shown and checked:

  1. the reference ink (what the logo should look like)
  2. the photo crop in grey, with the area outside the photo filled with paper colour
  3. the paper estimate: a 161 px closing wipes out every dark shape smaller
     than the kernel, so what is left is the paper (and its shading)
  4. "how much darker than the paper" per pixel, and the Otsu cut on it
  5. the photo ink mask (after a 3x3 opening that removes specks)
  6. raw difference with the reference, no slack at all
  7. the same after allowing 2 px of alignment slack = the final map
  8. per-part scores (icon / wordmark / tagline) and how much was cut off

Plus why the kernel is 161 px: the same steps with the first-run 45 px kernel.

It also runs the original main() with its output folder pointed at
steps/rerun_original/ to compare the saved files byte for byte.
Nothing in outputs/exp010/ is touched.

Outputs (in outputs/exp010/steps/):
  01_reference_ink.png        reference ink mask (green_ink_mask) with the part boxes
  02_grey_filled/<id>.png     grey crop, outside-the-photo area filled with the median
  03_paper_estimate/<id>.png  closing with a 161 px disc = estimated paper
  04_darker/<id>.png          paper minus grey, with the Otsu threshold
  05_photo_ink/<id>.png       photo ink mask after Otsu + 3x3 opening
  06_raw_difference/<id>.png  red = reference ink not in photo, blue = photo ink not in reference, no slack
  07_after_slack/<id>.png     same with 2 px slack (final map; black = outside the photo)
  08_scores/<id>.png          missing / extra % per part
  combined/<id>.png           all steps for one photo
  kernel_45_vs_161_<id>.png   paper estimate and ink with a 45 px vs a 161 px kernel
  kernel_45_vs_161.csv        icon / word / tagline missing % for both kernels, all photos
  steps.csv                   per photo: Otsu threshold, pixel counts per step, scores, match with saved
  rerun_original/             exp010_template_diff.main() output, for the byte-for-byte check

Run: python src/exp010_steps.py [--kernel-demo 001]
"""
import argparse
import csv

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import exp010_template_diff
from common import OUTPUTS, green_ink_mask, load_reference
from exp010_template_diff import SLACK, ink_mask, region_masks

OUT = OUTPUTS / "exp010" / "steps"
SAVED = OUTPUTS / "exp010"      # read only
PARTS = ("icon", "word", "tagline")
PART_COLOURS = {"icon": "#e69f00", "word": "#009e73", "tagline": "#cc79a7"}


def ink_stages(bgr, outside, size=161):
    """ink_mask() from exp010_template_diff.py, split up so every stage can be kept."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = np.where(outside, np.median(gray[~outside]), gray).astype(np.uint8)
    paper = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
    darker = cv2.subtract(paper, gray)
    otsu, raw = cv2.threshold(darker, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    opened = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return {"gray": gray, "paper": paper, "darker": darker, "otsu": otsu,
            "ink_raw": (raw > 0) & ~outside, "ink": (opened > 0) & ~outside}


def compare(ref_ink, ink, outside, kernel):
    """The comparison of exp010_template_diff.main(): 2 px slack both ways."""
    ref_grown = cv2.dilate(ref_ink.astype(np.uint8), kernel) > 0
    ink_grown = cv2.dilate(ink.astype(np.uint8), kernel) > 0
    missing = ref_ink & ~ink_grown & ~outside
    extra = ink & ~ref_grown
    return missing, extra


def scores(ref_ink, missing, extra, outside, regions):
    """Same formulas (and rounding) as the row in exp010_template_diff.main()."""
    row = {"cut_pct": round(100 * (ref_ink & outside).sum() / ref_ink.sum(), 1)}
    for group, reg in regions.items():
        n = max(1, (ref_ink & reg).sum())
        row[f"{group}_missing_pct"] = round(100 * (missing & reg).sum() / n, 2)
        row[f"{group}_extra_pct"] = round(100 * (extra & reg).sum() / n, 2)
    row["worst_part_pct"] = max(row[f"{g}_missing_pct"] + row[f"{g}_extra_pct"] for g in regions)
    return row


def diff_picture(ref_ink, ink, missing, extra, outside):
    """Colours as in the original sheets: grey = matching ink, red = missing, blue = extra, black = outside."""
    pic = np.full(ref_ink.shape + (3,), 255, np.uint8)
    pic[ref_ink & ink] = (200, 200, 200)
    pic[missing] = (255, 0, 0)
    pic[extra] = (0, 0, 255)
    pic[outside] = (0, 0, 0)
    return pic


def boxes(ax, regions):
    for group, reg in regions.items():
        ys, xs = np.nonzero(reg)
        ax.add_patch(plt.Rectangle((xs.min(), ys.min()), xs.max() - xs.min(), ys.max() - ys.min(),
                                   fill=False, ec=PART_COLOURS[group], lw=1.5))
        ax.text(xs.min() + 3, ys.min() + 14, group, color=PART_COLOURS[group], fontsize=9, weight="bold")


# ---------- panels ----------

def show(ax, img, title, cmap=None, **kw):
    im = ax.imshow(img, cmap=cmap, **kw)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    return im


def panel_scores(ax, row):
    x = np.arange(len(PARTS))
    miss = [row[f"{g}_missing_pct"] for g in PARTS]
    extra = [row[f"{g}_extra_pct"] for g in PARTS]
    ax.bar(x - 0.18, miss, 0.34, color="#e03030", label="missing (red)")
    ax.bar(x + 0.18, extra, 0.34, color="#3050e0", label="extra (blue)")
    for i, (m, e) in enumerate(zip(miss, extra)):
        ax.text(i - 0.18, m, f"{m:.1f}", ha="center", va="bottom", fontsize=9)
        ax.text(i + 0.18, e, f"{e:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, ["icon", "wordmark", "tagline"])
    ax.set_ylim(0, max(5, max(miss + extra) * 1.2))
    ax.set_ylabel("% of that part's reference ink")
    ax.set_title(f"8. Scores: worst part {row['worst_part_pct']:.1f} %, cut off by the frame {row['cut_pct']:.1f} %",
                 fontsize=10)
    ax.legend(fontsize=8), ax.grid(axis="y", alpha=0.3)


def save_single(stage, name, draw, size=(9.2, 2.9)):
    folder = OUT / stage
    folder.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=size)
    draw(ax)
    fig.tight_layout()
    fig.savefig(folder / f"{name}.png", dpi=100)
    plt.close(fig)


def one_photo(path, ref_ink, regions, kernel):
    name = path.stem
    crop = cv2.imread(str(path))
    outside = crop.max(axis=2) == 0
    st = ink_stages(crop, outside)
    assert np.array_equal(st["ink"], ink_mask(crop, outside))    # same as the original function
    ink = st["ink"]
    raw_missing = ref_ink & ~ink & ~outside
    raw_extra = ink & ~ref_ink
    missing, extra = compare(ref_ink, ink, outside, kernel)
    row = scores(ref_ink, missing, extra, outside, regions)

    titles = {
        "gray": "2. Grey crop (outside the photo filled with paper grey)",
        "paper": "3. Paper estimate: closing with a 161 px disc",
        "darker": f"4. Paper minus grey; Otsu cut at {st['otsu']:.0f}",
        "ink": f"5. Photo ink: darker > {st['otsu']:.0f}, then 3x3 opening ({ink.sum()} px)",
        "raw": f"6. Raw difference, no slack: {raw_missing.sum()} missing, {raw_extra.sum()} extra px",
        "final": f"7. With 2 px slack (final map): {missing.sum()} missing, {extra.sum()} extra px",
    }
    raw_pic = diff_picture(ref_ink, ink, raw_missing, raw_extra, outside)
    final_pic = diff_picture(ref_ink, ink, missing, extra, outside)

    save_single("02_grey_filled", name, lambda ax: show(ax, st["gray"], titles["gray"], "gray", vmin=0, vmax=255))
    save_single("03_paper_estimate", name, lambda ax: show(ax, st["paper"], titles["paper"], "gray", vmin=0, vmax=255))
    save_single("04_darker", name, lambda ax: show(ax, st["darker"], titles["darker"], "magma"))
    save_single("05_photo_ink", name, lambda ax: show(ax, ink, titles["ink"], "gray_r"))
    save_single("06_raw_difference", name, lambda ax: show(ax, raw_pic, titles["raw"]))
    save_single("07_after_slack", name, lambda ax: (show(ax, final_pic, titles["final"]), boxes(ax, regions)))
    save_single("08_scores", name, lambda ax: panel_scores(ax, row), (6.5, 4))

    fig, ax = plt.subplots(3, 3, figsize=(21, 9.5))
    show(ax[0, 0], cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), "Rectified photo crop (input, from EXP-001)")
    show(ax[0, 1], ref_ink, f"1. Reference ink ({ref_ink.sum()} px)", "gray_r")
    show(ax[0, 2], st["gray"], titles["gray"], "gray", vmin=0, vmax=255)
    show(ax[1, 0], st["paper"], titles["paper"], "gray", vmin=0, vmax=255)
    show(ax[1, 1], st["darker"], titles["darker"], "magma")
    show(ax[1, 2], ink, titles["ink"], "gray_r")
    show(ax[2, 0], raw_pic, titles["raw"])
    show(ax[2, 1], final_pic, titles["final"])
    boxes(ax[2, 1], regions)
    panel_scores(ax[2, 2], row)
    fig.suptitle(f"EXP-010 photo {name}: red = reference ink missing in the photo, blue = photo ink the reference "
                 f"does not have, grey = matching ink, black = outside the photo", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    (OUT / "combined").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "combined" / f"{name}.png", dpi=80)
    plt.close(fig)

    return {"image": name, **row, "otsu_threshold": int(st["otsu"]), "photo_ink_px": int(ink.sum()),
            "specks_removed_by_opening_px": int(st["ink_raw"].sum() - ink.sum()),
            "raw_missing_px": int(raw_missing.sum()), "raw_extra_px": int(raw_extra.sum()),
            "missing_px_after_slack": int(missing.sum()), "extra_px_after_slack": int(extra.sum())}


def kernel_demo(path, ref_ink, regions, kernel):
    """The first run used a 45 px kernel. Show what that does to the paper estimate and the icon."""
    crop = cv2.imread(str(path))
    outside = crop.max(axis=2) == 0
    fig, ax = plt.subplots(2, 3, figsize=(20, 6.2))
    for i, size in enumerate((45, 161)):
        st = ink_stages(crop, outside, size)
        missing, extra = compare(ref_ink, st["ink"], outside, kernel)
        row = scores(ref_ink, missing, extra, outside, regions)
        show(ax[i, 0], st["paper"], f"{size} px kernel: paper estimate", "gray", vmin=0, vmax=255)
        show(ax[i, 1], st["ink"], f"{size} px kernel: photo ink", "gray_r")
        show(ax[i, 2], diff_picture(ref_ink, st["ink"], missing, extra, outside),
             f"{size} px: icon missing {row['icon_missing_pct']:.1f} %, word {row['word_missing_pct']:.1f} %, "
             f"tagline {row['tagline_missing_pct']:.1f} %")
    fig.suptitle(f"Why 161 px: photo {path.stem}. A 45 px disc fits inside the solid icon halves, so the closing "
                 "does not remove them: the icon is taken as paper and comes out 'missing'.", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / f"kernel_45_vs_161_{path.stem}.png", dpi=85)
    plt.close(fig)


def rerun_original():
    """Run exp010_template_diff.main() as is, but writing into steps/rerun_original/."""
    exp010_template_diff.OUT = OUT / "rerun_original"
    exp010_template_diff.main()
    return {f: (exp010_template_diff.OUT / f).read_bytes() == (SAVED / f).read_bytes()
            for f in ("diff_scores.csv", "diff_sheet_1.jpg", "diff_sheet_2.jpg")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel-demo", default="001", help="photo used for the 45 vs 161 px figure")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    ref = load_reference(1)
    ref_ink = green_ink_mask(ref) > 0
    regions = region_masks(ref)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * SLACK + 1, 2 * SLACK + 1))

    fig, ax = plt.subplots(figsize=(9.2, 2.9))
    show(ax, ref_ink, f"1. Reference ink mask ({ref_ink.sum()} px) and the three scored parts", "gray_r")
    boxes(ax, regions)
    fig.tight_layout()
    fig.savefig(OUT / "01_reference_ink.png", dpi=110)
    plt.close(fig)

    paths = sorted((OUTPUTS / "exp001" / "rectified").glob("*.png"))
    saved = {r["image"]: r for r in csv.DictReader(open(SAVED / "diff_scores.csv"))}
    rows, kernel_rows = [], []
    for path in paths:
        row = one_photo(path, ref_ink, regions, kernel)
        s = saved.get(row["image"])
        row["same_as_saved"] = s is not None and all(abs(float(s[k]) - row[k]) < 1e-9 for k in s if k != "image")
        rows.append(row)
        print(f"{row['image']}: otsu {row['otsu_threshold']} worst {row['worst_part_pct']:.2f} % "
              f"cut {row['cut_pct']} % same_as_saved={row['same_as_saved']}")

        # icon / word / tagline missing with the old 45 px kernel, for the "why 161" table
        crop = cv2.imread(str(path))
        outside = crop.max(axis=2) == 0
        k45 = ink_stages(crop, outside, 45)["ink"]
        r45 = scores(ref_ink, *compare(ref_ink, k45, outside, kernel), outside, regions)
        kernel_rows.append({"image": path.stem,
                            **{f"{g}_missing_45": r45[f"{g}_missing_pct"] for g in PARTS},
                            **{f"{g}_missing_161": row[f"{g}_missing_pct"] for g in PARTS}})

    for name, data in (("steps.csv", rows), ("kernel_45_vs_161.csv", kernel_rows)):
        with open(OUT / name, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
    kernel_demo(next(p for p in paths if p.stem == args.kernel_demo), ref_ink, regions, kernel)

    same = rerun_original()
    icon45 = [r["icon_missing_45"] for r in kernel_rows]
    print(f"\nper-photo scores equal to saved diff_scores.csv: {sum(r['same_as_saved'] for r in rows)}/{len(rows)}")
    print("original script re-run, byte-identical to saved files: " + ", ".join(f"{k}={v}" for k, v in same.items()))
    print(f"icon missing with a 45 px kernel: median {np.median(icon45):.1f} %, min {min(icon45):.1f} %")
    print(f"photos with any difference: {[r['image'] for r in rows if r['worst_part_pct'] > 0]}")


if __name__ == "__main__":
    main()
