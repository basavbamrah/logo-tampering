"""
EXP-008 intermediate results: what the optical-flow letter-shift test does, step by step.

exp008_flow.py only saves the end numbers (one row per photo + a text summary).
This script re-runs the same steps with the same functions and settings and
saves the pictures in between, so each step can be shown and checked:

  1. which reference pixels belong to which letter (the masks the shifts are averaged in)
  2. what the flow algorithm actually receives: grey, lighting removed
  3. the flow field: for every reference pixel, how far it moved sideways in the photo
  4. the same flow as arrows on the photo
  5. one number per letter (median sideways shift) against the "genuine" band
  6. shift vs letter position: the slope that a wordmark stretch would produce

It also redoes the synthetic evaluation (known truth) to draw it, and runs the
original main() with its output folder pointed at steps/rerun_original/ so the
saved files can be compared byte for byte. Nothing in outputs/exp008/ is touched.

Outputs (in outputs/exp008/steps/):
  01_reference_letters.png   reference (half size) with the 6 wordmark + 14 tagline letter masks
  02_prepared/<id>.png       what DIS sees: reference and photo after removing lighting
  03_flow_colour/<id>.png    horizontal flow (half px), red = moved right, blue = moved left
  04_flow_arrows/<id>.png    flow arrows on the photo, magnified so small shifts are visible
  05_letter_shifts/<id>.png  per-letter shift bars, median line and genuine band (max_dev)
  06_slope/<id>.png          shift vs letter position + fitted line (slope)
  combined/<id>.png          all of the above for one photo
  overview_max_dev.png       max_dev of all 49 photos against the threshold
  synthetic_<reg|pose>.png   synthetic check: measured vs true shift, detection rate,
                             genuine vs re-spaced max_dev, genuine vs stretched slope
  synthetic_<reg|pose>.csv   one row per synthetic sample used
  steps.csv                  per photo: letter shifts, median, max_dev, slope, flag, match with saved
  rerun_original/            exp008_flow.main() output, for the byte-for-byte check

Run: python src/exp008_steps.py
"""
import csv
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import exp008_flow
from common import OUTPUTS, ROOT, auc, load_reference
from exp008_flow import H, W, letter_masks, letter_shifts, prepare, summarise

OUT = OUTPUTS / "exp008" / "steps"
SAVED = OUTPUTS / "exp008"          # read only
DATA = ROOT / "data" / "synth"
FLOW_LIMIT = 2.0                    # colour scale of the flow maps (half px); the biggest real shift is ~2.5
ARROW_GAIN = 10                     # arrows drawn 10x longer than the real shift, else they are invisible
LETTER_COLOURS = plt.cm.tab10.colors


def rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def flow_field(ref_small, crop, dis):
    """Same call as inside letter_shifts(), kept separately because that function only returns medians."""
    return dis.calc(prepare(ref_small), prepare(crop), None)


# ---------- synthetic evaluation (same selection rules as exp008_flow.main) ----------

def synthetic(crops, dis, masks):
    word = [c for g, c, _ in masks if g == "word"]
    rows = []
    for r in csv.DictReader(open(DATA / "labels.csv")):
        if r["set"] != "D2":
            continue
        if r["label"] == "0":
            kind = "genuine"
        elif r["tamper"] == "respace" and r["target"].startswith("word:"):
            kind = "respace_word"
        elif r["tamper"] == "stretch" and r["target"] == "word" and r["detail"].startswith("x "):
            kind = "stretch_word_x"
        else:
            continue
        cand_path = DATA / "D2" / f"{r['id']}{'_cand_reg.png' if crops == 'reg' else '_cand.png'}"
        if not cand_path.exists():   # registration failed on this synthetic photo
            continue
        shifts, centres = letter_shifts(cv2.imread(str(DATA / "D2" / f"{r['id']}_ref.png")),
                                        cv2.imread(str(cand_path)), masks, dis)
        max_dev, slope = summarise(shifts, centres)
        row = {"id": r["id"], "kind": kind, "max_dev": max_dev, "slope": slope,
               "magnitude": float(r["magnitude"]), "letter": "", "true_dx": "", "measured_dx": ""}
        if kind == "respace_word":
            letter = r["target"].split(":")[1]
            dx = shifts["word"]
            row.update(letter=letter, true_dx=float(r["detail"].replace("px", "")) / 2,   # half-size px
                       measured_dx=float(dx[word.index(letter)] - np.median(dx)))
        rows.append(row)
    return rows


def synthetic_figure(rows, crops, path):
    gen = [r for r in rows if r["kind"] == "genuine"]
    resp = [r for r in rows if r["kind"] == "respace_word"]
    stretch = [r for r in rows if r["kind"] == "stretch_word_x"]
    thr = np.percentile([r["max_dev"] for r in gen], 95)

    fig, ax = plt.subplots(2, 2, figsize=(13, 9.5))
    fig.suptitle(f"EXP-008 synthetic NORVIA, crops = {crops} "
                 f"({'made by the real registration code' if crops == 'reg' else 'made from the known pose'})",
                 fontsize=13)

    # a) does the measured shift of the moved letter match the injected one?
    a = ax[0, 0]
    t = np.array([r["true_dx"] for r in resp])
    m = np.array([r["measured_dx"] for r in resp])
    a.scatter(t, m, s=22, color="#2a6fb0", alpha=0.8)
    lim = max(abs(t).max(), abs(m).max()) + 0.3
    a.plot([-lim, lim], [-lim, lim], color="grey", lw=1, ls="--", label="measured = true")
    a.set_xlim(-lim, lim), a.set_ylim(-lim, lim)
    err = m - t
    # the signed mean error hides a shrink: + and - shifts cancel. A fitted line shows it.
    k, b0 = np.polyfit(t, m, 1)
    a.plot([-lim, lim], [k * -lim + b0, k * lim + b0], color="#d0542b", lw=1.5, label=f"fit: measured = {k:.2f} x true")
    a.set_title(f"a) moved letter: measured vs injected shift (n={len(resp)})\n"
                f"error mean {err.mean():+.2f}, std {err.std():.2f}; mean |error| {np.abs(err).mean():.2f} half px",
                fontsize=10)
    a.set_xlabel("injected shift (half-size px)"), a.set_ylabel("measured shift minus median of letters (half px)")
    a.legend(loc="upper left", fontsize=9), a.grid(alpha=0.3)

    # b) detection rate by shift size, at 5 % false alarms on genuine
    b = ax[0, 1]
    bins = ((1, 2), (2, 4), (4, 6.01))
    rates, labels = [], []
    for lo, hi in bins:
        sel = [r["max_dev"] for r in resp if lo <= r["magnitude"] < hi]
        rates.append(100 * np.mean(np.array(sel) > thr) if sel else 0)
        labels.append(f"{lo}-{hi:.0f} px\n(n={len(sel)})")
    bars = b.bar(labels, rates, color="#2a6fb0", width=0.6)
    for bar, v in zip(bars, rates):
        b.text(bar.get_x() + bar.get_width() / 2, v + 2, f"{v:.0f} %", ha="center", fontsize=10)
    b.set_ylim(0, 112), b.set_ylabel("re-spaced samples detected (%)")
    b.set_xlabel("injected shift (full 920 px frame)")
    b.set_title(f"b) detection rate, threshold = genuine 95th pct = {thr:.2f} half px", fontsize=10)
    b.grid(axis="y", alpha=0.3)

    # c) the max_dev score itself: genuine vs re-spaced
    c = ax[1, 0]
    edges = np.linspace(0, max(np.percentile([r["max_dev"] for r in resp], 98), thr * 2), 40)
    c.hist([r["max_dev"] for r in gen], edges, color="#888888", alpha=0.7, density=True, label=f"genuine (n={len(gen)})")
    c.hist([r["max_dev"] for r in resp], edges, color="#d0542b", alpha=0.6, density=True, label=f"re-spaced (n={len(resp)})")
    c.axvline(thr, color="black", ls="--", lw=1.2, label=f"threshold {thr:.2f}")
    c.set_title(f"c) max_dev, AUC re-spaced vs genuine = "
                f"{auc([r['max_dev'] for r in resp], [r['max_dev'] for r in gen]):.3f}", fontsize=10)
    c.set_xlabel("max_dev (half px)"), c.set_ylabel("density"), c.legend(fontsize=9), c.grid(alpha=0.3)

    # d) |slope|: genuine vs wordmark x-stretch
    d = ax[1, 1]
    gs = np.abs([r["slope"] for r in gen])
    ss = np.abs([r["slope"] for r in stretch])
    d.hist(gs, np.linspace(0, max(gs.max(), ss.max()) * 1.05, 40), color="#888888", alpha=0.7,
           label=f"genuine (n={len(gen)})")
    d.set_yscale("log")   # most genuine samples sit in the first bin; log keeps the tail visible
    for s in ss:
        d.axvline(s, color="#d0542b", lw=1.5, alpha=0.8)
    d.plot([], [], color="#d0542b", label=f"wordmark x-stretch (n={len(ss)}, one line each)")
    d.set_title(f"d) |slope|, AUC stretch vs genuine = {auc(ss, gs):.3f}", fontsize=10)
    d.set_xlabel("|slope| (shift per px of position)"), d.set_ylabel("genuine samples (log)"), d.legend(fontsize=9), d.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return thr, float(np.percentile(gs, 95))


# ---------- per-photo panels (each drawn on a given axis, reused for the combined figure) ----------

def outline(ax, masks, colour="black"):
    for _, _, m in masks:
        ax.contour(m, levels=[0.5], colors=colour, linewidths=0.6)


def panel_letters(ax, ref_small, masks):
    over = rgb(ref_small).astype(float) / 255
    word_i = 0
    for group, char, m in masks:
        col = LETTER_COLOURS[word_i % 10] if group == "word" else (0.55, 0.55, 0.55)
        over[m] = 0.35 * over[m] + 0.65 * np.array(col)
        if group == "word":
            ys, xs = np.nonzero(m)
            ax.text(xs.mean(), ys.min() - 4, char, ha="center", va="bottom", fontsize=11, weight="bold", color=col)
            word_i += 1
    ax.imshow(over)
    ax.set_title("1. Reference letter masks (shift is measured inside each; grey = tagline, not scored)", fontsize=10)
    ax.axis("off")


def panel_prepared(ax, ref_small, crop):
    both = np.vstack([prepare(ref_small), np.full((4, W), 255, np.uint8), prepare(crop)])
    ax.imshow(both, cmap="gray", vmin=0, vmax=255)
    ax.set_title("2. What the flow sees: top reference, bottom photo (grey, lighting removed)", fontsize=10)
    ax.axis("off")


def panel_flow_colour(ax, flow, masks, fig):
    im = ax.imshow(flow[..., 0], cmap="RdBu_r", vmin=-FLOW_LIMIT, vmax=FLOW_LIMIT)
    outline(ax, masks)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, label="half px")
    ax.set_title("3. Horizontal flow per reference pixel: red = moved right, blue = moved left", fontsize=10)
    ax.axis("off")


def panel_arrows(ax, crop, flow, masks):
    ax.imshow(rgb(crop))
    ink = np.zeros((H, W), bool)
    for _, _, m in masks:
        ink |= m
    ys, xs = np.mgrid[3:H:6, 3:W:6]
    keep = ink[ys, xs]                     # only on letters: the flat paper has no texture to track
    u, v = flow[ys, xs, 0][keep], flow[ys, xs, 1][keep]
    ax.quiver(xs[keep], ys[keep], u * ARROW_GAIN, v * ARROW_GAIN, color="#ff00aa",
              angles="xy", scale_units="xy", scale=1, width=0.0018, headwidth=3)
    ax.set_title(f"4. Flow arrows on the photo (drawn {ARROW_GAIN}x longer than the real shift)", fontsize=10)
    ax.axis("off")


def panel_bars(ax, chars, dx, thr):
    med = np.median(dx)
    dev = np.abs(dx - med)
    colours = ["#d0542b" if d > thr else "#2a6fb0" for d in dev]
    ax.bar(chars, dx, color=colours, width=0.6)
    ax.axhspan(med - thr, med + thr, color="#999999", alpha=0.25, label=f"median +/- {thr:.2f} (genuine 95th pct)")
    ax.axhline(med, color="black", lw=1, label=f"median {med:+.2f}")
    for i, v in enumerate(dx):
        ax.text(i, v + (0.06 if v >= 0 else -0.06), f"{v:+.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
    lim = max(1.0, np.abs(dx).max() + 0.5)
    ax.set_ylim(-lim, lim)
    ax.set_ylabel("sideways shift (half px)")
    worst = int(np.argmax(dev))
    ax.set_title(f"5. Per-letter shift. max_dev = {dev.max():.2f} half px "
                 f"(letter {chars[worst]}, {2 * (dx[worst] - med):+.1f} px in the 920 frame)"
                 + ("  -> ABOVE threshold" if dev.max() > thr else "  -> within genuine band"), fontsize=10)
    ax.legend(fontsize=8, loc="lower left"), ax.grid(axis="y", alpha=0.3)


def panel_slope(ax, chars, dx, x, slope, slope_ref):
    ax.scatter(x, dx, color="#2a6fb0", s=40, zorder=3)
    for c, xi, v in zip(chars, x, dx):
        ax.annotate(c, (xi, v), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=10)
    k, b = np.polyfit(x, dx, 1)
    xs = np.array([0, W])
    ax.plot(xs, k * xs + b, color="#d0542b", lw=1.5, label=f"fit: slope {slope:+.5f} per px")
    lim = max(1.0, np.abs(dx).max() + 0.5)
    ax.set_ylim(-lim, lim), ax.set_xlim(0, W)
    ax.set_xlabel("letter centre x (half-size px)"), ax.set_ylabel("shift (half px)")
    ax.set_title(f"6. Shift vs position. |slope| {abs(slope):.5f}; genuine synthetic 95th pct {slope_ref:.5f} "
                 "(context only, the original script sets no slope threshold)", fontsize=9)
    ax.legend(fontsize=8, loc="lower left"), ax.grid(alpha=0.3)


def save_single(stage, name, draw, size=(10, 3.4)):
    folder = OUT / stage
    folder.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=size)
    draw(ax, fig)
    fig.tight_layout()
    fig.savefig(folder / f"{name}.png", dpi=100)
    plt.close(fig)


def one_photo(path, ref_small, masks, dis, thr, slope_ref):
    name = path.stem
    crop = cv2.resize(cv2.imread(str(path)), (W, H), interpolation=cv2.INTER_AREA)
    shifts, centres = letter_shifts(ref_small, crop, masks, dis)
    max_dev, slope = summarise(shifts, centres)
    flow = flow_field(ref_small, crop, dis)
    word = [(c, m) for g, c, m in masks if g == "word"]
    chars = [c for c, _ in word]
    dx, x = shifts["word"], centres["word"]
    # letters next to the black out-of-photo area are skipped (NaN) by letter_shifts; the
    # charts show only the measured letters and name the skipped ones
    ok = ~np.isnan(dx)
    skipped = [c for c, good in zip(chars, ok) if not good]
    # the picture uses its own flow call; make sure it is the same flow the numbers came from
    assert np.allclose([np.median(flow[..., 0][m]) for (_, m), good in zip(word, ok) if good], dx[ok])
    all_chars, all_dx = chars, dx
    chars, dx, x = [c for c, good in zip(chars, ok) if good], dx[ok], x[ok]

    save_single("02_prepared", name, lambda ax, f: panel_prepared(ax, ref_small, crop), (10, 6.2))
    save_single("03_flow_colour", name, lambda ax, f: panel_flow_colour(ax, flow, masks, f))
    save_single("04_flow_arrows", name, lambda ax, f: panel_arrows(ax, crop, flow, masks))
    save_single("05_letter_shifts", name, lambda ax, f: panel_bars(ax, chars, dx, thr), (10, 4.5))
    save_single("06_slope", name, lambda ax, f: panel_slope(ax, chars, dx, x, slope, slope_ref), (10, 4.5))

    fig = plt.figure(figsize=(20, 14))
    grid = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.15])
    panel_letters(fig.add_subplot(grid[0, 0]), ref_small, masks)
    panel_prepared(fig.add_subplot(grid[0, 1]), ref_small, crop)
    panel_flow_colour(fig.add_subplot(grid[1, 0]), flow, masks, fig)
    panel_arrows(fig.add_subplot(grid[1, 1]), crop, flow, masks)
    panel_bars(fig.add_subplot(grid[2, 0]), chars, dx, thr)
    panel_slope(fig.add_subplot(grid[2, 1]), chars, dx, x, slope, slope_ref)
    fig.suptitle(f"EXP-008 photo {name}: max_dev {max_dev:.2f} half px, slope {slope:+.5f}"
                 f"  ({'FLAGGED' if max_dev > thr else 'not flagged'}; threshold {thr:.2f})"
                 + (f"  - skipped (touch the area outside the photo): {', '.join(skipped)}" if skipped else ""), fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    (OUT / "combined").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "combined" / f"{name}.png", dpi=80)
    plt.close(fig)

    med = float(np.median(dx))
    worst = int(np.argmax(np.abs(dx - med)))
    return {"image": name, "max_dev_px": round(max_dev, 3), "slope": round(slope, 5),
            **{f"dx_{c}": round(float(v), 2) for c, v in zip(all_chars, all_dx)},   # nan = skipped
            "median_dx": round(med, 3), "worst_letter": chars[worst],
            "worst_shift_full_px": round(2 * float(dx[worst] - med), 2),
            "flagged": max_dev > thr}


def overview(rows, thr, path):
    rows = sorted(rows, key=lambda r: -r["max_dev_px"])
    fig, ax = plt.subplots(figsize=(15, 4.8))
    v = [r["max_dev_px"] for r in rows]
    ax.bar([r["image"] for r in rows], v, color=["#d0542b" if x > thr else "#2a6fb0" for x in v], width=0.7)
    ax.axhline(thr, color="black", ls="--", lw=1, label=f"threshold {thr:.2f} half px = 95th pct of genuine synthetic (reg)")
    ax.set_ylabel("max_dev (half px)")
    ax.set_title(f"EXP-008: max_dev of the 49 rectified real photos; {sum(x > thr for x in v)} above the threshold (orange)")
    ax.tick_params(axis="x", rotation=90, labelsize=8)
    ax.legend(), ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def rerun_original():
    """Run exp008_flow.main() as is, but writing into steps/rerun_original/ instead of outputs/exp008/."""
    exp008_flow.OUT = OUT / "rerun_original"
    for crops in ("reg", "pose"):
        sys.argv = ["exp008_flow.py", "--crops", crops]
        exp008_flow.main()
    same = {}
    for f in ("real_letter_shifts.csv", "summary_reg.txt", "summary_pose.txt"):
        same[f] = (exp008_flow.OUT / f).read_bytes() == (SAVED / f).read_bytes()
    return same


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    masks = letter_masks()
    ref_small = cv2.resize(load_reference(1), (W, H), interpolation=cv2.INTER_AREA)

    fig, ax = plt.subplots(figsize=(10, 3.4))
    panel_letters(ax, ref_small, masks)
    fig.tight_layout()
    fig.savefig(OUT / "01_reference_letters.png", dpi=110)
    plt.close(fig)

    # synthetic first: the "genuine" threshold used on the real photos comes from it
    for crops in ("pose", "reg"):
        rows = synthetic(crops, dis, masks)
        with open(OUT / f"synthetic_{crops}.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        thr, slope_ref = synthetic_figure(rows, crops, OUT / f"synthetic_{crops}.png")
        print(f"synthetic {crops}: {len(rows)} samples, genuine 95th pct max_dev {thr:.3f}")
    # thr and slope_ref now hold the 'reg' values, as in the EXP-008 write-up

    saved = {r["image"]: r for r in csv.DictReader(open(SAVED / "real_letter_shifts.csv"))}
    rows = []
    for path in sorted((OUTPUTS / "exp001" / "rectified").glob("*.png")):
        row = one_photo(path, ref_small, masks, dis, thr, slope_ref)
        s = saved.get(row["image"])
        # nan (a skipped letter) counts as equal to nan
        row["same_as_saved"] = s is not None and all(
            np.isclose(float(s[k]), row[k], rtol=0, atol=1e-9, equal_nan=True) for k in s if k != "image")
        rows.append(row)
        print(f"{row['image']}: max_dev {row['max_dev_px']:.3f} slope {row['slope']:+.5f} "
              f"worst {row['worst_letter']} flagged={row['flagged']} same_as_saved={row['same_as_saved']}")
    with open(OUT / "steps.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    overview(rows, thr, OUT / "overview_max_dev.png")

    same = rerun_original()
    print(f"\nper-photo numbers equal to saved real_letter_shifts.csv: "
          f"{sum(r['same_as_saved'] for r in rows)}/{len(rows)}")
    print("original script re-run, byte-identical to saved files: " + ", ".join(f"{k}={v}" for k, v in same.items()))
    print(f"flagged (max_dev > {thr:.2f}): {[r['image'] for r in rows if r['flagged']]}")


if __name__ == "__main__":
    main()
