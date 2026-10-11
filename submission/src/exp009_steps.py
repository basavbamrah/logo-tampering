"""
EXP-009 intermediate results: how the font of the logo was identified, as pictures and numbers.

exp009_font_match.py only saves the ranking and one overlay sheet of the
wordmark letters. This script re-runs the same functions (it imports them)
and saves the steps in between:

  01_reference_letters.png   the 20 letters cut out of ref_1 (boxes on the logo + each mask)
  02_overlays/<font>.png     every reference letter with the same letter of that font drawn on top
                             (scaled to the same height, centred). yellow = both, red = reference
                             only, green = font only. Wordmark top row, tagline bottom row.
  03_iou_by_font.png         mean shape overlap (IoU) per font, wordmark and tagline
  04_iou_per_letter.png      IoU of every letter in every font: is the winner best on every letter?
  05_whole_word_dejavu.png   the whole words set in DejaVu Sans Bold on top of the reference,
                             spacing included (the per-letter test ignores spacing)
  06_noise_floor.png         the same letters cut from ref_2 and ref_3 compared with ref_1:
                             what "the same font" scores when only the JPEG differs
  07_zoom_top3_fonts.png     letters R and A of the wordmark, large, in the three best fonts
  steps.csv                  per font and letter: IoU and width error
  noise_floor.csv            ref_2 / ref_3 vs ref_1 per letter

It does NOT change anything exp009_font_match.py saved. It reads
outputs/exp009/font_ranking.csv only to check that it got the same numbers.

Run: python src/exp009_steps.py
"""
import csv

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from common import OUTPUTS, TAGLINE, WORDMARK, available_fonts, load_font, load_reference, reference_parts
from exp009_font_match import compare, render_letter

OUT = OUTPUTS / "exp009" / "steps"
SAVED = OUTPUTS / "exp009" / "font_ranking.csv"   # read only
CHOSEN = "DejaVuSans-Bold"


def overlay_tile(a, b, height=100):
    """yellow = both, red = reference only, green = font only (same colours as letter_overlays.png)."""
    tile = np.full(a.shape + (3,), 255, np.uint8)
    tile[a] = (0, 0, 255)
    tile[b] = (0, 200, 0)
    tile[a & b] = (0, 220, 255)
    return cv2.resize(tile, (max(1, int(tile.shape[1] * height / tile.shape[0])), height),
                      interpolation=cv2.INTER_NEAREST)


def strip(tiles, gap=8):
    pad = lambda t: np.pad(t, ((0, 0), (gap // 2, gap // 2), (0, 0)), constant_values=255)
    return np.hstack([pad(t) for t in tiles])


def save_fig(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def render_text_mask(font_path, text, height):
    """Render a whole line of text, crop it tight and scale it to the given height."""
    font = load_font(font_path, 400)
    canvas = Image.new("L", (400 * (len(text) + 2), 800), 0)
    ImageDraw.Draw(canvas).text((200, 200), text, font=font, fill=255)
    mask = np.array(canvas) > 128
    ys, xs = np.nonzero(mask)
    mask = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8) * 255
    return cv2.resize(mask, (round(mask.shape[1] * height / mask.shape[0]), height), interpolation=cv2.INTER_AREA)


def whole_word(ref, letters, group, text, font_path):
    """DejaVu text placed on the reference text's bounding box; returns overlay + IoU + width error."""
    boxes = [p["box"] for p in letters if p["group"] == group]
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x1, y1 = max(b[0] + b[2] for b in boxes), max(b[1] + b[3] for b in boxes)
    ref_mask = np.zeros((y1 - y0, x1 - x0), bool)
    for p in letters:
        if p["group"] == group:
            x, y, w, h = p["box"]
            ref_mask[y - y0:y - y0 + h, x - x0:x - x0 + w] |= p["mask"] > 0
    font = render_text_mask(font_path, text, y1 - y0) > 127
    width = max(ref_mask.shape[1], font.shape[1])
    a = np.zeros((y1 - y0, width), bool)
    b = np.zeros((y1 - y0, width), bool)
    a[:, :ref_mask.shape[1]] = ref_mask      # left-aligned: a spacing error then grows to the right
    b[:, :font.shape[1]] = font
    iou = (a & b).sum() / (a | b).sum()
    return a, b, iou, 100.0 * (font.shape[1] - ref_mask.shape[1]) / ref_mask.shape[1]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ref = load_reference(1)
    parts = [p for p in reference_parts(ref) if p["char"]]
    # reference_parts sorts groups alphabetically (tagline before word); put the wordmark first
    letters = [p for p in parts if p["group"] == "word"] + [p for p in parts if p["group"] == "tagline"]
    fonts = available_fonts()

    # ---- 1. the reference letters --------------------------------------------------
    boxed = ref.copy()
    for p in letters:
        x, y, w, h = p["box"]
        cv2.rectangle(boxed, (x, y), (x + w, y + h), (0, 0, 255), 1)
    word = strip([cv2.cvtColor(cv2.resize(255 - p["mask"], (int(p["mask"].shape[1] * 100 / p["mask"].shape[0]), 100)),
                               cv2.COLOR_GRAY2BGR) for p in letters if p["group"] == "word"])
    tag = strip([cv2.cvtColor(cv2.resize(255 - p["mask"], (int(p["mask"].shape[1] * 100 / p["mask"].shape[0]), 100)),
                              cv2.COLOR_GRAY2BGR) for p in letters if p["group"] == "tagline"])
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [2.2, 1, 1]})
    axes[0].imshow(cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"ref_1: {len(letters)} letters found by connected components "
                      f"(green-ink mask, see common.reference_parts)", fontsize=10)
    axes[1].imshow(word)
    axes[1].set_title(f"wordmark letters, each scaled to 100 px high (real height "
                      f"{letters[0]['box'][3]} px)", fontsize=10)
    tag_h = [p["box"][3] for p in letters if p["group"] == "tagline"]
    axes[2].imshow(tag)
    axes[2].set_title(f"tagline letters, each scaled to 100 px high (real height {min(tag_h)}-{max(tag_h)} px)",
                      fontsize=10)
    for ax in axes:
        ax.axis("off")
    save_fig(fig, OUT / "01_reference_letters.png")

    # ---- 2. every font on every letter -----------------------------------------------
    rows, per_font = [], {}
    for name, path in fonts.items():
        word_tiles, tag_tiles, ious = [], [], {"word": [], "tagline": []}
        for p in letters:
            iou, werr, a, b = compare(p["mask"], render_letter(path, p["char"]))
            ious[p["group"]].append(iou)
            (word_tiles if p["group"] == "word" else tag_tiles).append(overlay_tile(a, b))
            rows.append({"font": name, "group": p["group"], "letter": p["char"],
                         "iou": round(float(iou), 4), "width_err_pct": round(float(werr), 2)})
        per_font[name] = {g: float(np.mean(v)) for g, v in ious.items()}
        per_font[name]["width"] = float(np.mean([abs(r["width_err_pct"]) for r in rows if r["font"] == name]))
        w, t = strip(word_tiles), strip(tag_tiles)
        sheet = np.full((210, max(w.shape[1], t.shape[1]), 3), 255, np.uint8)
        sheet[:100, :w.shape[1]] = w
        sheet[110:, :t.shape[1]] = t
        fig, ax = plt.subplots(figsize=(14, 3))
        ax.imshow(cv2.cvtColor(sheet, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{name}: IoU wordmark {per_font[name]['word']:.3f}, tagline {per_font[name]['tagline']:.3f}, "
                     f"mean |width error| {per_font[name]['width']:.1f} %   "
                     f"(yellow = both, red = reference only, green = font only)", fontsize=10)
        ax.axis("off")
        save_fig(fig, OUT / "02_overlays" / f"{name}.png")

    order = sorted(fonts, key=lambda n: -per_font[n]["word"])     # same order as font_ranking.csv

    # ---- 3. IoU per font ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(order))[::-1]
    ax.barh(y + 0.2, [per_font[n]["word"] for n in order], 0.38, color="#0072B2", label="wordmark (6 letters)")
    ax.barh(y - 0.2, [per_font[n]["tagline"] for n in order], 0.38, color="#E69F00", label="tagline (14 letters)")
    for yi, n in zip(y, order):
        ax.text(per_font[n]["word"] + 0.005, yi + 0.2, f"{per_font[n]['word']:.3f}", va="center", fontsize=8)
        ax.text(per_font[n]["tagline"] + 0.005, yi - 0.2, f"{per_font[n]['tagline']:.3f}", va="center", fontsize=8)
    ax.set_yticks(y, order)
    ax.set_xlim(0.6, 1.02)
    ax.set_xlabel("mean shape overlap with the reference letters (IoU, 1 = identical)")
    ax.set_title("Which font matches the NORVIA letters?")
    ax.legend(fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    save_fig(fig, OUT / "03_iou_by_font.png")

    # ---- 4. IoU per letter ----------------------------------------------------------------
    chars = [p["char"] for p in letters]
    grid = np.array([[r["iou"] for r in rows if r["font"] == n] for n in order])
    fig, ax = plt.subplots(figsize=(13, 4.5))
    im = ax.imshow(grid, cmap="Blues", vmin=0.5, vmax=1.0, aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if grid[i, j] > 0.85 else "black")
    ax.set_xticks(range(len(chars)), chars)
    ax.set_yticks(range(len(order)), order)
    ax.axvline(len(WORDMARK) - 0.5, color="black", lw=2)
    ax.set_title("IoU per letter (left of the line: wordmark, right: tagline)")
    fig.colorbar(im, ax=ax, fraction=0.02)
    save_fig(fig, OUT / "04_iou_per_letter.png")
    best_on = sum(int(np.argmax(grid[:, j]) == order.index(CHOSEN)) for j in range(grid.shape[1]))

    # ---- 5. whole words in the chosen font -------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(12, 5))
    words = {}
    for ax, (group, text) in zip(axes, (("word", WORDMARK), ("tagline", "PURE BOTANICALS"))):
        a, b, iou, werr = whole_word(ref, letters, group, text, fonts[CHOSEN])
        words[group] = (iou, werr)
        ax.imshow(cv2.cvtColor(overlay_tile(a, b, height=a.shape[0] * 2), cv2.COLOR_BGR2RGB))
        ax.set_title(f'"{text}" typed in {CHOSEN} (default spacing) on the reference: IoU {iou:.3f}, '
                     f"width {werr:+.1f} %", fontsize=10)
        ax.axis("off")
    fig.suptitle("Whole words, left-aligned: if the spacing were different, the letters would drift apart "
                 "towards the right", fontsize=10)
    save_fig(fig, OUT / "05_whole_word_dejavu.png")

    # ---- 6. noise floor: the same letters in the other two reference images -----------------
    floor_rows = []
    for k in (2, 3):
        other = {(q["group"], q["char"], i): q
                 for i, q in enumerate(q for q in reference_parts(load_reference(k)) if q["char"])}
        for i, p in enumerate(parts):
            q = other.get((p["group"], p["char"], i))
            if q is None:
                continue
            iou, werr, _, _ = compare(p["mask"], q["mask"])
            floor_rows.append({"ref": f"ref_{k}", "group": p["group"], "letter": p["char"],
                               "iou": round(float(iou), 4), "width_err_pct": round(float(werr), 2)})
    floor = {g: float(np.mean([r["iou"] for r in floor_rows if r["group"] == g])) for g in ("word", "tagline")}
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = ["ref_2/3 vs ref_1\n(same font)", f"{CHOSEN}\nvs ref_1", f"{order[1]}\nvs ref_1"]
    x = np.arange(3)
    vals_w = [floor["word"], per_font[CHOSEN]["word"], per_font[order[1]]["word"]]
    vals_t = [floor["tagline"], per_font[CHOSEN]["tagline"], per_font[order[1]]["tagline"]]
    ax.bar(x - 0.2, vals_w, 0.38, color="#0072B2", label="wordmark")
    ax.bar(x + 0.2, vals_t, 0.38, color="#E69F00", label="tagline")
    for xi, vw, vt in zip(x, vals_w, vals_t):
        ax.text(xi - 0.2, vw + 0.004, f"{vw:.3f}", ha="center", fontsize=8)
        ax.text(xi + 0.2, vt + 0.004, f"{vt:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylim(0.6, 1.02)
    ax.set_ylabel("mean IoU")
    ax.set_title("What a perfect font match can score here: the same letters from the other references")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    save_fig(fig, OUT / "06_noise_floor.png")

    # ---- 7. zoom: two wordmark letters in the top three fonts -----------------------------
    zoom = [p for p in letters if p["group"] == "word" and p["char"] in "RA"]
    fig, axes = plt.subplots(len(zoom), 3, figsize=(12, 8))
    for i, p in enumerate(zoom):
        for j, n in enumerate(order[:3]):
            iou, werr, a, b = compare(p["mask"], render_letter(fonts[n], p["char"]))
            axes[i, j].imshow(cv2.cvtColor(overlay_tile(a, b, height=300), cv2.COLOR_BGR2RGB))
            axes[i, j].set_title(f"{p['char']} in {n}\nIoU {iou:.3f}, width {werr:+.1f} %", fontsize=9)
            axes[i, j].axis("off")
    fig.suptitle("Zoom (yellow = both, red = reference only, green = font only)", fontsize=10)
    save_fig(fig, OUT / "07_zoom_top3_fonts.png")

    for fname, data in (("steps.csv", rows), ("noise_floor.csv", floor_rows)):
        with open(OUT / fname, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=data[0].keys())
            w.writeheader()
            w.writerows(data)

    # ---- check against the saved ranking ----------------------------------------------------
    saved = {r["font"]: r for r in csv.DictReader(open(SAVED))}
    same = [n for n in fonts if n in saved
            and round(per_font[n]["word"], 4) == float(saved[n]["mean_iou_wordmark"])
            and round(per_font[n]["tagline"], 4) == float(saved[n]["mean_iou_tagline"])
            and round(per_font[n]["width"], 2) == float(saved[n]["mean_abs_width_err_pct"])]
    print(f"{'font':24s} {'IoU word':>9s} {'IoU tag':>8s} {'|width err| %':>14s}")
    for n in order:
        print(f"{n:24s} {per_font[n]['word']:9.3f} {per_font[n]['tagline']:8.3f} {per_font[n]['width']:14.2f}")
    print(f"\n{CHOSEN} is the best font on {best_on}/{len(chars)} letters")
    print(f"whole words in {CHOSEN}: " + ", ".join(f"{g} IoU {i:.3f} width {e:+.1f} %" for g, (i, e) in words.items()))
    print(f"noise floor (ref_2/3 vs ref_1, {len(floor_rows)} letters): wordmark {floor['word']:.3f}, "
          f"tagline {floor['tagline']:.3f}")
    print(f"same numbers as saved for {len(same)}/{len(fonts)} fonts; outputs in {OUT}")


if __name__ == "__main__":
    main()
