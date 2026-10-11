"""
EXP-013: one detailed diagram of the whole experiment, for the write-up and the panel.

Data generation -> the two model inputs -> the model (with real tensor shapes,
taken from a forward pass, not typed by hand) -> training and tuning -> evaluation.

Run: python src/exp013_diagram.py   -> outputs/exp013/exp013_diagram.png
"""
import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from common import OUTPUTS  # noqa: E402
from models import build  # noqa: E402

W_IN, H_IN = 448, 192


def shapes():
    """Tensor shapes inside siamese_diff depth 1 for a 448 x 192 input (batch of 1)."""
    model = build("siamese_diff", pretrained=False, depth=1).eval()
    x = torch.zeros(1, 3, H_IN, W_IN)
    with torch.no_grad():
        a = model.encoder(x)
        c = model.compare(torch.cat([(a - a).abs(), a * a], 1))
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return tuple(a.shape[1:]), tuple(c.shape[1:]), params


def box(ax, x, y, w, h, text, colour, size=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.012",
                                fc=colour, ec="#333333", lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, wrap=True)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, lw=1.1, color="#333333"))


def main():
    feat, comp, params = shapes()
    fig, ax = plt.subplots(figsize=(17, 12.5))
    ax.set_xlim(0, 1), ax.set_ylim(0, 1), ax.axis("off")
    GEN, INP, MOD, TRN, EVA = "#e3f0fb", "#fdf1d8", "#e6f5e3", "#f4e6f7", "#fbe3e3"

    # band titles
    for y, t in ((0.965, "1. DATA GENERATION (data/synth_v2, 10 000 samples: 5000 NORVIA + 5000 multi-font, half genuine / half tampered)"),
                 (0.745, "2. THE TWO MODEL INPUTS (both cropped with ~30 % background, 448 x 192)"),
                 (0.535, f"3. MODEL: siamese_diff, depth 1 ({params:,} trainable parameters)"),
                 (0.255, "4. TRAINING + TUNING"), (0.105, "5. EVALUATION")):
        ax.text(0.01, y, t, fontsize=11, weight="bold")

    # 1. generation
    xs = [0.01, 0.15, 0.29, 0.43, 0.57, 0.71, 0.85]
    texts = ["Clean logo\nNORVIA: ink from ref_1/2/3\nmulti-font: made-up word in\none of 9 fonts (Lato held out)",
             "Tamper (label known)\nnone, stretch, recolour,\nrespace, glyph swap,\nfont swap, overlap, cut",
             "Scene\nrandom paper, bands,\nlight spots, filler text",
             "Virtual camera\nyaw +-35, pitch +-28,\nroll +-10 deg; logo\n380-1000 px wide",
             "Photo effects\nlighting, blur, noise,\ncolour cast; edge NOT\nmirrored",
             "Render square, resize to\n960x1280 / 1280x960 /\n1024x1024, JPEG 78-96",
             "<id>_photo.jpg\n+ labels.csv\n(type, target, size)\nsplit 70/15/15"]
    for i, (x, t) in enumerate(zip(xs, texts)):
        box(ax, x, 0.80, 0.125, 0.14, t, GEN, 8)
        if i:
            arrow(ax, xs[i - 1] + 0.125, 0.87, x, 0.87)

    # 2. inputs
    box(ax, 0.03, 0.575, 0.36, 0.14,
        "REFERENCE input  (<id>_ref_raw.png)\nthe clean, front-on logo of that sample with paper around it:\n"
        "upright box around the logo, +15 % per side, grown to 448:192\n(NORVIA: from the reference image; multi-font: clean render)", INP)
    box(ax, 0.45, 0.575, 0.52, 0.14,
        "TAMPERED input  (<id>_cand_raw.png)  = the image being checked (genuine in half the samples)\n"
        "1) SIFT + RANSAC find WHERE the logo is in the photo (template corners through H)\n"
        "2) upright box around those corners, +15 % per side, grown to 448:192, NOT warped:\n"
        "the logo stays tilted / in perspective as the camera saw it; black where the box leaves the photo\n"
        "Photos where the logo can't be found get no crop (like 'can't register' for a real photo)", INP)
    arrow(ax, 0.92, 0.80, 0.80, 0.715)
    arrow(ax, 0.07, 0.80, 0.15, 0.715)

    # 3. model
    box(ax, 0.03, 0.39, 0.17, 0.11, f"reference\n3 x {H_IN} x {W_IN}", MOD)
    box(ax, 0.03, 0.27, 0.17, 0.11, f"tampered\n3 x {H_IN} x {W_IN}", MOD)
    box(ax, 0.25, 0.30, 0.17, 0.18, "shared encoder (same weights)\nResNet-18 (ImageNet):\nconv1 + maxpool + layer1\n"
        f"-> feature map\n{feat[0]} x {feat[1]} x {feat[2]}", MOD)
    box(ax, 0.47, 0.30, 0.15, 0.18, "compare, location by\nlocation:\n|a - b|  and  a * b\n"
        f"-> {2 * feat[0]} x {feat[1]} x {feat[2]}", MOD)
    box(ax, 0.66, 0.30, 0.13, 0.18, f"compare block\n(conv layers)\n-> {comp[0]} x {comp[1]} x {comp[2]}\nthen max + mean\npooling", MOD)
    box(ax, 0.83, 0.405, 0.15, 0.085, "head 1: tampered?\nyes / no (main score)", MOD)
    box(ax, 0.83, 0.295, 0.15, 0.085, "head 2: which tamper?\n8 classes (helper)", MOD)
    arrow(ax, 0.20, 0.445, 0.25, 0.42), arrow(ax, 0.20, 0.325, 0.25, 0.36)
    arrow(ax, 0.42, 0.39, 0.47, 0.39), arrow(ax, 0.62, 0.39, 0.66, 0.39)
    arrow(ax, 0.79, 0.41, 0.83, 0.445), arrow(ax, 0.79, 0.37, 0.83, 0.335)
    arrow(ax, 0.21, 0.575, 0.12, 0.50)
    # tampered input -> model: routed around the reference box and the encoder
    ax.plot([0.71, 0.71, 0.225, 0.225], [0.575, 0.52, 0.52, 0.325], color="#333333", lw=1.1)
    arrow(ax, 0.225, 0.325, 0.20, 0.325)

    # 4. training
    box(ax, 0.03, 0.135, 0.30, 0.10, "Fixed: AdamW, lr 3e-4, cosine schedule,\n15 epochs, batch 32, ImageNet start;\n"
        "loss = BCE(tampered?) + 0.3 x CE(which tamper?)", TRN)
    box(ax, 0.36, 0.135, 0.36, 0.10, "Tuning, step by step on VALIDATION AUC (keep only if > +0.01):\n"
        "A baseline -> B weight decay 1e-3 / 1e-2 -> C dropout 0.3 ->\nD augmentation (photometric + small camera move of the\n"
        "tampered crop) -> E label smoothing 0.1 / focal loss -> F seed 1", TRN)
    box(ax, 0.75, 0.135, 0.22, 0.10, "Logged per run: TensorBoard\n(loss per step, AUC per epoch),\nbest.pt, metrics.json,\n"
        "predictions.csv, training log", TRN)

    # 5. evaluation
    box(ax, 0.03, 0.01, 0.30, 0.085, "New test sets (labels known):\nNORVIA, multi-font, unseen font (Lato)\nAUC, false alarms, caught per tamper", EVA)
    box(ax, 0.36, 0.01, 0.30, 0.085, "OLD NORVIA test set (206 pairs): like-for-like\nwith exp012_raw (raw, 4K data) and\nnorvia_448 (rectified)", EVA)
    box(ax, 0.69, 0.01, 0.28, 0.085, "50 real photos (no labels):\nmean over the 3 references,\ngenuine / don't know / tampered", EVA)
    arrow(ax, 0.90, 0.295, 0.86, 0.235)
    arrow(ax, 0.18, 0.135, 0.18, 0.095), arrow(ax, 0.54, 0.135, 0.51, 0.095), arrow(ax, 0.86, 0.135, 0.83, 0.095)

    ax.text(0.5, 1.0, "EXP-013: reference + tampered input (no warping), new 10K dataset, depth-1 siamese model",
            ha="center", fontsize=14, weight="bold")
    out = OUTPUTS / "exp013"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "exp013_diagram.png", dpi=110, bbox_inches="tight")
    print("diagram ->", out / "exp013_diagram.png", "| shapes", feat, comp, "| params", params)


if __name__ == "__main__":
    main()
