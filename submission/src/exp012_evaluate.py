"""
EXP-012 results: un-straightened input (exp012_raw) vs straightened input (norvia_448).

The two runs differ in ONE thing: what the model gets as the second image.
  norvia_448  the EXP-001 rectified crop (logo warped onto the reference), 448 x 128
  exp012_raw  a loose upright box cut from the photo, nothing warped, 448 x 192
Same model (siamese_diff depth 1), same NORVIA samples and splits, same
optimiser, learning rate, batch, epochs, loss and seed.

What it measures:
  1. synthetic NORVIA test (labels known): AUC, false alarms, share caught per
     tamper type, whole-logo vs one-part stretch, the three-way decision counts
  2. the 50 real photos: score of each model, mean over the 3 references,
     three-way decision with each model's own thresholds
Thresholds come from each run's own validation set (metrics.json), as for every run.

Writes outputs/exp012/: results.md, synthetic_comparison.csv, real_comparison.csv,
caught_per_tamper.png, real_scores.png
Run: python src/exp012_evaluate.py
"""
import csv
import json

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import OUTPUTS, ROOT, auc, list_images, load_reference  # noqa: E402
from infer_real import half_size, load_run  # noqa: E402
from train_pairs import to_tensor  # noqa: E402

OUT = OUTPUTS / "exp012"
RUNS = {"norvia_448": "straightened (control)", "exp012_raw": "un-straightened (EXP-012)"}
TAMPERS = ["stretch", "recolour", "respace", "glyph_swap", "font_swap", "overlap", "cut"]


def share(text):
    """'7/15' -> 0.467 (0 if there are no samples)."""
    a, b = (int(v) for v in text.split("/"))
    return a / b if b else 0.0


def band(p, metrics):
    if p >= metrics["threshold_5pct_false_alarm"]:
        return "tampered"
    if p <= metrics["threshold_low"]:
        return "genuine"
    return "don't know"


def synthetic(run, metrics, labels):
    """Numbers on the synthetic NORVIA test split, from the run's saved predictions."""
    rows = [r for r in csv.DictReader(open(OUTPUTS / "exp007" / run / "predictions.csv"))
            if r["split"].startswith("test") and r["set"] == "D2"]
    hi = metrics["threshold_5pct_false_alarm"]
    gen = [float(r["prob_tampered"]) for r in rows if r["label"] == "0"]
    tam = [float(r["prob_tampered"]) for r in rows if r["label"] == "1"]
    out = {"run": run, "n_genuine": len(gen), "n_tampered": len(tam), "auc": round(auc(tam, gen), 3),
           "false_alarms": f"{sum(p >= hi for p in gen)}/{len(gen)}",
           "caught_all": f"{sum(p >= hi for p in tam)}/{len(tam)}"}
    for t in TAMPERS:
        sel = [float(r["prob_tampered"]) >= hi for r in rows if r["tamper"] == t]
        out[t] = f"{sum(sel)}/{len(sel)}"
    for target, name in (("whole", "stretch_whole"), (None, "stretch_part")):
        sel = [float(r["prob_tampered"]) >= hi for r in rows if r["tamper"] == "stretch"
               and ((labels[r["id"]]["target"] == "whole") == (target == "whole"))]
        out[name] = f"{sum(sel)}/{len(sel)}"
    for b in ("genuine", "don't know", "tampered"):
        out[f"genuine_pairs_{b}"] = sum(band(p, metrics) == b for p in gen)
    return out


@torch.no_grad()
def real(run, model, metrics, device):
    """Mean probability over the 3 references for each real photo, same procedure as infer_real.py."""
    size = tuple(metrics["input_size"])
    if run == "exp012_raw":
        refs = [cv2.imread(str(OUT / f"reference_{i}_raw.png")) for i in (1, 2, 3)]
        crop_of = lambda name: cv2.imread(str(OUT / "real_crops" / f"{name}.png"))
    else:
        refs = [half_size(load_reference(i)) for i in (1, 2, 3)]
        crop_of = lambda name: (lambda c: None if c is None else half_size(c))(
            cv2.imread(str(OUTPUTS / "exp001" / "rectified" / f"{name}.png")))
    ref_batch = torch.stack([to_tensor(r, size) for r in refs]).to(device)
    out = {}
    for path in list_images():
        crop = crop_of(path.stem)
        if crop is None:                       # 029: could not be located / registered
            out[path.stem] = (None, "can't register")
            continue
        cand = to_tensor(crop, size)[None].to(device).expand(3, -1, -1, -1)
        logit, _ = model(ref_batch, cand)
        p = float(torch.sigmoid(logit.float()).mean())
        out[path.stem] = (p, band(p, metrics))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    labels = {r["id"]: r for f in ("data/synth", "data/synth_more")
              for r in csv.DictReader(open(ROOT / f / "labels.csv"))}

    syn, rl, metrics_of = [], {}, {}
    for run in RUNS:
        model, metrics = load_run(run, device)
        metrics_of[run] = metrics
        syn.append(synthetic(run, metrics, labels))
        rl[run] = real(run, model, metrics, device)

    with open(OUT / "synthetic_comparison.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=syn[0].keys())
        writer.writeheader()
        writer.writerows(syn)
    names = sorted(rl["norvia_448"])
    with open(OUT / "real_comparison.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image"] + [f"{r}_{c}" for r in RUNS for c in ("p", "decision")])
        for n in names:
            writer.writerow([n] + [v for r in RUNS for v in (
                "" if rl[r][n][0] is None else round(rl[r][n][0], 4), rl[r][n][1])])

    # chart: share caught per tamper on the synthetic test
    fig, ax = plt.subplots(figsize=(11, 4.2))
    cols = TAMPERS[:1] + ["stretch_whole", "stretch_part"] + TAMPERS[1:]
    x = np.arange(len(cols))
    for k, s in enumerate(syn):
        vals = [share(s[c]) for c in cols]
        ax.bar(x + (k - 0.5) * 0.38, vals, 0.38, label=f"{RUNS[s['run']]}: AUC {s['auc']}, false alarms {s['false_alarms']}")
        for xi, v, txt in zip(x, vals, [s[c] for c in cols]):
            ax.text(xi + (k - 0.5) * 0.38, v + 0.02, txt, ha="center", fontsize=7)
    ax.set_xticks(x, [c.replace("_", "\n") for c in cols])
    ax.set_ylim(0, 1.15), ax.set_ylabel("share caught (synthetic NORVIA test)")
    ax.set_title("EXP-012: straightened vs un-straightened input, same model and settings")
    ax.legend(fontsize=8, loc="lower right"), ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(), fig.savefig(OUT / "caught_per_tamper.png", dpi=110), plt.close(fig)

    # chart: real photo scores, both models
    fig, ax = plt.subplots(figsize=(15, 4.2))
    x = np.arange(len(names))
    for k, run in enumerate(RUNS):
        v = [rl[run][n][0] or 0 for n in names]
        ax.bar(x + (k - 0.5) * 0.4, v, 0.4, label=RUNS[run])
        ax.axhline(metrics_of[run]["threshold_5pct_false_alarm"], ls="--", lw=1,
                   color=f"C{k}", label=f"{run} 'tampered' threshold")
    ax.set_xticks(x, names, rotation=90, fontsize=8), ax.set_ylabel("p(tampered), mean of 3 refs")
    ax.set_title("EXP-012: the 50 real photos (no ground truth)")
    ax.legend(fontsize=8), fig.tight_layout(), fig.savefig(OUT / "real_scores.png", dpi=100), plt.close(fig)

    # markdown summary
    lines = ["# EXP-012 results (generated by `src/exp012_evaluate.py`)", "",
             "## Synthetic NORVIA test (labels known)", "",
             "| | " + " | ".join(RUNS[s["run"]] for s in syn) + " |", "|---|---|---|"]
    for key in ["auc", "false_alarms", "caught_all"] + TAMPERS[:1] + ["stretch_whole", "stretch_part"] + TAMPERS[1:] \
            + ["genuine_pairs_genuine", "genuine_pairs_don't know", "genuine_pairs_tampered"]:
        lines.append(f"| {key} | " + " | ".join(str(s[key]) for s in syn) + " |")
    lines += ["", "## Real photos (no ground truth)", ""]
    for run in RUNS:
        ds = [d for _, d in rl[run].values()]
        lines.append(f"- **{RUNS[run]}**: " + ", ".join(f"{b} {ds.count(b)}" for b in ("tampered", "don't know", "genuine", "can't register"))
                     + "; tampered: " + " ".join(n for n in names if rl[run][n][1] == "tampered"))
    both = [n for n in names if rl["norvia_448"][n][1] == "tampered" and rl["exp012_raw"][n][1] == "tampered"]
    lines.append(f"- tampered by both: {' '.join(both)}")
    for a, b in (("norvia_448", "exp012_raw"), ("exp012_raw", "norvia_448")):
        only = [n for n in names if rl[a][n][1] == "tampered" and rl[b][n][1] != "tampered"]
        lines.append(f"- tampered only by {a}: {' '.join(only) or '-'}")
    (OUT / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
