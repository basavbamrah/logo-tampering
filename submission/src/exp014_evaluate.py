"""
EXP-014 results: e14_consistent (all crops registered, data/synth_v2) vs
sd_depth1_more (the best model so far; NORVIA crops registered, multi-font
crops made with the known camera pose).

Same model and recipe for both (siamese_diff depth 1, 448 x 128, AdamW 3e-4,
wd 1e-4, BCE + 0.3 CE, 15 epochs, batch 32). Each model uses its own thresholds
from its own validation set.

Comparisons (agreed with the candidate, D49):
  1. OLD NORVIA test set: the 206 pairs sd_depth1_more was measured on
  2. NEW test sets (data/synth_v2, every crop registered): BOTH models scored
     on the same pairs, incl. multi-font stretch (whole logo vs one part)
  3. the 50 real photos

Writes outputs/exp014/: results.md, caught_per_tamper.png, real_comparison.csv
Run: python src/exp014_evaluate.py
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
from train_pairs import load_rows, to_tensor  # noqa: E402

OUT = OUTPUTS / "exp014"
NEW, OLD = "e14_consistent", "sd_depth1_more"
TAMPERS = ["stretch", "recolour", "respace", "glyph_swap", "font_swap", "overlap", "cut"]


def band(p, m):
    return "tampered" if p >= m["threshold_5pct_false_alarm"] else ("genuine" if p <= m["threshold_low"] else "don't know")


def frac(sel):
    return f"{sum(sel)}/{len(sel)}"


@torch.no_grad()
def score(model, m, rows, device):
    """p(tampered) for each row (ref_path, cand_path), crops read from disk as in training."""
    size = tuple(m["input_size"])
    out = []
    for i in range(0, len(rows), 64):
        chunk = rows[i:i + 64]
        ref = torch.stack([to_tensor(cv2.imread(str(r["ref_path"])), size) for r in chunk]).to(device)
        cand = torch.stack([to_tensor(cv2.imread(str(r["cand_path"])), size) for r in chunk]).to(device)
        logit, _ = model(ref, cand)
        out += torch.sigmoid(logit.float()).cpu().tolist()
    return out


def summary(rows, probs, m):
    """AUC, false alarms and share caught per tamper type at the model's own threshold."""
    hi = m["threshold_5pct_false_alarm"]
    g = [p for r, p in zip(rows, probs) if r["label"] == "0"]
    t = [p for r, p in zip(rows, probs) if r["label"] == "1"]
    d = {"pairs": f"{len(g)} genuine / {len(t)} tampered", "AUC": f"{auc(t, g):.3f}",
         "false alarms": frac([p >= hi for p in g]), "caught": frac([p >= hi for p in t])}
    for tt in TAMPERS:
        d[tt] = frac([p >= hi for r, p in zip(rows, probs) if r["tamper"] == tt])
    d["stretch whole"] = frac([p >= hi for r, p in zip(rows, probs) if r["tamper"] == "stretch" and r["target"] == "whole"])
    d["stretch one part"] = frac([p >= hi for r, p in zip(rows, probs) if r["tamper"] == "stretch" and r["target"] != "whole"])
    return d


@torch.no_grad()
def real(model, m, device):
    """Same procedure as infer_real.py: rectified crop vs each of the 3 references, mean probability."""
    size = tuple(m["input_size"])
    refs = torch.stack([to_tensor(half_size(load_reference(i)), size) for i in (1, 2, 3)]).to(device)
    out = {}
    for path in list_images():
        crop = cv2.imread(str(OUTPUTS / "exp001" / "rectified" / f"{path.stem}.png"))
        if crop is None:
            out[path.stem] = ("", "can't register")
            continue
        cand = to_tensor(half_size(crop), size)[None].to(device).expand(3, -1, -1, -1)
        logit, _ = model(refs, cand)
        p = float(torch.sigmoid(logit.float()).mean())
        out[path.stem] = (round(p, 4), band(p, m))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = {name: load_run(name, device) for name in (NEW, OLD)}

    # 1. old NORVIA test set (rectified crops, as sd_depth1_more was tested)
    old_rows, _ = load_rows(ROOT / "data" / "synth", {"test"}, {"D2"})
    t1 = {name: summary(old_rows, score(mod, m, old_rows, device), m) for name, (mod, m) in models.items()}

    # 2. new test sets, every crop registered; both models on the same pairs
    t2 = {}
    for label, split, sets in (("NORVIA", "test", {"D2"}), ("multi-font", "test", {"D1"}), ("unseen font", "test_unseen_font", {"D1"})):
        rows, _ = load_rows(ROOT / "data" / "synth_v2", {split}, sets)
        for name, (mod, m) in models.items():
            t2[f"{label}: {name}"] = summary(rows, score(mod, m, rows, device), m)

    # 3. real photos
    r3 = {name: real(mod, m, device) for name, (mod, m) in models.items()}
    names = sorted(r3[NEW])
    with open(OUT / "real_comparison.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", f"{NEW}_p", f"{NEW}_decision", f"{OLD}_p", f"{OLD}_decision"])
        for n in names:
            w.writerow([n, *r3[NEW][n], *r3[OLD][n]])

    # chart: caught per tamper on the new NORVIA and multi-font tests, both models
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.2), sharey=True)
    cols = ["stretch whole", "stretch one part"] + TAMPERS[1:]
    for ax, label in zip(axes, ("NORVIA", "multi-font")):
        x = np.arange(len(cols))
        for k, name in enumerate((OLD, NEW)):
            d = t2[f"{label}: {name}"]
            vals = [int(d[c].split("/")[0]) / max(1, int(d[c].split("/")[1])) for c in cols]
            ax.bar(x + (k - 0.5) * 0.38, vals, 0.38, label=f"{name}: AUC {d['AUC']}, false alarms {d['false alarms']}")
        ax.set_xticks(x, [c.replace(" ", "\n").replace("_", "\n") for c in cols], fontsize=8)
        ax.set_title(f"new {label} test (every crop registered)"), ax.grid(axis="y", alpha=0.3), ax.legend(fontsize=7)
    axes[0].set_ylabel("share caught")
    fig.tight_layout(), fig.savefig(OUT / "caught_per_tamper.png", dpi=105), plt.close(fig)

    def table(title, d):
        cols_ = list(d)
        keys = list(d[cols_[0]])
        return [f"### {title}", "", "| | " + " | ".join(cols_) + " |", "|---|" + "---|" * len(cols_)] + \
               [f"| {k} | " + " | ".join(str(d[c][k]) for c in cols_) + " |" for k in keys] + [""]

    lines = ["# EXP-014 results (generated by `src/exp014_evaluate.py`)", "",
             f"{NEW}: every crop registered (data/synth_v2). {OLD}: NORVIA registered, multi-font known-pose (old data). "
             "Same model and recipe; each model uses its own validation thresholds.", ""]
    lines += table("1. Old NORVIA test set (206 pairs, like-for-like)", t1)
    lines += table("2. New test sets, every crop registered (both models on the same pairs)", t2)
    lines += ["### 3. Real photos (no ground truth)", ""]
    for name in (NEW, OLD):
        ds = [d for _, d in r3[name].values()]
        lines.append(f"- **{name}**: " + ", ".join(f"{b} {ds.count(b)}" for b in ("tampered", "don't know", "genuine", "can't register"))
                     + "; tampered: " + " ".join(n for n in names if r3[name][n][1] == "tampered"))
    (OUT / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
