"""
What the training data did to accuracy: the same model (siamese_diff, depth 1)
trained on four datasets, all scored on the SAME 206 NORVIA test pairs
(data/synth, aligned by registration), each at its own validation threshold.

  original        NORVIA + other fonts, 2 020 pairs                 (sd_depth1)
  NORVIA only     NORVIA + extra NORVIA, 3 982 pairs                (norvia_448, batch 16)
  mixed + extra   NORVIA + other fonts + extra, 7 789 pairs;
                  other-font crops made with the known camera       (sd_depth1_more)
  consistent      every crop aligned like a real photo, 6 770 pairs (e14_consistent)

The first three runs were tested on these pairs during training (test_D2 in
their metrics.json). e14_consistent was trained on data/synth_v2, so it is
scored here on the old pairs, as in src/exp014_evaluate.py.

Writes outputs/exp014/evidence/data_progression.png
Run: python src/exp014_data_chart.py
"""
import json

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import OUTPUTS, ROOT, auc, fix_seeds  # noqa: E402
from infer_real import load_run  # noqa: E402
from train_pairs import load_rows, to_tensor  # noqa: E402

RUNS = [("sd_depth1", "original\n2 020 pairs"),
        ("norvia_448", "NORVIA only\n3 982 pairs"),
        ("sd_depth1_more", "mixed + extra\n7 789 pairs"),
        ("e14_consistent", "consistent\n6 770 pairs")]


@torch.no_grad()
def old_test_scores(name, device):
    """AUC and false alarms of one run on the 206 shared NORVIA test pairs."""
    model, m = load_run(name, device)
    rows, _ = load_rows(ROOT / "data" / "synth", {"test"}, ["D2"])
    probs = []
    for i in range(0, len(rows), 64):
        chunk = rows[i:i + 64]
        ref = torch.stack([to_tensor(cv2.imread(str(r["ref_path"])), tuple(m["input_size"])) for r in chunk]).to(device)
        cand = torch.stack([to_tensor(cv2.imread(str(r["cand_path"])), tuple(m["input_size"])) for r in chunk]).to(device)
        probs += torch.sigmoid(model(ref, cand)[0].float()).cpu().tolist()
    probs = np.array(probs)
    tampered = np.array([r["label"] == "1" for r in rows])
    false_alarms = int((probs[~tampered] >= m["threshold_5pct_false_alarm"]).sum())
    return auc(probs[tampered], probs[~tampered]), false_alarms, int((~tampered).sum())


def main():
    fix_seeds()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    aucs, alarms = [], []
    for name, _ in RUNS:
        a, fa, n = old_test_scores(name, device)
        aucs.append(a)
        alarms.append(fa)
        print(f"{name}: AUC {a:.3f}, false alarms {fa}/{n}")

    labels = [label for _, label in RUNS]
    colours = ["#9e9e9e", "#9e9e9e", "#64b5f6", "#ef6c00"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    bars = ax1.bar(labels, aucs, color=colours)
    ax1.bar_label(bars, [f"{a:.3f}" for a in aucs])
    ax1.set_ylim(0.8, 1.0)
    ax1.set_ylabel("AUC (higher is better)")
    ax1.set_title("Shared NORVIA test (206 pairs): AUC")
    bars = ax2.bar(labels, alarms, color=colours)
    ax2.bar_label(bars, [f"{fa}/105" for fa in alarms])
    ax2.set_ylim(0, max(alarms) + 3)
    ax2.set_ylabel("genuine pairs flagged (lower is better)")
    ax2.set_title("False alarms at the model's own threshold")
    for ax in (ax1, ax2):
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Same model and settings, different training data (the NORVIA-only run used batch 16)", fontsize=12)
    fig.tight_layout()
    out = OUTPUTS / "exp014" / "evidence" / "data_progression.png"
    fig.savefig(out, dpi=110)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
