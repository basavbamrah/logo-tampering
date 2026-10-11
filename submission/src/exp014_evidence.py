"""
Evidence pictures for the final model e14_consistent (EXP-014), for the write-up and slides.

  dataset_sample_D2.jpg / _D1.jpg   the final dataset: reference | tampered input (registered,
                                    exactly as the model sees it), one row per tamper type
  training_curve.png                training vs validation loss and AUC per epoch, best epoch marked
  roc_test_sets.png                 ROC on the NORVIA, multi-font and unseen-font test sets
  score_histograms.png              p(tampered) for genuine vs tampered test pairs, with the thresholds
  confusion_kind.png                "which tamper?" head on tampered test pairs
  examples_missed.jpg / examples_false_alarms.jpg / examples_correct.jpg
                                    reference | tampered input | difference heatmap for test pairs

Run: python src/exp014_evidence.py      -> outputs/exp014/evidence/
"""
import csv
import json

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import OUTPUTS, ROOT  # noqa: E402
from infer_real import load_run  # noqa: E402
from models import TAMPER_TYPES  # noqa: E402
from train_pairs import to_tensor  # noqa: E402

RUN = "e14_consistent"
DATA = ROOT / "data" / "synth_v2"
OUT = OUTPUTS / "exp014" / "evidence"
TAMPERS = ["none", "stretch", "recolour", "respace", "glyph_swap", "font_swap", "overlap", "cut"]
TESTS = {"NORVIA test": "test_D2", "multi-font test": "test_D1", "unseen font test": "test_unseen_font_D1"}


def label_bar(width, text, h=26):
    bar = np.full((h, width, 3), 255, np.uint8)
    cv2.putText(bar, text[:110], (5, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return bar


def dataset_sample(labels, registered):
    """One genuine + one of each tamper per set: reference | tampered input as the model sees it."""
    for s in ("D2", "D1"):
        tiles = []
        for t in TAMPERS:
            r = next((r for r in labels if r["set"] == s and r["tamper"] == t and r["id"] in registered), None)
            if r is None:
                continue
            ref = cv2.imread(str(DATA / s / f"{r['id']}_ref.png"))
            cand = cv2.imread(str(DATA / s / f"{r['id']}_cand_reg.png"))
            pair = np.hstack([ref, np.full((ref.shape[0], 6, 3), 255, np.uint8), cand])
            text = "GENUINE" if t == "none" else f"TAMPERED: {t}  {r['target']}  {r['detail']}"
            tiles.append(np.vstack([label_bar(pair.shape[1], f"{r['id']} ({r['font']}) {text}"), pair,
                                    np.full((6, pair.shape[1], 3), 255, np.uint8)]))
        head = label_bar(tiles[0].shape[1], f"Final dataset ({'NORVIA' if s == 'D2' else 'multi-font'}): "
                                            "REFERENCE (left) | TAMPERED input, registered like a real photo (right)", 34)
        cv2.imwrite(str(OUT / f"dataset_sample_{s}.jpg"), np.vstack([head] + tiles))


def training_curve(m):
    h = m["history"]
    ep = [r["epoch"] for r in h]
    tr = next(k for k in h[0] if k.startswith("train_") and k.endswith("/auc"))[:-4]
    va = next(k for k in h[0] if k.startswith("val_") and k.endswith("/auc"))[:-4]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    a1.plot(ep, [r[f"{tr}/bce"] for r in h], "o-", label="training pairs")
    a1.plot(ep, [r[f"{va}/bce"] for r in h], "o-", label="validation pairs")
    a1.set_title("yes/no loss per epoch (validation flat = no harmful overfitting)", fontsize=9)
    a2.plot(ep, [r[f"{tr}/auc"] for r in h], "o-", label="training AUC")
    a2.plot(ep, [r[f"{va}/auc"] for r in h], "o-", label="validation AUC")
    a2.axvline(m["best_epoch"], color="grey", ls="--", label=f"best epoch {m['best_epoch']}")
    a2.set_title(f"AUC per epoch; best validation {m['best_val_auc']:.3f}", fontsize=9)
    for a in (a1, a2):
        a.set_xlabel("epoch"), a.grid(alpha=0.3), a.legend(fontsize=8)
    fig.suptitle(f"{RUN}: siamese_diff depth 1, consistent data (every crop registered)", fontsize=10)
    fig.tight_layout(), fig.savefig(OUT / "training_curve.png", dpi=100), plt.close(fig)


def roc_points(scores, labels):
    order = np.argsort(-scores)
    l = labels[order]
    tpr = np.concatenate([[0], np.cumsum(l) / max(1, l.sum())])
    fpr = np.concatenate([[0], np.cumsum(1 - l) / max(1, (1 - l).sum())])
    return fpr, tpr, float(np.trapezoid(tpr, fpr))


def test_charts(rows, m):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    fig2, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    for (name, split), a2 in zip(TESTS.items(), axes):
        sel = [r for r in rows if r["split"] == split]
        s = np.array([float(r["prob_tampered"]) for r in sel])
        y = np.array([int(r["label"]) for r in sel])
        fpr, tpr, a = roc_points(s, y)
        ax.plot(fpr, tpr, label=f"{name}: AUC {a:.3f}")
        a2.hist(s[y == 0], bins=30, range=(0, 1), alpha=0.6, label="genuine")
        a2.hist(s[y == 1], bins=30, range=(0, 1), alpha=0.6, label="tampered")
        a2.axvline(m["threshold_5pct_false_alarm"], color="red", ls="--", label="'tampered' threshold")
        a2.axvline(m["threshold_low"], color="green", ls="--", label="'genuine' threshold")
        a2.set_title(name, fontsize=9), a2.set_xlabel("p(tampered)"), a2.legend(fontsize=7)
    ax.plot([0, 1], [0, 1], ":", color="grey")
    ax.set_xlabel("false alarm rate"), ax.set_ylabel("share of tampers caught"), ax.legend(fontsize=8), ax.grid(alpha=0.3)
    ax.set_title(f"{RUN}: ROC on the test sets", fontsize=10)
    fig.tight_layout(), fig.savefig(OUT / "roc_test_sets.png", dpi=100), plt.close(fig)
    fig2.tight_layout(), fig2.savefig(OUT / "score_histograms.png", dpi=100), plt.close(fig2)

    tam = [r for r in rows if r["split"] in TESTS.values() and r["label"] == "1"]
    cm = np.zeros((len(TAMPER_TYPES), len(TAMPER_TYPES)), int)
    for r in tam:
        cm[TAMPER_TYPES.index(r["tamper"]), TAMPER_TYPES.index(r["pred_kind"])] += 1
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm, cmap="Blues")
    for i in range(len(TAMPER_TYPES)):
        for j in range(len(TAMPER_TYPES)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(TAMPER_TYPES)), TAMPER_TYPES, rotation=45), ax.set_yticks(range(len(TAMPER_TYPES)), TAMPER_TYPES)
    ax.set_xlabel("predicted (head 2)"), ax.set_ylabel("true tamper")
    ax.set_title(f"'Which tamper?' on tampered test pairs: right {np.trace(cm)}/{cm.sum()}", fontsize=9)
    fig.tight_layout(), fig.savefig(OUT / "confusion_kind.png", dpi=100), plt.close(fig)


@torch.no_grad()
def examples(rows, m, model, device, labels):
    size = tuple(m["input_size"])
    hi = m["threshold_5pct_false_alarm"]
    test = [r for r in rows if r["split"] == "test_D2"]
    groups = {"missed": [r for r in test if r["label"] == "1" and float(r["prob_tampered"]) < hi],
              "false_alarms": [r for r in test if r["label"] == "0" and float(r["prob_tampered"]) >= hi],
              "correct": [r for r in test if r["label"] == "1" and float(r["prob_tampered"]) >= hi][::12]}
    for name, sel in groups.items():
        tiles = []
        for r in sel[:16]:
            folder = DATA / r["set"]
            ref, cand = cv2.imread(str(folder / f"{r['id']}_ref.png")), cv2.imread(str(folder / f"{r['id']}_cand_reg.png"))
            heat = model.heatmap(to_tensor(ref, size)[None].to(device), to_tensor(cand, size)[None].to(device))[0].float().cpu().numpy()
            heat = cv2.resize(heat / (heat.max() + 1e-6), (cand.shape[1], cand.shape[0]))
            heat_img = cv2.addWeighted(cand, 0.5, cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_JET), 0.5, 0)
            row = np.hstack([ref, cand, heat_img])
            lab = labels[r["id"]]
            text = f"{r['id']}  true: {r['tamper']} {lab['target']} {lab['detail']}  p={float(r['prob_tampered']):.2f}  predicted kind: {r['pred_kind']}"
            tiles.append(np.vstack([label_bar(row.shape[1], text), row]))
        if tiles:
            head = label_bar(tiles[0].shape[1], f"{RUN}, NORVIA test, {name.replace('_', ' ')} "
                                                f"({len(sel)} in total): reference | tampered input | where the model sees a difference", 32)
            cv2.imwrite(str(OUT / f"examples_{name}.jpg"), np.vstack([head] + tiles))
        print(f"{name}: {len(sel)}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    labels_list = list(csv.DictReader(open(DATA / "labels.csv")))
    labels = {r["id"]: r for r in labels_list}
    registered = {r["id"] for r in csv.DictReader(open(DATA / "registration_rect.csv")) if r["ok"] == "True"}
    model, m = load_run(RUN, device)
    rows = list(csv.DictReader(open(OUTPUTS / "exp007" / RUN / "predictions.csv")))
    dataset_sample(labels_list, registered)
    training_curve(m)
    test_charts(rows, m)
    examples(rows, m, model, device, labels)
    print("evidence ->", OUT)


if __name__ == "__main__":
    main()
