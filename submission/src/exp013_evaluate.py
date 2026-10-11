"""
EXP-013 results and intermediate evidence (raw model: reference + tampered input).

Reads the finished tuning runs (e13_*) and writes everything to outputs/exp013/:

  dataset/            what the new dataset contains (counts per set, split, class,
                      tamper type; registration success) and example sheets of
                      reference | tampered input for each tamper type
  training_curves/    per run: training vs validation loss and AUC per epoch
                      (the overfitting picture), plus an overview of all runs
  test_results/       best run on the NEW test sets: AUC, false alarms, share caught
                      per tamper (whole vs one-part stretch), three-way decisions
  old_test_set/       best run on the OLD NORVIA test set (data/synth), where
                      exp012_raw (raw, 4K data) and norvia_448 (rectified) were
                      measured: the only like-for-like comparison of the three
  real_photos/        best run on the 50 real photos, next to the rectified models
  results.md          all numbers in one place

Run: python src/exp013_evaluate.py
"""
import csv
import json

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import OUTPUTS, ROOT, auc, list_images  # noqa: E402
from infer_real import load_run  # noqa: E402
from train_pairs import load_rows, to_tensor  # noqa: E402

OUT = OUTPUTS / "exp013"
DATA = ROOT / "data" / "synth_v2"
TAMPERS = ["stretch", "recolour", "respace", "glyph_swap", "font_swap", "overlap", "cut"]


def band(p, m):
    return "tampered" if p >= m["threshold_5pct_false_alarm"] else ("genuine" if p <= m["threshold_low"] else "don't know")


def frac(sel):
    return f"{sum(sel)}/{len(sel)}"


# ---------------------------------------------------------------- dataset
def dataset_evidence():
    out = OUT / "dataset"
    out.mkdir(parents=True, exist_ok=True)
    labels = list(csv.DictReader(open(DATA / "labels.csv")))
    reg = {r["id"]: r["registered"] == "True" for r in csv.DictReader(open(DATA / "registration.csv"))}
    lines = ["# EXP-013 dataset (data/synth_v2)", "",
             f"Generated: {len(labels)} samples. Usable (logo found in the photo, so a tampered crop exists): "
             f"{sum(reg.values())}.", "",
             "| set | split | genuine | tampered | usable |", "|---|---|---|---|---|"]
    for s in ("D2", "D1"):
        for sp in ("train", "val", "test", "test_unseen_font"):
            sel = [r for r in labels if r["set"] == s and r["split"] == sp]
            if sel:
                lines.append(f"| {s} | {sp} | {sum(r['label'] == '0' for r in sel)} | {sum(r['label'] == '1' for r in sel)} | "
                             f"{sum(reg[r['id']] for r in sel)} |")
    lines += ["", "| set | " + " | ".join(["none"] + TAMPERS) + " |", "|---|" + "---|" * (len(TAMPERS) + 1)]
    for s in ("D2", "D1"):
        lines.append(f"| {s} | " + " | ".join(str(sum(r["set"] == s and r["tamper"] == t for r in labels))
                                            for t in ["none"] + TAMPERS) + " |")
    lines += ["", "Registration success per tamper (the tampered crop needs the logo to be found):", "",
              "| set | " + " | ".join(["none"] + TAMPERS) + " |", "|---|" + "---|" * (len(TAMPERS) + 1)]
    for s in ("D2", "D1"):
        lines.append(f"| {s} | " + " | ".join(frac([reg[r["id"]] for r in labels if r["set"] == s and r["tamper"] == t])
                                            for t in ["none"] + TAMPERS) + " |")
    fonts = {}
    for r in labels:
        if r["set"] == "D1":
            fonts[r["font"]] = fonts.get(r["font"], 0) + 1
    lines += ["", "Multi-font samples per font: " + ", ".join(f"{k} {v}" for k, v in sorted(fonts.items(), key=lambda x: -x[1]))]
    (out / "dataset_stats.md").write_text("\n".join(lines) + "\n")

    # example sheet: for each set, one genuine + one of each tamper: reference | tampered input
    for s in ("D2", "D1"):
        tiles = []
        for t in ["none"] + TAMPERS:
            r = next((r for r in labels if r["set"] == s and r["tamper"] == t and reg[r["id"]]), None)
            if r is None:
                continue
            ref = cv2.imread(str(DATA / s / f"{r['id']}_ref_raw.png"))
            cand = cv2.imread(str(DATA / s / f"{r['id']}_cand_raw.png"))
            pair = np.hstack([ref, np.full((ref.shape[0], 6, 3), 255, np.uint8), cand])
            bar = np.full((30, pair.shape[1], 3), 255, np.uint8)
            text = "genuine" if t == "none" else f"{t}: {r['target']} {r['detail']}"
            cv2.putText(bar, f"{r['id']} {r['font']} | {text}"[:90], (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            tiles.append(np.vstack([bar, pair, np.full((8, pair.shape[1], 3), 255, np.uint8)]))
        head = np.full((40, tiles[0].shape[1], 3), 255, np.uint8)
        cv2.putText(head, f"{s}: REFERENCE input (left) | TAMPERED input (right, the image being checked)",
                    (6, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
        cv2.imwrite(str(out / f"examples_{s}.jpg"), np.vstack([head] + tiles))


# ---------------------------------------------------------------- curves
def training_curves(runs):
    out = OUT / "training_curves"
    out.mkdir(parents=True, exist_ok=True)
    fig_all, ax_all = plt.subplots(figsize=(9, 5))
    for k, name in enumerate(runs):
        m = json.loads((OUTPUTS / "exp007" / name / "metrics.json").read_text())
        h = m["history"]
        ep = [r["epoch"] for r in h]
        tr = next(k_ for k_ in h[0] if k_.startswith("train_") and k_.endswith("/auc"))[:-4]
        va = next(k_ for k_ in h[0] if k_.startswith("val_") and k_.endswith("/auc"))[:-4]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
        a1.plot(ep, [r[f"{tr}/bce"] for r in h], "o-", label="training pairs")
        a1.plot(ep, [r[f"{va}/bce"] for r in h], "o-", label="validation pairs")
        a1.set_title("yes/no loss (BCE) per epoch: validation going UP while training goes down = overfitting", fontsize=9)
        a2.plot(ep, [r[f"{tr}/auc"] for r in h], "o-", label="training AUC")
        a2.plot(ep, [r[f"{va}/auc"] for r in h], "o-", label="validation AUC")
        a2.axvline(m["best_epoch"], color="grey", ls="--", label=f"best epoch {m['best_epoch']}")
        a2.set_title(f"AUC per epoch; best validation {m['best_val_auc']:.3f}", fontsize=9)
        for a in (a1, a2):
            a.set_xlabel("epoch"), a.grid(alpha=0.3), a.legend(fontsize=8)
        fig.suptitle(f"{name}: {json.dumps({k_: m['config'][k_] for k_ in ('weight_decay', 'dropout', 'loss', 'label_smoothing', 'augment', 'geo_augment', 'seed') if k_ in m['config']})}", fontsize=9)
        fig.tight_layout(), fig.savefig(out / f"{name}.png", dpi=90), plt.close(fig)
        ax_all.plot(ep, [r[f"{va}/auc"] for r in h], "o-", ms=3, label=name, color=f"C{k}")
        ax_all.plot(ep, [r[f"{tr}/auc"] for r in h], ":", color=f"C{k}")
    ax_all.set_title("All EXP-013 runs: validation AUC (solid) and training AUC (dotted)")
    ax_all.set_xlabel("epoch"), ax_all.set_ylabel("AUC"), ax_all.grid(alpha=0.3), ax_all.legend(fontsize=7)
    fig_all.tight_layout(), fig_all.savefig(out / "overview.png", dpi=100), plt.close(fig_all)


# ---------------------------------------------------------------- new test sets
def new_test(best, m, labels):
    rows = [r for r in csv.DictReader(open(OUTPUTS / "exp007" / best / "predictions.csv")) if r["split"].startswith("test")]
    hi = m["threshold_5pct_false_alarm"]
    res = {}
    # split names in predictions.csv carry the sets: test_D2, test_D1, test_unseen_font_D1
    for name, keep in (("NORVIA test", lambda r: r["split"] == "test_D2"),
                       ("multi-font test", lambda r: r["split"] == "test_D1"),
                       ("unseen font test", lambda r: r["split"] == "test_unseen_font_D1")):
        sel = [r for r in rows if keep(r)]
        g = [float(r["prob_tampered"]) for r in sel if r["label"] == "0"]
        t = [float(r["prob_tampered"]) for r in sel if r["label"] == "1"]
        d = {"n": f"{len(g)} genuine / {len(t)} tampered", "AUC": f"{auc(t, g):.3f}",
             "false alarms": frac([p >= hi for p in g]), "caught": frac([p >= hi for p in t])}
        for tt in TAMPERS:
            d[tt] = frac([float(r["prob_tampered"]) >= hi for r in sel if r["tamper"] == tt])
        d["stretch whole"] = frac([float(r["prob_tampered"]) >= hi for r in sel if r["tamper"] == "stretch" and labels[r["id"]]["target"] == "whole"])
        d["stretch one part"] = frac([float(r["prob_tampered"]) >= hi for r in sel if r["tamper"] == "stretch" and labels[r["id"]]["target"] != "whole"])
        d["genuine pairs: genuine / don't know / tampered"] = " / ".join(str(sum(band(p, m) == b for p in g)) for b in ("genuine", "don't know", "tampered"))
        res[name] = d
    return res


# ---------------------------------------------------------------- old NORVIA test set
@torch.no_grad()
def score_pairs(model, m, rows, device):
    size = tuple(m["input_size"])
    probs = []
    for i in range(0, len(rows), 32):
        chunk = rows[i:i + 32]
        ref = torch.stack([to_tensor(cv2.imread(str(r["ref_path"])), size) for r in chunk]).to(device)
        cand = torch.stack([to_tensor(cv2.imread(str(r["cand_path"])), size) for r in chunk]).to(device)
        logit, _ = model(ref, cand)
        probs += torch.sigmoid(logit.float()).cpu().tolist()
    return probs


def old_test(best, model, m, device):
    """Same 206 NORVIA test pairs that exp012_raw and norvia_448 were measured on."""
    old = ROOT / "data" / "synth"
    rows, _ = load_rows(old, {"test"}, {"D2"}, raw=True)
    probs = score_pairs(model, m, rows, device)
    old_labels = {r["id"]: r for r in csv.DictReader(open(old / "labels.csv"))}
    out = {}
    for run, source in ((best, None), ("exp012_raw", "exp012_raw"), ("norvia_448", "norvia_448")):
        if source is None:
            p = {r["id"]: pr for r, pr in zip(rows, probs)}
            thr = m["threshold_5pct_false_alarm"]
        else:
            pm = json.loads((OUTPUTS / "exp007" / source / "metrics.json").read_text())
            p = {r["id"]: float(r["prob_tampered"]) for r in csv.DictReader(open(OUTPUTS / "exp007" / source / "predictions.csv"))
                 if r["split"] == "test_D2"}
            thr = pm["threshold_5pct_false_alarm"]
        ids = [r["id"] for r in rows if r["id"] in p]
        g = [p[i] for i in ids if old_labels[i]["label"] == "0"]
        t = [p[i] for i in ids if old_labels[i]["label"] == "1"]
        d = {"pairs": len(ids), "AUC": f"{auc(t, g):.3f}", "false alarms": frac([x >= thr for x in g]),
             "caught": frac([x >= thr for x in t])}
        for tt in TAMPERS:
            d[tt] = frac([p[i] >= thr for i in ids if old_labels[i]["tamper"] == tt])
        out[run] = d
    return out


# ---------------------------------------------------------------- real photos
@torch.no_grad()
def real_photos(best, model, m, device):
    out = OUT / "real_photos"
    out.mkdir(parents=True, exist_ok=True)
    size = tuple(m["input_size"])
    refs = torch.stack([to_tensor(cv2.imread(str(OUTPUTS / "exp012" / f"reference_{i}_raw.png")), size)
                        for i in (1, 2, 3)]).to(device)
    others = {}
    for run, f in (("sd_depth1_more", OUTPUTS / "exp007" / "evidence" / "real_photos" / "real_predictions_sd_depth1_more.csv"),
                   ("norvia_448", OUTPUTS / "exp012" / "real_comparison.csv"), ("exp012_raw", OUTPUTS / "exp012" / "real_comparison.csv")):
        for r in csv.DictReader(open(f)):
            key = "decision" if run == "sd_depth1_more" else f"{run}_decision"
            others.setdefault(r["image"], {})[run] = r[key].split(" +")[0]
    rows = []
    for path in list_images():
        crop = cv2.imread(str(OUTPUTS / "exp012" / "real_crops" / f"{path.stem}.png"))
        if crop is None:
            rows.append({"image": path.stem, "p": "", "decision": "can't register", **others.get(path.stem, {})})
            continue
        cand = to_tensor(crop, size)[None].to(device).expand(3, -1, -1, -1)
        logit, _ = model(refs, cand)
        p = float(torch.sigmoid(logit.float()).mean())
        rows.append({"image": path.stem, "p": round(p, 4), "decision": band(p, m), **others.get(path.stem, {})})
    with open(out / f"real_predictions_{best}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "p", "decision", "sd_depth1_more", "norvia_448", "exp012_raw"])
        w.writeheader()
        w.writerows(rows)
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    best_info = json.loads((OUT / "best_setting.json").read_text())
    best = best_info["best_run"]
    runs = sorted(p.parent.name for p in (OUTPUTS / "exp007").glob("e13_*/metrics.json"))
    labels = {r["id"]: r for r in csv.DictReader(open(DATA / "labels.csv"))}

    dataset_evidence()
    training_curves(runs)
    model, m = load_run(best, device)
    new = new_test(best, m, labels)
    old = old_test(best, model, m, device)
    real = real_photos(best, model, m, device)

    def table(title, d):
        cols = list(d)
        keys = list(d[cols[0]])
        return [f"### {title}", "", "| | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)] + \
               [f"| {k} | " + " | ".join(str(d[c][k]) for c in cols) + " |" for k in keys] + [""]

    dec = [r["decision"] for r in real]
    lines = ["# EXP-013 results (generated by `src/exp013_evaluate.py`)", "",
             f"Best run (chosen on validation): **{best}**, settings {json.dumps(best_info['settings'])}, "
             f"val AUC {best_info['val_auc']:.3f}. Tuning: `experiments/EXP-013_tuning_log.md`.", ""]
    lines += table("New test sets (data/synth_v2)", new)
    lines += table("OLD NORVIA test set: the like-for-like comparison (same 206 pairs for all three)", old)
    lines += ["### Real photos (no ground truth)", "",
              f"{best}: " + ", ".join(f"{b} {dec.count(b)}" for b in ("tampered", "don't know", "genuine", "can't register")),
              "tampered: " + " ".join(r["image"] for r in real if r["decision"] == "tampered"), ""]
    (OUT / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
