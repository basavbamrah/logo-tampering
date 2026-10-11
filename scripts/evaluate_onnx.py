"""
Check that the ONNX model gives the same accuracy as the trained PyTorch model.

It scores every test pair with ONNX Runtime and reports, per test set:
  AUC                  chance that a random tampered pair scores above a random genuine one
  false alarms         genuine pairs flagged (p >= the tampered threshold from validation, 0.493)
  caught               tampered pairs flagged at the same threshold, also per tamper type
  ONNX vs PyTorch      largest |p_onnx - p_pytorch| and how many decisions differ
  decision bands       how many pairs land in TAMPERED / GENUINE / DON'T KNOW and how often
                       each band is right: this is the confidence shown with every decision

Data:
  default              test_data/norvia_test (bundled): the shared 206-pair NORVIA test set.
                       PyTorch's scores are stored in its labels.csv, so no PyTorch is needed.
  --data DIR           a dataset folder of the main repository, e.g. ../data/synth_v2: its test
                       sets (NORVIA, multi-font, unseen font), aligned crops only. Add --pytorch
                       to score the same pairs with best.pt as well (needs torch).

Run (from the submission folder):
    python scripts/evaluate_onnx.py
    python scripts/evaluate_onnx.py --data ../data/synth_v2 --pytorch --out model/onnx_check_full.md \
        --save-confidence model/confidence.json
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

SUB = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SUB / "app"))
import pipeline as P     # noqa: E402  (same preprocessing and ONNX session as the app)

TAMPERS = ["stretch", "recolour", "respace", "glyph_swap", "font_swap", "overlap", "cut"]
SET_NAMES = {"D2": "NORVIA", "D1": "multi-font"}
GROUP_ORDER = ["NORVIA (shared test, 206 pairs)", "NORVIA", "multi-font", "unseen font (Lato)"]


def auc(pos, neg):
    """Mann-Whitney form of the ROC AUC (ties count half)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(wins / (len(pos) * len(neg)))


def bundled_pairs(folder):
    rows = list(csv.DictReader(open(folder / "labels.csv")))
    for r in rows:
        r["ref_path"], r["cand_path"], r["group"] = folder / r["ref"], folder / r["cand"], "NORVIA (shared test, 206 pairs)"
    return rows


def repo_pairs(folder):
    """Test pairs of a main-repository dataset, aligned crops only (as in EXP-014)."""
    if (folder / "registration.csv").exists():             # data/synth_v2: both sets aligned
        reg_file, sets = folder / "registration.csv", {"D1", "D2"}
    else:                                                   # data/synth: only NORVIA was aligned
        reg_file, sets = folder / "registration_D2.csv", {"D2"}
    registered = {r["id"] for r in csv.DictReader(open(reg_file)) if r["registered"] == "True"}
    rows = []
    for r in csv.DictReader(open(folder / "labels.csv")):
        if not r["split"].startswith("test") or r["set"] not in sets or r["id"] not in registered:
            continue
        r["ref_path"] = folder / r["set"] / f"{r['id']}_ref.png"
        r["cand_path"] = folder / r["set"] / f"{r['id']}_cand_reg.png"
        r["group"] = "unseen font (Lato)" if r["split"] == "test_unseen_font" else SET_NAMES[r["set"]]
        rows.append(r)
    return rows


def score_onnx(rows, batch=64):
    out = []
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        ref = np.stack([P.to_input(cv2.imread(str(r["ref_path"]))) for r in chunk]).astype(np.float32)
        cand = np.stack([P.to_input(cv2.imread(str(r["cand_path"]))) for r in chunk]).astype(np.float32)
        logit, _, _ = P.session().run(None, {"reference": ref, "tampered": cand})
        out += (1 / (1 + np.exp(-logit))).tolist()
    return np.array(out)


def score_pytorch(rows, batch=64):
    import torch
    sys.path.insert(0, str(SUB / "scripts"))
    from models import build
    torch.manual_seed(0)
    metrics = json.loads((SUB / "model" / "metrics.json").read_text())
    model = build("siamese_diff", pretrained=False, **metrics["model_options"])
    model.load_state_dict(torch.load(SUB / "model" / "best.pt", map_location="cpu"))
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), batch):
            chunk = rows[i:i + batch]
            ref = torch.from_numpy(np.stack([P.to_input(cv2.imread(str(r["ref_path"]))) for r in chunk]))
            cand = torch.from_numpy(np.stack([P.to_input(cv2.imread(str(r["cand_path"]))) for r in chunk]))
            out += torch.sigmoid(model(ref, cand)[0]).tolist()
    return np.array(out)


def bands(p, tampered):
    """Share of pairs in each decision band and how often the band is right (labels known)."""
    hi, lo = P.INFO["threshold_tampered"], P.INFO["threshold_genuine"]
    t, g = p >= hi, p <= lo
    d = ~t & ~g
    return {"pairs": int(len(p)), "share_tampered_in_data": float(tampered.mean()),
            "tampered_band": {"pairs": int(t.sum()), "right": float(tampered[t].mean())},
            "genuine_band": {"pairs": int(g.sum()), "right": float((~tampered[g]).mean())},
            "dont_know_band": {"pairs": int(d.sum()), "share_tampered": float(tampered[d].mean())}}


def frac(flags):
    return f"{int(np.sum(flags))}/{len(flags)}"


def report(rows, p_onnx, p_torch):
    t = P.INFO["threshold_tampered"]
    groups = [g for g in GROUP_ORDER if any(r["group"] == g for r in rows)]
    lines = ["| | " + " | ".join(groups) + " |", "|---|" + "---|" * len(groups)]
    table = {}
    for g in groups:
        idx = np.array([r["group"] == g for r in rows])
        sub = [r for r, k in zip(rows, idx) if k]
        p = p_onnx[idx]
        lab = np.array([r["label"] == "1" for r in sub])
        col = {"pairs": f"{(~lab).sum()} genuine / {lab.sum()} tampered",
               "AUC (ONNX)": f"{auc(p[lab], p[~lab]):.3f}"}
        if p_torch is not None:
            q = p_torch[idx]
            col["AUC (PyTorch)"] = f"{auc(q[lab], q[~lab]):.3f}"
        col["false alarms"] = frac(p[~lab] >= t)
        col["caught"] = frac(p[lab] >= t)
        for kind in TAMPERS:
            col[kind] = frac(np.array([pp >= t for pp, r in zip(p, sub) if r["tamper"] == kind]))
        b = bands(p, lab)
        col["TAMPERED band: pairs, right"] = f"{b['tampered_band']['pairs']}, {b['tampered_band']['right']:.1%}"
        col["GENUINE band: pairs, right"] = f"{b['genuine_band']['pairs']}, {b['genuine_band']['right']:.1%}"
        col["DON'T KNOW band: pairs, share tampered"] = f"{b['dont_know_band']['pairs']}, {b['dont_know_band']['share_tampered']:.1%}"
        if p_torch is not None:
            col["max \\|p ONNX − p PyTorch\\|"] = f"{np.abs(p - q).max():.1e}"
            col["decisions that differ"] = f"{int(((p >= t) != (q >= t)).sum())}/{len(p)}"
        table[g] = col
    for key in table[groups[0]]:
        lines.append(f"| {key} | " + " | ".join(table[g][key] for g in groups) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="main-repository dataset folder (default: the bundled test set)")
    ap.add_argument("--pytorch", action="store_true", help="also score with best.pt (needs torch)")
    ap.add_argument("--out", default=None, help="also write the table to this markdown file")
    ap.add_argument("--save-confidence", default=None,
                    help="write the band right-rates (all pairs together and per test set) to this JSON file")
    args = ap.parse_args()
    np.random.seed(0)

    if args.data:
        rows = repo_pairs(Path(args.data).resolve())
        source = f"{args.data} (test sets, aligned crops)"
    else:
        rows = bundled_pairs(SUB / "test_data" / "norvia_test")
        source = "test_data/norvia_test (bundled)"
    print(f"scoring {len(rows)} pairs from {source} with {P.MODEL_DIR.name}/e14_consistent.onnx ...", flush=True)
    p_onnx = score_onnx(rows)
    if args.pytorch:
        p_torch = score_pytorch(rows)
    elif "p_pytorch" in rows[0]:
        p_torch = np.array([float(r["p_pytorch"]) for r in rows])
    else:
        p_torch = None

    text = (f"ONNX model check on {source}. Threshold for \"tampered\": p ≥ {P.INFO['threshold_tampered']:.3f} "
            f"(from validation).\n\n{report(rows, p_onnx, p_torch)}\n")
    print("\n" + text)
    if args.save_confidence:
        lab = np.array([r["label"] == "1" for r in rows])
        per_set = {g: bands(p_onnx[np.array([r["group"] == g for r in rows])],
                            lab[np.array([r["group"] == g for r in rows])]) for g in dict.fromkeys(r["group"] for r in rows)}
        conf = {"source": f"{source}, labels known", "all_test_sets": bands(p_onnx, lab), "per_test_set": per_set,
                "note": "synthetic test data, about half tampered; on real photos the share of tampered logos is unknown, "
                        "so these rates are a guide, not a guarantee"}
        Path(args.save_confidence).write_text(json.dumps(conf, indent=1))
        print(f"band right-rates written to {args.save_confidence}")
    if args.out:
        Path(args.out).write_text(f"# ONNX vs PyTorch accuracy check\n\nGenerated by `scripts/evaluate_onnx.py`.\n\n{text}")
        print(f"written to {args.out}")


if __name__ == "__main__":
    main()
