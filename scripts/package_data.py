"""
Copy the data this folder needs to run on its own, from the main repository:

  samples/              the 50 photos of the assignment (to try the app)
  samples/other_methods.csv
                        per photo, what the independent reference comparisons of the main
                        pipeline found (outputs/final/final_results.csv): cut-off (EXP-010),
                        difference map (EXP-010), colour check (EXP-005)
  test_data/norvia_test/
      labels.csv        one row per test pair: label, tamper type, image paths,
                        and p_pytorch = the PyTorch model's score for that pair
      cand/<id>.png     the aligned tampered crop (460 x 130, as made in EXP-014)
      ref/<name>.png    the reference crops (identical ones stored once)

The test set is the shared NORVIA test set of EXP-014 / MODEL_COMPARISON.md:
the 206 pairs from data/synth (test split, NORVIA, aligned by registration).
p_pytorch lets scripts/evaluate_onnx.py check the ONNX model against PyTorch
without PyTorch installed.

Run (from the submission folder, needs the main repository + torch):
    python scripts/package_data.py
"""
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

SUB = Path(__file__).resolve().parent.parent
REPO = SUB.parent
sys.path.insert(0, str(SUB / "app"))
from pipeline import to_input        # noqa: E402  (the app's preprocessing, used for every score)
from models import build             # noqa: E402

SYNTH = REPO / "data" / "synth"
OUT = SUB / "test_data" / "norvia_test"


def copy_samples():
    (SUB / "samples").mkdir(exist_ok=True)
    photos = sorted((REPO / "candidate_package" / "images").glob("*.jpg"))
    for p in photos:
        shutil.copy(p, SUB / "samples" / p.name)
    print(f"copied {len(photos)} sample photos to samples/")


def copy_other_methods():
    """The non-model evidence per photo, from the main pipeline's final run (bash run_final.sh)."""
    rows = []
    for r in csv.DictReader(open(REPO / "outputs" / "final" / "final_results.csv")):
        found = [e for e in ("cut-off", "difference map", "colour") if e in r["evidence"]]
        rows.append({"image": r["image"], "aligned": r["registered"], "cut_pct": r["cut_pct"],
                     "difference_map_worst_part_pct": r["shape_worst_part_pct"], "colour_flag": r["hue_flag"],
                     "other_methods_flagging": "; ".join(found) or "none"})
    with open(SUB / "samples" / "other_methods.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"other methods' results for {len(rows)} photos -> samples/other_methods.csv")


def copy_test_set():
    registered = {r["id"] for r in csv.DictReader(open(SYNTH / "registration_D2.csv")) if r["registered"] == "True"}
    rows = [r for r in csv.DictReader(open(SYNTH / "labels.csv"))
            if r["split"] == "test" and r["set"] == "D2" and r["id"] in registered]
    (OUT / "cand").mkdir(parents=True, exist_ok=True)
    (OUT / "ref").mkdir(exist_ok=True)
    out_rows = []
    for r in rows:
        cand = f"cand/{r['id']}.png"
        shutil.copy(SYNTH / "D2" / f"{r['id']}_cand_reg.png", OUT / cand)
        ref_file = SYNTH / "D2" / f"{r['id']}_ref.png"
        ref = f"ref/{hashlib.md5(ref_file.read_bytes()).hexdigest()[:12]}.png"   # most pairs share a reference
        if not (OUT / ref).exists():
            shutil.copy(ref_file, OUT / ref)
        out_rows.append({"id": r["id"], "label": r["label"], "tamper": r["tamper"], "target": r["target"],
                         "magnitude": r["magnitude"], "unit": r["unit"], "ref": ref, "cand": cand})
    return out_rows


@torch.no_grad()
def add_pytorch_scores(rows):
    metrics = json.loads((SUB / "model" / "metrics.json").read_text())
    model = build("siamese_diff", pretrained=False, **metrics["model_options"])
    model.load_state_dict(torch.load(SUB / "model" / "best.pt", map_location="cpu"))
    model.eval()
    for i in range(0, len(rows), 64):
        chunk = rows[i:i + 64]
        ref = torch.from_numpy(np.stack([to_input(cv2.imread(str(OUT / r["ref"]))) for r in chunk]))
        cand = torch.from_numpy(np.stack([to_input(cv2.imread(str(OUT / r["cand"]))) for r in chunk]))
        logit, _ = model(ref, cand)
        for r, p in zip(chunk, torch.sigmoid(logit).tolist()):
            r["p_pytorch"] = f"{p:.6f}"


def main():
    torch.manual_seed(0)
    copy_samples()
    copy_other_methods()
    rows = copy_test_set()
    add_pytorch_scores(rows)
    with open(OUT / "labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    n_ref = len(list((OUT / "ref").glob("*.png")))
    print(f"test set: {len(rows)} pairs ({sum(r['label'] == '1' for r in rows)} tampered), {n_ref} distinct references "
          f"-> {OUT.relative_to(SUB)}")


if __name__ == "__main__":
    main()
