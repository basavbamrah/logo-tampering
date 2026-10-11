"""
EXP-007: run trained pair models on the 50 real photos, against all 3 references.

For every real photo:
  - EXP-001 could not line it up (e.g. 029)       -> "flagged: can't register"
  - otherwise the rectified crop (outputs/exp001/rectified/<image>.png) is
    compared with EACH of the 3 clean references. The three references share
    the same layout (logo at the same position, checked in common.py), so the
    one crop lines up with all three. That gives 3 probabilities per photo:
        p_mean    average of the three   -> used for the decision
        p_min/max lowest/highest         -> how much the reference choice matters
    Decision, with thresholds chosen on the synthetic VALIDATION set of that run:
        p_mean >= threshold_5pct_false_alarm   -> "tampered"
        p_mean <= threshold_low                -> "genuine"
        in between                             -> "don't know"


With several runs, a summary table puts them side by side and counts how many
runs call each photo "tampered".

For siamese_diff runs, --heatmaps saves, for each photo, the crop with a map
of where the model's two feature maps differ most (averaged over the 3
references), plus one contact sheet of all photos.

These thresholds come from synthetic data: on real photos they are a starting
point, not a guarantee.

Run:
  python src/infer_real.py --runs sd_base                    # one run
  python src/infer_real.py --runs sd_base,sd_aug --heatmaps  # several runs + heatmaps
  python src/infer_real.py --best 3                          # the 3 runs with the best val AUC
Output: outputs/exp007/<run>/real_predictions.csv, outputs/exp007/real_summary.csv
"""
import argparse
import csv
import json

import cv2
import numpy as np
import torch

from common import OUTPUTS, fix_seeds, load_reference
from models import TAMPER_TYPES, build
from train_pairs import to_tensor

CUT_PCT = 1.0          # % of reference ink outside the photo (EXP-010) above which a logo counts as cut off
RUNS_DIR = OUTPUTS / "exp007"


def half_size(img):
    # The synthetic crops were saved at 460 x 130; take real crops (and the
    # references) through the same size first so they are resized the same way.
    return cv2.resize(img, (460, 130), interpolation=cv2.INTER_AREA)


def finished_runs():
    runs = []
    for f in RUNS_DIR.glob("*/metrics.json"):
        m = json.loads(f.read_text())
        runs.append((m["best_val_auc"], m["run"]))
    return [name for _, name in sorted(runs, reverse=True)]


def load_run(name, device):
    folder = RUNS_DIR / name
    metrics = json.loads((folder / "metrics.json").read_text())
    model = build(metrics["arch"], pretrained=False, **metrics.get("model_options", {})).to(device)
    model.load_state_dict(torch.load(folder / "best.pt", map_location=device))
    model.eval()
    return model, metrics


def heatmap_tile(crop, heat, scale, label):
    """
    Crop on top, the same crop with the difference map blended in below.
    `scale` is ONE value shared by all photos, so the colours mean the same in
    every tile (red = among the largest differences seen in any photo).
    Note: this is the raw difference between the two feature maps, a rough
    guide to where the model looks, not an exact explanation of its decision.
    """
    heat = cv2.resize(heat, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_LINEAR)
    heat = np.uint8(255 * np.clip(heat / scale, 0, 1))
    blended = cv2.addWeighted(crop, 0.5, cv2.applyColorMap(heat, cv2.COLORMAP_JET), 0.5, 0)
    tile = np.vstack([crop, blended])
    cv2.rectangle(tile, (0, 0), (300, 22), (255, 255, 255), -1)
    cv2.putText(tile, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return tile


def cut_percentages():
    """EXP-010's share of reference ink outside the photo, per image (empty if EXP-010 has not run)."""
    f = OUTPUTS / "exp010" / "diff_scores.csv"
    return {r["image"]: float(r["cut_pct"]) for r in csv.DictReader(open(f))} if f.exists() else {}


@torch.no_grad()
def score_run(name, device, refs, registration, make_heatmaps):
    model, metrics = load_run(name, device)
    size = tuple(metrics["input_size"])
    t_high, t_low = metrics["threshold_5pct_false_alarm"], metrics["threshold_low"]
    full = size[0] > 460                   # full-resolution model: skip the half-size step
    ref_batch = torch.stack([to_tensor(load_reference(i) if full else r, size)
                             for i, r in zip((1, 2, 3), refs)]).to(device)  # 3 references

    rows, heats = [], []
    cut = cut_percentages()
    heat_dir = RUNS_DIR / name / "real_heatmaps"
    if make_heatmaps and hasattr(model, "heatmap"):
        heat_dir.mkdir(exist_ok=True)

    for r in registration:
        image = r["image"]
        if r["registered"] != "True":
            rows.append({"image": image, "p_ref1": "", "p_ref2": "", "p_ref3": "", "p_mean": "",
                         "p_min": "", "p_max": "", "kind": "", "decision": "flagged: can't register"})
            continue
        crop = cv2.imread(str(OUTPUTS / "exp001" / "rectified" / f"{image}.png"))
        if not full:
            crop = half_size(crop)
        cand = to_tensor(crop, size)[None].to(device).expand(3, -1, -1, -1)  # same crop vs each reference
        logit, kind_logits = model(ref_batch, cand)
        p = torch.sigmoid(logit.float()).cpu().numpy()
        kinds = kind_logits.argmax(1).cpu().numpy()
        kind = TAMPER_TYPES[np.bincount(kinds, minlength=len(TAMPER_TYPES)).argmax()]   # majority of 3
        p_mean = float(p.mean())

        if p_mean >= t_high:
            decision = "tampered"
        elif p_mean <= t_low:
            decision = "genuine"
        else:
            decision = "don't know"
        if cut.get(image, 0.0) > CUT_PCT:
            decision += f" + flagged: logo partly outside the photo ({cut[image]:.1f} % of the ink)"

        rows.append({"image": image, "p_ref1": round(float(p[0]), 4), "p_ref2": round(float(p[1]), 4),
                     "p_ref3": round(float(p[2]), 4), "p_mean": round(p_mean, 4),
                     "p_min": round(float(p.min()), 4), "p_max": round(float(p.max()), 4),
                     "kind": kind, "decision": decision})

        if make_heatmaps and hasattr(model, "heatmap"):
            heat = model.heatmap(ref_batch, cand).float().mean(0).cpu().numpy()
            heats.append((image, crop, heat, f"{image}  p={p_mean:.2f}  {kind}"))

    with open(RUNS_DIR / name / "real_predictions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    tiles = []
    if heats:
        scale = float(np.percentile(np.concatenate([h.ravel() for _, _, h, _ in heats]), 99.5))
        for image, crop, heat, label in heats:
            tile = heatmap_tile(cv2.resize(crop, (920, 260)), heat, scale, label)
            cv2.imwrite(str(heat_dir / f"{image}.jpg"), tile)
            tiles.append(cv2.resize(tile, (460, 260)))
    if tiles:
        while len(tiles) % 5:
            tiles.append(np.full_like(tiles[0], 255))
        sheet = np.vstack([np.hstack(tiles[i:i + 5]) for i in range(0, len(tiles), 5)])
        cv2.imwrite(str(RUNS_DIR / name / "real_heatmaps_sheet.jpg"), sheet)

    print(f"\n== {name} ({metrics['arch']}, val AUC {metrics['best_val_auc']:.3f}; "
          f"tampered if p_mean >= {t_high:.3f}, genuine if <= {t_low:.3f})")
    counts = {}
    for row in rows:
        counts[row["decision"].split(" +")[0]] = counts.get(row["decision"].split(" +")[0], 0) + 1
    print("   decisions:", counts)
    for row in sorted(rows, key=lambda x: -(x["p_mean"] if x["p_mean"] != "" else -1))[:12]:
        print(f"   {row['image']}  p_mean {row['p_mean']!s:>6}  (refs {row['p_ref1']}, {row['p_ref2']}, "
              f"{row['p_ref3']})  {row['kind']:10s} {row['decision']}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="", help="comma-separated run names (folders under outputs/exp007)")
    ap.add_argument("--best", type=int, default=0, help="use the N finished runs with the best validation AUC")
    ap.add_argument("--heatmaps", action="store_true", help="save difference heatmaps (siamese_diff only)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    fix_seeds()   # same result on every re-run

    runs = [r for r in args.runs.split(",") if r]
    if args.best:
        runs += [r for r in finished_runs()[:args.best] if r not in runs]
    if not runs:
        raise SystemExit(f"no runs given; finished runs: {finished_runs()}")

    refs = [half_size(load_reference(i)) for i in (1, 2, 3)]
    registration = list(csv.DictReader(open(OUTPUTS / "exp001" / "registration.csv")))
    results = {name: score_run(name, args.device, refs, registration, args.heatmaps) for name in runs}

    # Side-by-side summary over runs.
    summary = []
    for i, r in enumerate(registration):
        row = {"image": r["image"]}
        votes = 0
        for name in runs:
            res = results[name][i]
            row[f"{name}_p"] = res["p_mean"]
            row[f"{name}_decision"] = res["decision"]
            votes += res["decision"].startswith("tampered")
        row["runs_saying_tampered"] = f"{votes}/{len(runs)}"
        summary.append(row)
    with open(RUNS_DIR / "real_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary[0].keys())
        w.writeheader()
        w.writerows(summary)
    print(f"\nsummary over {len(runs)} run(s): {RUNS_DIR / 'real_summary.csv'}")


if __name__ == "__main__":
    main()
