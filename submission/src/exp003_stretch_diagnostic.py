"""
Why is the "implied stretch" (EXP-001) so noisy on synthetic genuine photos?

The number comes from splitting the homography into camera pose and the logo's
own x/y scale. That split needs the camera's focal length. EXP-001 estimates
it from each photo ("solved"), which can be shaky. Here we compare three ways of
choosing the focal length on the synthetic NORVIA photos, where we know the truth:
  solved   estimate it per photo (what EXP-001 does)
  oracle   use the true focal length of that synthetic photo (best case)
  median   use one focal length for all photos (what we could do on the real set
           IF the real photos share a camera; not known)
and ask how far the estimated stretch is from the true one (1.0 for genuine).

We also look at how stable the solved focal length is on the REAL photos.

Run: python src/exp003_stretch_diagnostic.py
"""
import csv
import json

import cv2
import numpy as np

from common import OUTPUTS, ROOT, load_reference
from exp001_register import fit_homography, implied_stretch, refine_with_ecc, to_gray

DATA = ROOT / "data" / "synth"
OUT = OUTPUTS / "exp003"


def square_h(img_shape, H):
    """Undo the 'rendered square, then resized' step (as EXP-001 does)."""
    h, w = img_shape[:2]
    return np.diag([h / w, 1.0, 1.0]) @ H, (h, h)


def main():
    ref_gray = to_gray(load_reference(1))
    sift = cv2.SIFT_create(nfeatures=5000)

    # ---- real: how stable is the solved focal length? --------------------------
    real_f = []
    for path in sorted((OUTPUTS / "exp001" / "rectified").glob("*_H.npy")):
        img = cv2.imread(str(ROOT / "candidate_package" / "images" / f"{path.stem[:-2]}.jpg"))
        sq, shape = square_h(img.shape, np.load(path))
        _, f, src = implied_stretch(sq, shape)
        real_f.append((path.stem[:-2], f / shape[0], src))
    solved = np.array([f for _, f, s in real_f if s == "solved"])

    # ---- synthetic: three ways to pick the focal length ------------------------
    labels = [r for r in csv.DictReader(open(DATA / "labels.csv")) if r["set"] == "D2"]
    genuine = [r for r in labels if r["label"] == "0"][:150]
    stretched = [r for r in labels if r["tamper"] == "stretch" and r["target"] == "whole"
                 and r["detail"].startswith("x ")]

    rows = []
    for r in genuine + stretched:
        img = cv2.imread(str(DATA / "D2" / f"{r['id']}_photo.jpg"))
        H, _ = fit_homography(ref_gray, to_gray(img), sift)
        if H is None:
            continue
        H, _ = refine_with_ecc(ref_gray, to_gray(img), H)
        sq, shape = square_h(img.shape, H)
        true = 1.0 if r["label"] == "0" else 1 + float(r["detail"].split()[1].replace("%", "")) / 100
        pose = json.loads(r["pose"])
        f_true = pose["focal"] * shape[0] / 1024
        s_solved, f_solved, src = implied_stretch(sq, shape)
        s_oracle, _, _ = implied_stretch(sq, shape, f_fixed=f_true)
        rows.append({"id": r["id"], "true": true, "solved": s_solved, "oracle": s_oracle,
                     "f_true_rel": f_true / shape[0], "f_solved_rel": f_solved / shape[0],
                     "src": src, "yaw": pose["yaw"], "pitch": pose["pitch"], "H": sq, "shape": shape})

    f_med = float(np.median([r["f_solved_rel"] for r in rows if r["src"] == "solved"]))
    for r in rows:
        r["median"], _, _ = implied_stretch(r["H"], r["shape"], f_fixed=f_med * r["shape"][0])

    lines = ["== real photos: solved focal length / image side ==",
             f"solved for {len(solved)}/{len(real_f)} photos (the rest fell back to a default); "
             f"median {np.median(solved):.2f}, 10-90th pct {np.percentile(solved, 10):.2f}-{np.percentile(solved, 90):.2f}",
             "", "== synthetic: error of the implied stretch (estimate - truth) =="]
    for group, sel in (("genuine", [r for r in rows if r["true"] == 1.0]),
                       ("whole-logo x-stretch", [r for r in rows if r["true"] != 1.0])):
        lines.append(f"{group} (n={len(sel)})")
        for mode in ("solved", "median", "oracle"):
            err = np.array([r[mode] - r["true"] for r in sel])
            mad = 1.4826 * np.median(np.abs(err - np.median(err)))
            lines.append(f"  focal = {mode:7s}: mean {err.mean():+.3f}  std {err.std():.3f}  robust std {mad:.3f}  max abs {np.abs(err).max():.3f}")
    st = [r for r in rows if r["true"] != 1.0]
    if len(st) > 3:
        for mode in ("solved", "median", "oracle"):
            lines.append(f"correlation(true, estimate), focal = {mode}: "
                         f"{np.corrcoef([r['true'] for r in st], [r[mode] for r in st])[0, 1]:.3f}")
    gen = [r for r in rows if r["true"] == 1.0]
    big = [r for r in gen if abs(r["solved"] - 1) > 0.1]
    lines.append(f"genuine photos where the solved estimate is off by >10%: {len(big)}/{len(gen)}; "
                 f"their median |yaw| {np.median([abs(r['yaw']) for r in big]) if big else float('nan'):.1f} deg "
                 f"vs all {np.median([abs(r['yaw']) for r in gen]):.1f} deg; "
                 f"median |pitch| {np.median([abs(r['pitch']) for r in big]) if big else float('nan'):.1f} vs {np.median([abs(r['pitch']) for r in gen]):.1f}")
    text = "\n".join(lines)
    print(text)
    (OUT / "stretch_diagnostic.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
