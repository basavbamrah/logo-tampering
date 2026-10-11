"""
EXP-003 check: do the synthetic NORVIA photos behave like the real ones?

We run the SAME registration code as EXP-001 on the synthetic photos and ask:

  1. Realism: how often does registration work, and how many inliers do we get,
     compared with the 50 real photos?
  2. Noise floor of the "implied stretch" number (EXP-001, doc section 1.3).
     Genuine synthetic photos have no stretch, so the spread of their
     implied stretch is how much that number wobbles on its own. If the real
     photos spread about the same, we can't call a real photo "stretched" on
     that evidence alone.
  3. Sensitivity: for synthetic whole-logo horizontal stretches we know the
     true stretch. Does the estimate follow it?

Output: outputs/exp003/realism.txt and outputs/exp003/stretch_pairs.npy (true vs estimated stretch)
Run:    python src/exp003_check_realism.py [--data data/synth] [--n 150]
"""
import argparse
import csv

import cv2
import numpy as np

from common import OUTPUTS, ROOT, load_reference
from exp001_register import fit_homography, implied_stretch, refine_with_ecc, to_gray

OUT = OUTPUTS / "exp003"


def register_photo(path, ref_gray, sift):
    """Same steps as EXP-001. Returns a dict of numbers, or None if registration failed."""
    img = cv2.imread(str(path))
    H, stats = fit_homography(ref_gray, to_gray(img), sift)
    if H is None:
        return None
    H, ecc = refine_with_ecc(ref_gray, to_gray(img), H)
    h_img, w_img = img.shape[:2]
    # undo the 'rendered square, then resized' step, as EXP-001 does
    square = np.diag([h_img / w_img, 1.0, 1.0]) @ H
    stretch, _, _ = implied_stretch(square, (h_img, h_img))
    return {"inliers": stats["inliers"], "err": stats["reproj_err_px"], "ecc": ecc, "stretch": stretch}


def summary(values):
    v = np.array(values, float)
    v = v[~np.isnan(v)]
    return f"n={len(v)} median={np.median(v):.3f} mean={v.mean():.3f} std={v.std():.3f} min={v.min():.3f} max={v.max():.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synth")
    ap.add_argument("--n", type=int, default=150, help="how many genuine photos to register")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    data = ROOT / args.data
    labels = [r for r in csv.DictReader(open(data / "labels.csv")) if r["set"] == "D2"]
    genuine = [r for r in labels if r["label"] == "0"][:args.n]
    # whole-logo horizontal stretches: detail looks like "x +3.2%"
    stretched = [r for r in labels if r["tamper"] == "stretch" and r["target"] == "whole"
                 and r["detail"].startswith("x ")]

    ref_gray = to_gray(load_reference(1))
    sift = cv2.SIFT_create(nfeatures=5000)

    def run(rows):
        results = []
        for r in rows:
            res = register_photo(data / "D2" / f"{r['id']}_photo.jpg", ref_gray, sift)
            results.append((r, res))
        return results

    gen = run(genuine)
    st = run(stretched)

    real = list(csv.DictReader(open(OUTPUTS / "exp001" / "registration.csv")))
    real_ok = [r for r in real if r["registered"] == "True"]

    lines = []
    ok = [res for _, res in gen if res]
    lines.append("== 1. Does registration behave like on the real photos? ==")
    lines.append(f"real      : registered {len(real_ok)}/{len(real)}  inliers {summary([r['inliers'] for r in real_ok])}")
    lines.append(f"synthetic : registered {len(ok)}/{len(gen)}  inliers {summary([r['inliers'] for r in ok])}")
    lines.append(f"real      reproj err px {summary([r['reproj_err_px'] for r in real_ok])}")
    lines.append(f"synthetic reproj err px {summary([r['err'] for r in ok])}")

    lines.append("\n== 2. Noise floor of the implied stretch (should be 1.0 for genuine) ==")
    lines.append(f"real photos (unknown mix of genuine/tampered): {summary([r['stretch_square_px'] for r in real_ok])}")
    lines.append(f"synthetic GENUINE photos                     : {summary([res['stretch'] for res in ok])}")

    lines.append("\n== 3. Does the estimate follow a known whole-logo stretch? ==")
    xs, ys = [], []
    for r, res in st:
        if res is not None:
            xs.append(1 + float(r["detail"].split()[1].replace("%", "")) / 100)   # true stretch
            ys.append(res["stretch"])                                              # estimated
    lines.append(f"whole-logo x-stretches registered: {len(xs)}/{len(st)}")
    if xs:
        xs, ys = np.array(xs), np.array(ys)
        err = ys - xs
        lines.append(f"estimate - truth: mean {err.mean():+.4f}  std {err.std():.4f}  max abs {np.abs(err).max():.4f}")
        lines.append(f"correlation(true, estimate) = {np.corrcoef(xs, ys)[0, 1]:.3f}")
        small = np.abs(xs - 1) < 0.03
        if small.any():
            lines.append(f"stretches under 3%: {small.sum()} samples, mean estimate - truth {err[small].mean():+.4f}")

    text = "\n".join(lines)
    print(text)
    (OUT / "realism.txt").write_text(text + "\n")
    np.save(OUT / "stretch_pairs.npy", np.array([xs, ys]))


if __name__ == "__main__":
    main()
