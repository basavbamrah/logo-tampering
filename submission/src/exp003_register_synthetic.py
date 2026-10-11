"""
F12 fix: rectify the synthetic NORVIA photos the same way as the real ones.

Why: the generator first made the candidate crops with the camera pose it knew
(plus a little random error). Real photos don't come with a known pose; we
line them up with SIFT + RANSAC + ECC (EXP-001). That registration:
  - has its own small error (~1 px), and
  - "absorbs" a stretch of the whole logo: if the whole logo is 5 % wider,
    the fitted transform simply includes that 5 %, and the rectified crop looks
    normal again.
A model should learn from crops made exactly like the real ones, so here we run
the EXP-001 code on every synthetic D2 photo and save `<id>_cand_reg.png`.

Photos that fail to register get no crop (in the real pipeline they would be
reported as "can't register"); they are listed in registration_D2.csv.

Run: python src/exp003_register_synthetic.py [--data data/synth] [--full]

--full saves crops at the full logo-frame size (920 x 260) as <id>_cand_reg_full.png,
plus <id>_ref_full.png: the full-size version of the reference used for that
sample (found by matching the saved half-size reference against the 3 originals).
"""
import argparse
import csv
from multiprocessing import Pool

import cv2
import numpy as np

from common import ROOT, load_reference
from exp001_register import fit_homography, refine_with_ecc, to_gray

SAVE_W, SAVE_H = 460, 130


def which_reference(half_ref, refs_half):
    """Index (1-3) of the original reference whose half-size version matches best."""
    diffs = [np.abs(half_ref.astype(np.float32) - r.astype(np.float32)).mean() for r in refs_half]
    return int(np.argmin(diffs)) + 1


def register_one(args):
    sid, folder, full = args
    cv2.setNumThreads(1)
    ref = load_reference(1)
    ref_gray = to_gray(ref)
    sift = cv2.SIFT_create(nfeatures=5000)
    img = cv2.imread(str(folder / f"{sid}_photo.jpg"))
    H, stats = fit_homography(ref_gray, to_gray(img), sift)
    row = {"id": sid, "registered": H is not None, "inliers": stats["inliers"],
           "reproj_err_px": stats["reproj_err_px"], "ecc_corr": np.nan, "ref_index": ""}
    if H is not None:
        H, ecc = refine_with_ecc(ref_gray, to_gray(img), H)
        crop = cv2.warpPerspective(img, np.linalg.inv(H), (ref.shape[1], ref.shape[0]))
        if full:
            cv2.imwrite(str(folder / f"{sid}_cand_reg_full.png"), crop)
            refs_half = [cv2.resize(load_reference(i), (SAVE_W, SAVE_H), interpolation=cv2.INTER_AREA) for i in (1, 2, 3)]
            idx = which_reference(cv2.imread(str(folder / f"{sid}_ref.png")), refs_half)
            cv2.imwrite(str(folder / f"{sid}_ref_full.png"), load_reference(idx))
            row["ref_index"] = idx
        else:
            cv2.imwrite(str(folder / f"{sid}_cand_reg.png"),
                        cv2.resize(crop, (SAVE_W, SAVE_H), interpolation=cv2.INTER_AREA))
        row["ecc_corr"] = ecc
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synth")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--full", action="store_true", help="save full-size (920x260) crops and references")
    args = ap.parse_args()
    data = ROOT / args.data
    labels = [r for r in csv.DictReader(open(data / "labels.csv")) if r["set"] == "D2"]

    with Pool(args.workers) as pool:
        rows = pool.map(register_one, [(r["id"], data / "D2", args.full) for r in labels], chunksize=8)

    out_csv = data / ("registration_D2_full.csv" if args.full else "registration_D2.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # how many failed, per tamper type
    by_id = {r["id"]: r for r in rows}
    print(f"registered {sum(r['registered'] for r in rows)}/{len(rows)} synthetic NORVIA photos")
    for tamper in sorted({r["tamper"] for r in labels}):
        sel = [by_id[r["id"]] for r in labels if r["tamper"] == tamper]
        ok = sum(s["registered"] for s in sel)
        print(f"  {tamper:11s} {ok:4d}/{len(sel)} registered")


if __name__ == "__main__":
    main()
