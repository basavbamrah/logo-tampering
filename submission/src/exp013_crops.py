"""
EXP-013: make the "tampered" input (the image being checked) for every sample
of data/synth_v2:

  <id>_cand_raw.png   nothing warped; a loose upright box around the found
                      logo with ~30 % background (EXP-012 rule), 448 x 192

Only this raw form is made. The candidate decided (D48) not to train the
rectified model again: it is already measured, and what needs proving is
reference + tampered image as they are.

Registration looks for the clean logo of that sample:
  D2 (NORVIA)     reference/ref_1.jpg, exactly as for the real photos
  D1 (multi-font) <id>_ref_full.png, that sample's own clean logo
Registration is used ONLY to find where to cut the box (the template's corners
through the RANSAC homography); the photo is not warped. Samples whose photo
can't be registered get no crop, as a real photo would be "can't register".

Output: the crops next to the photos, and data/synth_v2/registration.csv
(id, set, registered, inliers).
Run: python src/exp013_crops.py [--data data/synth_v2]
"""
import argparse
import csv
from multiprocessing import get_context
from pathlib import Path

import cv2
import numpy as np

from common import ROOT, load_reference
from exp001_register import fit_homography, to_gray
from exp012_make_crops import cut, loose_box


def one_sample(job):
    sid, kind, folder = job
    cv2.setNumThreads(1)
    folder = Path(folder)
    template = load_reference(1) if kind == "D2" else cv2.imread(str(folder / f"{sid}_ref_full.png"))
    t_gray = to_gray(template)
    img = cv2.imread(str(folder / f"{sid}_photo.jpg"))
    img_gray = to_gray(img)
    H, stats = fit_homography(t_gray, img_gray, cv2.SIFT_create(nfeatures=5000))
    row = {"id": sid, "set": kind, "registered": H is not None, "inliers": stats["inliers"],
           "reproj_err_px": stats["reproj_err_px"]}
    if H is None:
        return row

    # where the logo is: the template's corners through the RANSAC homography (same as EXP-012)
    h, w = t_gray.shape
    corners = cv2.perspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2), H)
    cv2.imwrite(str(folder / f"{sid}_cand_raw.png"), cut(img, loose_box(corners.reshape(-1, 2))))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synth_v2")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    data = ROOT / args.data if not Path(args.data).is_absolute() else Path(args.data)
    labels = list(csv.DictReader(open(data / "labels.csv")))
    jobs = [(r["id"], r["set"], str(data / r["set"])) for r in labels]
    with get_context("spawn").Pool(args.workers) as pool:      # spawn: forking after OpenCV threads can hang
        rows = pool.map(one_sample, jobs, chunksize=16)

    with open(data / "registration.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    by_id = {r["id"]: r for r in rows}
    for kind in ("D2", "D1"):
        sel = [r for r in labels if r["set"] == kind]
        print(f"{kind}: registered {sum(by_id[r['id']]['registered'] for r in sel)}/{len(sel)}")
        for t in sorted({r["tamper"] for r in sel}):
            s = [by_id[r["id"]]["registered"] for r in sel if r["tamper"] == t]
            print(f"   {t:11s} {sum(s)}/{len(s)}")


if __name__ == "__main__":
    main()
