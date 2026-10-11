"""
EXP-014: a CONSISTENT dataset, every tampered input made by real registration.

The best model so far (sd_depth1_more) trained on two kinds of crop: NORVIA
crops made by registration (like real photos) and multi-font crops made with
the known camera pose (their photos were not kept). data/synth_v2 keeps the
photos of both sets, so here every sample gets the same treatment as a real
photo: the full EXP-001 pipeline (SIFT + RANSAC + ECC with the out-of-photo
fix), then the photo is warped onto the 920 x 260 logo frame and saved at
460 x 130 like all earlier rectified crops.

Template, as in exp013_crops.py: NORVIA -> reference/ref_1.jpg (as for real
photos); multi-font -> that sample's own clean logo (<id>_ref_full.png).
Only samples that registered in exp013_crops.py are used (registration.csv),
so EXP-013 and EXP-014 use the same samples.

Writes (new files only): data/synth_v2/<set>/<id>_cand_reg.png and
data/synth_v2/registration_rect.csv (id, set, inliers, ECC score).
Run: python src/exp014_rectify.py
"""
import csv
from multiprocessing import get_context
from pathlib import Path

import cv2
import numpy as np

from common import ROOT, load_reference
from exp001_register import fit_homography, refine_with_ecc, to_gray
from exp003_make_data import SAVE_H, SAVE_W

DATA = ROOT / "data" / "synth_v2"


def one_sample(job):
    sid, kind = job
    cv2.setNumThreads(1)
    folder = DATA / kind
    template = load_reference(1) if kind == "D2" else cv2.imread(str(folder / f"{sid}_ref_full.png"))
    t_gray = to_gray(template)
    img = cv2.imread(str(folder / f"{sid}_photo.jpg"))
    img_gray = to_gray(img)
    H, stats = fit_homography(t_gray, img_gray, cv2.SIFT_create(nfeatures=5000))
    if H is None:                       # should not happen: these samples registered before
        return {"id": sid, "set": kind, "ok": False, "inliers": stats["inliers"], "ecc_corr": ""}
    H, ecc = refine_with_ecc(t_gray, img_gray, H)
    h, w = t_gray.shape
    crop = cv2.warpPerspective(img, np.linalg.inv(H), (w, h))
    cv2.imwrite(str(folder / f"{sid}_cand_reg.png"), cv2.resize(crop, (SAVE_W, SAVE_H), interpolation=cv2.INTER_AREA))
    return {"id": sid, "set": kind, "ok": True, "inliers": stats["inliers"], "ecc_corr": ecc}


def main():
    jobs = [(r["id"], r["set"]) for r in csv.DictReader(open(DATA / "registration.csv")) if r["registered"] == "True"]
    with get_context("spawn").Pool(15) as pool:        # spawn: forking after OpenCV threads can hang; 15 of 16 cores
        rows = pool.map(one_sample, jobs, chunksize=16)
    with open(DATA / "registration_rect.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    bad = [r["id"] for r in rows if not r["ok"]]
    ecc = [float(r["ecc_corr"]) for r in rows if r["ok"] and r["ecc_corr"] not in ("", None) and not np.isnan(float(r["ecc_corr"]))]
    print(f"rectified {len(rows) - len(bad)}/{len(rows)}; not registered this time: {bad}; median ECC {np.median(ecc):.3f}")


if __name__ == "__main__":
    main()
