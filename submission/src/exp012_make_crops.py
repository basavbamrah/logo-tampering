"""
EXP-012: make UN-straightened crops for the model

Until now the model saw the photo after EXP-001 had warped the logo onto the
reference (the rectified crop). Here it sees the logo as the camera saw it:
tilted, turned and in perspective. Only a loose, upright box is cut out of the
photo; nothing is warped.

How the box is found: the same SIFT + RANSAC step as EXP-001 locates the logo
(the template's 4 corners pushed through H). We take the upright bounding box
of those corners, add 15 % on every side (~30 % background in total), grow it
to the 448 x 192 shape WITHOUT squeezing the content, cut it from the photo
(black where the box leaves the photo) and resize it. The logo keeps whatever
shape the camera and the photo resize gave it.

The straight image (model input 1) is the same kind of box cut from the clean
reference image around its logo, so both inputs show the logo with some paper
around it.

Only NORVIA (D2) samples can be made: the multi-font set has no saved photos.
Samples whose photo failed registration are skipped, exactly as for the
rectified crops, so EXP-012 uses the same samples as the control run norvia_448.

Writes NEW files only (nothing existing is changed):
  data/synth/D2/<id>_cand_raw.png, <id>_ref_raw.png      (and the same in data/synth_more)
  outputs/exp012/real_crops/<id>.png                     real photos, same rule
  outputs/exp012/crop_examples.jpg                       a few examples to check by eye
  outputs/exp012/crops.csv                               box per sample

Run: python src/exp012_make_crops.py
"""
import csv
import multiprocessing as mp

import cv2
import numpy as np

from common import OUTPUTS, REF_LOGO_BOX, REFERENCE_DIR, ROOT, list_images, load_reference
from exp001_register import fit_homography, to_gray
from exp003_register_synthetic import SAVE_H, SAVE_W, which_reference

OUT = OUTPUTS / "exp012"
OUT_W, OUT_H = 448, 192         # the input size agreed for EXP-012
MARGIN = 0.15                   # extra room on every side, as a share of the box size
FOLDERS = ["data/synth", "data/synth_more"]


def loose_box(corners):
    """
    Upright box around the 4 logo corners, 15 % bigger on every side, then
    grown (never shrunk) to the OUT_W:OUT_H shape around the same centre.
    Returns x0, y0, x1, y1 as floats (may reach outside the image).
    """
    (x0, y0), (x1, y1) = corners.min(axis=0), corners.max(axis=0)
    w, h = (x1 - x0) * (1 + 2 * MARGIN), (y1 - y0) * (1 + 2 * MARGIN)
    aspect = OUT_W / OUT_H
    if w / h < aspect:
        w = h * aspect
    else:
        h = w / aspect
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def cut(img, box):
    """Cut the box out of the image (black outside the image) and resize to OUT_W x OUT_H."""
    x0, y0, x1, y1 = box
    # an affine map from the box to the output; warpAffine fills outside with black
    scale = OUT_W / (x1 - x0)
    m = np.float32([[scale, 0, -x0 * scale], [0, scale, -y0 * scale]])
    flags = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.warpAffine(img, m, (OUT_W, OUT_H), flags=flags, borderValue=(0, 0, 0))


def reference_crops():
    """The straight input: the same kind of loose box around the logo in each clean reference image."""
    x0, y0, x1, y1 = REF_LOGO_BOX
    corners = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    out = {}
    for i in (1, 2, 3):
        full = cv2.imread(str(REFERENCE_DIR / f"ref_{i}.jpg"))
        out[i] = cut(full, loose_box(corners))
    return out


def logo_corners(img, ref_gray, sift):
    """Where the logo is in the photo: template corners through the RANSAC homography (no warping)."""
    H, stats = fit_homography(ref_gray, to_gray(img), sift)
    if H is None:
        return None, stats
    h, w = ref_gray.shape
    c = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(c, H).reshape(-1, 2), stats


def one_sample(job):
    sid, folder = job
    cv2.setNumThreads(1)
    ref_gray = to_gray(load_reference(1))
    sift = cv2.SIFT_create(nfeatures=5000)
    img = cv2.imread(str(folder / f"{sid}_photo.jpg"))
    corners, _ = logo_corners(img, ref_gray, sift)
    if corners is None:
        return {"id": sid, "ok": False}
    box = loose_box(corners)
    cv2.imwrite(str(folder / f"{sid}_cand_raw.png"), cut(img, box))
    # which clean reference this sample uses as input 1 (same one as its rectified pair)
    refs_half = [cv2.resize(load_reference(i), (SAVE_W, SAVE_H), interpolation=cv2.INTER_AREA) for i in (1, 2, 3)]
    idx = which_reference(cv2.imread(str(folder / f"{sid}_ref.png")), refs_half)
    cv2.imwrite(str(folder / f"{sid}_ref_raw.png"), REF_CROPS[idx])
    return {"id": sid, "ok": True, "ref_index": idx, "box": [round(float(v), 1) for v in box]}


REF_CROPS = reference_crops()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "real_crops").mkdir(exist_ok=True)

    # ---- synthetic NORVIA photos (only those the rectified pipeline also used) ----
    rows = []
    for folder in FOLDERS:
        data = ROOT / folder
        registered = {r["id"] for r in csv.DictReader(open(data / "registration_D2.csv")) if r["registered"] == "True"}
        jobs = [(r["id"], data / "D2") for r in csv.DictReader(open(data / "labels.csv"))
                if r["set"] == "D2" and r["id"] in registered]
        with mp.get_context("spawn").Pool(8) as pool:      # spawn: forking after OpenCV threads can hang
            done = pool.map(one_sample, jobs, chunksize=16)
        missing = [d["id"] for d in done if not d["ok"]]
        print(f"{folder}: {sum(d['ok'] for d in done)}/{len(jobs)} raw crops made; not located: {missing}")
        rows += [{**d, "folder": folder} for d in done]

    # ---- real photos: same rule ----
    ref_gray = to_gray(load_reference(1))
    sift = cv2.SIFT_create(nfeatures=5000)
    for path in list_images():
        img = cv2.imread(str(path))
        corners, _ = logo_corners(img, ref_gray, sift)
        if corners is None:
            rows.append({"id": path.stem, "ok": False, "folder": "real"})
            continue
        box = loose_box(corners)
        cv2.imwrite(str(OUT / "real_crops" / f"{path.stem}.png"), cut(img, box))
        rows.append({"id": path.stem, "ok": True, "folder": "real", "box": [round(float(v), 1) for v in box]})
    for i, crop in REF_CROPS.items():
        cv2.imwrite(str(OUT / f"reference_{i}_raw.png"), crop)

    with open(OUT / "crops.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["folder", "id", "ok", "ref_index", "box"])
        writer.writeheader()
        writer.writerows(rows)

    # a sheet to check by eye: rectified crop (old input) vs raw crop (new input), plus the reference
    data = ROOT / "data/synth" / "D2"
    picks = [r["id"] for r in rows if r["folder"] == "data/synth" and r["ok"]][:8]
    tiles = []
    for sid in picks:
        old = cv2.resize(cv2.imread(str(data / f"{sid}_cand_reg.png")), (OUT_W, int(OUT_W * 130 / 460)))
        new = cv2.imread(str(data / f"{sid}_cand_raw.png"))
        ref = cv2.imread(str(data / f"{sid}_ref_raw.png"))
        pad = np.full((OUT_H - old.shape[0], OUT_W, 3), 255, np.uint8)
        tiles.append(np.hstack([ref, np.vstack([old, pad]), new]))
    sheet = np.vstack([np.pad(t, ((4, 4), (0, 0), (0, 0)), constant_values=255) for t in tiles])
    bar = np.full((40, sheet.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, "straight input (reference) | OLD input: rectified crop | NEW input: raw crop (EXP-012)",
                (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.imwrite(str(OUT / "crop_examples.jpg"), np.vstack([bar, sheet]))


if __name__ == "__main__":
    main()
