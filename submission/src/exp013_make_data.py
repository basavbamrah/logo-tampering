"""
EXP-013: a new, bigger synthetic dataset (data/synth_v2), kept for good.

What is new compared with data/synth (EXP-003):
  - ~10 000 samples: 5000 NORVIA (D2) + 5000 multi-font (D1); half genuine,
    half tampered (7 tamper types in equal turns), as agreed (D47).
  - The full photo is saved for BOTH sets, so the raw "tampered" input can be
    cut for every sample (the multi-font photos were not kept in EXP-003).
  - The camera no longer mirrors the logo in when it looks past the edge of
    the artwork 
  - Splits: 70 % train / 15 % val / 15 % test; D1 samples in Lato only go to
    "test_unseen_font" (a font never seen in training), as before.

The generator itself is unchanged (src/synth.py): same logos, tampers,
camera, lighting, blur, noise, JPEG. Different seeds from the old datasets.

Files per sample, in data/synth_v2/<set>/:
  <id>_photo.jpg      the synthetic photo (what a phone would give)
  <id>_ref.png        the clean logo at 460 x 130 (kept for compatibility; not used in EXP-013)
  <id>_ref_raw.png    the REFERENCE input, 448 x 192: the clean logo with paper
                      around it, cut by the same rule as the tampered input
  <id>_ref_full.png   D1 only: the clean logo at 920 x 260, the template that
                      registration looks for (D2 uses reference/ref_1.jpg, as for real photos)
The TAMPERED input (<id>_cand_raw.png) is made afterwards by exp013_crops.py,
because finding the logo in the photo needs registration.

Run: python src/exp013_make_data.py            (~10 000 samples)
"""
import argparse
import csv
import json
from multiprocessing import get_context
from pathlib import Path

import cv2
import numpy as np

import synth
from common import REF_LOGO_BOX, REFERENCE_DIR, ROOT, available_fonts, load_reference
from exp003_make_data import SAVE_H, SAVE_W, TAMPER_TYPES, UNSEEN_FONT, make_jobs
from exp012_make_crops import OUT_H, OUT_W, loose_box

def cut_replicate(img, box):
    """Like exp012_make_crops.cut, but the edge is extended instead of black (references only)."""
    x0, y0, x1, y1 = box
    scale = OUT_W / (x1 - x0)
    m = np.float32([[scale, 0, -x0 * scale], [0, scale, -y0 * scale]])
    return cv2.warpAffine(img, m, (OUT_W, OUT_H), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)


def clean_canvas(logo, rng):
    """The untampered logo front-on on plain paper, with the canvas margin kept (for the raw reference)."""
    paper = synth.hsv_to_bgr(rng.uniform(0, 360), rng.uniform(0.02, 0.08), rng.uniform(0.85, 0.95))
    a = logo["alpha"][..., None]
    img = paper * (1 - a) + (logo["ink"] / 255) * a
    img = img + rng.normal(0, 0.004, img.shape).astype(np.float32)
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


def raw_reference(canvas, logo_x, logo_y):
    """Loose box around the 920 x 260 logo frame placed at (logo_x, logo_y) in `canvas`."""
    w, h = synth.LOGO_W, synth.LOGO_H
    corners = np.float32([[logo_x, logo_y], [logo_x + w, logo_y], [logo_x + w, logo_y + h], [logo_x, logo_y + h]])
    return cut_replicate(canvas, loose_box(corners))


def make_sample(job):
    cv2.setNumThreads(1)
    rng = np.random.default_rng(job["seed"])
    kind, tamper = job["set"], job["tamper"]
    out = Path(job["out"]) / kind      # from the job: spawned workers don't see main()'s settings
    sid = job["id"]

    # ---- the genuine logo and the two reference inputs ------------------------
    ref_idx = ""
    if kind == "D2":
        ref_idx, cand_idx = (int(i) for i in rng.choice([1, 2, 3], size=2, replace=True))
        logo = synth.logo_from_reference(cand_idx)
        ref_full = load_reference(ref_idx)
        full_ref_image = cv2.imread(str(REFERENCE_DIR / f"ref_{ref_idx}.jpg"))
        ref_raw = raw_reference(full_ref_image, REF_LOGO_BOX[0], REF_LOGO_BOX[1])
        font_name, word = "DejaVuSans-Bold", "NORVIA"
    else:
        fonts = available_fonts()
        logo = None
        while logo is None:   # retry until the letters of the random word don't touch
            font_name = str(rng.choice(sorted(fonts)))
            word, tagline = str(rng.choice(synth.WORDS)), str(rng.choice(synth.TAGLINES))
            logo = synth.logo_from_text(word, tagline, font_name, fonts[font_name])
        canvas = clean_canvas(logo, rng)
        ref_full = canvas[synth.PAD:synth.PAD + synth.LOGO_H, synth.PAD:synth.PAD + synth.LOGO_W]
        ref_raw = raw_reference(canvas, synth.PAD, synth.PAD)
        cv2.imwrite(str(out / f"{sid}_ref_full.png"), ref_full)

    # ---- tamper, photograph (no mirrored copies), finish ---------------------
    tampered, info = synth.TAMPERS[tamper](logo, rng)
    cut_frac = None
    if tamper == "cut":
        cut_frac = float(rng.uniform(0.08, 0.45))
        info.update({"magnitude": cut_frac * 100, "unit": "percent_outside", "detail": f"{cut_frac:.2f}"})
    art, origin = synth.render_artwork(tampered, rng)
    h_logo, pose = synth.camera_pose(rng, cut_frac)
    photo_f, photo_info = synth.photograph(art, origin, h_logo, rng, border=cv2.BORDER_REPLICATE)
    photo, shape, jpeg_q = synth.finish_photo(photo_f, rng)

    cv2.imwrite(str(out / f"{sid}_photo.jpg"), photo, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(str(out / f"{sid}_ref.png"), cv2.resize(ref_full, (SAVE_W, SAVE_H), interpolation=cv2.INTER_AREA))
    cv2.imwrite(str(out / f"{sid}_ref_raw.png"), ref_raw)

    u = np.random.default_rng(job["seed"] + 7).random()
    split = "train" if u < 0.70 else "val" if u < 0.85 else "test"
    if kind == "D1" and font_name == UNSEEN_FONT:
        split = "test_unseen_font"
    return {"id": sid, "set": kind, "split": split, "seed": job["seed"],
            "label": 0 if tamper == "none" else 1, "tamper": info["type"] if tamper != "cut" else "cut",
            "target": info["target"], "magnitude": round(info["magnitude"], 3), "unit": info["unit"],
            "detail": info["detail"], "font": font_name, "word": word, "ref_index": ref_idx,
            "photo_w": shape[0], "photo_h": shape[1], "jpeg_q": jpeg_q, "blur": photo_info["blur"],
            "gains": json.dumps(photo_info["gains"]), "pose": json.dumps(pose, default=str)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-d2", type=int, default=5000)
    ap.add_argument("--n-d1", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="data/synth_v2", help="output folder (a small test run can go elsewhere)")
    args = ap.parse_args()
    out = ROOT / args.out
    for kind in ("D1", "D2"):
        (out / kind).mkdir(parents=True, exist_ok=True)

    # seeds 777 / 555: different from data/synth (2026 / 1010) and data/synth_more (offset 100003)
    jobs = make_jobs("D2", args.n_d2, 777, out) + make_jobs("D1", args.n_d1, 555, out)
    rows = []
    with get_context("spawn").Pool(args.workers) as pool:
        for i, row in enumerate(pool.imap_unordered(make_sample, jobs, chunksize=8), 1):
            rows.append(row)
            if i % 500 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} samples", flush=True)

    rows.sort(key=lambda r: r["id"])
    with open(out / "labels.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"done: {len(rows)} samples in {out}; tamper types {TAMPER_TYPES}")


if __name__ == "__main__":
    main()
