"""
EXP-003: build the synthetic dataset.

Two sets, same tampers, same camera:
  D2  the real NORVIA logo (taken from the reference images), tampered or not.
  D1  made-up words set in different fonts (5 Google Fonts + DejaVu), tampered or not.

Each sample is a pair:
  ref   the clean, front-on logo        ("head 1" of the two-head model)
  cand  the logo as seen in a "photo", undone back to the logo frame ("head 2")
plus a row in labels.csv saying what we did to it (type, target, size).
About half of the samples are genuine (only camera effects, no tamper).

Run:   python src/exp003_make_data.py --n-d2 1500 --n-d1 1500
Quick: python src/exp003_make_data.py --n-d2 40 --n-d1 40 --out data/synth_test
More training data (different random seeds, everything in the train split):
       python src/exp003_make_data.py --n-d2 3000 --n-d1 3000 --out data/synth_more --seed-offset 1 --all-train
"""
import argparse
import csv
import json
from multiprocessing import Pool

import cv2
import numpy as np

import synth
from common import ROOT, available_fonts, load_reference

TAMPER_TYPES = ["stretch", "recolour", "respace", "glyph_swap", "font_swap", "overlap", "cut"]
UNSEEN_FONT = "Lato-Bold"       # D1 samples in this font only go to the "test_unseen_font" split
SAVE_W, SAVE_H = 460, 130        # crops are saved at half size


def make_sample(job):
    cv2.setNumThreads(1)
    rng = np.random.default_rng(job["seed"])
    kind, tamper = job["set"], job["tamper"]
    fonts = available_fonts()

    # ---- the genuine logo and the clean "head 1" image --------------------
    if kind == "D2":
        ref_idx, cand_idx = (int(i) for i in rng.choice([1, 2, 3], size=2, replace=True))
        logo = synth.logo_from_reference(cand_idx)
        ref_img, font_name, word = load_reference(ref_idx), "DejaVuSans-Bold", "NORVIA"
    else:
        logo = None
        while logo is None:   # retry until the letters of the random word don't touch
            font_name = str(rng.choice(sorted(fonts)))
            word, tagline = str(rng.choice(synth.WORDS)), str(rng.choice(synth.TAGLINES))
            logo = synth.logo_from_text(word, tagline, font_name, fonts[font_name])
        ref_img = synth.clean_render(logo, rng)

    # ---- tamper -----------------------------------------------------------
    tampered, info = synth.TAMPERS[tamper](logo, rng)
    cut_frac = None
    if tamper == "cut":
        cut_frac = float(rng.uniform(0.08, 0.45))
        info.update({"magnitude": cut_frac * 100, "unit": "percent_outside", "detail": f"{cut_frac:.2f}"})

    # ---- photograph it, then undo the camera ------------------------------
    art, origin = synth.render_artwork(tampered, rng)
    h_logo, pose = synth.camera_pose(rng, cut_frac)
    photo_f, photo_info = synth.photograph(art, origin, h_logo, rng)
    photo, shape, jpeg_q = synth.finish_photo(photo_f, rng)
    cand_img = synth.rectify_with_known_pose(photo, h_logo, shape, rng)

    # ---- save ---------------------------------------------------------------
    out = job["out"] / kind
    out.mkdir(parents=True, exist_ok=True)
    sid = job["id"]
    for name, img in (("ref", ref_img), ("cand", cand_img)):
        cv2.imwrite(str(out / f"{sid}_{name}.png"), cv2.resize(img, (SAVE_W, SAVE_H), interpolation=cv2.INTER_AREA))
    if kind == "D2":   # full photos are only kept for the NORVIA set (to test registration later)
        cv2.imwrite(str(out / f"{sid}_photo.jpg"), photo, [cv2.IMWRITE_JPEG_QUALITY, 95])

    seed_split = np.random.default_rng(job["seed"] + 7).random()
    split = "train" if seed_split < 0.70 else "val" if seed_split < 0.85 else "test"
    if kind == "D1" and font_name == UNSEEN_FONT:
        split = "test_unseen_font"
    if job["all_train"]:
        # Extra data for training only. Keep the held-out font out of it too,
        # so the "unseen font" test stays honest.
        split = "skip_unseen_font" if (kind == "D1" and font_name == UNSEEN_FONT) else "train"

    return {"id": sid, "set": kind, "split": split, "seed": job["seed"],
            "label": 0 if tamper == "none" else 1, "tamper": info["type"] if tamper != "cut" else "cut",
            "target": info["target"], "magnitude": round(info["magnitude"], 3), "unit": info["unit"],
            "detail": info["detail"], "font": font_name, "word": word,
            "photo_w": shape[0], "photo_h": shape[1], "jpeg_q": jpeg_q, "blur": photo_info["blur"],
            "gains": json.dumps(photo_info["gains"]), "pose": json.dumps(pose)}


def make_jobs(kind, n, start_seed, out, all_train=False):
    rng = np.random.default_rng(start_seed)
    jobs = []
    for i in range(n):
        tamper = "none" if i % 2 == 0 else TAMPER_TYPES[(i // 2) % len(TAMPER_TYPES)]
        jobs.append({"id": f"{kind}_{i:05d}", "set": kind, "seed": int(rng.integers(1 << 31)),
                     "tamper": tamper, "out": out, "all_train": all_train})
    return jobs


def contact_sheet(rows, out, kind, n=24):
    """Side-by-side: clean ref | candidate, with the label written on top."""
    tiles = []
    for r in [r for r in rows if r["set"] == kind][:n]:
        ref = cv2.imread(str(out / kind / f"{r['id']}_ref.png"))
        cand = cv2.imread(str(out / kind / f"{r['id']}_cand.png"))
        pair = np.hstack([ref, cand])
        cv2.rectangle(pair, (0, 0), (SAVE_W * 2, 22), (255, 255, 255), -1)
        text = "genuine" if r["label"] == 0 else f"{r['tamper']} {r['target']} {r['detail']}"
        cv2.putText(pair, f"{r['id'][-5:]} {r['font']} | {text}"[:80], (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        tiles.append(np.pad(pair, ((2, 2), (2, 2), (0, 0)), constant_values=200))
    cols = 2
    rows_img = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles) - cols + 1, cols)]
    cv2.imwrite(str(out / f"sheet_{kind}.jpg"), np.vstack(rows_img))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-d2", type=int, default=1500)
    ap.add_argument("--n-d1", type=int, default=1500)
    ap.add_argument("--out", default="data/synth")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="change this to get a different set of samples (0 = the original dataset)")
    ap.add_argument("--all-train", action="store_true", help="put every sample in the train split")
    args = ap.parse_args()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    off = 100003 * args.seed_offset
    jobs = (make_jobs("D2", args.n_d2, 2026 + off, out, args.all_train) +
            make_jobs("D1", args.n_d1, 1010 + off, out, args.all_train))

    rows = []
    with Pool(args.workers) as pool:
        for i, row in enumerate(pool.imap_unordered(make_sample, jobs, chunksize=4), 1):
            rows.append(row)
            if i % 100 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} samples", flush=True)

    rows.sort(key=lambda r: r["id"])
    with open(out / "labels.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    for kind in ("D2", "D1"):
        if any(r["set"] == kind for r in rows):
            contact_sheet(rows, out, kind)
    print(f"done: {len(rows)} samples in {out}")


if __name__ == "__main__":
    main()
