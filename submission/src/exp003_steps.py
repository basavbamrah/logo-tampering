"""
EXP-003 intermediate results: how one synthetic sample is made, stage by stage.

exp003_make_data.py only saves the end products (ref, cand, photo). The
generator is deterministic by seed, so here we re-run it for rows of
data/synth/labels.csv and keep every stage, so each one can be shown:

  1. the clean logo alpha (from the NORVIA reference)
  2. the tamper, with a difference image against the clean logo
  3. the flat artwork (paper, bands, filler text)
  4. the camera: perspective only
  5. lighting, blur, noise, colour cast
  6. resize to the photo shape + JPEG
  7. the crop made with the known pose (old) next to the crop made by the
     EXP-001 registration (the "F12" crop the models train on)

Before trusting the pictures we check that the re-run gives the SAME photo as
the file stored in data/synth (pixel for pixel).

It also makes:
  - a realism figure: real photos next to synthetic genuine photos, plus a few
    measured statistics (sharpness, brightness, ink colour, inliers)
  - a count of photos that show a mirrored copy of the logo (generator side
    effect, see mirror_job)
  - the stretch diagnostic as a plot (estimated vs true whole-logo stretch).
    The saved outputs/exp003/stretch_pairs.npy turned out to be empty (it was
    overwritten by a small test run), so the pairs are recomputed here.

Nothing under data/ or outputs/exp003/ is changed; everything goes to
outputs/exp003/steps/.

Run: python src/exp003_steps.py [--workers 8]
"""
import argparse
import copy
import csv
import json
from multiprocessing import get_context

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import synth
from common import OUTPUTS, ROOT, green_ink_mask, list_images, load_reference
from exp001_register import fit_homography, refine_with_ecc, to_gray
from exp003_check_realism import register_photo

DATA = ROOT / "data" / "synth"            # read only
OUT = OUTPUTS / "exp003" / "steps"
TAMPER_TYPES = ["stretch", "recolour", "respace", "glyph_swap", "font_swap", "overlap", "cut"]
LOGO_CORNERS = np.float32([[0, 0], [synth.LOGO_W, 0], [synth.LOGO_W, synth.LOGO_H], [0, synth.LOGO_H]])


# =============================================================================
# Re-running the generator
# =============================================================================

def photograph_steps(art, origin, h_logo, rng):
    """
    synth.photograph() split into its stages. Same code, same order of random
    draws, so the last stage must equal what synth.photograph() returns
    (checked in rebuild()).
    """
    ox, oy = origin
    to_art = np.array([[1, 0, ox], [0, 1, oy], [0, 0, 1]], np.float64)
    h_scene = h_logo @ np.linalg.inv(to_art)
    img = cv2.warpPerspective(art, h_scene, (synth.PHOTO, synth.PHOTO), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT)
    stages = [("perspective", img)]

    yy, xx = np.mgrid[0:synth.PHOTO, 0:synth.PHOTO].astype(np.float32) / synth.PHOTO
    ang = rng.uniform(0, 2 * np.pi)
    light = 1 + rng.uniform(-0.12, 0.12) * ((xx * np.cos(ang) + yy * np.sin(ang)) - 0.5)
    img = img * light[..., None]
    stages.append(("lighting", img))

    blur = "none"
    if rng.random() < 0.5:
        sigma = rng.uniform(0.4, 1.6)
        if rng.random() < 0.15:
            length = int(rng.integers(5, 13))
            kernel = np.zeros((length, length), np.float32)
            kernel[length // 2, :] = 1.0 / length
            rot = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), rng.uniform(0, 180), 1)
            kernel = cv2.warpAffine(kernel, rot, (length, length))
            img = cv2.filter2D(img, -1, kernel / max(kernel.sum(), 1e-6))
            blur = f"motion{length}"
        else:
            img = cv2.GaussianBlur(img, (0, 0), sigma)
            blur = f"gauss{sigma:.1f}"
    stages.append((f"blur ({blur})", img))

    noise_sd = rng.uniform(0, 0.012)
    img = img + rng.normal(0, noise_sd, img.shape).astype(np.float32)
    stages.append((f"noise (sd {noise_sd * 255:.1f}/255)", img))

    gains = rng.uniform(0.94, 1.06, 3) * rng.uniform(0.9, 1.08)
    img = img * gains.astype(np.float32)
    stages.append((f"colour cast (BGR gains {gains[0]:.2f} {gains[1]:.2f} {gains[2]:.2f})", img))
    return [(name, np.clip(im, 0, 1)) for name, im in stages], np.clip(img, 0, 1)


def rebuild(row):
    """
    Re-run exp003_make_data.make_sample() for one D2 row of labels.csv, keeping
    every stage. The random draws must happen in exactly the same order as
    there, otherwise the seed gives a different sample.
    """
    assert row["set"] == "D2", "only the NORVIA set is rebuilt here"
    rng = np.random.default_rng(int(row["seed"]))
    ref_idx, cand_idx = (int(i) for i in rng.choice([1, 2, 3], size=2, replace=True))
    logo = synth.logo_from_reference(cand_idx)
    tampered, info = synth.TAMPERS[row["tamper"]](logo, rng)
    cut_frac = float(rng.uniform(0.08, 0.45)) if row["tamper"] == "cut" else None

    art, origin = synth.render_artwork(tampered, rng)
    h_logo, pose = synth.camera_pose(rng, cut_frac)

    # Run the original photograph() on a copy of the generator, and our split
    # version on the real one: both must agree, or the stage pictures lie.
    official, _ = synth.photograph(art, origin, h_logo, copy.deepcopy(rng))
    stages, photo_f = photograph_steps(art, origin, h_logo, rng)
    assert np.array_equal(official, photo_f), "split photograph() differs from synth.photograph()"

    photo, shape, quality = synth.finish_photo(photo_f, rng)
    known_crop = synth.rectify_with_known_pose(photo, h_logo, shape, rng)
    # make_sample saves the photo once more as JPEG 95; do the same to compare with the file
    ok, buf = cv2.imencode(".jpg", photo, [cv2.IMWRITE_JPEG_QUALITY, 95])
    as_saved = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    w, h = shape
    # logo frame -> final photo. The resize maps pixel centres: x' = (x + 0.5) * s - 0.5
    sx, sy = w / synth.PHOTO, h / synth.PHOTO
    resize = np.array([[sx, 0, 0.5 * (sx - 1)], [0, sy, 0.5 * (sy - 1)], [0, 0, 1]])
    return {"row": row, "ref_idx": ref_idx, "cand_idx": cand_idx, "logo": logo, "tampered": tampered,
            "info": info, "cut_frac": cut_frac, "art": art, "origin": origin, "h_logo": h_logo,
            "pose": pose, "stages": stages, "photo_f": photo_f, "resized": cv2.resize(photo_f, shape, interpolation=cv2.INTER_AREA),
            "photo": photo, "shape": shape, "quality": quality, "known_crop": known_crop,
            "as_saved": as_saved, "H_true": resize @ h_logo}


def compare_with_stored(s):
    """Max abs pixel difference between the re-run photo and the stored file (0 = identical)."""
    stored = cv2.imread(str(DATA / "D2" / f"{s['row']['id']}_photo.jpg"))
    if stored is None or stored.shape != s["as_saved"].shape:
        return None
    return int(np.abs(stored.astype(int) - s["as_saved"].astype(int)).max())


def flat_logo(logo):
    """The logo canvas on white paper, overlays included (to see what a tamper changed)."""
    a = logo["alpha"][..., None]
    img = 0.97 * (1 - a) + (logo["ink"] / 255) * a
    for tight, left, baseline, base_row, grey in logo["overlays"]:
        layer = np.zeros(logo["alpha"].shape, np.float32)
        synth.paste_max(layer, tight, left, baseline, base_row)
        img = img * (1 - 0.85 * layer[..., None]) + grey * 0.85 * layer[..., None]
    return np.clip(img, 0, 1).astype(np.float32)


def register(photo_bgr):
    """The EXP-001 registration, exactly as exp003_register_synthetic.py runs it."""
    ref = load_reference(1)
    ref_gray = to_gray(ref)
    H, stats = fit_homography(ref_gray, to_gray(photo_bgr), cv2.SIFT_create(nfeatures=5000))
    if H is None:
        return None, None, stats
    H_ecc, ecc = refine_with_ecc(ref_gray, to_gray(photo_bgr), H)
    crop = cv2.warpPerspective(photo_bgr, np.linalg.inv(H_ecc), (ref.shape[1], ref.shape[0]))
    stats["ecc_corr"] = ecc
    return H_ecc, crop, stats


def corner_error(H_a, H_b):
    """Max distance (photo px) between the logo-box corners pushed through two homographies."""
    a = cv2.perspectiveTransform(LOGO_CORNERS.reshape(-1, 1, 2), H_a).reshape(-1, 2)
    b = cv2.perspectiveTransform(LOGO_CORNERS.reshape(-1, 1, 2), H_b).reshape(-1, 2)
    return float(np.linalg.norm(a - b, axis=1).max())


# =============================================================================
# Drawing
# =============================================================================

def rgb(img):
    """float 0..1 or uint8 BGR -> RGB for matplotlib."""
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def outline(img, H, colour=(0, 220, 255), width=3):
    """Draw the logo box (pushed through H) on a copy of the image."""
    out = img.copy() if img.dtype == np.uint8 else (np.clip(img, 0, 1) * 255).astype(np.uint8)
    quad = cv2.perspectiveTransform(LOGO_CORNERS.reshape(-1, 1, 2), H).reshape(-1, 2)
    cv2.polylines(out, [quad.astype(np.int32)], True, colour, width, cv2.LINE_AA)
    return out


def ref_outline(crop):
    """Reference ink edges in red on a 920x260 crop: red on the letter edges = aligned."""
    edges = cv2.Canny(green_ink_mask(load_reference(1)), 50, 150) > 0
    out = crop.copy()
    out[edges] = (0, 0, 255)
    return out


def show(ax, img, title, **kw):
    ax.imshow(img, **kw)
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_xticks([])
    ax.set_yticks([])


def logo_patch(img, H, pad=40):
    """Cut the logo area out of a photo-sized image (for zoomed views of subtle effects)."""
    quad = cv2.perspectiveTransform(LOGO_CORNERS.reshape(-1, 1, 2), H).reshape(-1, 2)
    x0, y0 = np.maximum(quad.min(0) - pad, 0).astype(int)
    x1, y1 = quad.max(0).astype(int) + pad
    return img[y0:y1, x0:x1]


def tamper_figure(s, reg, match, path):
    """One figure, 3 x 3, with every stage of one sample."""
    r, info = s["row"], s["info"]
    clean, tampered = flat_logo(s["logo"]), flat_logo(s["tampered"])
    diff = np.abs(clean - tampered).max(axis=2)
    H_ecc, reg_crop, stats = reg

    fig, ax = plt.subplots(3, 3, figsize=(19, 13))
    show(ax[0, 0], rgb(clean), f"1. Clean logo (reference {s['cand_idx']} ink, on white)")
    show(ax[0, 1], rgb(tampered), f"2. Tampered: {r['tamper']} {info['target']} | {info['detail']}")
    changed = int((diff > 0.1).sum())
    extra = "\n(cut is a camera position: the artwork is unchanged)" if r["tamper"] == "cut" else ""
    show(ax[0, 2], diff, f"Difference |clean - tampered|: {changed} px changed by > 0.1\n"
         f"magnitude {float(r['magnitude']):.2f} {r['unit']}{extra}", cmap="magma", vmin=0, vmax=max(0.05, float(diff.max())))
    fig.colorbar(ax[0, 2].images[0], ax=ax[0, 2], fraction=0.025)

    art = (s["art"] * 255).astype(np.uint8)
    ox, oy = s["origin"]
    cv2.rectangle(art, (ox, oy), (ox + synth.LOGO_W, oy + synth.LOGO_H), (0, 220, 255), 4)
    show(ax[1, 0], rgb(art), "3. Flat artwork 1800x1800 (paper, light, bands, fillers);\nyellow = logo frame")
    p = s["pose"]
    show(ax[1, 1], rgb(outline(s["stages"][0][1], s["h_logo"])),
         f"4. Camera only: yaw {p['yaw']} pitch {p['pitch']} roll {p['roll']} deg,\n"
         f"focal {p['focal']} px, logo {p['logo_width_px']} px wide" + (f", cut {p['cut_side']}" if p["cut_side"] else ""))
    show(ax[1, 2], rgb(s["photo_f"]), "5. + lighting, " + ", ".join(n for n, _ in s["stages"][2:]).replace("colour cast", "\ncolour cast"))

    show(ax[2, 0], rgb(outline(s["as_saved"], s["H_true"])),
         f"6. Resized to {s['shape'][0]}x{s['shape'][1]}, JPEG q{s['quality']} (stored file;"
         f" re-run {'IDENTICAL' if match == 0 else f'max diff {match}'}).\nYellow = true logo box")
    show(ax[2, 1], rgb(ref_outline(s["known_crop"])), "7a. Crop with the KNOWN pose + 0.8 px jitter (old *_cand.png)\nred = reference ink edges")
    if reg_crop is None:
        ax[2, 2].axis("off")
        ax[2, 2].set_title(f"7b. Registration FAILED ({stats['inliers']} inliers): no crop", fontsize=10, loc="left")
    else:
        err = corner_error(H_ecc, s["H_true"])
        show(ax[2, 2], rgb(ref_outline(reg_crop)),
             f"7b. Crop by EXP-001 registration (F12, what models train on):\n{stats['inliers']} inliers, "
             f"ECC {stats['ecc_corr']}, corner error vs truth {err:.1f} px")
    fig.suptitle(f"{r['id']}  ({r['tamper']})  seed {r['seed']}", fontsize=14, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def overview_figure(s, reg, match, path):
    """A genuine sample with the subtle photo effects zoomed onto the logo."""
    H_ecc, reg_crop, stats = reg
    fig, ax = plt.subplots(3, 5, figsize=(26, 13))
    alpha = s["logo"]["alpha"]
    show(ax[0, 0], rgb(load_reference(s["cand_idx"])), f"0. Reference image {s['cand_idx']} (crop 920x260)")
    show(ax[0, 1], alpha, "1. Ink alpha (0 = paper, 1 = full ink)\n= blurred green mask, 120 px spare border", cmap="gray_r")
    show(ax[0, 2], rgb(flat_logo(s["logo"])), "2. Tamper: none (genuine)")
    show(ax[0, 3], rgb(s["art"]), "3. Flat artwork")
    show(ax[0, 4], rgb(outline(s["stages"][0][1], s["h_logo"])), "4. Camera pose (true logo box)")

    # PHOTO-sized stages, all cut to the same logo window so changes are comparable
    for i, (name, img) in enumerate(s["stages"]):
        show(ax[1, i], rgb(logo_patch(img, s["h_logo"])), f"5.{i} {name}" if i else "5.0 perspective only (zoom)")

    resized_patch = logo_patch((s["resized"] * 255).astype(np.uint8), s["H_true"])
    jpeg_patch = logo_patch(s["photo"], s["H_true"])
    show(ax[2, 0], rgb(resized_patch), f"6a. Resized to {s['shape'][0]}x{s['shape'][1]} (zoom)")
    show(ax[2, 1], rgb(jpeg_patch), f"6b. JPEG q{s['quality']} (zoom); re-run vs stored: "
         f"{'IDENTICAL' if match == 0 else f'max diff {match}'}")
    jd = np.abs(resized_patch.astype(int) - jpeg_patch.astype(int)).max(axis=2)
    show(ax[2, 2], jd, f"6c. |before - after JPEG|, mean {jd.mean():.1f}/255", cmap="magma")
    show(ax[2, 3], rgb(ref_outline(s["known_crop"])), "7a. Known-pose crop")
    if reg_crop is not None:
        show(ax[2, 4], rgb(ref_outline(reg_crop)), f"7b. Registration crop (F12): {stats['inliers']} inliers,\n"
             f"corner error vs truth {corner_error(H_ecc, s['H_true']):.1f} px")
    fig.suptitle(f"How one synthetic genuine sample is made: {s['row']['id']} (seed {s['row']['seed']})",
                 fontsize=15, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)


# =============================================================================
# Realism and stretch diagnostic
# =============================================================================

def crop_stats(crop):
    """Sharpness, ink hue/saturation and paper brightness of a rectified 920x260 crop."""
    ink = cv2.erode(green_ink_mask(load_reference(1)), np.ones((5, 5), np.uint8)) > 0
    paper = cv2.dilate(green_ink_mask(load_reference(1)), np.ones((15, 15), np.uint8)) == 0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    valid = crop.max(axis=2) > 0          # outside the photo the crop is black
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return {"sharpness": float(cv2.Laplacian(gray, cv2.CV_64F)[valid].var()),
            "ink_hue": float(np.median(hsv[..., 0][ink & valid]) * 2),      # degrees 0-360
            "ink_sat": float(np.median(hsv[..., 1][ink & valid])),
            "paper_brightness": float(np.median(gray[paper & valid]))}


def realism(labels):
    real_paths = list_images()
    real_reg = {r["image"]: r for r in csv.DictReader(open(OUTPUTS / "exp001" / "registration.csv"))}
    syn_reg = {r["id"]: r for r in csv.DictReader(open(DATA / "registration_D2.csv"))}
    genuine = [r for r in labels if r["set"] == "D2" and r["label"] == "0"][:150]

    rows = []
    for p in real_paths:
        crop = cv2.imread(str(OUTPUTS / "exp001" / "rectified" / f"{p.stem}.png"))
        img = cv2.imread(str(p))
        row = {"source": "real", "id": p.stem, "photo_brightness": float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean()),
               "inliers": float(real_reg[p.stem]["inliers"]) if real_reg[p.stem]["registered"] == "True" else np.nan}
        row.update(crop_stats(crop) if crop is not None else {})
        rows.append(row)
    for r in genuine:
        img = cv2.imread(str(DATA / "D2" / f"{r['id']}_photo.jpg"))
        crop = cv2.imread(str(DATA / "D2" / f"{r['id']}_cand_reg_full.png"))
        row = {"source": "synthetic", "id": r["id"], "photo_brightness": float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean()),
               "inliers": float(syn_reg[r["id"]]["inliers"]) if syn_reg[r["id"]]["registered"] == "True" else np.nan}
        row.update(crop_stats(crop) if crop is not None else {})
        rows.append(row)

    keys = ["source", "id", "photo_brightness", "paper_brightness", "sharpness", "ink_hue", "ink_sat", "inliers"]
    with open(OUT / "realism_stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: (round(row[k], 2) if isinstance(row.get(k), float) else row.get(k, "")) for k in keys})

    # side-by-side pictures: 8 real (spread over the set) and 8 synthetic genuine
    real_pick = real_paths[::6][:8]
    syn_pick = genuine[:8]
    fig, ax = plt.subplots(2, 8, figsize=(26, 8))
    for i, p in enumerate(real_pick):
        show(ax[0, i], rgb(cv2.imread(str(p))), f"real {p.stem}")
    for i, r in enumerate(syn_pick):
        show(ax[1, i], rgb(cv2.imread(str(DATA / "D2" / f"{r['id']}_photo.jpg"))), f"synthetic {r['id']}")
    fig.suptitle("Top: real photos (every 6th). Bottom: synthetic GENUINE photos (first 8 in data/synth).",
                 fontsize=14, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "realism_side_by_side.jpg", dpi=70)
    plt.close(fig)

    # histograms of the measured numbers
    names = [("photo_brightness", "whole-photo brightness (grey 0-255)"),
             ("paper_brightness", "paper brightness near the logo (0-255)"),
             ("sharpness", "sharpness: Laplacian variance of the logo crop"),
             ("ink_hue", "ink hue (deg)"), ("ink_sat", "ink saturation (0-255)"),
             ("inliers", "RANSAC inliers")]
    summary = []
    fig, ax = plt.subplots(2, 3, figsize=(18, 9))
    for a, (k, label) in zip(ax.ravel(), names):
        real = np.array([r[k] for r in rows if r["source"] == "real" and k in r], float)
        syn = np.array([r[k] for r in rows if r["source"] == "synthetic" and k in r], float)
        real, syn = real[~np.isnan(real)], syn[~np.isnan(syn)]
        bins = np.linspace(min(real.min(), syn.min()), max(real.max(), syn.max()), 25)
        a.hist(real, bins, density=True, alpha=0.6, label=f"real (n={len(real)})", color="#1f77b4")
        a.hist(syn, bins, density=True, alpha=0.6, label=f"synthetic genuine (n={len(syn)})", color="#ff7f0e")
        a.set_title(label, fontsize=11)
        a.legend(fontsize=9)
        summary.append((label, real, syn))
    fig.tight_layout()
    fig.savefig(OUT / "realism_stats.png", dpi=80)
    plt.close(fig)
    return summary


def stretch_job(r):
    cv2.setNumThreads(1)
    ref_gray = to_gray(load_reference(1))
    res = register_photo(DATA / "D2" / f"{r['id']}_photo.jpg", ref_gray, cv2.SIFT_create(nfeatures=5000))
    true = 1.0 if r["label"] == "0" else 1 + float(r["detail"].split()[1].replace("%", "")) / 100
    return {"id": r["id"], "true": true, "estimate": res["stretch"] if res else np.nan,
            "yaw": json.loads(r["pose"])["yaw"]}


def stretch_plot(labels, workers):
    """Same estimator and samples as exp003_check_realism.py (150 genuine + whole-logo x-stretches)."""
    d2 = [r for r in labels if r["set"] == "D2"]
    genuine = [r for r in d2 if r["label"] == "0"][:150]
    stretched = [r for r in d2 if r["tamper"] == "stretch" and r["target"] == "whole" and r["detail"].startswith("x ")]
    with get_context("spawn").Pool(workers) as pool:
        rows = pool.map(stretch_job, genuine + stretched)
    with open(OUT / "stretch_pairs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    gen = np.array([r["estimate"] for r in rows if r["true"] == 1.0 and not np.isnan(r["estimate"])])
    st = np.array([(r["true"], r["estimate"]) for r in rows if r["true"] != 1.0 and not np.isnan(r["estimate"])])
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    lim = [0.85, 1.25]
    ax[0].plot(lim, lim, "k--", lw=1, label="perfect estimate")
    ax[0].scatter(st[:, 0], st[:, 1], s=25, label=f"whole-logo x-stretch (n={len(st)})")
    ax[0].axhspan(np.percentile(gen, 5), np.percentile(gen, 95), color="grey", alpha=0.2,
                  label="5-95 % range of GENUINE estimates")
    ax[0].set_xlim(0.9, 1.1)
    ax[0].set_ylim(*lim)
    ax[0].set_xlabel("true stretch (1.05 = 5 % wider)")
    ax[0].set_ylabel("implied stretch from the homography")
    r_corr = np.corrcoef(st[:, 0], st[:, 1])[0, 1]
    ax[0].set_title(f"Estimated vs true stretch (corr {r_corr:.2f})")
    ax[0].legend(fontsize=9)
    ax[1].hist(gen, bins=40, color="grey")
    ax[1].axvline(1.0, color="k", ls="--")
    for x in (0.92, 0.99, 1.01, 1.08):
        ax[1].axvline(x, color="#d62728", ls=":", lw=1)
    ax[1].set_title(f"Genuine photos (true = 1.0): std {gen.std():.3f}, n={len(gen)}\nred = tamper range 1-8 %")
    ax[1].set_xlabel("implied stretch")
    fig.tight_layout()
    fig.savefig(OUT / "stretch_diagnostic.png", dpi=80)
    plt.close(fig)
    return {"n_gen": len(gen), "gen_std": float(gen.std()), "gen_p5": float(np.percentile(gen, 5)),
            "gen_p95": float(np.percentile(gen, 95)), "n_st": len(st), "n_st_total": len(stretched),
            "corr": float(r_corr), "st_err_std": float((st[:, 1] - st[:, 0]).std())}


def mirror_job(row):
    """
    Does the photo show a MIRRORED copy of the logo? photograph() warps the
    artwork with BORDER_REFLECT, so where the camera sees past the 1800 px
    artwork, the scene (logo included) is reflected back in. We only need the
    pose for this, not the photo: follow a grid of photo pixels back into the
    artwork and count those that land outside it and reflect onto logo ink.
    """
    cv2.setNumThreads(1)
    rng = np.random.default_rng(int(row["seed"]))
    _, cand_idx = (int(i) for i in rng.choice([1, 2, 3], size=2, replace=True))
    logo = synth.logo_from_reference(cand_idx)
    tampered, _ = synth.TAMPERS[row["tamper"]](logo, rng)
    cut_frac = float(rng.uniform(0.08, 0.45)) if row["tamper"] == "cut" else None
    art, origin = synth.render_artwork(tampered, rng)
    h_logo, _ = synth.camera_pose(rng, cut_frac)

    ox, oy = origin
    to_art = np.array([[1, 0, ox], [0, 1, oy], [0, 0, 1]], np.float64)
    step = 4
    ys, xs = np.mgrid[0:synth.PHOTO:step, 0:synth.PHOTO:step].astype(np.float32)
    src = cv2.perspectiveTransform(np.dstack([xs, ys]).reshape(-1, 1, 2), np.linalg.inv(h_logo @ np.linalg.inv(to_art))).reshape(-1, 2)
    outside = ((src < 0) | (src >= synth.SCENE)).any(axis=1)
    refl = np.where(src < 0, -src - 1, np.where(src >= synth.SCENE, 2 * synth.SCENE - src - 1, src))
    # ink of the (tampered) logo, in artwork coordinates
    ink = np.zeros((synth.SCENE, synth.SCENE), bool)
    cx, cy = ox - synth.PAD, oy - synth.PAD
    ink[cy:cy + synth.CANVAS_H, cx:cx + synth.CANVAS_W] = tampered["alpha"] > 0.5
    r = np.clip(refl.round().astype(int), 0, synth.SCENE - 1)
    mirrored = outside & ink[r[:, 1], r[:, 0]]
    return {"id": row["id"], "tamper": row["tamper"], "photo_px_outside_artwork_pct": round(100 * outside.mean(), 1),
            "mirrored_ink_px": int(mirrored.sum() * step * step)}


def mirror_check(labels, workers):
    d2 = [r for r in labels if r["set"] == "D2"]
    with get_context("spawn").Pool(workers) as pool:
        rows = pool.map(mirror_job, d2, chunksize=8)
    with open(OUT / "mirrored_logo_check.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    hit = [r for r in rows if r["mirrored_ink_px"] > 0]
    big = [r for r in rows if r["mirrored_ink_px"] > 2000]
    text = (f"D2 photos (render 1024x1024) that see past the artwork edge: "
            f"{sum(r['photo_px_outside_artwork_pct'] > 0 for r in rows)}/{len(rows)}\n"
            f"... and show reflected logo ink: {len(hit)}/{len(rows)}; more than 2000 px of ink: {len(big)}/{len(rows)}\n"
            f"examples: {', '.join(r['id'] for r in sorted(rows, key=lambda r: -r['mirrored_ink_px'])[:8])}\n")
    (OUT / "mirrored_logo_check.txt").write_text(text)
    return text


# =============================================================================

def example_job(row):
    cv2.setNumThreads(1)
    s = rebuild(row)
    match = compare_with_stored(s)
    reg = register(cv2.imread(str(DATA / "D2" / f"{row['id']}_photo.jpg")))
    name = "overview_genuine" if row["tamper"] == "none" else f"tamper_{row['tamper']}"
    if row["tamper"] == "none":
        overview_figure(s, reg, match, OUT / f"00_{name}_{row['id']}.jpg")
    else:
        tamper_figure(s, reg, match, OUT / f"{TAMPER_TYPES.index(row['tamper']) + 1:02d}_{name}_{row['id']}.jpg")

    # is our registration crop the same as the stored F12 crop?
    stored_reg = cv2.imread(str(DATA / "D2" / f"{row['id']}_cand_reg_full.png"))
    reg_same = None if (stored_reg is None or reg[1] is None) else int(np.abs(stored_reg.astype(int) - reg[1].astype(int)).max())
    return {"id": row["id"], "tamper": row["tamper"], "target": s["info"]["target"], "detail": s["info"]["detail"],
            "magnitude": row["magnitude"], "unit": row["unit"], "photo_max_diff_vs_stored": match,
            "reg_inliers": reg[2]["inliers"], "reg_crop_max_diff_vs_stored": reg_same,
            "corner_err_px": round(corner_error(reg[0], s["H_true"]), 2) if reg[0] is not None else ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    labels = list(csv.DictReader(open(DATA / "labels.csv")))
    d2 = [r for r in labels if r["set"] == "D2"]

    # the first D2 sample of each kind: one genuine overview + one per tamper type
    examples = [next(r for r in d2 if r["tamper"] == t) for t in ["none"] + TAMPER_TYPES]
    with get_context("spawn").Pool(min(args.workers, len(examples))) as pool:
        rows = pool.map(example_job, examples)
    with open(OUT / "examples.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)

    summary = realism(labels)
    with open(OUT / "realism_summary.txt", "w") as f:
        f.write(f"{'measure':48s} {'real median [10-90%]':>26s} {'synthetic median [10-90%]':>28s}\n")
        for label, real, syn in summary:
            q = lambda v: f"{np.median(v):7.1f} [{np.percentile(v, 10):.1f}-{np.percentile(v, 90):.1f}]"
            f.write(f"{label:48s} {q(real):>26s} {q(syn):>28s}\n")
    print(open(OUT / "realism_summary.txt").read())

    print(stretch_plot(labels, args.workers))
    print(mirror_check(labels, args.workers))


if __name__ == "__main__":
    main()
