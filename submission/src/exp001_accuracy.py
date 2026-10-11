"""
EXP-001 accuracy: how do we know RANSAC found the correct match?

On the 50 real photos we can only check alignment by eye. On the synthetic
NORVIA photos (EXP-003) we know the true camera, so we know exactly where the
logo is in each photo. Here we run the SAME registration code as EXP-001
(fit_homography + refine_with_ecc, same settings) on stored synthetic photos
and measure how far the found logo box is from the true one.

The true mapping (reference crop -> photo) is rebuilt from the seed stored in
data/synth/labels.csv (exp003_steps.rebuild). We first check the rebuilt photo
is pixel-identical to the stored file, so the truth belongs to that exact photo.
The three reference images agree to within 0.03 px, so it does not matter
which one the generator drew the ink from.

Measures, per photo:
  corner error   max distance (photo px) between the 4 corners of the 920x260
                 logo box pushed through the estimated and the true homography
                 (corners include the white margin, so this is the worst place)
  ink error      mean misalignment over the logo's ink pixels, in REFERENCE px
                 (what the later pixel-comparison methods actually feel)
both before ECC (RANSAC only) and after ECC.

The key question: when registration PASSES the checks (>= 15 inliers, sane
outline), is it actually right? And when it fails, would the crop have been wrong?

Whole-logo stretches are a special case: registration is expected to absorb
the stretch, so their error is also measured against the "stretched truth".

Output: outputs/exp001/accuracy/ (accuracy.csv, figures, summary.txt)
Run:    python src/exp001_accuracy.py [--n 300 --workers 8]
"""
import argparse
import csv
import json
from multiprocessing import get_context

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import synth
from common import OUTPUTS, ROOT, green_ink_mask, load_reference
from exp001_register import MIN_INLIERS, RANSAC_PX, RATIO_TEST, fit_homography, looks_sane, refine_with_ecc, to_gray
from exp003_steps import LOGO_CORNERS, compare_with_stored, outline, rebuild, rgb

DATA = ROOT / "data" / "synth"        # read only
OUT = OUTPUTS / "exp001" / "accuracy"
WRONG_PX = 10.0                       # a corner more than this far off = "wrong" registration
ECC_GATE = 0.9                        # candidate quality gate tested below
# Samples are grouped by how the tamper relates to the geometry:
#  - the logo outline is unchanged or changed only locally: registration should be exact
#  - font swap: the whole wordmark changes shape; cut: part of the logo is missing
#  - a stretch of one part; a whole-logo stretch (expected to be absorbed)
GROUPS = ["genuine / local tamper", "font_swap", "cut", "part stretch"]
GROUP_COLOURS = {"genuine / local tamper": "#2ca02c", "font_swap": "#d62728", "cut": "#ff7f0e",
                 "part stretch": "#1f77b4", "whole-logo stretch (vs camera truth)": "#9467bd"}
TAMPER_ORDER = ["none", "stretch", "recolour", "respace", "glyph_swap", "font_swap", "overlap", "cut"]


def group(r):
    if r["tamper"] == "stretch":
        return "whole-logo stretch (vs camera truth)" if r["target"] == "whole" else "part stretch"
    return r["tamper"] if r["tamper"] in ("font_swap", "cut") else GROUPS[0]


def raw_ransac(ref_gray, img_gray, sift):
    """
    fit_homography() without the final checks, so we also get the homography
    of photos that FAIL (to see whether rejecting them was right).
    Same matching rules, same RANSAC call.
    """
    kp_r, des_r = sift.detectAndCompute(ref_gray, None)
    kp_i, des_i = sift.detectAndCompute(img_gray, None)
    if des_i is None or len(kp_i) < 10:
        return None, 0, 0
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des_r, des_i, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < RATIO_TEST * n.distance]
    best = {}
    for m in good:
        if m.trainIdx not in best or m.distance < best[m.trainIdx].distance:
            best[m.trainIdx] = m
    good = list(best.values())
    if len(good) < 4:                 # a homography needs 4 pairs
        return None, len(good), 0
    src = np.float32([kp_r[m.queryIdx].pt for m in good])
    dst = np.float32([kp_i[m.trainIdx].pt for m in good])
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_PX)
    return H, len(good), (int(mask.sum()) if mask is not None else 0)


def corner_errors(H_est, H_true):
    a = cv2.perspectiveTransform(LOGO_CORNERS.reshape(-1, 1, 2), H_est).reshape(-1, 2)
    b = cv2.perspectiveTransform(LOGO_CORNERS.reshape(-1, 1, 2), H_true).reshape(-1, 2)
    d = np.linalg.norm(a - b, axis=1)
    return float(d.max()), float(d.mean())


def ink_error(H_est, H_true, ink_pts):
    """Mean distance (reference px) between where the true and estimated H put each ink pixel."""
    p = cv2.perspectiveTransform(ink_pts, H_est)
    back = cv2.perspectiveTransform(p, np.linalg.inv(H_true))
    return float(np.linalg.norm(back - ink_pts, axis=2).mean())


def stretch_matrix(s):
    """
    For a whole-logo stretch: the 3x3 map (reference frame) the tamper applied,
    rebuilt from tamper_stretch() in synth.py. The 'stretched truth' is H_true @ T.
    """
    axis, pct = s["info"]["axis"], float(s["info"]["detail"].split()[1].rstrip("%"))
    scale = 1 + pct / 100
    ys, xs = np.nonzero(s["logo"]["alpha"] > 0.001)
    if axis == "x":
        t = np.array([[scale, 0, xs.min() * (1 - scale)], [0, 1, 0], [0, 0, 1]])
    else:
        mid = (ys.min() + ys.max()) / 2
        t = np.array([[1, 0, 0], [0, scale, mid * (1 - scale)], [0, 0, 1]])
    pad = np.array([[1, 0, synth.PAD], [0, 1, synth.PAD], [0, 0, 1]], float)   # reference frame -> canvas
    return np.linalg.inv(pad) @ t @ pad


def tilt_deg(pose):
    """Angle between the camera axis and the logo's normal (0 = facing the camera)."""
    r = synth.rotation(pose["yaw"], pose["pitch"], pose["roll"])
    return float(np.degrees(np.arccos(abs(r[2, 2]))))


def one(job):
    row, extra = job
    cv2.setNumThreads(1)
    s = rebuild(row)
    H_true = s["H_true"]
    photo = cv2.imread(str(DATA / "D2" / f"{row['id']}_photo.jpg"))

    ref_gray = to_gray(load_reference(1))
    ink = np.argwhere(green_ink_mask(load_reference(1)) > 0)[::7, ::-1].astype(np.float32).reshape(-1, 1, 2)
    img_gray = to_gray(photo)
    sift = cv2.SIFT_create(nfeatures=5000)

    H, stats = fit_homography(ref_gray, img_gray, sift)        # the real EXP-001 call
    H_raw, n_good, raw_inl = raw_ransac(ref_gray, img_gray, sift)

    pose = s["pose"]
    mid = cv2.perspectiveTransform(np.float32([[[0, 130]], [[920, 130]]]), H_true).reshape(-1, 2)
    out = {"id": row["id"], "extra_failed_sample": extra, "tamper": row["tamper"], "target": row["target"],
           "detail": row["detail"], "magnitude": row["magnitude"],
           "yaw": pose["yaw"], "pitch": pose["pitch"], "roll": pose["roll"], "tilt_deg": round(tilt_deg(pose), 1),
           "logo_width_in_photo_px": round(float(np.linalg.norm(mid[1] - mid[0]))),
           "photo_identical_to_stored": compare_with_stored(s) == 0,
           "good_matches": stats["good_matches"], "inliers": stats["inliers"] or raw_inl,
           "reproj_err_px": stats["reproj_err_px"], "passed": H is not None}

    if H is None:
        if n_good < MIN_INLIERS:
            reason = f"only {n_good} matches"
        elif raw_inl < MIN_INLIERS:
            reason = f"{raw_inl} inliers < {MIN_INLIERS}"
        elif H_raw is not None and not looks_sane(H_raw, ref_gray.shape):
            reason = "outline not sane"
        else:
            reason = "no H"
        out["fail_reason"] = reason
    else:
        out["fail_reason"] = ""
        # our copy of the matching must give the same answer as the real function
        out["raw_matches_official"] = bool(np.allclose(H, H_raw))

    if H_raw is not None:                       # error the crop WOULD have had (also for failures)
        out["corner_err_raw_px"] = round(corner_errors(H_raw, H_true)[0], 2)
    if H is not None:
        H_ecc, ecc = refine_with_ecc(ref_gray, img_gray, H)
        out["ecc_corr"] = ecc
        for tag, h in (("ransac", H), ("ecc", H_ecc)):
            mx, mean = corner_errors(h, H_true)
            out[f"corner_max_{tag}_px"] = round(mx, 2)
            out[f"corner_mean_{tag}_px"] = round(mean, 2)
            out[f"ink_err_{tag}_refpx"] = round(ink_error(h, H_true, ink), 3)
        if row["tamper"] == "stretch" and row["target"] == "whole":
            H_stretched = H_true @ stretch_matrix(s)
            out["corner_max_ecc_vs_stretched_truth_px"] = round(corner_errors(H_ecc, H_stretched)[0], 2)
        out["H_ecc"] = H_ecc
    out["H_true"] = H_true
    return out


# =============================================================================

def save_csv(rows):
    keys = ["id", "extra_failed_sample", "tamper", "target", "detail", "magnitude", "yaw", "pitch", "roll",
            "tilt_deg", "logo_width_in_photo_px", "photo_identical_to_stored", "good_matches", "inliers",
            "reproj_err_px", "passed", "fail_reason", "raw_matches_official", "ecc_corr", "corner_err_raw_px",
            "corner_max_ransac_px", "corner_mean_ransac_px", "ink_err_ransac_refpx",
            "corner_max_ecc_px", "corner_mean_ecc_px", "ink_err_ecc_refpx", "corner_max_ecc_vs_stretched_truth_px"]
    with open(OUT / "accuracy.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def col(rows, key):
    return np.array([float(r[key]) if r.get(key, "") not in ("", None) else np.nan for r in rows])


def pct(v, q):
    v = v[~np.isnan(v)]
    return float(np.percentile(v, q)) if len(v) else float("nan")


def figures(rows):
    main = [r for r in rows if not r["extra_failed_sample"]]
    ok = [r for r in main if r["passed"]]
    normal = [r for r in ok if not (r["tamper"] == "stretch" and r["target"] == "whole")]

    # 1. distribution of the error before and after ECC
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    for key, label, c in (("corner_max_ransac_px", "RANSAC only", "#1f77b4"), ("corner_max_ecc_px", "after ECC", "#ff7f0e")):
        v = np.sort(col(normal, key))
        ax[0].plot(v, np.arange(1, len(v) + 1) / len(v), label=f"{label}: median {np.median(v):.2f}, 95th {pct(v, 95):.2f} px", color=c)
    ax[0].axvline(WRONG_PX, color="#d62728", ls="--", label=f"'wrong' threshold {WRONG_PX:.0f} px")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("max corner error of the 920x260 logo box (photo px, log)")
    ax[0].set_ylabel("share of registered photos")
    ax[0].set_title(f"Registration error vs ground truth (n={len(normal)} passed; whole-logo stretches excluded)")
    ax[0].legend(fontsize=9)
    for key, label, c in (("ink_err_ransac_refpx", "RANSAC only", "#1f77b4"), ("ink_err_ecc_refpx", "after ECC", "#ff7f0e")):
        v = np.sort(col(normal, key))
        ax[1].plot(v, np.arange(1, len(v) + 1) / len(v), label=f"{label}: median {np.median(v):.2f}, 95th {pct(v, 95):.2f}", color=c)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("mean misalignment of the logo INK (reference px, log)")
    ax[1].set_title("Same, measured on the ink in the logo frame")
    ax[1].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "01_error_before_after_ecc.png", dpi=80)
    plt.close(fig)

    # 2. error per tamper type
    fig, ax = plt.subplots(figsize=(14, 5.5))
    data, labels = [], []
    for t in TAMPER_ORDER:
        sel = [r for r in ok if r["tamper"] == t]
        for key in ("corner_max_ransac_px", "corner_max_ecc_px"):
            data.append(col(sel, key))
            labels.append(f"{t}\n{'RANSAC' if 'ransac' in key else 'ECC'} n={len(sel)}")
    bp = ax.boxplot(data, tick_labels=labels, showfliers=True, patch_artist=True)
    for i, b in enumerate(bp["boxes"]):
        b.set_facecolor("#9ecae1" if i % 2 == 0 else "#fdae6b")
    ax.axhline(WRONG_PX, color="#d62728", ls="--")
    ax.set_yscale("log")
    ax.set_ylabel("max corner error (photo px, log)")
    ax.set_title("Error per tamper type, RANSAC only (blue) vs after ECC (orange). Whole-logo stretches ARE included in 'stretch'.")
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "02_error_by_tamper.png", dpi=80)
    plt.close(fig)

    # 3. error vs pose, coloured by kind of sample
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))
    for a, key, label in zip(ax, ("yaw", "pitch", "tilt_deg"), ("|yaw| (deg)", "|pitch| (deg)", "total tilt from facing the camera (deg)")):
        for g in GROUPS:
            sel = [r for r in normal if group(r) == g]
            a.scatter(np.abs(col(sel, key)), col(sel, "corner_max_ecc_px"), s=14, alpha=0.7, c=GROUP_COLOURS[g], label=f"{g} (n={len(sel)})")
        clean = [r for r in normal if group(r) == GROUPS[0]]
        x, e = np.abs(col(clean, key)), col(clean, "corner_max_ecc_px")
        rank = np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(e)))[0, 1]
        a.set_yscale("log")
        a.axhline(WRONG_PX, color="#d62728", ls="--")
        a.set_xlabel(label)
        a.set_ylabel("max corner error after ECC (photo px)")
        a.set_title(f"rank correlation, {GROUPS[0]} only: {rank:+.2f}", fontsize=10)
    ax[0].legend(fontsize=8)
    fig.suptitle("Does a stronger camera angle make registration worse? (passed, whole-logo stretches excluded)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "03_error_vs_pose.png", dpi=80)
    plt.close(fig)

    # 4. error vs inliers, including what failed photos would have given
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for g in GROUPS + ["whole-logo stretch (vs camera truth)"]:
        sel = [r for r in rows if r["passed"] and group(r) == g]
        ax.scatter(col(sel, "inliers"), np.maximum(col(sel, "corner_max_ecc_px"), 0.05), s=22, alpha=0.7,
                   c=GROUP_COLOURS[g], label=f"passed, {g}: n={len(sel)}")
    sel = [r for r in rows if not r["passed"]]
    ax.scatter(col(sel, "inliers"), col(sel, "corner_err_raw_px"), s=30, c="k", marker="x",
               label=f"FAILED (raw RANSAC H, no checks): n={len(sel)}")
    ax.axvline(MIN_INLIERS, color="k", ls="--", label=f"EXP-001 minimum: {MIN_INLIERS} inliers")
    ax.axhline(WRONG_PX, color="#d62728", ls=":", label=f"{WRONG_PX:.0f} px")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("RANSAC inliers (log)")
    ax.set_ylabel("max corner error (photo px, log); passed = after ECC")
    ax.set_title("Inliers vs error. Failed photos: all 23 failures of the 1500 (error of the H that was rejected)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "04_error_vs_inliers.png", dpi=80)
    plt.close(fig)

    # 6. can the ECC score tell us when not to trust the result?
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for g in GROUPS + ["whole-logo stretch (vs camera truth)"]:
        sel = [r for r in ok if group(r) == g]
        ax.scatter(col(sel, "ecc_corr"), col(sel, "corner_max_ecc_px"), s=22, alpha=0.7, c=GROUP_COLOURS[g], label=f"{g}: n={len(sel)}")
    ax.axvline(ECC_GATE, color="k", ls="--", label=f"ECC score {ECC_GATE}")
    ax.axhline(WRONG_PX, color="#d62728", ls=":", label=f"{WRONG_PX:.0f} px")
    ax.set_yscale("log")
    ax.set_xlabel("ECC correlation score (EXP-001 'ecc_corr')")
    ax.set_ylabel("max corner error after ECC (photo px, log)")
    ax.set_title("ECC score vs true error (passed registrations)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "06_ecc_score_vs_error.png", dpi=80)
    plt.close(fig)


def worst_cases(rows, n=6):
    """Passed registrations with the largest error: true box (yellow) vs found box (red)."""
    ok = [r for r in rows if r["passed"] and not (r["tamper"] == "stretch" and r["target"] == "whole")]
    ok.sort(key=lambda r: -r["corner_max_ecc_px"])
    fig, ax = plt.subplots(2, n, figsize=(4.2 * n, 9))
    for i, r in enumerate(ok[:n]):
        photo = cv2.imread(str(DATA / "D2" / f"{r['id']}_photo.jpg"))
        pic = outline(outline(photo, r["H_true"], (0, 220, 255), 6), r["H_ecc"], (0, 0, 255), 3)
        ax[0, i].imshow(rgb(pic))
        ax[0, i].set_title(f"{r['id']} {r['tamper']}\ncorner err {r['corner_max_ecc_px']:.1f} px, "
                           f"ink err {r['ink_err_ecc_refpx']:.2f} ref px", fontsize=9, loc="left")
        crop = cv2.warpPerspective(photo, np.linalg.inv(r["H_ecc"]), (920, 260))
        edges = cv2.Canny(green_ink_mask(load_reference(1)), 50, 150) > 0
        crop[edges] = (0, 0, 255)
        ax[1, i].imshow(rgb(crop))
        ax[1, i].set_title(f"crop + reference edges; {r['inliers']} inliers, ECC {r['ecc_corr']}", fontsize=9, loc="left")
        for a in ax[:, i]:
            a.set_xticks([])
            a.set_yticks([])
    fig.suptitle("Largest errors among PASSED registrations. Yellow = true logo box, red = found box.", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "05_worst_passed.jpg", dpi=70)
    plt.close(fig)


def summary(rows):
    main = [r for r in rows if not r["extra_failed_sample"]]
    ok = [r for r in main if r["passed"]]
    whole = [r for r in ok if r["tamper"] == "stretch" and r["target"] == "whole"]
    normal = [r for r in ok if r not in whole]
    lines = [f"samples: {len(main)} (first D2 rows of data/synth) + {len(rows) - len(main)} extra = the other photos that "
             f"failed registration in registration_D2.csv",
             f"rebuilt photo identical to the stored file: {sum(r['photo_identical_to_stored'] for r in rows)}/{len(rows)}",
             f"our raw RANSAC copy gives the same H as fit_homography: {sum(r.get('raw_matches_official', False) for r in rows if r['passed'])}"
             f"/{sum(r['passed'] for r in rows)}",
             "", f"== success rate (main {len(main)}) ==",
             f"passed the checks: {len(ok)}/{len(main)} ({100 * len(ok) / len(main):.1f} %)"]
    for t in TAMPER_ORDER:
        sel = [r for r in main if r["tamper"] == t]
        lines.append(f"  {t:11s} {sum(r['passed'] for r in sel):3d}/{len(sel)}")

    lines += ["", f"== error of PASSED registrations vs ground truth (n={len(normal)}, whole-logo stretches excluded) =="]
    for key, label in (("corner_max_ransac_px", "max corner err, RANSAC only (photo px)"),
                       ("corner_max_ecc_px", "max corner err, after ECC  (photo px)"),
                       ("corner_mean_ransac_px", "mean corner err, RANSAC only (photo px)"),
                       ("corner_mean_ecc_px", "mean corner err, after ECC  (photo px)"),
                       ("ink_err_ransac_refpx", "ink err, RANSAC only (reference px)"),
                       ("ink_err_ecc_refpx", "ink err, after ECC  (reference px)")):
        v = col(normal, key)
        lines.append(f"  {label:42s} median {np.median(v):6.2f}  95th {pct(v, 95):6.2f}  max {np.nanmax(v):7.2f}")
    better = np.mean(col(normal, "corner_max_ecc_px") < col(normal, "corner_max_ransac_px"))
    lines.append(f"  ECC reduced the corner error in {100 * better:.0f} % of photos")

    lines += ["", "== same, after ECC, per group (max corner err photo px | ink err reference px) =="]
    for g in GROUPS:
        sel = [r for r in normal if group(r) == g]
        e, ink = col(sel, "corner_max_ecc_px"), col(sel, "ink_err_ecc_refpx")
        e0 = col(sel, "corner_max_ransac_px")
        lines.append(f"  {g:24s} n={len(sel):3d}  RANSAC median {np.median(e0):5.2f} | ECC median {np.median(e):5.2f} "
                     f"95th {pct(e, 95):6.2f} max {np.nanmax(e):6.2f} px | ink median {np.median(ink):5.2f} max {np.nanmax(ink):5.2f}")

    lines += ["", f"== does 'passed' mean 'correct'? (wrong = max corner error > {WRONG_PX:.0f} px) =="]
    for key, label in (("corner_max_ransac_px", "RANSAC only"), ("corner_max_ecc_px", "after ECC")):
        bad = [r for r in normal if r[key] > WRONG_PX]
        lines.append(f"  {label}: {len(bad)}/{len(normal)} passed-but-wrong: " +
                     ", ".join(f"{r['id']} ({r['tamper']}, {r[key]:.1f} px, ink {r['ink_err_' + key.split('_')[2] + '_refpx']:.2f} ref px, "
                               f"{r['inliers']} inl, ECC {r.get('ecc_corr')})" for r in bad))
    bad_ink = [r for r in normal if r["ink_err_ecc_refpx"] > 2]
    lines.append(f"  after ECC, ink misaligned by > 2 reference px: {len(bad_ink)}/{len(normal)}: " +
                 ", ".join(f"{r['id']} ({r['tamper']}, {r['ink_err_ecc_refpx']:.2f})" for r in bad_ink))

    lines += ["", f"== would an ECC-score gate (>= {ECC_GATE}) catch the wrong ones? =="]
    ecc = col(normal, "ecc_corr")
    wrong = col(normal, "corner_max_ecc_px") > WRONG_PX
    low = ~(ecc >= ECC_GATE)                    # nan (ECC failed) counts as low
    lines.append(f"  wrong after ECC: {wrong.sum()}; of those, ECC < {ECC_GATE} (or failed): {(wrong & low).sum()}")
    lines.append(f"  wrong but ECC >= {ECC_GATE}: " + ", ".join(f"{r['id']} ({r['tamper']} {r['target']}, ecc {r['ecc_corr']})"
                                                         for r, w, l in zip(normal, wrong, low) if w and not l))
    lines.append(f"  correct (<= {WRONG_PX:.0f} px) but ECC < {ECC_GATE}: {(~wrong & low).sum()}/{(~wrong).sum()}")

    lines += ["", f"== whole-logo stretches (n={len(whole)}): registration absorbs the stretch =="]
    for r in whole:
        lines.append(f"  {r['id']} {r['detail']:10s} vs camera truth {r['corner_max_ecc_px']:6.2f} px, "
                     f"vs stretched truth {r['corner_max_ecc_vs_stretched_truth_px']:6.2f} px")

    failed = [r for r in rows if not r["passed"]]
    lines += ["", f"== failed registrations (all {len(failed)}, incl. extra): would the crop have been wrong? =="]
    for r in failed:
        lines.append(f"  {r['id']} {r['tamper']:10s} {r['fail_reason']:22s} raw RANSAC corner err "
                     f"{r.get('corner_err_raw_px', 'n/a')}")

    lines += ["", "== inliers vs error (passed, after ECC, whole-logo stretches excluded) =="]
    inl, err = col(normal, "inliers"), col(normal, "corner_max_ecc_px")
    clean = np.array([group(r) == GROUPS[0] for r in normal])
    for lo, hi in ((15, 30), (30, 50), (50, 80), (80, 1000)):
        sel = (inl >= lo) & (inl < hi)
        if sel.any():
            lines.append(f"  {lo:3d}-{hi:<4d} inliers: n={sel.sum():3d}  median {np.median(err[sel]):.2f}  "
                         f"95th {np.percentile(err[sel], 95):.2f}  max {err[sel].max():.2f} px; "
                         f"{GROUPS[0]} only: n={(sel & clean).sum()} max {err[sel & clean].max():.2f} px")
    fails_inl = col(failed, "inliers")
    raw = col(failed, "corner_err_raw_px")
    lines.append(f"  failed photos whose rejected H was within {WRONG_PX:.0f} px anyway: {(raw <= WRONG_PX).sum()}/{len(failed)}")
    lines.append(f"  lowest inlier count among passed: {int(np.nanmin(inl))}; failed photos had "
                 f"{', '.join(str(int(x)) for x in sorted(fails_inl))} raw inliers")

    lines += ["", "== RANSAC settings ==",
              f"EXP-001 call: cv2.findHomography(src, dst, cv2.RANSAC, ransacReprojThreshold={RANSAC_PX})",
              "not passed, so OpenCV defaults apply: maxIters=2000, confidence=0.995",
              "(C++ signature in opencv2/calib3d.hpp: findHomography(src, dst, method=0, ransacReprojThreshold=3, "
              "mask=noArray(), maxIters=2000, confidence=0.995); OpenCV adapts the number of iterations to the "
              "inlier ratio, so 2000 is only the cap. Version used here: " + cv2.__version__ + ")"]
    text = "\n".join(lines)
    (OUT / "summary.txt").write_text(text + "\n")
    print(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="how many D2 samples (from the first rows)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    d2 = [r for r in csv.DictReader(open(DATA / "labels.csv")) if r["set"] == "D2"]
    main_rows = d2[:args.n]
    # add every other photo that failed in the F12 run, to see what failures look like
    failed_ids = {r["id"] for r in csv.DictReader(open(DATA / "registration_D2.csv")) if r["registered"] == "False"}
    extra = [r for r in d2[args.n:] if r["id"] in failed_ids]
    jobs = [(r, False) for r in main_rows] + [(r, True) for r in extra]

    with get_context("spawn").Pool(args.workers) as pool:
        rows = []
        for i, r in enumerate(pool.imap(one, jobs, chunksize=2), 1):
            rows.append(r)
            if i % 50 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)}", flush=True)

    save_csv(rows)
    np.save(OUT / "homographies.npy", {r["id"]: {"true": r["H_true"], "ecc": r.get("H_ecc")} for r in rows}, allow_pickle=True)
    figures(rows)
    worst_cases(rows)
    summary(rows)


if __name__ == "__main__":
    main()
