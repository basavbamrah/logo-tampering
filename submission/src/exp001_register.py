"""
EXP-001: can we line up every photo with the reference logo?

For each of the 50 photos:
  1. find SIFT keypoints in the photo and in the reference logo
  2. match them, and fit a homography (a perspective transform) with RANSAC,
     which ignores the wrong matches
  3. refine the fit with ECC (pixel-level alignment, insensitive to brightness)
  4. warp the photo into the reference frame ("rectify" it) and save the crop

IMPORTANT: this step does NOT decide whether a logo is genuine. "Registered"
only means "we could line it up". A tampered logo still lines up, because most
of its points (icon, untouched letters) still match. Finding the differences
is done afterwards on the rectified crops (EXP-008 flow, EXP-010 difference
map, EXP-007 models). One known blind spot: if the WHOLE logo was stretched
evenly, the fitted transform absorbs the stretch and the crop looks normal.

We record how well each step worked, so we know which photos can be trusted
for any method that compares against the reference.

There is also one exploratory number per image, "implied_stretch" (see
implied_stretch() below). It was tested on synthetic photos with a known
stretch (src/exp003_stretch_diagnostic.py) and turned out too noisy to detect
1-8 % stretches, so it is kept only for the record.

Output: outputs/exp001/registration.csv, rectified crops, check sheets.
Run:    python src/exp001_register.py
"""
import csv

import cv2
import numpy as np

from common import OUTPUTS, fix_seeds, green_ink_mask, list_images, load_reference

OUT = OUTPUTS / "exp001"
RATIO_TEST = 0.75      # Lowe's ratio test for keeping a match
RANSAC_PX = 4.0        # reprojection tolerance for RANSAC, in pixels
MIN_INLIERS = 15       # below this we call the registration failed


def to_gray(bgr):
    # Contrast-normalised grey makes SIFT and ECC less sensitive to lighting.
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def fit_homography(ref_gray, img_gray, sift):
    """SIFT + ratio test + RANSAC. Returns H (ref -> image) and match stats."""
    kp_r, des_r = sift.detectAndCompute(ref_gray, None)
    kp_i, des_i = sift.detectAndCompute(img_gray, None)
    stats = {"kp_ref": len(kp_r), "kp_img": len(kp_i), "good_matches": 0, "inliers": 0,
             "inlier_ratio": 0.0, "reproj_err_px": np.nan}
    if des_i is None or len(kp_i) < 10:
        return None, stats

    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des_r, des_i, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < RATIO_TEST * n.distance]

    # Each photo keypoint may be used only once (keep its best match).
    # Without this, a few images had dozens of reference points all matched
    # to ONE photo point, and RANSAC happily "fitted" a homography that
    # squashes the whole logo into that point.
    best = {}
    for m in good:
        if m.trainIdx not in best or m.distance < best[m.trainIdx].distance:
            best[m.trainIdx] = m
    good = list(best.values())
    stats["good_matches"] = len(good)
    if len(good) < MIN_INLIERS:
        return None, stats

    src = np.float32([kp_r[m.queryIdx].pt for m in good])
    dst = np.float32([kp_i[m.trainIdx].pt for m in good])
    H, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_PX)
    if H is None:
        return None, stats

    inl = inlier_mask.ravel().astype(bool)
    projected = cv2.perspectiveTransform(src[inl].reshape(-1, 1, 2), H).reshape(-1, 2)
    stats["inliers"] = int(inl.sum())
    stats["inlier_ratio"] = round(float(inl.mean()), 3)
    stats["reproj_err_px"] = round(float(np.linalg.norm(projected - dst[inl], axis=1).mean()), 3)
    if stats["inliers"] < MIN_INLIERS or not looks_sane(H, ref_gray.shape):
        return None, stats
    return H, stats


def looks_sane(H, ref_shape):
    """
    Reject homographies that cannot be a photo of a flat logo: the projected
    logo box must keep its corner order (no folding) and keep a reasonable size.
    """
    h, w = ref_shape[:2]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    quad = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    if not cv2.isContourConvex(quad.astype(np.float32).reshape(-1, 1, 2)):
        return False
    area_ratio = cv2.contourArea(quad) / float(w * h)
    return 0.05 < area_ratio < 20


def refine_with_ecc(ref_gray, img_gray, H):
    """
    Start from the RANSAC homography and let ECC nudge it to sub-pixel accuracy.
    Returns (refined H, correlation score). If ECC fails we keep the old H.
    """
    h, w = ref_gray.shape
    # ECC works on the image already warped into the reference frame, so we
    # refine a small correction on top of H.
    warped = cv2.warpPerspective(img_gray, np.linalg.inv(H), (w, h))

    inside = cv2.warpPerspective(np.full(img_gray.shape, 255, np.uint8), np.linalg.inv(H), (w, h))
    inside = cv2.erode(inside, np.ones((5, 5), np.uint8))
    correction = np.eye(3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6)
    try:
        score, correction = cv2.findTransformECC(
            ref_gray.astype(np.float32), warped.astype(np.float32), correction,
            cv2.MOTION_HOMOGRAPHY, criteria, inside, 5)
    except cv2.error:
        return H, np.nan
    # warpPerspective with H^-1 gives warped(x) = img(H x). ECC finds a small
    # correction C with ref(x) ~ warped(C x) = img(H C x), so the new H is H @ C.
    return H @ correction.astype(np.float64), round(float(score), 4)


def implied_stretch(H, img_shape, f_fixed=None):
    """
    Exploratory. Decompose H as K [r1 r2 t] with a pinhole camera K
    (focal length f, centre at the image centre, square pixels).
    For a real camera r1 and r2 have equal length; their ratio is the
    horizontal stretch the homography "needs" relative to the reference.
    f is solved from r1 being perpendicular to r2; if that is unstable
    (nearly straight-on view) we fall back to a typical phone focal length.
    f_fixed lets the caller supply the focal length instead (used by the
    diagnostic). Returns (stretch, focal length used, how it was chosen).
    """
    h_img, w_img = img_shape[:2]
    cx, cy = w_img / 2.0, h_img / 2.0
    h1, h2 = H[:, 0], H[:, 1]
    a1 = np.array([h1[0] - cx * h1[2], h1[1] - cy * h1[2]])
    a2 = np.array([h2[0] - cx * h2[2], h2[1] - cy * h2[2]])

    f_default = 1.0 * max(w_img, h_img)
    f, f_source = f_default, "default"
    denom = h1[2] * h2[2]
    if f_fixed is not None:                      # focal length given by the caller
        f, f_source = float(f_fixed), "fixed"
    elif abs(denom) > 1e-12:
        f2 = -(a1 @ a2) / denom
        if f2 > 0 and 0.4 * max(w_img, h_img) < np.sqrt(f2) < 4.0 * max(w_img, h_img):
            f, f_source = float(np.sqrt(f2)), "solved"

    r1 = np.array([a1[0] / f, a1[1] / f, h1[2]])
    r2 = np.array([a2[0] / f, a2[1] / f, h2[2]])
    return round(float(np.linalg.norm(r1) / np.linalg.norm(r2)), 4), round(f, 1), f_source


def logo_inside_frame(H, ref_shape, img_shape):
    """
    Is the whole reference logo box projected inside the photo? (False = cut off)
    Known to over-report: the box includes a margin around the logo, so a logo
    close to the edge counts as "cut" even when the ink is complete. EXP-010's
    cut_pct (share of logo ink outside the photo) is the better measure.
    """
    h, w = ref_shape[:2]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    pts = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    H_img, W_img = img_shape[:2]
    return bool((pts[:, 0] >= 0).all() and (pts[:, 0] < W_img).all()
                and (pts[:, 1] >= 0).all() and (pts[:, 1] < H_img).all())


def edge_overlay(ref_bgr, rectified):
    """Rectified crop with the reference logo outline drawn in red, to eyeball alignment."""
    edges = cv2.Canny(green_ink_mask(ref_bgr), 50, 150) > 0
    out = rectified.copy()
    out[edges] = (0, 0, 255)
    return out


def contact_sheet(tiles, names, cols=5, tile_w=460):
    rows = []
    for i in range(0, len(tiles), cols):
        row = []
        for tile, name in zip(tiles[i:i + cols], names[i:i + cols]):
            t = cv2.resize(tile, (tile_w, int(tile.shape[0] * tile_w / tile.shape[1])))
            cv2.rectangle(t, (0, 0), (70, 26), (0, 255, 255), -1)
            cv2.putText(t, name, (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
            row.append(np.pad(t, ((3, 3), (3, 3), (0, 0)), constant_values=255))
        while len(row) < cols:
            row.append(np.full_like(row[0], 255))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def main():
    fix_seeds()   # same result on every re-run
    (OUT / "rectified").mkdir(parents=True, exist_ok=True)
    (OUT / "overlay").mkdir(parents=True, exist_ok=True)
    # Remove results of earlier runs. (An old run once left a garbage crop for
    # an image that no longer registers, and later experiments picked it up.)
    for old in list((OUT / "rectified").glob("*")) + list((OUT / "overlay").glob("*")):
        old.unlink()

    ref = load_reference(1)
    ref_gray = to_gray(ref)
    sift = cv2.SIFT_create(nfeatures=5000)

    rows, overlays, names = [], [], []
    for path in list_images():
        name = path.stem
        img = cv2.imread(str(path))
        H, stats = fit_homography(ref_gray, to_gray(img), sift)
        row = {"image": name, **stats, "registered": H is not None}

        if H is not None:
            H, ecc = refine_with_ecc(ref_gray, to_gray(img), H)
            stretch, f, f_src = implied_stretch(H, img.shape)
            # Hypothesis check (see notes): the stretch values first came out at
            # ~0.75 / 1.0 / 1.33, i.e. exactly width/height of the photo. That
            # looks like whole photos rendered square and then resized to 3:4 or
            # 4:3. Undo that resize (make the pixels square again) and recompute.
            h_img, w_img = img.shape[:2]
            unsquash = np.diag([h_img / w_img, 1.0, 1.0])
            stretch_sq, _, _ = implied_stretch(unsquash @ H, (h_img, h_img))
            rectified = cv2.warpPerspective(img, np.linalg.inv(H), (ref.shape[1], ref.shape[0]))
            cv2.imwrite(str(OUT / "rectified" / f"{name}.png"), rectified)
            overlay = edge_overlay(ref, rectified)
            cv2.imwrite(str(OUT / "overlay" / f"{name}.png"), overlay)
            np.save(OUT / "rectified" / f"{name}_H.npy", H)
            row.update({"ecc_corr": ecc, "logo_inside_frame": logo_inside_frame(H, ref.shape, img.shape),
                        "implied_stretch": stretch, "stretch_square_px": stretch_sq,
                        "focal_px": f, "focal_source": f_src})
            overlays.append(overlay)
        else:
            row.update({"ecc_corr": np.nan, "logo_inside_frame": None,
                        "implied_stretch": np.nan, "stretch_square_px": np.nan,
                        "focal_px": np.nan, "focal_source": ""})
            overlays.append(np.full((ref.shape[0], ref.shape[1], 3), 200, np.uint8))
        names.append(name)
        rows.append(row)
        print(f"{name}: registered={row['registered']} inliers={row['inliers']} "
              f"err={row['reproj_err_px']} ecc={row['ecc_corr']} stretch={row['implied_stretch']} "
              f"square_px={row.get('stretch_square_px')}")

    with open(OUT / "registration.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    cv2.imwrite(str(OUT / "overlay_sheet_1.jpg"), contact_sheet(overlays[:25], names[:25]))
    cv2.imwrite(str(OUT / "overlay_sheet_2.jpg"), contact_sheet(overlays[25:], names[25:]))

    ok = sum(r["registered"] for r in rows)
    print(f"\nregistered {ok}/{len(rows)} images; results in {OUT}")


if __name__ == "__main__":
    main()
