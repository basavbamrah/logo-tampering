"""
EXP-001 intermediate results: every step of the registration, as pictures and numbers.

exp001_register.py only saves the end result (rectified crop, overlay, H).
This script re-runs the same steps with the same settings and saves what
happens in between, so each step can be shown and checked:

  1. the photo as given
  2. what SIFT looks at (grey + contrast boost) and the keypoints it finds
  3. matching: which template keypoint was paired with which photo keypoint,
     and which pairs RANSAC kept (green) or threw out (red)
  4. the found logo: where it is, how much it is rotated, how big it is and
     how much one side is further away than the other (perspective)
  5. the rectified crop using the RANSAC homography only
  6. the rectified crop after the ECC fine-tuning (this is what later
     experiments use)

It does NOT change anything exp001_register.py saved. It reads the saved
H matrices (outputs/exp001/rectified/<id>_H.npy) only to check that it got the
same answer, and writes everything into outputs/exp001/steps/.

Outputs (in outputs/exp001/steps/), one image per photo in each stage folder:
  01_grey_clahe/          what SIFT receives: grey + local contrast boost
  02_keypoints/           every SIFT keypoint drawn on the photo (circle = patch it
                          describes, line = its main direction) - BEFORE matching
  03_matches_ratio_test/  pairs left after the ratio test (yellow)
  04_matches_unique/      pairs left after "each photo keypoint used once" (blue)
  04b_unique_removed/     what that step threw away: orange lines = template points that
                          picked an already-taken photo point and lost (a worse fingerprint match)
  05_ransac/              AFTER RANSAC: green = kept (inliers), red = rejected
  06_found_logo/          found logo outline + rotation arrow
  07_rectified_ransac/    rectified crop with the RANSAC homography only (+ red reference outline)
  08_rectified_ecc/       rectified crop after ECC (+ red reference outline) = what later steps use
  10_ransac_vs_ecc_zoom/  the "N" and "O", 4x zoom: RANSAC-only crop vs ECC crop, red = reference outline
  09_whole_photo_flat/    left: photo as given; right: the WHOLE photo flattened with the same H
                          (shows background text gets "un-squashed" together with the logo)
  combined/               all steps for one photo in a single figure
  template_keypoints.jpg  the template's keypoints
  keypoints_explained_<id>.jpg  matched pairs shown as little squares side by side
  found_logo_sheet_1/2.jpg      stage 06 for all 50 photos
  steps.csv               counts after each step + position / angle / size per photo

Run: python src/exp001_steps.py [--explain 041]
"""
import argparse
import csv

import cv2
import numpy as np

from common import OUTPUTS, list_images, load_reference
from exp001_register import (MIN_INLIERS, RANSAC_PX, RATIO_TEST, contact_sheet, edge_overlay,
                             looks_sane, refine_with_ecc, to_gray)

OUT = OUTPUTS / "exp001" / "steps"
SAVED = OUTPUTS / "exp001" / "rectified"   # read only
PANEL_W = 1000
FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN, RED, YELLOW, MAGENTA, GREY = (0, 200, 0), (0, 0, 255), (0, 220, 255), (255, 0, 255), (150, 150, 150)


def match(ref_gray, img_gray, sift):
    """Same matching rules as fit_homography() in exp001_register.py, but keeps every list for drawing."""
    kp_r, des_r = sift.detectAndCompute(ref_gray, None)
    kp_i, des_i = sift.detectAndCompute(img_gray, None)
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des_r, des_i, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < RATIO_TEST * n.distance]
    best = {}
    for m in good:
        if m.trainIdx not in best or m.distance < best[m.trainIdx].distance:
            best[m.trainIdx] = m
    unique = list(best.values())

    H, inl = None, np.zeros(len(unique), bool)
    if len(unique) >= MIN_INLIERS:
        src = np.float32([kp_r[m.queryIdx].pt for m in unique])
        dst = np.float32([kp_i[m.trainIdx].pt for m in unique])
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_PX)
        if mask is not None:
            inl = mask.ravel().astype(bool)
    found = H is not None and inl.sum() >= MIN_INLIERS and looks_sane(H, ref_gray.shape)
    return kp_r, kp_i, good, unique, inl, (H if found else None)


def logo_position(H, ref_shape):
    """
    Plain-language numbers about where the logo is, read off the 4 corners of
    the template after pushing them through H (all in photo pixels, photo as given):
      angle_deg      rotation of the logo's middle line; + = clockwise (right end lower)
      width_x, height_x   how many times larger than the reference it appears
      right_vs_left  length of the right edge / left edge. 1 = facing the camera;
                     < 1 = the right side is further away (label turned away on the right)
      top_vs_bottom  same for top edge / bottom edge
    """
    h, w = ref_shape[:2]
    pts = np.float32([[0, 0], [w, 0], [w, h], [0, h],          # corners
                      [0, h / 2], [w, h / 2], [w / 2, 0], [w / 2, h]])  # edge midpoints
    tl, tr, br, bl, ml, mr, mt, mb = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H).reshape(-1, 2)
    length = lambda a, b: float(np.linalg.norm(a - b))
    return {
        "centre_x": round(float((ml[0] + mr[0]) / 2)), "centre_y": round(float((ml[1] + mr[1]) / 2)),
        "angle_deg": round(float(np.degrees(np.arctan2(mr[1] - ml[1], mr[0] - ml[0]))), 1),
        "width_x": round(length(ml, mr) / w, 2), "height_x": round(length(mt, mb) / h, 2),
        "right_vs_left": round(length(tr, br) / length(tl, bl), 2),
        "top_vs_bottom": round(length(tl, tr) / length(bl, br), 2),
    }, (tl, tr, br, bl, ml, mr)


def overlap_score(ref_gray, crop_gray):
    """
    How well a rectified crop overlaps the reference: correlation of the two
    grey images (1 = identical up to brightness/contrast). This is the quantity
    ECC maximises, so it shows what ECC gained over RANSAC alone.
    """
    a = ref_gray.astype(np.float64).ravel()
    b = crop_gray.astype(np.float64).ravel()
    a, b = a - a.mean(), b - b.mean()
    return round(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)), 4)


def zoom_compare(ref, crop_ransac, crop_ecc, box=(180, 20, 470, 180), factor=4):
    """Zoom on part of the logo frame (default: "N" and "O") for both crops, reference outline in red."""
    x0, y0, x1, y1 = box
    tiles = []
    for crop in (crop_ransac, crop_ecc):
        z = edge_overlay(ref, crop)[y0:y1, x0:x1]
        tiles.append(cv2.resize(z, None, fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST))
    gap = np.full((tiles[0].shape[0], 12, 3), 255, np.uint8)
    return np.hstack([tiles[0], gap, tiles[1]])


def corner_shift(H_a, H_b, ref_shape):
    """How far (photo px, mean over the 4 corners) the logo box moves between two homographies."""
    h, w = ref_shape[:2]
    c = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    a = cv2.perspectiveTransform(c, H_a).reshape(-1, 2)
    b = cv2.perspectiveTransform(c, H_b).reshape(-1, 2)
    return round(float(np.linalg.norm(a - b, axis=1).mean()), 2)


# ---------- drawing helpers ----------

def titled(img, title, width):
    """Resize to the given width and put a white title bar on top."""
    scale = width / img.shape[1]
    img = cv2.resize(img, (width, int(round(img.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    bar = np.full((44, width, 3), 255, np.uint8)
    cv2.putText(bar, title, (8, 30), FONT, 0.72, (0, 0, 0), 2, cv2.LINE_AA)
    return np.vstack([bar, img])


def text_panel(lines, shape):
    panel = np.full((shape[0], shape[1], 3), 255, np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(panel, line, (14, 44 + 40 * i), FONT, 0.85, (0, 0, 0), 2, cv2.LINE_AA)
    return panel


def matching_picture(ref, img, kp_r, kp_i, matches, kept, H, colour=None):
    """
    Template on the left, photo on the right, lines between matched keypoints.
    With colour=None the lines are green (kept by RANSAC) or red (rejected);
    otherwise every line gets that colour. H (if given) draws the found logo.
    """
    h = max(ref.shape[0], img.shape[0])
    canvas = np.full((h, ref.shape[1] + img.shape[1], 3), 255, np.uint8)
    canvas[:ref.shape[0], :ref.shape[1]] = ref
    canvas[:img.shape[0], ref.shape[1]:] = img
    dx = ref.shape[1]
    for k in kp_i:
        cv2.circle(canvas, (int(k.pt[0]) + dx, int(k.pt[1])), 2, GREY, -1)
    for keep in (False, True):                      # rejected first, kept ones on top
        for m, is_in in zip(matches, kept):
            if is_in == keep:
                a = tuple(int(v) for v in kp_r[m.queryIdx].pt)
                b = (int(kp_i[m.trainIdx].pt[0]) + dx, int(kp_i[m.trainIdx].pt[1]))
                cv2.line(canvas, a, b, colour or (GREEN if is_in else RED), 1, cv2.LINE_AA)
    if H is not None:
        rh, rw = ref.shape[:2]
        corners = np.float32([[0, 0], [rw, 0], [rw, rh], [0, rh]]).reshape(-1, 1, 2)
        quad = cv2.perspectiveTransform(corners, H).reshape(-1, 2) + [dx, 0]
        cv2.polylines(canvas, [quad.astype(np.int32)], True, YELLOW, 4, cv2.LINE_AA)
    return canvas


def found_logo_picture(img, pts, pos):
    tl, tr, br, bl, ml, mr = pts
    out = img.copy()
    cv2.polylines(out, [np.int32([tl, tr, br, bl])], True, YELLOW, 4, cv2.LINE_AA)
    cv2.arrowedLine(out, tuple(np.int32(ml)), tuple(np.int32(mr)), MAGENTA, 4, cv2.LINE_AA, tipLength=0.04)
    cv2.putText(out, f"angle {pos['angle_deg']:+.1f} deg, size x{pos['width_x']:.2f}", (20, 50),
                FONT, 1.4, MAGENTA, 3, cv2.LINE_AA)
    return out


def whole_photo_flat(img, H, ref_shape, limit=2500):
    """
    Flatten the whole photo (not just the logo) with H^-1, in the logo frame's
    scale. Areas far outside the logo are clipped to `limit` px so a strongly
    tilted photo can't make a giant image.
    """
    h, w = img.shape[:2]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    flat = cv2.perspectiveTransform(corners, np.linalg.inv(H)).reshape(-1, 2)
    x0, y0 = np.clip(flat.min(axis=0), -limit, limit)
    x1, y1 = np.clip(flat.max(axis=0), -limit, ref_shape[1] + limit)
    shift = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], dtype=np.float64)
    out = cv2.warpPerspective(img, shift @ np.linalg.inv(H), (int(x1 - x0), int(y1 - y0)),
                              borderValue=(255, 255, 255))
    rh, rw = ref_shape[:2]       # mark where the logo frame is
    cv2.rectangle(out, (int(-x0), int(-y0)), (int(-x0 + rw), int(-y0 + rh)), YELLOW, 3)
    return out


def letterbox(img, size=600):
    """Fit into a white square so photos of different shapes can share a sheet."""
    scale = size / max(img.shape[:2])
    small = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    out = np.full((size, size, 3), 255, np.uint8)
    y, x = (size - small.shape[0]) // 2, (size - small.shape[1]) // 2
    out[y:y + small.shape[0], x:x + small.shape[1]] = small
    return out


def save(stage, name, img):
    folder = OUT / stage
    folder.mkdir(parents=True, exist_ok=True)
    ext = ".png" if stage.startswith(("07", "08")) else ".jpg"   # crops lossless, the rest jpg
    cv2.imwrite(str(folder / f"{name}{ext}"), img, [cv2.IMWRITE_JPEG_QUALITY, 85])


def failed_panel(shape, text):
    panel = np.full((shape[0], shape[1], 3), 225, np.uint8)
    cv2.putText(panel, text, (20, shape[0] // 2), FONT, 1.0, (0, 0, 160), 2, cv2.LINE_AA)
    return panel


# ---------- per photo ----------

def one_photo(path, ref, ref_gray, sift):
    name = path.stem
    img = cv2.imread(str(path))
    img_gray = to_gray(img)
    kp_r, kp_i, good, unique, inl, H = match(ref_gray, img_gray, sift)
    row = {"image": name, "photo_size": f"{img.shape[1]}x{img.shape[0]}",
           "photo_keypoints": len(kp_i), "template_keypoints": len(kp_r),
           "after_ratio_test": len(good), "after_unique": len(unique), "ransac_inliers": int(inl.sum()),
           "found": H is not None}

    # stages 01-05 exist for every photo, found or not
    save("01_grey_clahe", name, img_gray)
    save("02_keypoints", name, cv2.drawKeypoints(img, kp_i, None, color=YELLOW,
                                                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))
    save("03_matches_ratio_test", name, matching_picture(ref, img, kp_r, kp_i, good, [True] * len(good), None, YELLOW))
    save("04_matches_unique", name, matching_picture(ref, img, kp_r, kp_i, unique, [True] * len(unique), None, (255, 120, 0)))
    removed = [m for m in good if m not in unique]
    save("04b_unique_removed", name, matching_picture(ref, img, kp_r, kp_i, removed, [True] * len(removed), None, (0, 140, 255)))
    save("05_ransac", name, matching_picture(ref, img, kp_r, kp_i, unique, inl, H))

    # step 1 + 2
    seen = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    for k in kp_i:
        cv2.circle(seen, tuple(int(v) for v in k.pt), 3, YELLOW, -1)
    row1 = np.hstack([titled(img, "1. Photo as given", PANEL_W),
                      titled(seen, f"2. What SIFT sees (grey + CLAHE): {len(kp_i)} keypoints (yellow)", PANEL_W)])

    # step 3
    row2 = titled(matching_picture(ref, img, kp_r, kp_i, unique, inl, H),
                  f"3. Matching: {len(unique)} template->photo pairs; green = {int(inl.sum())} agree on one "
                  f"geometry (RANSAC), red = rejected", 2 * PANEL_W)

    if H is None:
        row.update({k: "" for k in ("centre_x", "centre_y", "angle_deg", "width_x", "height_x",
                                    "right_vs_left", "top_vs_bottom", "ecc_corr", "ecc_corner_shift_px",
                                    "overlap_ransac", "overlap_ecc",
                                    "same_as_saved_H")})
        lines = ["NOT FOUND", f"only {int(inl.sum())} pairs agree on one geometry,",
                 f"at least {MIN_INLIERS} are needed.", "No crop is made, so no wrong crop",
                 "reaches the later experiments.", "Reported as \"can't register\"."]
        half = (img.shape[0] * PANEL_W // img.shape[1], PANEL_W)
        row3 = np.hstack([titled(failed_panel(half, "4. logo not found"), "4. Found logo", PANEL_W),
                          titled(text_panel(lines, half), "Numbers", PANEL_W)])
        crop_shape = (ref.shape[0] * PANEL_W // ref.shape[1], PANEL_W)
        row4 = np.hstack([titled(failed_panel(crop_shape, "no crop"), "5. Rectified (RANSAC only)", PANEL_W),
                          titled(failed_panel(crop_shape, "no crop"), "6. Rectified after ECC", PANEL_W)])
        return np.vstack([row1, row2, row3, row4]), row, failed_panel(img.shape, "NOT FOUND")

    # step 4
    pos, pts = logo_position(H, ref.shape)
    found_pic = found_logo_picture(img, pts, pos)

    # step 5 + 6
    size = (ref.shape[1], ref.shape[0])
    crop_ransac = cv2.warpPerspective(img, np.linalg.inv(H), size)
    H_ecc, ecc = refine_with_ecc(ref_gray, img_gray, H)
    crop_ecc = cv2.warpPerspective(img, np.linalg.inv(H_ecc), size)
    shift = corner_shift(H, H_ecc, ref.shape)
    save("06_found_logo", name, found_pic)
    save("07_rectified_ransac", name, edge_overlay(ref, crop_ransac))
    save("08_rectified_ecc", name, edge_overlay(ref, crop_ecc))
    score_ransac = overlap_score(ref_gray, to_gray(crop_ransac))
    score_ecc = overlap_score(ref_gray, to_gray(crop_ecc))
    zoom = zoom_compare(ref, crop_ransac, crop_ecc)
    save("10_ransac_vs_ecc_zoom", name, titled(zoom, f"left: RANSAC only (overlap {score_ransac})   "
                                                     f"right: after ECC (overlap {score_ecc})", zoom.shape[1]))
    save("09_whole_photo_flat", name, np.hstack([letterbox(img, 1000), letterbox(whole_photo_flat(img, H_ecc, ref.shape), 1000)]))

    saved = SAVED / f"{name}_H.npy"
    same = bool(np.allclose(np.load(saved), H_ecc, rtol=1e-6, atol=1e-6)) if saved.exists() else "no saved H"
    row.update({**pos, "ecc_corr": ecc, "ecc_corner_shift_px": shift,
                "overlap_ransac": score_ransac, "overlap_ecc": score_ecc, "same_as_saved_H": same})

    lines = [f"Logo centre in photo: x={pos['centre_x']}, y={pos['centre_y']}",
             f"Rotation: {pos['angle_deg']:+.1f} deg (+ = clockwise)",
             f"Size vs reference: x{pos['width_x']:.2f} wide, x{pos['height_x']:.2f} tall",
             f"Right edge / left edge: {pos['right_vs_left']:.2f} (1 = facing camera)",
             f"Top edge / bottom edge: {pos['top_vs_bottom']:.2f}",
             f"ECC moved the corners by {shift:.2f} px", f"ECC similarity score: {ecc}"]
    row3 = np.hstack([titled(found_pic, "4. Found logo: template corners pushed through H (yellow)", PANEL_W),
                      titled(text_panel(lines, (found_pic.shape[0], found_pic.shape[1])), "Numbers", PANEL_W)])
    row4 = np.hstack([titled(edge_overlay(ref, crop_ransac), "5. Rectified, RANSAC only (red = reference outline)", PANEL_W),
                      titled(edge_overlay(ref, crop_ecc), f"6. Rectified after ECC (used later), score {ecc}", PANEL_W)])
    return np.vstack([row1, row2, row3, row4]), row, found_pic


# ---------- what a keypoint is ----------

def patch(img, kp, out=140):
    """Square around a keypoint, sized by the keypoint's own scale, resized for display."""
    half = int(max(10, 2.5 * kp.size))
    x, y = int(kp.pt[0]), int(kp.pt[1])
    padded = cv2.copyMakeBorder(img, half, half, half, half, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    return cv2.resize(padded[y:y + 2 * half, x:x + 2 * half], (out, out), interpolation=cv2.INTER_CUBIC)


def keypoints_explained(path, ref, ref_gray, sift):
    img = cv2.imread(str(path))
    kp_r, kp_i, _, unique, inl, H = match(ref_gray, to_gray(img), sift)
    # top: template keypoints with circle = the area the keypoint describes, line = its orientation
    rich = cv2.drawKeypoints(ref, kp_r, None, color=(0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    top = titled(rich, f"Template keypoints ({len(kp_r)}): circle = patch the keypoint describes, "
                       "line = its main direction", 1200)

    # bottom: 8 kept pairs spread over the logo + 4 rejected pairs
    kept = sorted([m for m, k in zip(unique, inl) if k], key=lambda m: kp_r[m.queryIdx].pt[0])
    kept = [kept[i] for i in np.linspace(0, len(kept) - 1, min(8, len(kept))).astype(int)] if kept else []
    rejected = [m for m, k in zip(unique, inl) if not k][:4]

    def pair_tile(m, label, colour):
        a, b = patch(ref, kp_r[m.queryIdx]), patch(img, kp_i[m.trainIdx])
        tile = np.hstack([a, np.full((140, 6, 3), 255, np.uint8), b])
        bar = np.full((30, tile.shape[1], 3), 255, np.uint8)
        cv2.putText(bar, label, (4, 21), FONT, 0.55, colour, 2, cv2.LINE_AA)
        return np.pad(np.vstack([bar, tile]), ((6, 6), (6, 6), (0, 0)), constant_values=255)

    tiles = [pair_tile(m, "kept: template | photo", GREEN) for m in kept]
    tiles += [pair_tile(m, "REJECTED: template | photo", RED) for m in rejected]
    rows = []
    for i in range(0, len(tiles), 4):
        chunk = tiles[i:i + 4]
        chunk += [np.full_like(tiles[0], 255)] * (4 - len(chunk))
        rows.append(np.hstack(chunk))
    grid = np.vstack(rows)
    grid = titled(grid, f"Photo {path.stem}: matched pairs. Left square = around a template keypoint, "
                        "right = its partner in the photo", 1200)
    return np.vstack([top, grid])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--explain", default="041", help="photo used for the keypoint explanation figure")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    ref = load_reference(1)
    ref_gray = to_gray(ref)
    sift = cv2.SIFT_create(nfeatures=5000)

    rows, found_pics, names = [], [], []
    for path in list_images():
        figure, row, found_pic = one_photo(path, ref, ref_gray, sift)
        save("combined", path.stem, figure)
        rows.append(row)
        found_pics.append(letterbox(found_pic))
        names.append(path.stem)
        print(f"{path.stem}: found={row['found']} inliers={row['ransac_inliers']} angle={row['angle_deg']} "
              f"size={row['width_x']} right/left={row['right_vs_left']} same_as_saved={row['same_as_saved_H']}")

    with open(OUT / "steps.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    cv2.imwrite(str(OUT / "found_logo_sheet_1.jpg"), contact_sheet(found_pics[:25], names[:25], tile_w=400))
    cv2.imwrite(str(OUT / "found_logo_sheet_2.jpg"), contact_sheet(found_pics[25:], names[25:], tile_w=400))
    kp_r = sift.detect(ref_gray, None)
    cv2.imwrite(str(OUT / "template_keypoints.jpg"),
                cv2.drawKeypoints(ref, kp_r, None, color=(0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))

    explain_path = next(p for p in list_images() if p.stem == args.explain)
    cv2.imwrite(str(OUT / f"keypoints_explained_{args.explain}.jpg"),
                keypoints_explained(explain_path, ref, ref_gray, sift), [cv2.IMWRITE_JPEG_QUALITY, 90])

    found = [r for r in rows if r["found"]]
    print(f"\nfound {len(found)}/{len(rows)}; same answer as the saved H for "
          f"{sum(r['same_as_saved_H'] is True for r in found)}/{len(found)}; outputs in {OUT}")


if __name__ == "__main__":
    main()
