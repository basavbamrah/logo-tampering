"""
Everything the app does, without the user interface:

  1. align the photo with the reference (EXP-001: grey + CLAHE -> SIFT -> ratio test ->
     one partner per photo point -> RANSAC homography -> sanity check -> ECC fine-tuning
     that ignores the area outside the photo -> warp onto the 920 x 260 logo frame)
  2. run the ONNX model on (reference, aligned photo): p(tampered), the tamper-type
     probabilities and the difference heatmap
  3. decide with the thresholds the model was validated with

The alignment code is copied from src/exp001_register.py (same settings), so the
app gives the same aligned crops as the experiments.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parent / "model"
INFO = json.loads((MODEL_DIR / "model_info.json").read_text())
# How often each decision band was right on the labelled synthetic test sets (scripts/evaluate_onnx.py)
CONFIDENCE = json.loads((MODEL_DIR / "confidence.json").read_text())["all_test_sets"]
REF_LOGO_BOX = tuple(INFO["ref_logo_box"])      # where the logo sits in the 1200 x 700 reference images
LOGO_W, LOGO_H = 920, 260
RATIO_TEST, RANSAC_PX, MIN_INLIERS = 0.75, 4.0, 15
CUT_PCT = 1.0                                   # % of reference ink outside the photo above which the logo counts as cut off
SEED = 0

# Fixed seeds, so the same photo always gives the same result (OpenCV's RANSAC
# already uses a fixed internal seed; ONNX Runtime on the CPU has no randomness).
cv2.setRNGSeed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------- alignment (EXP-001)
def to_gray(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def logo_template(reference_bgr):
    """The reference images are 1200 x 700 with the logo at a fixed place; anything else is taken as a logo crop."""
    if reference_bgr.shape[:2] == (700, 1200):
        x0, y0, x1, y1 = REF_LOGO_BOX
        return reference_bgr[y0:y1, x0:x1].copy()
    return cv2.resize(reference_bgr, (LOGO_W, LOGO_H), interpolation=cv2.INTER_AREA)


def looks_sane(H, shape):
    h, w = shape[:2]
    quad = cv2.perspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2), H)
    if not cv2.isContourConvex(quad):
        return False
    return 0.05 < cv2.contourArea(quad) / float(w * h) < 20


def fit_homography(t_gray, img_gray):
    sift = cv2.SIFT_create(nfeatures=5000)
    kp_t, des_t = sift.detectAndCompute(t_gray, None)
    kp_i, des_i = sift.detectAndCompute(img_gray, None)
    if des_i is None or len(kp_i) < 10:
        return None, 0
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des_t, des_i, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < RATIO_TEST * n.distance]
    best = {}
    for m in good:                       # each photo keypoint may be used once
        if m.trainIdx not in best or m.distance < best[m.trainIdx].distance:
            best[m.trainIdx] = m
    good = list(best.values())
    if len(good) < MIN_INLIERS:
        return None, len(good)
    src = np.float32([kp_t[m.queryIdx].pt for m in good])
    dst = np.float32([kp_i[m.trainIdx].pt for m in good])
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_PX)
    inliers = int(mask.sum()) if mask is not None else 0
    if H is None or inliers < MIN_INLIERS or not looks_sane(H, t_gray.shape):
        return None, inliers
    return H, inliers


def refine_with_ecc(t_gray, img_gray, H):
    h, w = t_gray.shape
    warped = cv2.warpPerspective(img_gray, np.linalg.inv(H), (w, h))
    inside = cv2.warpPerspective(np.full(img_gray.shape, 255, np.uint8), np.linalg.inv(H), (w, h))
    inside = cv2.erode(inside, np.ones((5, 5), np.uint8))       # only pixels that come from the photo count
    C = np.eye(3, dtype=np.float32)
    try:
        score, C = cv2.findTransformECC(t_gray.astype(np.float32), warped.astype(np.float32), C,
                                        cv2.MOTION_HOMOGRAPHY, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6),
                                        inside, 5)
    except cv2.error:
        return H, float("nan")
    return H @ C.astype(np.float64), float(score)


def green_ink_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return (hsv[..., 1] > 80) & (hsv[..., 2] < 200)


def align(reference_bgr, photo_bgr):
    """Returns a dict: ok, aligned crop (920 x 260), inliers, ECC score, % of logo ink outside the photo, outline in the photo."""
    template = logo_template(reference_bgr)
    t_gray, img_gray = to_gray(template), to_gray(photo_bgr)
    H, inliers = fit_homography(t_gray, img_gray)
    if H is None:
        return {"ok": False, "inliers": inliers}
    H, ecc = refine_with_ecc(t_gray, img_gray, H)
    crop = cv2.warpPerspective(photo_bgr, np.linalg.inv(H), (LOGO_W, LOGO_H))
    outside = crop.max(axis=2) == 0                              # black = outside the photo
    ink = green_ink_mask(template)
    cut_pct = 100.0 * (ink & outside).sum() / max(1, ink.sum())
    corners = cv2.perspectiveTransform(np.float32([[0, 0], [LOGO_W, 0], [LOGO_W, LOGO_H], [0, LOGO_H]]).reshape(-1, 1, 2), H)
    return {"ok": True, "crop": crop, "template": template, "inliers": inliers, "ecc": ecc,
            "cut_pct": cut_pct, "outline": corners.reshape(-1, 2)}


def overlay_outline(template, crop):
    """The aligned crop with the reference's ink outline in red (the visual check of EXP-001)."""
    edges = cv2.Canny(green_ink_mask(template).astype(np.uint8) * 255, 50, 150) > 0
    out = crop.copy()
    out[edges] = (0, 0, 255)
    return out


# ---------------------------------------------------------------- model (ONNX)
_session = None


def session():
    global _session
    if _session is None:
        _session = ort.InferenceSession(str(MODEL_DIR / "e14_consistent.onnx"), providers=["CPUExecutionProvider"])
    return _session


def to_input(bgr):
    """Same preprocessing as training: via 460 x 130, then 448 x 128, RGB, ImageNet-normalised."""
    w, h = INFO["input_size_wh"]
    img = cv2.resize(bgr, (460, 130), interpolation=cv2.INTER_AREA)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    return ((rgb - np.array(INFO["mean"], np.float32)) / np.array(INFO["std"], np.float32)).transpose(2, 0, 1)


def predict(templates, crop):
    """
    Score the aligned crop against one or more reference logo crops (mean, as in the experiments).
    Returns p(tampered), class probabilities, mean heatmap (32 x 112) and the decision.
    """
    refs = np.stack([to_input(t) for t in templates]).astype(np.float32)
    cand = np.stack([to_input(crop)] * len(templates)).astype(np.float32)
    logit, kind, heat = session().run(None, {"reference": refs, "tampered": cand})
    p = float((1 / (1 + np.exp(-logit))).mean())
    e = np.exp(kind - kind.max(axis=1, keepdims=True))
    probs = (e / e.sum(axis=1, keepdims=True)).mean(axis=0)
    if p >= INFO["threshold_tampered"]:
        decision = "TAMPERED"
    elif p <= INFO["threshold_genuine"]:
        decision = "GENUINE"
    else:
        decision = "DON'T KNOW"
    return {"p": p, "per_reference": (1 / (1 + np.exp(-logit))).tolist(),
            "class_probs": dict(zip(INFO["classes"], probs.tolist())), "heatmap": heat.mean(axis=0), "decision": decision}


def confidence_note(decision):
    """The confidence that goes with a decision: how often that band was right on synthetic test pairs."""
    c, n = CONFIDENCE, CONFIDENCE["pairs"]
    if decision == "TAMPERED":
        b = c["tampered_band"]
        return f"right for {b['right']:.0%} of the {b['pairs']} synthetic test pairs that landed in this band"
    if decision == "GENUINE":
        b = c["genuine_band"]
        return f"right for {b['right']:.0%} of the {b['pairs']} synthetic test pairs that landed in this band"
    b = c["dont_know_band"]
    return (f"no call: {b['share_tampered']:.0%} of the {b['pairs']} synthetic test pairs in this band were tampered "
            f"(out of {n} pairs, about half tampered)")


def heatmap_overlay(crop, heat):
    """Heatmap on the aligned crop, on the fixed colour scale (red = among the largest differences seen on real photos)."""
    heat = cv2.resize(heat, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_LINEAR)
    heat = np.uint8(255 * np.clip(heat / INFO["heatmap_scale"], 0, 1))
    return cv2.addWeighted(crop, 0.5, cv2.applyColorMap(heat, cv2.COLORMAP_JET), 0.5, 0)
