"""
Synthetic data for EXP-003.

The idea, in five steps:
  1. A logo is a flat "alpha" layer (how much ink at each pixel) plus an ink
     colour. We get it either from the real NORVIA reference (set "D2") or by
     rendering a made-up word in one of 9 fonts: 5 Google Fonts + 4 already on
     this machine, including DejaVu Sans Bold, the logo's own font (set "D1").
  2. A TAMPER edits that layer: stretch it, recolour it, move a letter,
     swap a glyph or the font, draw text over it, or push it out of the frame.
     We always know what we did and by how much.
  3. The logo is put on a random scene (coloured paper, bands, light, small
     filler text).
  4. The scene is "photographed": a physically plausible camera pose
     (perspective), lighting, blur, noise, colour cast, JPEG.
  5. We undo the camera with the known pose plus a little error, to get the
     same kind of rectified crop that EXP-001 makes for the real photos.
     (For the NORVIA set, src/exp003_register_synthetic.py also makes crops
     with the real registration code: the "F12 fix". Models train on those.)

All sizes are in "logo frame" pixels: the reference crop is 920 x 260.
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path

from common import (available_fonts, green_ink_mask, load_font, load_reference,
                    parts_from_mask, reference_parts)

# --- sizes (see common.REF_LOGO_BOX) ----------------------------------------
LOGO_W, LOGO_H = 920, 260                 # the reference crop
PAD = 120                                 # spare room around it so tampers can grow
CANVAS_W, CANVAS_H = LOGO_W + 2 * PAD, LOGO_H + 2 * PAD
WORD_CAP, WORD_LEFT, WORD_BASE = 113, 198, 161   # wordmark: cap height, left edge, baseline
TAG_CAP, TAG_LEFT, TAG_BASE = 30, 190, 238       # tagline
INK_BGR = np.array([79, 120, 29], np.float32)    # median ink colour of the 3 references
SCENE = 1800                                     # side of the flat "artwork" canvas
PHOTO = 1024                                     # square render before the final resize
FILLER_FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")

WORDS = ["NORVIA", "ALTARA", "VELORA", "SOLENA", "KAIRON", "ZENTRA", "LUMINA", "ORVEN",
         "MAVIRA", "TERRAL", "AURIC", "BOTANA", "LIVORA", "PELORA", "VANTIS", "ELMORA"]
TAGLINES = ["PURE BOTANICALS", "NATURAL CARE", "FRESH HERBS", "GARDEN FRESH", "WILD HONEY"]
FILLERS = ["NET WT 250g", "BATCH 0921", "STORE IN A COOL DRY PLACE", "100% NATURAL",
           "MADE WITH CARE"]
# letters that are easy to mistake for each other
LOOKALIKES = {"O": "0QD", "I": "l1T", "R": "BPK", "N": "MH", "V": "UY", "A": "4R",
              "P": "R", "E": "F", "B": "8R", "T": "I", "C": "G", "L": "I", "S": "5", "U": "V"}


def log_uniform(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def hsv_to_bgr(h_deg, s, v):
    px = np.array([[[h_deg % 360, s, v]]], np.float32)
    return cv2.cvtColor(px, cv2.COLOR_HSV2BGR)[0, 0]


# =============================================================================
# 1. Logos as alpha layers
# =============================================================================

def render_text(text, font_path, cap_h):
    """
    Draw text with a font and return (alpha, baseline_row): a tight alpha image
    scaled so a capital 'H' is cap_h pixels tall, and the row of the baseline.
    """
    big = 300
    font = load_font(font_path, big)
    box = font.getbbox("H")
    cap_big = box[3] - box[1]
    img = Image.new("L", (int(font.getlength(text)) + big, big * 2), 0)
    base_y = big * 3 // 2
    ImageDraw.Draw(img).text((big // 2, base_y), text, font=font, fill=255, anchor="ls")
    a = np.asarray(img, np.float32) / 255
    ys, xs = np.nonzero(a > 0.02)
    tight = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    scale = cap_h / cap_big
    tight = cv2.resize(tight, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return tight, (base_y - ys.min()) * scale


def paste_max(canvas, tight, left, baseline, baseline_row):
    """Put a tight alpha image on the canvas, with its baseline on `baseline`."""
    x, y = int(round(left)), int(round(baseline - baseline_row))
    h, w = tight.shape
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, canvas.shape[1]), min(y + h, canvas.shape[0])
    if x1 > x0 and y1 > y0:
        canvas[y0:y1, x0:x1] = np.maximum(canvas[y0:y1, x0:x1], tight[y0 - y:y1 - y, x0 - x:x1 - x])


def new_logo(alpha, parts, font_name, word, tagline):
    return {"alpha": alpha, "ink": np.tile(INK_BGR, (CANVAS_H, CANVAS_W, 1)), "parts": parts,
            "font": font_name, "word": word, "tagline": tagline, "overlays": []}


def logo_from_reference(index):
    """The real NORVIA artwork, taken from reference image `index`."""
    ref = load_reference(index)
    mask = cv2.GaussianBlur(green_ink_mask(ref), (0, 0), 0.8).astype(np.float32) / 255
    alpha = np.zeros((CANVAS_H, CANVAS_W), np.float32)
    alpha[PAD:PAD + LOGO_H, PAD:PAD + LOGO_W] = mask
    parts = reference_parts(ref)
    for p in parts:
        x, y, w, h = p["box"]
        p["box"] = (x + PAD, y + PAD, w, h)
    # The real logo is in DejaVu Sans Bold (EXP-009 found a 0.98 shape match).
    return new_logo(alpha, parts, "DejaVuSans-Bold", "NORVIA", "PURE BOTANICALS")


def logo_from_text(word, tagline, font_name, font_path):
    """A made-up logo: `word` and `tagline` set in one font. None if letters touch."""
    alpha = np.zeros((CANVAS_H, CANVAS_W), np.float32)
    tight, base_row = render_text(word, font_path, WORD_CAP)
    if PAD + WORD_LEFT + tight.shape[1] > CANVAS_W - 20:
        return None
    paste_max(alpha, tight, PAD + WORD_LEFT, PAD + WORD_BASE, base_row)
    tight, base_row = render_text(tagline, font_path, TAG_CAP)
    paste_max(alpha, tight, PAD + TAG_LEFT, PAD + TAG_BASE, base_row)

    mask = ((alpha > 0.5) * 255).astype(np.uint8)
    parts = parts_from_mask(mask, word=word, tagline=tagline, warn=False)
    named = [p for p in parts if p["char"]]
    if len(named) != len(word) + len(tagline.replace(" ", "")):   # letters merged
        return None
    return new_logo(alpha, parts, font_name, word, tagline)


# =============================================================================
# 2. Tampers. Each takes (logo, rng) and returns (new logo, info about what we did)
# =============================================================================

def part_region(part, grow=2):
    x, y, w, h = part["box"]
    m = np.zeros((CANVAS_H, CANVAS_W), np.uint8)
    m[y:y + h, x:x + w] = part["mask"]
    if grow:
        m = cv2.dilate(m, np.ones((2 * grow + 1, 2 * grow + 1), np.uint8))
    return m > 0


def group_region(logo, group):
    region = np.zeros((CANVAS_H, CANVAS_W), bool)
    for p in logo["parts"]:
        if p["group"] == group:
            region |= part_region(p)
    return region


def group_baseline(logo, group):
    return float(np.median([p["box"][1] + p["box"][3] for p in logo["parts"] if p["group"] == group]))


def copy_logo(logo):
    out = dict(logo)
    out["alpha"] = logo["alpha"].copy()
    out["ink"] = logo["ink"].copy()
    out["overlays"] = list(logo["overlays"])
    return out


def move_region(alpha, region, matrix):
    """Warp only the ink inside `region` with a 2x3 matrix; leave the rest."""
    inside = alpha * region
    moved = cv2.warpAffine(inside, matrix, (CANVAS_W, CANVAS_H), flags=cv2.INTER_LINEAR)
    return np.maximum(alpha - inside, moved)


def groups_of(logo):
    return sorted({p["group"] for p in logo["parts"]})


def tamper_stretch(logo, rng):
    """Stretch the whole logo, or just one part, by 1-8 %."""
    out = copy_logo(logo)
    groups = groups_of(logo)
    target = str(rng.choice(["whole"] + groups, p=[0.4] + [0.6 / len(groups)] * len(groups)))
    axis = "x" if rng.random() < 0.8 else "y"
    pct = log_uniform(rng, 1, 8) * rng.choice([-1, 1])
    region = logo["alpha"] > 0.001 if target == "whole" else group_region(logo, target)
    ys, xs = np.nonzero(region)
    scale = 1 + pct / 100
    if axis == "x":      # grow/shrink from the left edge, like a wider font would
        matrix = np.float32([[scale, 0, xs.min() * (1 - scale)], [0, 1, 0]])
    else:                # grow/shrink around the vertical middle
        mid = (ys.min() + ys.max()) / 2
        matrix = np.float32([[1, 0, 0], [0, scale, mid * (1 - scale)]])
    out["alpha"] = move_region(logo["alpha"], region, matrix)
    return out, {"type": "stretch", "target": target, "axis": axis, "magnitude": abs(pct),
                 "unit": "percent", "detail": f"{axis} {pct:+.1f}%"}


def tamper_recolour(logo, rng):
    """Rotate the hue of the ink (whole logo or one part) by 3-15 degrees."""
    out = copy_logo(logo)
    groups = groups_of(logo)
    target = str(rng.choice(["whole"] + groups, p=[0.5] + [0.5 / len(groups)] * len(groups)))
    dh = log_uniform(rng, 3, 15) * rng.choice([-1, 1])

    old = (INK_BGR / 255)[None, None, :].astype(np.float32)
    hsv = cv2.cvtColor(old, cv2.COLOR_BGR2HSV)
    hsv[..., 0] = (hsv[..., 0] + dh) % 360
    new = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    delta_e = float(np.linalg.norm(cv2.cvtColor(new, cv2.COLOR_BGR2LAB) -
                                   cv2.cvtColor(old, cv2.COLOR_BGR2LAB)))

    region = np.ones((CANVAS_H, CANVAS_W), bool) if target == "whole" else group_region(logo, target)
    m = cv2.GaussianBlur(region.astype(np.float32), (0, 0), 1.0)[..., None]
    out["ink"] = logo["ink"] * (1 - m) + (new[0, 0] * 255) * m
    return out, {"type": "recolour", "target": target, "magnitude": abs(dh), "unit": "hue_deg",
                 "detail": f"{dh:+.1f} deg, deltaE {delta_e:.1f}"}


def tamper_respace(logo, rng):
    """Slide one letter sideways by 1-6 px."""
    out = copy_logo(logo)
    group = "word" if rng.random() < 0.8 else "tagline"
    letters = [p for p in logo["parts"] if p["group"] == group and p["char"]]
    letter = letters[int(rng.integers(len(letters)))]
    dx = log_uniform(rng, 1, 6) * rng.choice([-1, 1])
    out["alpha"] = move_region(logo["alpha"], part_region(letter), np.float32([[1, 0, dx], [0, 1, 0]]))
    return out, {"type": "respace", "target": f"{group}:{letter['char']}", "magnitude": abs(dx),
                 "unit": "px", "detail": f"{dx:+.1f}px"}


def other_font(logo, rng):
    fonts = {n: p for n, p in available_fonts().items() if n != logo["font"]}
    name = str(rng.choice(sorted(fonts)))
    return name, fonts[name]


def tamper_glyph_swap(logo, rng):
    """Replace one letter by a look-alike (other font, or a similar character)."""
    out = copy_logo(logo)
    group = "word" if rng.random() < 0.7 else "tagline"
    letters = [p for p in logo["parts"] if p["group"] == group and p["char"]]
    letter = letters[int(rng.integers(len(letters)))]
    cap = WORD_CAP if group == "word" else TAG_CAP

    char = letter["char"]
    if rng.random() < 0.5 and char in LOOKALIKES:
        char = str(rng.choice(list(LOOKALIKES[char])))
    donor_name, donor_path = other_font(logo, rng)
    tight, base_row = render_text(char, donor_path, cap)

    x, y, w, h = letter["box"]
    new = np.zeros_like(logo["alpha"])
    paste_max(new, tight, x + w / 2 - tight.shape[1] / 2, group_baseline(logo, group), base_row)

    old_mask = part_region(letter, grow=0)
    iou = float((old_mask & (new > 0.5)).sum() / max(1, (old_mask | (new > 0.5)).sum()))
    out["alpha"] = np.maximum(logo["alpha"] * ~part_region(letter), new)
    return out, {"type": "glyph_swap", "target": f"{group}:{letter['char']}", "magnitude": 1 - iou,
                 "unit": "1-IoU", "detail": f"{letter['char']}->{char} in {donor_name}"}


def tamper_font_swap(logo, rng):
    """Re-set the whole wordmark in a different font."""
    out = copy_logo(logo)
    donor_name, donor_path = other_font(logo, rng)
    tight, base_row = render_text(logo["word"], donor_path, WORD_CAP)
    words = [p for p in logo["parts"] if p["group"] == "word"]
    left = min(p["box"][0] for p in words)
    old_w = max(p["box"][0] + p["box"][2] for p in words) - left
    new = np.zeros_like(logo["alpha"])
    paste_max(new, tight, left, group_baseline(logo, "word"), base_row)
    out["alpha"] = np.maximum(logo["alpha"] * ~group_region(logo, "word"), new)
    change = 100 * abs(tight.shape[1] / old_w - 1)
    return out, {"type": "font_swap", "target": "word", "magnitude": change, "unit": "width_pct",
                 "detail": f"{logo['font']}->{donor_name}"}


def tamper_overlap(logo, rng):
    """Small text drawn across the wordmark (the scene draws it on top of the ink)."""
    out = copy_logo(logo)
    text = str(rng.choice(FILLERS))
    cap = float(rng.uniform(14, 28))
    tight, base_row = render_text(text, FILLER_FONT, cap)
    words = [p for p in logo["parts"] if p["group"] == "word"]
    x0 = min(p["box"][0] for p in words)
    x1 = max(p["box"][0] + p["box"][2] for p in words)
    top = min(p["box"][1] for p in words)
    left = rng.uniform(x0, max(x0 + 1, x1 - tight.shape[1]))
    baseline = top + rng.uniform(25, WORD_CAP)
    grey = float(rng.uniform(0.02, 0.15) if rng.random() < 0.75 else rng.uniform(0.8, 0.95))
    out["overlays"].append((tight, left, baseline, base_row, grey))
    return out, {"type": "overlap", "target": "word", "magnitude": cap, "unit": "text_px",
                 "detail": text}


def tamper_none(logo, rng):
    return logo, {"type": "none", "target": "", "magnitude": 0.0, "unit": "", "detail": ""}


# "cut" is not an edit of the artwork: it is a camera position that leaves part
# of the logo outside the photo. It is chosen when the scene is photographed.
TAMPERS = {"stretch": tamper_stretch, "recolour": tamper_recolour, "respace": tamper_respace,
           "glyph_swap": tamper_glyph_swap, "font_swap": tamper_font_swap,
           "overlap": tamper_overlap, "cut": tamper_none, "none": tamper_none}


# =============================================================================
# 3. The scene: paper colour, light, bands, filler text
# =============================================================================

def make_background(rng):
    s = SCENE
    base = hsv_to_bgr(rng.uniform(0, 360), rng.uniform(0.02, 0.14), rng.uniform(0.62, 0.95))
    yy, xx = np.mgrid[0:s, 0:s].astype(np.float32) / s

    angle = rng.uniform(0, 2 * np.pi)
    ramp = (xx * np.cos(angle) + yy * np.sin(angle)) - 0.5
    bg = base + ramp[..., None] * (rng.uniform(-0.12, 0.12) + rng.uniform(-0.02, 0.02, 3))

    cx, cy, r = rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.6)
    spot = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r))
    bg += spot[..., None] * rng.uniform(-0.08, 0.12)

    if rng.random() < 0.4:                       # a shiny streak of light
        a = rng.uniform(1.2, 1.9)
        along = xx * np.cos(a) + yy * np.sin(a)
        streak = np.exp(-((along - rng.uniform(0.2, 0.8)) ** 2) / (2 * rng.uniform(0.02, 0.07) ** 2))
        bg += streak[..., None] * rng.uniform(0.04, 0.15)

    for _ in range(int(rng.integers(0, 3))):     # coloured bands, roughly horizontal
        phi = rng.uniform(-0.2, 0.2)
        d = yy * np.cos(phi) + xx * np.sin(phi)
        centre, half = rng.uniform(0.2, 0.8), rng.uniform(0.03, 0.09)
        col = hsv_to_bgr(rng.uniform(0, 360), rng.uniform(0.03, 0.14), rng.uniform(0.6, 0.9))
        a = ((np.abs(d - centre) < half) * rng.uniform(0.5, 0.9))[..., None]
        bg = bg * (1 - a) + col * a

    bg += rng.normal(0, 0.006, (s, s, 1))        # paper grain
    return np.clip(bg, 0, 1).astype(np.float32)


def draw_fillers(art, rng, keep_clear):
    """A few small grey texts away from the logo, like 'NET WT 250g' on real packs."""
    x0, y0, x1, y1 = keep_clear
    for text in rng.choice(FILLERS, size=int(rng.integers(1, 4)), replace=False):
        tight, base_row = render_text(str(text), FILLER_FONT, rng.uniform(12, 20))
        for _ in range(20):          # try a few random spots that don't touch the logo area
            x = rng.uniform(20, SCENE - tight.shape[1] - 20)
            y = rng.uniform(40, SCENE - 40)   # y is the text's baseline
            if x > x1 or x + tight.shape[1] < x0 or y < y0 or y > y1 + 60:
                break
        else:
            continue                 # no free spot: skip this text rather than draw it on the logo
        layer = np.zeros(art.shape[:2], np.float32)
        paste_max(layer, tight, x, y, base_row)
        grey = rng.uniform(0.3, 0.5)
        art[:] = art * (1 - 0.7 * layer[..., None]) + grey * 0.7 * layer[..., None]


def render_artwork(logo, rng):
    """The flat, front-on scene. Returns (artwork, top-left of the logo frame in it)."""
    art = make_background(rng)
    ox, oy = (SCENE - CANVAS_W) // 2, (SCENE - CANVAS_H) // 2
    a = logo["alpha"][..., None]
    region = art[oy:oy + CANVAS_H, ox:ox + CANVAS_W]
    region[:] = region * (1 - a) + (logo["ink"] / 255) * a

    draw_fillers(art, rng, (ox, oy, ox + CANVAS_W, oy + CANVAS_H))
    for tight, left, baseline, base_row, grey in logo["overlays"]:
        layer = np.zeros(art.shape[:2], np.float32)
        paste_max(layer, tight, ox + left, oy + baseline, base_row)
        art[:] = art * (1 - 0.85 * layer[..., None]) + grey * 0.85 * layer[..., None]
    return art, (ox + PAD, oy + PAD)


# =============================================================================
# 4. The camera
# =============================================================================

def rotation(yaw, pitch, roll):
    yaw, pitch, roll = np.radians([yaw, pitch, roll])
    rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
    ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    rz = np.array([[np.cos(roll), -np.sin(roll), 0], [np.sin(roll), np.cos(roll), 0], [0, 0, 1]])
    return rz @ ry @ rx


def camera_pose(rng, cut_frac=None):
    """
    A pinhole camera looking at the flat logo. Returns H mapping logo-frame
    pixels to pixels of the PHOTO x PHOTO render, and the pose numbers.
    If cut_frac is given, the logo is placed so about that fraction of it
    falls outside the picture.
    """
    f = rng.uniform(0.8, 1.3) * PHOTO
    yaw = float(np.clip(rng.normal(0, 14), -35, 35))
    pitch = float(np.clip(rng.normal(0, 10), -28, 28))
    roll = float(rng.uniform(-10, 10))
    r = rotation(yaw, pitch, roll)
    k = np.array([[f, 0, PHOTO / 2], [0, f, PHOTO / 2], [0, 0, 1]])
    centre = np.array([[1, 0, -LOGO_W / 2], [0, 1, -LOGO_H / 2], [0, 0, 1]])
    corners = np.float32([[0, 0], [LOGO_W, 0], [LOGO_W, LOGO_H], [0, LOGO_H]]).reshape(-1, 1, 2)

    width_px = rng.uniform(380, 1000)
    margin = 25
    for _ in range(12):
        z = f * LOGO_W / width_px
        h0 = k @ np.column_stack([r[:, 0], r[:, 1], [0, 0, z]]) @ centre
        box = cv2.perspectiveTransform(corners, h0).reshape(-1, 2)
        (x0, y0), (x1, y1) = box.min(0), box.max(0)
        if cut_frac is not None or (x1 - x0 < PHOTO - 2 * margin and y1 - y0 < PHOTO - 2 * margin):
            break
        width_px *= 0.9

    def pick(lo, hi):
        return rng.uniform(lo, hi) if lo < hi else (lo + hi) / 2

    dx, dy = pick(margin - x0, PHOTO - margin - x1), pick(margin - y0, PHOTO - margin - y1)
    side = None
    if cut_frac is not None:
        side = str(rng.choice(["left", "right", "top", "bottom"]))
        bw, bh = x1 - x0, y1 - y0
        if side == "right":
            dx = PHOTO - (1 - cut_frac) * bw - x0
        elif side == "left":
            dx = (1 - cut_frac) * bw - x1
        elif side == "bottom":
            dy = PHOTO - (1 - cut_frac) * bh - y0
        else:
            dy = (1 - cut_frac) * bh - y1
    h = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]]) @ h0
    pose = {"yaw": round(yaw, 1), "pitch": round(pitch, 1), "roll": round(roll, 1),
            "focal": round(float(f)), "logo_width_px": round(float(width_px)), "cut_side": side}
    return h, pose


def photograph(art, origin, h_logo, rng, border=cv2.BORDER_REFLECT):
    """
    Turn the flat artwork into a square photo (PHOTO x PHOTO, float 0..1) seen
    through the camera h_logo. Returns (photo, info about the effects used).

    border: what the camera sees beyond the edge of the artwork. The default
    (REFLECT) is what data/synth and data/synth_more were made with; it mirrors
    the logo into the picture when the camera looks past the edge (found
    2026-10-10, 19 % of D2 photos). New data (EXP-013) passes BORDER_REPLICATE,
    which just extends the edge paper. Kept as an option so old seeds still
    reproduce exactly.
    """
    ox, oy = origin
    to_art = np.array([[1, 0, ox], [0, 1, oy], [0, 0, 1]], np.float64)   # logo frame -> artwork
    h_scene = h_logo @ np.linalg.inv(to_art)                              # artwork -> photo
    img = cv2.warpPerspective(art, h_scene, (PHOTO, PHOTO), flags=cv2.INTER_LINEAR,
                              borderMode=border)

    # lighting: smooth brighter/darker areas
    yy, xx = np.mgrid[0:PHOTO, 0:PHOTO].astype(np.float32) / PHOTO
    ang = rng.uniform(0, 2 * np.pi)
    light = 1 + rng.uniform(-0.12, 0.12) * ((xx * np.cos(ang) + yy * np.sin(ang)) - 0.5)
    img = img * light[..., None]

    info = {}
    if rng.random() < 0.5:                                  # blur
        sigma = rng.uniform(0.4, 1.6)
        if rng.random() < 0.15:                             # motion blur
            length = int(rng.integers(5, 13))
            kernel = np.zeros((length, length), np.float32)
            kernel[length // 2, :] = 1.0 / length
            rot = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), rng.uniform(0, 180), 1)
            kernel = cv2.warpAffine(kernel, rot, (length, length))
            img = cv2.filter2D(img, -1, kernel / max(kernel.sum(), 1e-6))
            info["blur"] = f"motion{length}"
        else:
            img = cv2.GaussianBlur(img, (0, 0), sigma)
            info["blur"] = f"gauss{sigma:.1f}"
    else:
        info["blur"] = "none"

    img = img + rng.normal(0, rng.uniform(0, 0.012), img.shape).astype(np.float32)
    gains = rng.uniform(0.94, 1.06, 3) * rng.uniform(0.9, 1.08)   # colour cast + brightness
    img = img * gains.astype(np.float32)
    info["gains"] = [round(float(g), 3) for g in gains]
    return np.clip(img, 0, 1), info


def finish_photo(img, rng):
    """Resize like the real data (rendered square, then to 3:4 / 4:3 / 1:1) and JPEG-compress."""
    shape = [(960, 1280), (1280, 960), (1024, 1024)][int(rng.choice(3, p=[0.5, 0.25, 0.25]))]
    img = cv2.resize(img, shape, interpolation=cv2.INTER_AREA)
    quality = int(rng.integers(78, 97))
    ok, buf = cv2.imencode(".jpg", (img * 255 + 0.5).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR), shape, quality


def rectify_with_known_pose(photo, h_logo, photo_shape, rng, jitter_px=0.8):
    """
    Undo the camera like EXP-001 does for real photos, but using the pose we
    know, plus a small random error (EXP-001's registration is ~1 px off).
    Returns the logo-frame crop (LOGO_W x LOGO_H).
    """
    w, h = photo_shape
    h_photo = np.diag([w / PHOTO, h / PHOTO, 1.0]) @ h_logo
    corners = np.float32([[0, 0], [LOGO_W, 0], [LOGO_W, LOGO_H], [0, LOGO_H]])
    wobble = cv2.getPerspectiveTransform(corners, corners + rng.normal(0, jitter_px, (4, 2)).astype(np.float32))
    return cv2.warpPerspective(photo, np.linalg.inv(h_photo @ wobble), (LOGO_W, LOGO_H), flags=cv2.INTER_LINEAR)


def clean_render(logo, rng):
    """The logo on plain pale paper, front-on: 'head 1' for the generic (D1) set."""
    paper = hsv_to_bgr(rng.uniform(0, 360), rng.uniform(0.02, 0.08), rng.uniform(0.85, 0.95))
    a = logo["alpha"][..., None]
    img = paper * (1 - a) + (logo["ink"] / 255) * a
    img = img + rng.normal(0, 0.004, img.shape).astype(np.float32)
    img = np.clip(img[PAD:PAD + LOGO_H, PAD:PAD + LOGO_W], 0, 1)
    return (img * 255 + 0.5).astype(np.uint8)
