"""
Shared helpers used by all experiment scripts.

Kept deliberately small: paths, loading the reference logo, splitting the
logo into its parts (icon / wordmark letters / tagline letters), and font
loading.
"""
from pathlib import Path

import cv2
import numpy as np
from PIL import ImageFont

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "candidate_package"
IMAGES_DIR = PACKAGE / "images"
REFERENCE_DIR = PACKAGE / "reference"
OUTPUTS = ROOT / "outputs"
FONTS_DIR = ROOT / "data" / "fonts"

# Where the logo sits inside the 1200x700 reference images.
# All three references put it at x=168..1030, y=240..458 (measured with the
# green mask below); we add a margin so the crop has some background around it.
REF_LOGO_BOX = (140, 220, 1060, 480)  # x0, y0, x1, y1

WORDMARK = "NORVIA"
TAGLINE = "PUREBOTANICALS"  # letters only, the space is not a blob


SEED = 0


def fix_seeds(seed=SEED):
    """
    Fix every random number generator used in the pipeline, so a re-run gives
    exactly the same result. OpenCV's RANSAC already uses a fixed internal seed
    and the model has no randomness at inference; this makes it explicit and
    also covers numpy, PyTorch and the GPU (cuDNN picks deterministic kernels).
    """
    cv2.setRNGSeed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def list_images():
    """The 50 test photos, sorted by number."""
    return sorted(IMAGES_DIR.glob("*.jpg"))


def load_reference(index=1):
    """Return the logo crop from ref_<index>.jpg as a BGR image."""
    img = cv2.imread(str(REFERENCE_DIR / f"ref_{index}.jpg"))
    x0, y0, x1, y1 = REF_LOGO_BOX
    return img[y0:y1, x0:x1].copy()


def green_ink_mask(bgr):
    """
    Pixels that belong to the green logo ink.

    The logo is a saturated, fairly dark green; the backgrounds are pale.
    A simple saturation + brightness rule is enough on the clean reference.
    (On real photos the colours shift, so the experiments don't rely on this
    rule there; it is only used to describe the reference.)
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = (hsv[..., 1] > 80) & (hsv[..., 2] < 200)
    return mask.astype(np.uint8) * 255


def reference_parts(ref_bgr):
    """Parts of the reference logo (icon, 6 wordmark letters, 14 tagline letters)."""
    return parts_from_mask(green_ink_mask(ref_bgr), icon_x_limit=170)


def parts_from_mask(mask, word=WORDMARK, tagline=TAGLINE, icon_x_limit=None, warn=True):
    """
    Split a logo mask into parts using connected components.

    Returns a list of dicts:
        {"group": "icon" | "word" | "tagline", "char": "N" or None,
         "box": (x, y, w, h), "mask": uint8 crop of that blob}
    sorted by group and then left to right.

    The rules are based on what the reference looks like:
      - wordmark letters are tall (cap height ~113 px)
      - tagline letters are short (~30 px) and sit below the wordmark
      - anything left of icon_x_limit is the icon (only for the real logo)
    Letters are named from `word` / `tagline` so that later code can say
    "the R moved by 3 px". If the blob count does not match the text, the
    letters are left unnamed and a warning is printed.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    parts = []
    for i in range(1, n):  # 0 is background
        x, y, w, h, area = (int(v) for v in stats[i])
        if area < 30:  # specks of noise
            continue
        if icon_x_limit is not None and x + w < icon_x_limit:
            group = "icon"
        elif h > 80:
            group = "word"
        else:
            group = "tagline"
        blob = (labels[y:y + h, x:x + w] == i).astype(np.uint8) * 255
        parts.append({"group": group, "box": (x, y, w, h), "mask": blob, "char": None})

    parts.sort(key=lambda p: (p["group"], p["box"][0]))

    for group, text in (("word", word), ("tagline", tagline.replace(" ", ""))):
        letters = [p for p in parts if p["group"] == group]
        if len(letters) == len(text):
            for p, ch in zip(letters, text):
                p["char"] = ch
        elif warn:
            print(f"warning: expected {len(text)} {group} blobs, found {len(letters)}")
    return parts


# --- fonts ------------------------------------------------------------------

# Five Google Fonts (downloaded by src/download_fonts.sh) plus fonts that are
# already installed on this machine. The variable Google fonts are set to
# weight 700 (bold) because the NORVIA logo is bold.
GOOGLE_FONTS = {
    "OpenSans-Bold": FONTS_DIR / "opensans" / "OpenSans[wdth,wght].ttf",
    "NotoSans-Bold": FONTS_DIR / "notosans" / "NotoSans[wdth,wght].ttf",
    "Roboto-Bold": FONTS_DIR / "roboto" / "Roboto[wdth,wght].ttf",
    "Lato-Bold": FONTS_DIR / "lato" / "Lato-Bold.ttf",
    "PTSans-Bold": FONTS_DIR / "ptsans" / "PT_Sans-Web-Bold.ttf",
}

SYSTEM_FONTS = {
    "DejaVuSans-Bold": Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    "LiberationSans-Bold": Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
    "Ubuntu-Bold": Path("/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf"),
    "NotoSans-Bold-system": Path("/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf"),
}


def load_font(path, size):
    """Load a TTF at a pixel size; for variable fonts switch to bold."""
    font = ImageFont.truetype(str(path), size)
    try:
        axes = font.get_variation_axes()
    except OSError:  # not a variable font
        return font
    values = []
    for axis in axes:
        name = axis["name"].decode() if isinstance(axis["name"], bytes) else axis["name"]
        if name.lower() == "weight":
            values.append(min(700, axis["maximum"]))
        else:
            values.append(axis["default"])
    font.set_variation_by_axes(values)
    return font


def available_fonts():
    """All candidate fonts that actually exist on disk."""
    fonts = {**GOOGLE_FONTS, **SYSTEM_FONTS}
    return {name: path for name, path in fonts.items() if path.exists()}


# --- small metrics helpers ------------------------------------------------------

def auc(pos_scores, neg_scores):
    """
    Area under the ROC curve: the chance that a random tampered sample scores
    higher than a random genuine one. 0.5 = no better than a coin, 1.0 = perfect.
    """
    pos, neg = np.asarray(pos_scores, float), np.asarray(neg_scores, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(wins / (len(pos) * len(neg)))
