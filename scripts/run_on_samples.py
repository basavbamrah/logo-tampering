"""
Run the final model (ONNX) on the 50 sample photos and save, for each photo, one
picture with everything needed to check the result by eye:

  top       the photo with the found logo outlined (yellow), and a text panel:
            decision, p(tampered), confidence, p per reference, logo outside the
            photo, alignment quality, "which tamper?" probabilities, and what the
            independent reference comparisons found (samples/other_methods.csv)
  middle    the reference logo | the aligned photo with the reference outline in red
  bottom    the aligned photo  | the heatmap (where the model sees a difference)

Same steps as the app (app/pipeline.py): align with the reference (EXP-001),
score against all 3 references (mean), decide with the validation thresholds.

Writes outputs/real_photos/:
  <photo>.jpg      one picture per photo
  results.csv      one row per photo
  overview.jpg     all 50 heatmaps on one sheet, with the decision

Run (from the submission folder):  python scripts/run_on_samples.py
"""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

SUB = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SUB / "app"))
import pipeline as P     # noqa: E402

OUT = SUB / "outputs" / "real_photos"
TILE_W, TILE_H = P.LOGO_W, P.LOGO_H            # 920 x 260, the reference frame
WIDTH = 2 * TILE_W                             # picture width: two tiles side by side
COLOURS = {"TAMPERED": (0, 0, 220), "GENUINE": (0, 150, 0), "DON'T KNOW": (0, 140, 255), "CAN'T VERIFY": (90, 90, 90)}


def put(img, text, xy, scale=0.8, colour=(0, 0, 0), thick=2):
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick, cv2.LINE_AA)


def labelled(tile, label):
    """A tile with a white caption strip on top."""
    strip = np.full((36, tile.shape[1], 3), 255, np.uint8)
    put(strip, label, (8, 26), 0.75)
    return np.vstack([strip, tile])


def blank(label):
    tile = np.full((TILE_H, TILE_W, 3), 235, np.uint8)
    put(tile, "logo could not be aligned", (250, 140), 1.0, (90, 90, 90))
    return labelled(tile, label)


def other_methods():
    f = SUB / "samples" / "other_methods.csv"
    return {r["image"]: r for r in csv.DictReader(open(f))} if f.exists() else {}


def picture(name, photo, a, r, decision, lines):
    """Assemble the picture of one photo."""
    # top: the photo (yellow outline) and the text panel
    shown = photo.copy()
    if a["ok"]:
        cv2.polylines(shown, [a["outline"].astype(np.int32)], True, (0, 220, 255), 6)
    scale = min(560 / shown.shape[0], 700 / shown.shape[1])
    shown = cv2.resize(shown, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    top = np.full((600, WIDTH, 3), 255, np.uint8)
    top[20:20 + shown.shape[0], 20:20 + shown.shape[1]] = shown
    x = 760
    put(top, f"Photo {name}", (x, 60), 1.2, thick=3)
    put(top, decision, (x, 120), 1.5, COLOURS[decision.split(" +")[0]], 4)
    for i, line in enumerate(lines):
        put(top, line, (x, 170 + 34 * i), 0.72)

    # middle and bottom: reference, aligned crop with outline, aligned crop, heatmap
    template = P.logo_template(cv2.imread(str(SUB / "app" / "references" / "ref_1.jpg")))
    ref_tile = labelled(template, "reference logo (ref_1)")
    if a["ok"]:
        outline = labelled(P.overlay_outline(a["template"], a["crop"]), "aligned photo + reference outline (red)")
        crop = labelled(a["crop"], "aligned photo (input to the model)")
        heat = labelled(P.heatmap_overlay(a["crop"], r["heatmap"]),
                        "heatmap: where the model sees a difference (red = large, fixed scale)")
    else:
        outline, crop, heat = blank("aligned photo + reference outline"), blank("aligned photo"), blank("heatmap")
    return np.vstack([top, np.hstack([ref_tile, outline]), np.hstack([crop, heat])])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    references = [cv2.imread(str(SUB / "app" / "references" / f"ref_{i}.jpg")) for i in (1, 2, 3)]
    templates = [P.logo_template(ref) for ref in references]
    others = other_methods()
    rows, thumbs = [], []

    for path in sorted((SUB / "samples").glob("*.jpg")):
        name = path.stem
        photo = cv2.imread(str(path))
        a = P.align(references[0], photo)
        other = others.get(name, {}).get("other_methods_flagging", "not available")
        if not a["ok"]:
            decision = "CAN'T VERIFY"
            lines = [f"The logo could not be aligned with the reference: only {a['inliers']} matching",
                     f"points agree (at least {P.MIN_INLIERS} are needed). No decision is possible."]
            r = None
            row = {"image": name, "decision": decision, "p_tampered": "", "p_ref1": "", "p_ref2": "", "p_ref3": "",
                   "confidence": "no decision", "cut_pct": "", "inliers": a["inliers"], "ecc": "",
                   "top_tamper_type": "", "other_methods_flagging": other, "methods_agreeing_tampered": ""}
        else:
            r = P.predict(templates, a["crop"])
            decision = r["decision"]
            cut = a["cut_pct"] > P.CUT_PCT
            if cut:
                decision += " + CUT OFF"
            per_ref = r["per_reference"]
            spread = max(per_ref) - min(per_ref)
            kinds = sorted(r["class_probs"].items(), key=lambda kv: -kv[1])[:3]
            agreeing = (r["decision"] == "TAMPERED") + sum(1 for e in ("cut-off", "difference map", "colour") if e in other)
            lines = [f"p(tampered) = {r['p']:.3f}   (tampered if >= {P.INFO['threshold_tampered']:.3f}, "
                     f"genuine if <= {P.INFO['threshold_genuine']:.3f})",
                     "Confidence: " + P.confidence_note(r["decision"]).replace(" synthetic test pairs", " test pairs"),
                     "    (rates measured on labelled synthetic test pairs, not on real photos)",
                     "p per reference: " + ", ".join(f"{p:.3f}" for p in per_ref)
                     + ("  (they agree)" if spread < 0.1 else f"  (spread {spread:.2f}: less sure)"),
                     f"Logo outside the photo: {a['cut_pct']:.1f} % of the ink" + ("  -> cut off, not genuine" if cut else ""),
                     f"Alignment: {a['inliers']} matching points, ECC {a['ecc']:.3f}",
                     "Which tamper? " + ", ".join(f"{k} {v:.2f}" for k, v in kinds),
                     f"Other methods (difference map, colour, cut-off): {other}",
                     f"Methods saying tampered: {agreeing} (model + other methods)"]
            row = {"image": name, "decision": decision, "p_tampered": round(r["p"], 4),
                   "p_ref1": round(per_ref[0], 4), "p_ref2": round(per_ref[1], 4), "p_ref3": round(per_ref[2], 4),
                   "confidence": P.confidence_note(r["decision"]), "cut_pct": round(a["cut_pct"], 2),
                   "inliers": a["inliers"], "ecc": round(a["ecc"], 4), "top_tamper_type": kinds[0][0],
                   "other_methods_flagging": other, "methods_agreeing_tampered": agreeing}
        rows.append(row)
        cv2.imwrite(str(OUT / f"{name}.jpg"), picture(name, photo, a, r, decision, lines), [cv2.IMWRITE_JPEG_QUALITY, 88])

        # small tile for the overview sheet: heatmap with the decision
        tile = P.heatmap_overlay(a["crop"], r["heatmap"]) if r else np.full((TILE_H, TILE_W, 3), 235, np.uint8)
        tile = cv2.resize(tile, (460, 130), interpolation=cv2.INTER_AREA)
        strip = np.full((30, 460, 3), 255, np.uint8)
        put(strip, f"{name}  {decision}" + (f"  p={r['p']:.2f}" if r else ""), (6, 22), 0.6,
            COLOURS[decision.split(" +")[0]], 2)
        thumbs.append(np.vstack([strip, tile]))
        print(f"{name}: {decision:24s} " + (f"p={r['p']:.3f}" if r else ""))

    with open(OUT / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    while len(thumbs) % 5:
        thumbs.append(np.full_like(thumbs[0], 255))
    sheet = np.vstack([np.hstack(thumbs[i:i + 5]) for i in range(0, len(thumbs), 5)])
    cv2.imwrite(str(OUT / "overview.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])

    counts = {}
    for row in rows:
        counts[row["decision"]] = counts.get(row["decision"], 0) + 1
    print(f"\n{len(rows)} photos -> {OUT.relative_to(SUB)}/ ; decisions: {counts}")


if __name__ == "__main__":
    main()
