"""
Convert the final model (e14_consistent) to ONNX, so the app runs without PyTorch.

The ONNX graph has two inputs and three outputs:
  inputs   reference, tampered   float32, (batch, 3, 128, 448), RGB, ImageNet-normalised
  outputs  logit_tampered        (batch,)     -> sigmoid = p(tampered)
           logits_kind           (batch, 8)   -> softmax = probability per class
                                               (none, stretch, recolour, respace, glyph_swap, font_swap, overlap, cut)
           heatmap               (batch, 32, 112) where the two feature maps differ most
It also writes model/model_info.json: thresholds, class names, input size and
the heatmap colour scale (99.5th percentile over the aligned sample photos, so
colours mean the same in every picture, as in the main repo).

Checks, both against PyTorch:
  - the 49 sample photos that can be aligned (aligned by the app's own code), vs the 3 references
  - the bundled test set (test_data/norvia_test, 206 pairs)

Run (from the submission folder, needs requirements-export.txt):
    python scripts/export_onnx.py
"""
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from models import TAMPER_TYPES, build

HERE = Path(__file__).resolve().parent.parent          # the submission folder
sys.path.insert(0, str(HERE / "app"))
import pipeline as P                                    # noqa: E402  (alignment + preprocessing of the app)

MODEL_DIR = HERE / "model"
W, H = 448, 128
MEAN = np.array([0.485, 0.456, 0.406], np.float32)     # ImageNet statistics, as in training
STD = np.array([0.229, 0.224, 0.225], np.float32)
REF_LOGO_BOX = (140, 220, 1060, 480)


class WithHeatmap(nn.Module):
    """The trained siamese model, returning the heatmap as a third output."""

    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, reference, tampered):
        a, b = self.m.encoder(reference), self.m.encoder(tampered)
        x = self.m.compare(torch.cat([(a - b).abs(), a * b], 1))
        pooled = torch.cat([x.amax(dim=(2, 3)), x.mean(dim=(2, 3))], 1)    # same pooling as models.pool
        logit, kind = self.m.heads(self.m.dropout(pooled))
        return logit, kind, (a - b).abs().mean(1)


def compare(sess, model, ref, cand):
    """Largest |ONNX - PyTorch| over both outputs, and the ONNX heatmaps."""
    o_logit, o_kind, o_heat = sess.run(None, {"reference": ref, "tampered": cand})
    with torch.no_grad():
        t_logit, t_kind = model(torch.from_numpy(ref), torch.from_numpy(cand))
    diff = max(float(np.abs(o_logit - t_logit.numpy()).max()), float(np.abs(o_kind - t_kind.numpy()).max()))
    return diff, o_heat


def main():
    torch.manual_seed(0)
    metrics = json.loads((MODEL_DIR / "metrics.json").read_text())
    model = build("siamese_diff", pretrained=False, **metrics["model_options"])
    model.load_state_dict(torch.load(MODEL_DIR / "best.pt", map_location="cpu"))
    model.eval()
    wrapped = WithHeatmap(model).eval()

    onnx_path = MODEL_DIR / "e14_consistent.onnx"
    dummy = torch.zeros(1, 3, H, W)
    # The classic (TorchScript) exporter: PyTorch prints a deprecation notice for it,
    # but it is still supported, gives a plain opset-17 graph, and is checked below.
    torch.onnx.export(wrapped, (dummy, dummy), str(onnx_path), input_names=["reference", "tampered"],
                      output_names=["logit_tampered", "logits_kind", "heatmap"],
                      dynamic_axes={n: {0: "batch"} for n in ("reference", "tampered", "logit_tampered", "logits_kind", "heatmap")},
                      opset_version=17, dynamo=False)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    # 1. the sample photos, aligned by the app's code, vs the 3 references (also gives the heatmap scale)
    x0, y0, x1, y1 = REF_LOGO_BOX
    references = [cv2.imread(str(HERE / "app" / "references" / f"ref_{i}.jpg")) for i in (1, 2, 3)]
    refs = np.stack([P.to_input(r[y0:y1, x0:x1]) for r in references]).astype(np.float32)
    max_diff, heats, n_photos = 0.0, [], 0
    for path in sorted((HERE / "samples").glob("*.jpg")):
        a = P.align(references[0], cv2.imread(str(path)))
        if not a["ok"]:
            continue
        cand = np.stack([P.to_input(a["crop"])] * 3).astype(np.float32)
        diff, heat = compare(sess, model, refs, cand)
        max_diff, n_photos = max(max_diff, diff), n_photos + 1
        heats.append(heat.mean(0))
    scale = float(np.percentile(np.concatenate([h.ravel() for h in heats]), 99.5)) if heats else 1.0

    # 2. the bundled synthetic test pairs
    folder = HERE / "test_data" / "norvia_test"
    rows = list(csv.DictReader(open(folder / "labels.csv")))
    for i in range(0, len(rows), 64):
        chunk = rows[i:i + 64]
        ref = np.stack([P.to_input(cv2.imread(str(folder / r["ref"]))) for r in chunk]).astype(np.float32)
        cand = np.stack([P.to_input(cv2.imread(str(folder / r["cand"]))) for r in chunk]).astype(np.float32)
        max_diff = max(max_diff, compare(sess, model, ref, cand)[0])

    info = {"model": "e14_consistent", "architecture": "siamese_diff, depth 1 (ResNet-18 stem + stage 1, shared)",
            "input_size_wh": [W, H], "mean": MEAN.tolist(), "std": STD.tolist(),
            "threshold_tampered": metrics["threshold_5pct_false_alarm"], "threshold_genuine": metrics["threshold_low"],
            "classes": TAMPER_TYPES, "heatmap_scale": scale, "ref_logo_box": list(REF_LOGO_BOX)}
    (MODEL_DIR / "model_info.json").write_text(json.dumps(info, indent=1))
    print(f"ONNX -> {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")
    print(f"checked on {n_photos} aligned sample photos + {len(rows)} test pairs: "
          f"max |ONNX - PyTorch| on the outputs = {max_diff:.2e}; heatmap scale {scale:.3f}")
    print("accuracy check: python scripts/evaluate_onnx.py")


if __name__ == "__main__":
    main()
