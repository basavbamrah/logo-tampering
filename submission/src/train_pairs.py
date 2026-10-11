"""
EXP-007: train and evaluate one of the three pair models on the synthetic data.

What it does:
  1. reads data/synth/labels.csv
       D2 (real NORVIA artwork): uses <id>_cand_reg.png, the crops made with
          the SAME registration as the real photos (the F12 fix). Photos that
          failed to register are skipped.
       D1 (made-up words, 9 fonts): uses <id>_cand.png (no photos were kept,
          so these crops come from the known camera pose).
  2. trains with two losses:
       "tampered?" (yes/no)          binary cross-entropy (or focal loss)
       "which tamper?" (8 classes)    cross-entropy, weighted by --kind-weight
  3. after every epoch evaluates train / val / test splits and logs everything
     to TensorBoard (outputs/tensorboard/<run>) and to history in metrics.json
  4. keeps the epoch with the best VALIDATION AUC (test numbers are only
     logged for monitoring, never used to choose anything)
  5. at the end reports, per test split, AUC / accuracy / false alarms and per
     tamper type how many are caught at the threshold that flags 5 % of
     genuine validation samples

Run:
  python src/train_pairs.py --arch siamese_diff --dry-run       # shape/memory check only
  python src/train_pairs.py --arch siamese_diff --run sd_base   # training
  tensorboard --logdir outputs/tensorboard                      # watch the curves
Archs: siamese_diff | early_fusion | dino_frozen
"""
import argparse
import csv
import json
import time
from datetime import datetime

from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.tensorboard import SummaryWriter

from common import OUTPUTS, ROOT, auc
from models import INPUT_H, INPUT_W, TAMPER_TYPES, build

MEAN = np.array([0.485, 0.456, 0.406], np.float32)   # ImageNet statistics (both backbones use them)
STD = np.array([0.229, 0.224, 0.225], np.float32)
GPU_CACHE_MB = 1800   # total budget for image caches on the GPU (the 4 GB card also holds the model)
_gpu_cache_used_mb = 0.0
EVAL_SPLITS = [("train", {"D1", "D2"}), ("val", {"D1", "D2"}), ("test", {"D2"}),
               ("test", {"D1"}), ("test_unseen_font", {"D1"})]


def to_tensor(bgr, size=(INPUT_W, INPUT_H)):
    """BGR uint8 image -> normalised 3 x H x W float tensor."""
    rgb = cv2.cvtColor(cv2.resize(bgr, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    x = (rgb.astype(np.float32) / 255 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


def load_rows(data, splits, sets, full_res=False, raw=False):
    """
    Rows of labels.csv for the given splits/sets, with the image paths filled in.
    full_res: use the 920x260 NORVIA crops (<id>_cand_reg_full.png / <id>_ref_full.png).
    raw: EXP-012, use the un-straightened crops (<id>_cand_raw.png / <id>_ref_raw.png,
         made by exp012_make_crops.py). Same samples as the rectified crops.
    """
    registered = set()
    reg_file = data / ("registration_D2_full.csv" if full_res else "registration_D2.csv")
    if reg_file.exists():
        registered = {r["id"] for r in csv.DictReader(open(reg_file)) if r["registered"] == "True"}
    # data/synth_v2 (EXP-013): photos and registration exist for BOTH sets, one file for all
    both_sets = (data / "registration.csv").exists()
    if both_sets:
        registered = {r["id"] for r in csv.DictReader(open(data / "registration.csv")) if r["registered"] == "True"}

    rows, skipped = [], 0
    for r in csv.DictReader(open(data / "labels.csv")):
        if r["split"] not in splits or r["set"] not in sets:
            continue
        if r["set"] == "D2" or both_sets:
            if r["id"] not in registered:      # failed registration: no crop exists
                skipped += 1
                continue
            folder = data / r["set"]
            if raw:
                r["cand_path"] = folder / f"{r['id']}_cand_raw.png"
                r["ref_path"] = folder / f"{r['id']}_ref_raw.png"
            else:
                r["cand_path"] = folder / f"{r['id']}_cand_reg{'_full' if full_res else ''}.png"
                r["ref_path"] = folder / f"{r['id']}_ref{'_full' if full_res else ''}.png"
        else:
            if raw:
                raise ValueError("raw (un-straightened) crops exist only for the NORVIA set (D2)")
            if full_res:
                raise ValueError("full-resolution crops exist only for the NORVIA set (D2)")
            r["cand_path"] = data / "D1" / f"{r['id']}_cand.png"
            r["ref_path"] = data / "D1" / f"{r['id']}_ref.png"
        rows.append(r)
    return rows, skipped


class PairCache:
    """
    All image pairs of one split, decoded ONCE into uint8 tensors.

    Why: decoding and resizing PNGs on the CPU every epoch was the bottleneck
    (the GPU sat idle while CPU workers struggled). Now each epoch only slices
    tensors that are already in memory: on the GPU if they fit in
    the GPU_CACHE_MB budget (shared by all caches), otherwise in pinned CPU
    memory (fast copy per batch).
    """

    def __init__(self, rows, size, device):
        self.rows = rows

        def load(r):
            out = []
            for path in (r["ref_path"], r["cand_path"]):
                img = cv2.resize(cv2.imread(str(path)), size, interpolation=cv2.INTER_AREA)
                out.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            return out

        with ThreadPoolExecutor(8) as pool:      # OpenCV releases the GIL, so threads are enough
            pairs = list(pool.map(load, rows))
        ref = torch.from_numpy(np.stack([p[0] for p in pairs])).permute(0, 3, 1, 2).contiguous()
        cand = torch.from_numpy(np.stack([p[1] for p in pairs])).permute(0, 3, 1, 2).contiguous()
        global _gpu_cache_used_mb
        mb = 2 * ref.numel() / 2**20
        on_gpu = device == "cuda" and _gpu_cache_used_mb + mb < GPU_CACHE_MB
        if on_gpu:
            _gpu_cache_used_mb += mb
        self.ref = ref.to(device) if on_gpu else (ref.pin_memory() if device == "cuda" else ref)
        self.cand = cand.to(device) if on_gpu else (cand.pin_memory() if device == "cuda" else cand)
        self.label = torch.tensor([float(r["label"]) for r in rows])
        self.kind = torch.tensor([TAMPER_TYPES.index(r["tamper"]) for r in rows])
        self.where = "GPU" if on_gpu else "CPU (pinned)"
        self.mb = mb

    def __len__(self):
        return len(self.rows)

    def batches(self, batch_size, shuffle):
        order = torch.randperm(len(self)) if shuffle else torch.arange(len(self))
        for i in range(0, len(self), batch_size):
            idx = order[i:i + batch_size]
            yield self.ref[idx.to(self.ref.device)], self.cand[idx.to(self.cand.device)], \
                self.label[idx], self.kind[idx], idx


_MEAN_T, _STD_T = torch.tensor(MEAN).view(1, 3, 1, 1), torch.tensor(STD).view(1, 3, 1, 1)


def normalise(x_uint8, device):
    """uint8 batch (B,3,H,W) -> float, ImageNet-normalised, on `device`."""
    x = x_uint8.to(device, non_blocking=True).float() / 255
    return (x - _MEAN_T.to(device)) / _STD_T.to(device)


def augment(x_uint8, device):
    """
    Mild extra camera noise for training only, done on the GPU.
    Deliberately NO colour change and NO shifting/resizing: those are exactly
    what the tampers look like, so augmenting them would teach the model to
    ignore real tampering.
    """
    x = x_uint8.to(device, non_blocking=True).float()
    b = x.shape[0]
    gain = torch.empty(b, 1, 1, 1, device=device).uniform_(0.9, 1.1)      # contrast
    bias = torch.empty(b, 1, 1, 1, device=device).uniform_(-10, 10)       # brightness
    x = x * gain + bias
    blur = torch.rand(b, device=device) < 0.3
    if blur.any():
        sigma = float(np.random.uniform(0.3, 0.8))
        x[blur] = TF.gaussian_blur(x[blur], kernel_size=5, sigma=sigma)
    noise_level = torch.empty(b, 1, 1, 1, device=device).uniform_(0, 4)  # sensor noise
    x = x + torch.randn_like(x) * noise_level
    return x.clamp(0, 255).round().to(torch.uint8)


def geo_augment(x_uint8, device):
    """
    EXP-013, raw input only: a small random camera move of the whole tampered
    crop (rotation up to 4 degrees, the SAME scale in x and y within 5 %, shift
    up to 4 %). The raw crop is not lined up with the reference anyway, so this
    only adds camera variety. It never changes one letter or one axis, so it
    can't imitate a re-spacing or a stretch. Must NOT be used on rectified crops.
    """
    x = x_uint8.to(device, non_blocking=True).float()
    b = x.shape[0]
    angle = torch.empty(b, device=device).uniform_(-4, 4) * np.pi / 180
    scale = torch.empty(b, device=device).uniform_(0.95, 1.05)
    shift = torch.empty(b, 2, device=device).uniform_(-0.04, 0.04) * 2   # grid units are -1..1
    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
    aspect = x.shape[3] / x.shape[2]                                      # keep rotation round on a wide image
    theta = torch.stack([torch.stack([cos, -sin / aspect, shift[:, 0]], 1),
                         torch.stack([sin * aspect, cos, shift[:, 1]], 1)], 1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return x.clamp(0, 255).round().to(torch.uint8)


def binary_loss(logit, label, kind, gamma, smoothing=0.0):
    """
    Plain BCE, or focal loss (gamma > 0) that down-weights easy samples.
    smoothing: label smoothing; targets become 0+s/2 and 1-s/2, so the model
    is not pushed towards 100 % certainty (a common remedy for overfitting).
    """
    if smoothing:
        label = label * (1 - smoothing) + 0.5 * smoothing
    if kind == "bce":
        return F.binary_cross_entropy_with_logits(logit, label)
    bce = F.binary_cross_entropy_with_logits(logit, label, reduction="none")
    p_true = torch.exp(-bce)                          # probability given to the correct answer
    return ((1 - p_true) ** gamma * bce).mean()


def losses(model, batch, device, args, train_augment=False):
    ref, cand, label, kind, _ = batch
    if train_augment and args.geo_augment:
        cand = geo_augment(cand, device)
    if train_augment and args.augment:
        cand = augment(cand, device)
    logit, kind_logits = model(normalise(ref, device), normalise(cand, device))
    label, kind = label.float().to(device), kind.to(device)
    loss_bin = binary_loss(logit.float(), label, args.loss, args.gamma, args.label_smoothing if train_augment else 0.0)
    loss_kind = F.cross_entropy(kind_logits.float(), kind)
    return loss_bin + args.kind_weight * loss_kind, loss_bin, loss_kind


@torch.no_grad()
def predict(model, cache, device, batch_size=64):
    """Probability of 'tampered', predicted kind and losses for every sample (in cache order)."""
    model.eval()
    probs, kinds, index, bin_losses = [], [], [], []
    for ref, cand, label, _, idx in cache.batches(batch_size, shuffle=False):
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
            logit, kind_logits = model(normalise(ref, device), normalise(cand, device))
        logit = logit.float()
        bin_losses.append(F.binary_cross_entropy_with_logits(logit, label.float().to(device), reduction="none").cpu())
        probs.append(torch.sigmoid(logit).cpu())
        kinds.append(kind_logits.argmax(1).cpu())
        index.append(idx)
    return (torch.cat(probs).numpy(), torch.cat(kinds).numpy(), torch.cat(index).numpy(),
            float(torch.cat(bin_losses).mean()))


def evaluate(rows, probs, kinds, threshold):
    """AUC, accuracy at 0.5, false alarms and per tamper type the share caught at `threshold`."""
    labels = np.array([int(r["label"]) for r in rows])
    tampered = [(k, r) for k, r in zip(kinds, rows) if r["label"] == "1"]
    out = {"n": len(rows),
           "auc": auc(probs[labels == 1], probs[labels == 0]),
           "accuracy_at_0.5": float(((probs >= 0.5) == labels).mean()),
           "false_alarm_rate_at_threshold": float((probs[labels == 0] >= threshold).mean()) if (labels == 0).any() else None,
           "kind_accuracy_on_tampered": float(np.mean([TAMPER_TYPES[k] == r["tamper"] for k, r in tampered]))
           if tampered else None,
           "caught_per_tamper": {}}
    for t in TAMPER_TYPES[1:]:
        sel = [p for p, r in zip(probs, rows) if r["tamper"] == t]
        if sel:
            out["caught_per_tamper"][t] = {"n": len(sel), "caught": float(np.mean(np.array(sel) >= threshold))}
    return out


def split_name(split, sets):
    return f"{split}_{'+'.join(sorted(sets))}"


def main():
    global GPU_CACHE_MB   # --gpu-cache-mb may change the cache budget
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["siamese_diff", "early_fusion", "dino_frozen"])
    ap.add_argument("--run", default=None, help="name of this run (folder under outputs/exp007)")
    ap.add_argument("--data", default="data/synth")
    ap.add_argument("--sets", default="D1,D2", help="which synthetic sets to TRAIN on")
    ap.add_argument("--extra-data", default="", help="comma-separated extra dataset folders (train split only)")
    ap.add_argument("--full-res", action="store_true",
                    help="use the 920x260 NORVIA crops (needs --sets D2; pair with --input-w 896 --input-h 256)")
    ap.add_argument("--raw-crops", action="store_true",
                    help="EXP-012: candidate = un-straightened crop of the photo (NORVIA only)")
    ap.add_argument("--augment", action="store_true", help="mild brightness/blur/noise on the candidate (train only)")
    ap.add_argument("--depth", type=int, default=2, choices=[1, 2, 3], help="siamese_diff: ResNet stages kept")
    ap.add_argument("--dropout", type=float, default=0.0, help="siamese_diff: dropout before the heads")
    ap.add_argument("--freeze-stem", action="store_true", help="do not train the first conv layer + layer1")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--loss", default="bce", choices=["bce", "focal"])
    ap.add_argument("--gamma", type=float, default=2.0, help="focal loss gamma")
    ap.add_argument("--label-smoothing", type=float, default=0.0, help="label smoothing for the yes/no loss (train only)")
    ap.add_argument("--geo-augment", action="store_true",
                    help="EXP-013: small random rotation/scale/shift of the tampered crop (raw crops only)")
    ap.add_argument("--kind-weight", type=float, default=0.3, help="weight of the 'which tamper?' loss")
    ap.add_argument("--input-w", type=int, default=INPUT_W)
    ap.add_argument("--input-h", type=int, default=INPUT_H)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-batch", type=int, default=64, help="batch size for evaluation (lower it for large inputs)")
    ap.add_argument("--gpu-cache-mb", type=float, default=GPU_CACHE_MB,
                    help="GPU memory allowed for cached images; 0 = keep them in CPU memory "
                         "(use when another training shares the GPU)")
    ap.add_argument("--no-pretrained", action="store_true", help="random init (for quick checks only)")
    ap.add_argument("--dry-run", action="store_true", help="run 2 batches forward+backward, no weight update, then stop")
    args = ap.parse_args()

    GPU_CACHE_MB = args.gpu_cache_mb   # (declared global at the top of main)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    run = args.run or f"{args.arch}_{datetime.now():%Y%m%d_%H%M%S}"
    data = ROOT / args.data
    out = OUTPUTS / "exp007" / run
    device = "cuda" if torch.cuda.is_available() else "cpu"
    size = (args.input_w, args.input_h)
    train_sets = args.sets.split(",")
    if args.geo_augment and not args.raw_crops:
        raise SystemExit("--geo-augment moves the whole crop; only valid with --raw-crops (rectified crops must stay lined up)")

    train_rows, skipped = load_rows(data, {"train"}, train_sets, args.full_res, args.raw_crops)
    for extra in filter(None, args.extra_data.split(",")):
        more, more_skipped = load_rows(ROOT / extra, {"train"}, train_sets, args.full_res, args.raw_crops)
        train_rows += more
        skipped += more_skipped
    print(f"run {run}: train {len(train_rows)} pairs ({skipped} D2 skipped: not registered), device {device}")

    options = {"depth": args.depth, "dropout": args.dropout} if args.arch == "siamese_diff" else {}
    model = build(args.arch, pretrained=not args.no_pretrained, **options).to(device)
    if args.freeze_stem:
        # Keep the pretrained low-level filters (edges, blobs) fixed; fewer
        # trainable weights = less over-fitting. BatchNorm statistics in the
        # frozen part are also kept fixed (see freeze_bn below).
        net = model.encoder if hasattr(model, "encoder") else model.net
        for layer in list(net.children())[:5]:          # conv1, bn1, relu, maxpool, layer1
            for p in layer.parameters():
                p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"{args.arch}: {sum(p.numel() for p in trainable) / 1e6:.2f} M trainable parameters, input {size}")

    t_load = time.time()
    train_cache = PairCache(train_rows, size, device)
    print(f"loaded {len(train_cache)} training pairs into {train_cache.where} "
          f"({train_cache.mb:.0f} MB) in {time.time() - t_load:.0f}s", flush=True)
    steps_per_epoch = (len(train_cache) + args.batch - 1) // args.batch

    def freeze_bn():
        # model.train() switches every BatchNorm to "update statistics"; put the
        # frozen ones back to fixed statistics.
        if args.freeze_stem:
            net = model.encoder if hasattr(model, "encoder") else model.net
            for layer in list(net.children())[:5]:
                for m in layer.modules():
                    if isinstance(m, torch.nn.BatchNorm2d):
                        m.eval()
    use_amp = device == "cuda"

    if args.dry_run:
        # Check that data, model, loss and backward pass all fit together.
        # No optimiser step: the weights stay exactly as loaded.
        model.train()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for step, batch in enumerate(train_cache.batches(args.batch, shuffle=True)):
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                loss, _, _ = losses(model, batch, device, args, train_augment=True)
            loss.backward()
            model.zero_grad(set_to_none=True)
            print(f"dry-run batch {step}: ref {tuple(batch[0].shape)} loss {loss.item():.3f}")
            if step == 1:
                break
        if device == "cuda":
            print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**20:.0f} MB")
        print("dry-run OK (no weights were updated, nothing saved)")
        return

    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({**vars(args), "run": run, "started": datetime.now().isoformat()}, indent=2))
    writer = SummaryWriter(str(OUTPUTS / "tensorboard" / run))
    writer.add_text("config", json.dumps(vars(args), indent=2))

    # Evaluation sets. The training split re-uses the training cache, but only
    # its first 1000 pairs, so judging "train" each epoch stays cheap.
    # When training on NORVIA only (--sets D2), every split is NORVIA-only too,
    # so validation (model selection, thresholds) matches what was trained.
    if set(train_sets) == {"D2"}:
        eval_splits = [("train", {"D2"}), ("val", {"D2"}), ("test", {"D2"})]
    else:
        eval_splits = EVAL_SPLITS
    val_name = split_name("val", next(s for sp, s in eval_splits if sp == "val"))
    train_name = split_name("train", next(s for sp, s in eval_splits if sp == "train"))
    eval_sets = {}
    for split, sets in eval_splits:
        name = split_name(split, sets)
        if split == "train":
            sub = train_rows[:1000]
            eval_sets[name] = (sub, PairCache(sub, size, device) if len(train_rows) > 1000 else train_cache)
            continue
        rows, _ = load_rows(data, {split}, sets, args.full_res, args.raw_crops)
        if rows:
            eval_sets[name] = (rows, PairCache(rows, size, device))

    optimiser = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs * steps_per_epoch)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_auc, best_epoch, history, step = -1.0, 0, [], 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        freeze_bn()
        t0, total, n_seen = time.time(), 0.0, 0
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for batch in train_cache.batches(args.batch, shuffle=True):
            optimiser.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                loss, loss_bin, loss_kind = losses(model, batch, device, args, train_augment=True)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            scheduler.step()
            step += 1
            total += loss.item() * len(batch[2])
            n_seen += len(batch[2])
            writer.add_scalar("step/loss_total", loss.item(), step)
            writer.add_scalar("step/loss_tampered", loss_bin.item(), step)
            writer.add_scalar("step/loss_kind", loss_kind.item(), step)
            writer.add_scalar("step/lr", scheduler.get_last_lr()[0], step)

        record = {"epoch": epoch, "train_loss_running": total / n_seen, "seconds": time.time() - t0}
        if device == "cuda":
            record["peak_gpu_mb"] = torch.cuda.max_memory_allocated() / 2**20   # includes the image cache
            writer.add_scalar("system/peak_gpu_mb", record["peak_gpu_mb"], epoch)
        writer.add_scalar("epoch/train_loss_running", record["train_loss_running"], epoch)

        # Evaluate every split with the weights at the end of this epoch.
        for name, (rows, ld) in eval_sets.items():
            probs, kinds, idx, bin_loss = predict(model, ld, device, args.eval_batch)
            rows_o = [rows[i] for i in idx]
            labels = np.array([int(r["label"]) for r in rows_o])
            res = {"auc": auc(probs[labels == 1], probs[labels == 0]),
                   "acc": float(((probs >= 0.5) == labels).mean()), "bce": bin_loss}
            for k, v in res.items():
                record[f"{name}/{k}"] = v
                if not np.isnan(v):
                    writer.add_scalar(f"{name}/{k}", v, epoch)
        history.append(record)
        writer.flush()

        val_auc = record[f"{val_name}/auc"]
        print(f"epoch {epoch:2d} | train loss {record['train_loss_running']:.4f} "
              f"| train AUC {record[f'{train_name}/auc']:.4f} | val AUC {val_auc:.4f} acc {record[f'{val_name}/acc']:.3f} "
              f"| test D2 AUC {record.get('test_D2/auc', float('nan')):.4f} "
              f"| unseen-font AUC {record.get('test_unseen_font_D1/auc', float('nan')):.4f} "
              f"| {record['seconds']:.0f}s, GPU {record.get('peak_gpu_mb', 0):.0f} MB", flush=True)
        if val_auc > best_auc:
            best_auc, best_epoch = val_auc, epoch
            torch.save(model.state_dict(), out / "best.pt")

    # ---- final evaluation with the best epoch -----------------------------------
    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    rows, ld = eval_sets[val_name]  # thresholds are chosen on validation only
    probs, kinds, idx, _ = predict(model, ld, device, args.eval_batch)
    val_o = [rows[i] for i in idx]
    is_genuine = np.array([r["label"] == "0" for r in val_o])
    threshold = float(np.percentile(probs[is_genuine], 95))    # flags 5 % of genuine validation samples
    low_threshold = float(np.percentile(probs[~is_genuine], 5))  # 95 % of tampered validation samples score above this

    metrics = {"run": run, "arch": args.arch, "model_options": options, "config": vars(args), "best_epoch": best_epoch,
               "best_val_auc": best_auc, "threshold_5pct_false_alarm": threshold,
               "threshold_low": low_threshold, "input_size": size, "history": history, "results": {}}
    pred_rows = []
    for name, (rows, ld) in eval_sets.items():
        if name.startswith("train"):
            continue
        probs, kinds, idx, _ = predict(model, ld, device, args.eval_batch)
        rows_o = [rows[i] for i in idx]
        metrics["results"][name] = evaluate(rows_o, probs, kinds, threshold)
        for r, p, k in zip(rows_o, probs, kinds):
            pred_rows.append({"split": name, "id": r["id"], "set": r["set"], "label": r["label"],
                              "tamper": r["tamper"], "magnitude": r["magnitude"],
                              "prob_tampered": round(float(p), 4), "pred_kind": TAMPER_TYPES[k]})
        for t, v in metrics["results"][name]["caught_per_tamper"].items():
            writer.add_scalar(f"final_{name}/caught_{t}", v["caught"], best_epoch)

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    with open(out / "predictions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=pred_rows[0].keys())
        w.writeheader()
        w.writerows(pred_rows)
    writer.close()
    print(f"best epoch {best_epoch} (val AUC {best_auc:.4f}); threshold at 5 % false alarms = {threshold:.4f}")
    print(json.dumps(metrics["results"], indent=2))


if __name__ == "__main__":
    main()
