"""
EXP-007: evidence pictures and tables for the review panel (inference only, no training).

Everything goes into outputs/exp007/evidence/; nothing that already exists is touched.

  real_photos/     the 49 rectified real photos scored by sd_depth1_more, sd_more,
                   ef_base (and sd_base again, only to check this script reproduces
                   the saved sd_base/real_predictions.csv); same procedure as
                   infer_real.py (3 references, mean probability, run thresholds).
                   Per-photo picture: photo crop | reference | difference map.
  training_curves/ train loss and val AUC per epoch from TensorBoard, best epoch marked
  test_results/    ROC curves, share caught per tamper type at the 5 % false-alarm
                   threshold, "which tamper" confusion matrix, score histograms
                   (all from the saved predictions.csv files)
  examples/        synthetic TEST pairs for sd_depth1_more: caught, missed, false alarms
  architecture/    block diagram of siamese_diff depth 1 and early_fusion, parameter counts

Run: python src/exp007_evidence.py
"""
import csv
import json

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyBboxPatch
from tensorboard.backend.event_processing import event_accumulator

from common import OUTPUTS, ROOT, load_reference
from infer_real import CUT_PCT, cut_percentages, half_size, load_run
from models import INPUT_H, INPUT_W, TAMPER_TYPES, build
from train_pairs import load_rows, to_tensor

RUNS_DIR = OUTPUTS / "exp007"
OUT = RUNS_DIR / "evidence"
BEST = "sd_depth1_more"
REAL_RUNS = ["sd_depth1_more", "sd_more", "ef_base", "sd_base"]   # sd_base only as a reproduction check
KEY_RUNS = ["sd_depth1_more", "sd_more", "sd_base", "ef_base", "dino_base", "norvia_448", "norvia_896"]
BAR_RUNS = ["sd_depth1_more", "sd_more", "sd_base", "ef_base", "dino_base"]
TEST_SPLITS = [("test_D2", "NORVIA test (D2)"), ("test_D1", "multi-font test (D1)"),
               ("test_unseen_font_D1", "unseen font (Lato, D1)")]
# Fixed colour per run (the run keeps its colour in every figure).
COLOURS = {"sd_depth1_more": "#2a78d6", "sd_more": "#eb6834", "sd_base": "#1baf7a", "ef_base": "#eda100",
           "dino_base": "#e87ba4", "norvia_448": "#008300", "norvia_896": "#4a3aa7"}
GENUINE_C, TAMPERED_C = "#2a78d6", "#eb6834"
device = "cuda" if torch.cuda.is_available() else "cpu"

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e4e4e4", "grid.linewidth": 0.6, "figure.dpi": 110})


def metrics_of(run):
    return json.loads((RUNS_DIR / run / "metrics.json").read_text())


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def show_heat(ax, crop_rgb, heat, scale):
    """Grey crop with the difference map in red on top; opacity follows the value,
    so places with little difference show the plain logo underneath."""
    h = cv2.resize(heat, (crop_rgb.shape[1], crop_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    h = np.clip(h / scale, 0, 1)
    ax.imshow(cv2.cvtColor(cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB))
    return ax.imshow(h, cmap="Reds", vmin=0, vmax=1, alpha=0.85 * h)


def clean(ax, title=None):
    ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=8, loc="left")


# ---- 1. real photos ---------------------------------------------------------------

def decide(p_mean, t_high, t_low, cut_pct):
    """Same decision rule as infer_real.py (cut-off flag from EXP-010's cut_pct)."""
    decision = "tampered" if p_mean >= t_high else "genuine" if p_mean <= t_low else "don't know"
    if cut_pct > CUT_PCT:
        decision += " + flagged: logo partly outside the photo"
    return decision


@torch.no_grad()
def real_photos():
    out = OUT / "real_photos"
    out.mkdir(parents=True, exist_ok=True)
    registration = read_csv(OUTPUTS / "exp001" / "registration.csv")
    refs = [half_size(load_reference(i)) for i in (1, 2, 3)]
    results = {}
    for run in REAL_RUNS:
        model, m = load_run(run, device)
        size = tuple(m["input_size"])
        t_high, t_low = m["threshold_5pct_false_alarm"], m["threshold_low"]
        ref_batch = torch.stack([to_tensor(r, size) for r in refs]).to(device)
        rows, heats = [], []
        cut = cut_percentages()
        for r in registration:
            if r["registered"] != "True":
                rows.append({"image": r["image"], "p_ref1": "", "p_ref2": "", "p_ref3": "", "p_mean": "",
                             "p_min": "", "p_max": "", "kind": "", "decision": "flagged: can't register"})
                continue
            crop = half_size(cv2.imread(str(OUTPUTS / "exp001" / "rectified" / f"{r['image']}.png")))
            cand = to_tensor(crop, size)[None].to(device).expand(3, -1, -1, -1)
            logit, kind_logits = model(ref_batch, cand)
            p = torch.sigmoid(logit.float()).cpu().numpy()
            kinds = np.bincount(kind_logits.argmax(1).cpu().numpy(), minlength=len(TAMPER_TYPES))
            cut_pct = cut.get(r["image"], 0.0)
            row = {"image": r["image"], "p_ref1": round(float(p[0]), 4), "p_ref2": round(float(p[1]), 4),
                   "p_ref3": round(float(p[2]), 4), "p_mean": round(float(p.mean()), 4),
                   "p_min": round(float(p.min()), 4), "p_max": round(float(p.max()), 4),
                   "kind": TAMPER_TYPES[kinds.argmax()], "decision": decide(float(p.mean()), t_high, t_low, cut_pct)}
            rows.append(row)
            if hasattr(model, "heatmap"):
                heats.append((row, crop, model.heatmap(ref_batch, cand).float().mean(0).cpu().numpy()))
        write_csv(out / f"real_predictions_{run}.csv", rows)
        results[run] = rows
        if heats and run != "sd_base":
            heat_pictures(run, heats, refs[0], t_high, t_low, out)
        del model
        torch.cuda.empty_cache()

    # Does this script reproduce what infer_real.py saved for sd_base?
    old = read_csv(RUNS_DIR / "sd_base" / "real_predictions.csv")
    diffs = [abs(float(a["p_mean"]) - float(b["p_mean"])) for a, b in zip(old, results["sd_base"]) if a["p_mean"]]
    same_decisions = sum(a["decision"] == b["decision"] for a, b in zip(old, results["sd_base"]))
    print(f"sd_base reproduction: max |p_mean diff| {max(diffs):.5f}, same decision {same_decisions}/{len(old)}")

    # One table over runs, with the saved (old) sd_base predictions as the reference column.
    show = ["sd_depth1_more", "sd_more", "ef_base"]
    summary = []
    for i, r in enumerate(registration):
        row = {"image": r["image"], "ecc_corr": r["ecc_corr"]}
        row["sd_base_saved_p"], row["sd_base_saved_decision"] = old[i]["p_mean"], old[i]["decision"]
        votes = old[i]["decision"].startswith("tampered")
        for run in show:
            res = results[run][i]
            row[f"{run}_p"], row[f"{run}_kind"], row[f"{run}_decision"] = res["p_mean"], res["kind"], res["decision"]
            votes += res["decision"].startswith("tampered")
        row["runs_saying_tampered"] = f"{votes}/{len(show) + 1}"
        summary.append(row)
    write_csv(out / "real_comparison.csv", summary)
    real_score_plot(results, old, registration, out)

    counts = {run: count_decisions(results[run]) for run in show}
    counts["sd_base (saved)"] = count_decisions(old)
    return counts, max(diffs), same_decisions


def count_decisions(rows):
    c = {}
    for r in rows:
        k = r["decision"].split(" +")[0]
        c[k] = c.get(k, 0) + 1
    return c


def heat_pictures(run, heats, ref, t_high, t_low, out):
    """Per-photo picture (crop | reference | difference map) and one contact sheet.
    One colour scale for all photos of this run, so red means the same everywhere."""
    folder = out / f"heatmaps_{run}"
    folder.mkdir(exist_ok=True)
    scale = float(np.percentile(np.concatenate([h.ravel() for _, _, h in heats]), 99.5))
    for row, crop, heat in heats:
        fig, axes = plt.subplots(1, 3, figsize=(12, 1.9))
        axes[0].imshow(rgb(crop)), clean(axes[0], f"real photo {row['image']} (rectified crop)")
        axes[1].imshow(rgb(ref)), clean(axes[1], "reference ref_1 (ref_2/3 share the layout)")
        im = show_heat(axes[2], rgb(crop), heat, scale)
        clean(axes[2], "feature difference |a-b|, mean over 3 refs (shared scale)")
        fig.colorbar(im, ax=axes[2], fraction=0.025, pad=0.01, ticks=[0, 1]).ax.set_yticklabels(["0", "max"])
        fig.suptitle(f"{run}: p_mean {row['p_mean']:.3f} (refs {row['p_ref1']:.3f} / {row['p_ref2']:.3f} / "
                     f"{row['p_ref3']:.3f})  kind: {row['kind']}  ->  {row['decision']}   "
                     f"[tampered >= {t_high:.3f}, genuine <= {t_low:.3f}]", fontsize=9, x=0.01, ha="left")
        fig.subplots_adjust(left=0.005, right=0.97, top=0.72, bottom=0.02, wspace=0.03)
        fig.savefig(folder / f"{row['image']}.jpg", dpi=110)
        plt.close(fig)

    # Contact sheet: difference map only, 5 per row, frame colour = decision.
    frame = {"tampered": "#e34948", "don't know": "#eda100", "genuine": "#1baf7a"}
    n_rows = (len(heats) + 4) // 5
    fig, axes = plt.subplots(n_rows, 5, figsize=(17, 1.25 * n_rows + 0.3))
    for ax in axes.ravel():
        clean(ax)
    for ax, (row, crop, heat) in zip(axes.ravel(), heats):
        show_heat(ax, rgb(crop), heat, scale)
        d = row["decision"].split(" +")[0]
        ax.set_title(f"{row['image']}  p={row['p_mean']:.2f}  {row['kind']}  [{d}{' +cut flag' if '+' in row['decision'] else ''}]",
                     fontsize=7.5, loc="left")
        for s in ax.spines.values():
            s.set_visible(True), s.set_color(frame[d]), s.set_linewidth(2.5)
    fig.suptitle(f"{run} on the 49 registered real photos (029 cannot be registered). Frame: red = tampered, "
                 f"amber = don't know, green = genuine. Red overlay = feature difference, one scale for all photos.",
                 fontsize=10, x=0.01, ha="left")
    fig.subplots_adjust(left=0.005, right=0.995, top=1 - 0.75 / (1.25 * n_rows), bottom=0.005, hspace=0.35, wspace=0.03)
    fig.savefig(out / f"contact_sheet_{run}.jpg", dpi=100)
    plt.close(fig)


def real_score_plot(results, old, registration, out):
    """p_mean per photo for each run, photos sorted by the recommended model's score."""
    order = sorted([i for i, r in enumerate(registration) if r["registered"] == "True"],
                   key=lambda i: results[BEST][i]["p_mean"])
    names = [registration[i]["image"] for i in order]
    fig, ax = plt.subplots(figsize=(13, 4))
    series = [(BEST, results[BEST]), ("sd_more", results["sd_more"]), ("ef_base", results["ef_base"]),
              ("sd_base", old)]
    for run, rows in series:
        ax.plot(range(len(order)), [float(rows[i]["p_mean"]) for i in order], "o", ms=5 if run == BEST else 4,
                color=COLOURS[run], label=run + (" (saved predictions)" if run == "sd_base" else ""),
                alpha=1 if run == BEST else 0.75)
    m = metrics_of(BEST)
    ax.axhline(m["threshold_5pct_false_alarm"], color=COLOURS[BEST], lw=1, ls="--")
    ax.axhline(m["threshold_low"], color=COLOURS[BEST], lw=1, ls=":")
    ax.text(0, m["threshold_5pct_false_alarm"] + 0.015, f"{BEST} 'tampered' threshold {m['threshold_5pct_false_alarm']:.3f}",
            fontsize=8, color="#52514e")
    ax.text(0, m["threshold_low"] + 0.015, f"{BEST} 'genuine' threshold {m['threshold_low']:.3f}", fontsize=8, color="#52514e")
    ax.set_xticks(range(len(order)), names, rotation=90, fontsize=7)
    ax.set_ylabel("p_mean (mean over 3 references)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Real photos: score per photo and run (sorted by sd_depth1_more). Each run has its own thresholds "
                 "(see real_comparison.csv).", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(0, 0.93), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "real_scores_by_run.png", dpi=120)
    plt.close(fig)


# ---- 2. training curves --------------------------------------------------------------

def tb_curves(run):
    acc = event_accumulator.EventAccumulator(str(OUTPUTS / "tensorboard" / run), size_guidance={"scalars": 0})
    acc.Reload()
    tags = acc.Tags()["scalars"]

    def series(tag):
        return {e.step: e.value for e in acc.Scalars(tag)} if tag in tags else {}
    val_tag = next(t for t in tags if t.startswith("val_") and t.endswith("/auc"))
    train_tag = next(t for t in tags if t.startswith("train_") and t.endswith("/auc"))
    return {"loss": series("epoch/train_loss_running"), "val": series(val_tag), "train_auc": series(train_tag),
            "test_d2": series("test_D2/auc"), "val_name": val_tag.split("/")[0]}


def training_curves():
    out = OUT / "training_curves"
    out.mkdir(parents=True, exist_ok=True)
    runs = sorted(p.name for p in (OUTPUTS / "tensorboard").iterdir() if (RUNS_DIR / p.name / "metrics.json").exists())
    table, curves = [], {}
    for run in runs:
        c, m = tb_curves(run), metrics_of(run)
        curves[run] = c
        best = m["best_epoch"]
        tb_best = max(c["val"], key=lambda e: (c["val"][e], -e))   # first epoch with the top value, like training
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.2))
        ep = sorted(c["loss"])
        a1.plot(ep, [c["loss"][e] for e in ep], "-o", ms=3, lw=2, color="#2a78d6")
        a1.set(xlabel="epoch", ylabel="training loss (BCE + kind x CE)", title="train loss (running mean per epoch)")
        for key, label, colour, style in (("train_auc", "train AUC (1000-pair subset)", "#8a8984", "--"),
                                          ("val", f"val AUC ({c['val_name']})", "#2a78d6", "-"),
                                          ("test_d2", "test NORVIA AUC (monitoring only)", "#eb6834", ":")):
            if c[key]:
                ep = sorted(c[key])
                a2.plot(ep, [c[key][e] for e in ep], style, marker="o", ms=3, lw=2 if key == "val" else 1.3,
                        color=colour, label=label)
        a2.axvline(best, color="#52514e", lw=1, ls="--")
        a2.annotate(f"best epoch {best}\nval AUC {m['best_val_auc']:.3f}", (best, m["best_val_auc"]),
                    xytext=(6, -32), textcoords="offset points", fontsize=8)
        a2.plot([best], [m["best_val_auc"]], "*", ms=12, color="#2a78d6")
        a2.set(xlabel="epoch", ylabel="AUC", title="AUC per epoch (best epoch chosen on val only)")
        a2.legend(fontsize=7.5, frameon=False, loc="lower right")
        fig.suptitle(f"{run}", fontsize=11, x=0.01, ha="left", fontweight="bold")
        fig.tight_layout()
        fig.savefig(out / f"{run}.png", dpi=110)
        plt.close(fig)
        table.append({"run": run, "best_epoch_metrics": best, "best_epoch_tensorboard": tb_best,
                      "best_val_auc": round(m["best_val_auc"], 4), "val_set": c["val_name"],
                      "final_epoch_val_auc": round(c["val"][max(c["val"])], 4),
                      "final_epoch_train_auc": round(c["train_auc"][max(c["train_auc"])], 4),
                      "train_loss_first": round(c["loss"][min(c["loss"])], 4),
                      "train_loss_last": round(c["loss"][max(c["loss"])], 4)})
    write_csv(out / "best_epochs.csv", table)
    mismatched = [t["run"] for t in table if t["best_epoch_metrics"] != t["best_epoch_tensorboard"]]
    print(f"training curves: {len(runs)} runs; best epoch TensorBoard vs metrics.json mismatches: {mismatched or 'none'}")

    # Overview: val AUC per epoch for the key runs.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [2.2, 1]})
    for ax, group, title in ((axes[0], KEY_RUNS[:5], "mixed validation set (D1+D2, 454 pairs)"),
                             (axes[1], KEY_RUNS[5:], "NORVIA-only validation set (D2, 243 pairs)")):
        for run in group:
            c, m = curves[run], metrics_of(run)
            ep = sorted(c["val"])
            ax.plot(ep, [c["val"][e] for e in ep], "-", lw=2, color=COLOURS[run])
            ax.plot([m["best_epoch"]], [m["best_val_auc"]], "o", ms=8, color=COLOURS[run], mec="white", mew=2)
            ax.annotate(f"{run} {m['best_val_auc']:.3f}", (ep[-1], c["val"][ep[-1]]), xytext=(4, 0),
                        textcoords="offset points", fontsize=8, va="center", color="#0b0b0b")
        ax.set(xlabel="epoch", ylabel="validation AUC", title=title, xlim=(0.5, 19.5))
        ax.set_xticks(range(1, 16))
    axes[0].text(0.01, 0.02, "dot = best epoch (the weights that were kept)", transform=axes[0].transAxes, fontsize=8,
                 color="#52514e")
    fig.suptitle("Validation AUC per epoch, key runs (the two panels use different validation sets: not comparable "
                 "across panels)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "overview_val_auc.png", dpi=120)
    plt.close(fig)
    return table


# ---- 3. test results ----------------------------------------------------------------

def roc(scores, labels):
    thresholds = np.r_[np.inf, np.unique(scores)[::-1]]
    pos, neg = scores[labels == 1], scores[labels == 0]
    return [(neg >= t).mean() for t in thresholds], [(pos >= t).mean() for t in thresholds]


def roc_auc(scores, labels):
    pos, neg = scores[labels == 1], scores[labels == 0]
    return float(((pos[:, None] > neg[None]) + 0.5 * (pos[:, None] == neg[None])).mean())


def wilson(k, n, z=1.96):
    """95 % interval for a proportion; honest error bars for ~15 samples per type."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def load_predictions(run):
    rows = read_csv(RUNS_DIR / run / "predictions.csv")
    by_split = {}
    for r in rows:
        by_split.setdefault(r["split"], []).append(r)
    return by_split


def test_results():
    out = OUT / "test_results"
    out.mkdir(parents=True, exist_ok=True)
    preds = {run: load_predictions(run) for run in KEY_RUNS}

    # Summary table, recomputed from predictions.csv and checked against metrics.json.
    table, worst_check = [], 0.0
    for run in KEY_RUNS:
        m = metrics_of(run)
        thr = m["threshold_5pct_false_alarm"]
        for split, rows in preds[run].items():
            s = np.array([float(r["prob_tampered"]) for r in rows])
            y = np.array([int(r["label"]) for r in rows])
            row = {"run": run, "split": split, "n": len(rows), "threshold": round(thr, 4),
                   "auc": round(roc_auc(s, y), 4), "false_alarm_rate": round(float((s[y == 0] >= thr).mean()), 4)}
            worst_check = max(worst_check, abs(row["auc"] - m["results"][split]["auc"]))
            tam = [r for r in rows if r["label"] == "1"]
            row["kind_accuracy_on_tampered"] = round(np.mean([r["pred_kind"] == r["tamper"] for r in tam]), 4)
            for t in TAMPER_TYPES[1:]:
                sel = [float(r["prob_tampered"]) >= thr for r in rows if r["tamper"] == t]
                row[f"caught_{t}"] = f"{sum(sel)}/{len(sel)}"
            table.append(row)
    write_csv(out / "test_summary.csv", table)
    print(f"test results: largest AUC difference vs metrics.json {worst_check:.5f} (rounding only if ~1e-4)")

    # ROC curves.
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    for ax, (split, title) in zip(axes, TEST_SPLITS):
        for run in KEY_RUNS:
            if split not in preds[run]:
                continue
            rows = preds[run][split]
            s = np.array([float(r["prob_tampered"]) for r in rows])
            y = np.array([int(r["label"]) for r in rows])
            fpr, tpr = roc(s, y)
            ax.plot(fpr, tpr, lw=2.4 if run == BEST else 1.4, color=COLOURS[run],
                    label=f"{run}  AUC {roc_auc(s, y):.3f}", zorder=3 if run == BEST else 2)
            if run == BEST:
                thr = metrics_of(run)["threshold_5pct_false_alarm"]
                fa, hit = (s[y == 0] >= thr).mean(), (s[y == 1] >= thr).mean()
                ax.plot([fa], [hit], "o", ms=8, color=COLOURS[run], mec="white", mew=2, zorder=4)
                ax.annotate(f"operating point (thr {thr:.3f}):\n{fa:.1%} false alarms, {hit:.1%} caught", (fa, hit),
                            xytext=(0.3, 0.6), textcoords="axes fraction", fontsize=7.5,
                            arrowprops={"arrowstyle": "-", "color": "#8a8984"})
        ax.plot([0, 1], [0, 1], color="#c3c2b7", lw=1, ls=":")
        n_g = sum(r["label"] == "0" for r in preds[BEST][split])
        ax.set(xlabel="false-alarm rate (genuine flagged)", ylabel="share of tampers caught",
               title=f"{title}: {len(preds[BEST][split])} pairs ({n_g} genuine)", xlim=(-0.01, 1), ylim=(0, 1.01))
        ax.legend(fontsize=7, loc="lower right", frameon=False)
    fig.suptitle("ROC on the synthetic test sets (from predictions.csv). norvia_* runs were trained/tested on NORVIA "
                 "only, so they appear only in the first panel.", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "roc_test_sets.png", dpi=120)
    plt.close(fig)

    # Share caught per tamper type at each run's own 5 % (validation) false-alarm threshold.
    types = TAMPER_TYPES[1:]
    fig, axes = plt.subplots(3, 1, figsize=(15, 10.5))
    width = 0.8 / len(BAR_RUNS)
    for ax, (split, title) in zip(axes, TEST_SPLITS):
        for j, run in enumerate(BAR_RUNS):
            thr = metrics_of(run)["threshold_5pct_false_alarm"]
            rows = preds[run][split]
            rates, lo, hi, ns = [], [], [], []
            for t in types:
                sel = [float(r["prob_tampered"]) >= thr for r in rows if r["tamper"] == t]
                k, n = sum(sel), len(sel)
                a, b = wilson(k, n)
                rates.append(k / n), lo.append(k / n - a), hi.append(b - k / n), ns.append(n)
            fa = np.mean([float(r["prob_tampered"]) >= thr for r in rows if r["label"] == "0"])
            x = np.arange(len(types)) + (j - (len(BAR_RUNS) - 1) / 2) * width
            ax.bar(x, rates, width * 0.92, color=COLOURS[run], label=f"{run} (false alarms here {fa:.1%})")
            ax.errorbar(x, rates, yerr=[lo, hi], fmt="none", ecolor="#52514e", elinewidth=0.7, capsize=0)
        ax.set_xticks(range(len(types)), [f"{t}\n(n={n})" for t, n in zip(types, ns)])
        ax.set(ylim=(0, 1.08), ylabel="share caught", title=title)
        ax.title.set_ha("left"), ax.title.set_x(0)
        ax.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False)
    fig.suptitle("Share of tampers caught at the threshold that flags 5 % of genuine VALIDATION pairs "
                 "(per run; set in train_pairs.py). Whiskers: 95 % Wilson interval.", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "caught_per_tamper.png", dpi=120)
    plt.close(fig)

    # "Which tamper?" head: confusion matrix for the recommended model.
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))
    for ax, (split, title) in zip(axes, TEST_SPLITS):
        cm = np.zeros((8, 8), int)
        for r in preds[BEST][split]:
            cm[TAMPER_TYPES.index(r["tamper"]), TAMPER_TYPES.index(r["pred_kind"])] += 1
        share = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        for i in range(8):
            for j in range(8):
                if cm[i, j]:
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                            color="white" if share[i, j] > 0.55 else "#0b0b0b")
        ax.set_xticks(range(8), TAMPER_TYPES, rotation=45, ha="right")
        ax.set_yticks(range(8), TAMPER_TYPES)
        ax.grid(False)
        ax.set(xlabel="predicted kind", ylabel="true kind",
               title=f"{title}: kind correct {np.trace(cm) / cm.sum():.0%} of all pairs")
    fig.suptitle(f"{BEST}: 'which tamper?' head (8 classes) on the test sets. Numbers = counts, shade = share of the row.",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / f"confusion_kind_{BEST}.png", dpi=120)
    plt.close(fig)

    # Score histograms genuine vs tampered.
    hist_runs = [BEST, "sd_base", "ef_base"]
    fig, axes = plt.subplots(len(hist_runs), 3, figsize=(14, 3 * len(hist_runs)), sharex=True)
    bins = np.linspace(0, 1, 26)
    for i, run in enumerate(hist_runs):
        thr = metrics_of(run)["threshold_5pct_false_alarm"]
        for ax, (split, title) in zip(axes[i], TEST_SPLITS):
            rows = preds[run][split]
            for lab, colour, name in (("0", GENUINE_C, "genuine"), ("1", TAMPERED_C, "tampered")):
                ax.hist([float(r["prob_tampered"]) for r in rows if r["label"] == lab], bins=bins, color=colour,
                        alpha=0.65, label=name, edgecolor="white", linewidth=0.5)
            ax.axvline(thr, color="#0b0b0b", lw=1, ls="--")
            ax.text(thr + 0.01, ax.get_ylim()[1] * 0.9, f"thr {thr:.2f}", fontsize=7.5)
            ax.set_title(f"{run}: {title}", fontsize=9, loc="left")
            if i == len(hist_runs) - 1:
                ax.set_xlabel("predicted probability of 'tampered'")
            if ax is axes[i][0]:
                ax.set_ylabel("pairs")
    axes[0][0].legend(fontsize=8, frameon=False)
    fig.suptitle("Score distributions on the test sets; dashed = 5 % false-alarm threshold from validation",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "score_histograms.png", dpi=110)
    plt.close(fig)
    return table


# ---- 4. synthetic test examples ---------------------------------------------------------

@torch.no_grad()
def examples():
    out = OUT / "examples"
    out.mkdir(parents=True, exist_ok=True)
    model, m = load_run(BEST, device)
    thr = m["threshold_5pct_false_alarm"]
    data = ROOT / "data" / "synth"
    rows, _ = load_rows(data, {"test", "test_unseen_font"}, {"D1", "D2"})
    by_id = {r["id"]: r for r in rows}
    saved = [p for p in read_csv(RUNS_DIR / BEST / "predictions.csv") if p["split"].startswith("test")]

    # Re-run the model on exactly the training-time inputs (same files, same resize)
    # so the heatmap belongs to the score; the score must match predictions.csv.
    items = []
    for i in range(0, len(saved), 16):
        chunk = saved[i:i + 16]
        imgs = [(cv2.imread(str(by_id[p["id"]]["ref_path"])), cv2.imread(str(by_id[p["id"]]["cand_path"]))) for p in chunk]
        ref = torch.stack([to_tensor(a) for a, _ in imgs]).to(device)
        cand = torch.stack([to_tensor(b) for _, b in imgs]).to(device)
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
            logit, _ = model(ref, cand)
        probs = torch.sigmoid(logit.float()).cpu().numpy()
        heats = model.heatmap(ref, cand).float().cpu().numpy()
        for p, (a, b), prob, heat in zip(chunk, imgs, probs, heats):
            small = lambda x: cv2.resize(x, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)
            items.append({**p, "prob_now": float(prob), "ref": small(a), "cand": small(b), "heat": heat,
                          "detail": by_id[p["id"]]["detail"], "unit": by_id[p["id"]]["unit"]})
    diff = max(abs(it["prob_now"] - float(it["prob_tampered"])) for it in items)
    print(f"examples: re-scored {len(items)} test pairs, max |p - saved p| = {diff:.4f}")
    scale = float(np.percentile(np.concatenate([it["heat"].ravel() for it in items]), 99.5))

    for it in items:
        it["flagged"] = float(it["prob_tampered"]) >= thr
    tp = [it for it in items if it["label"] == "1" and it["flagged"]]
    fn = [it for it in items if it["label"] == "1" and not it["flagged"]]
    fp = [it for it in items if it["label"] == "0" and it["flagged"]]
    tn = [it for it in items if it["label"] == "0" and not it["flagged"]]

    # Caught: two per tamper type from the NORVIA test set (first two by id), plus two genuine passes.
    correct = []
    for t in TAMPER_TYPES[1:]:
        correct += [it for it in tp if it["tamper"] == t and it["split"] == "test_D2"][:2]
    correct += [it for it in tn if it["split"] == "test_D2"][:2]
    order = {s: i for i, (s, _) in enumerate(TEST_SPLITS)}
    key = lambda it: (order[it["split"]], it["tamper"], float(it["prob_tampered"]))
    sheets = {"correct": (correct, "caught tampers (2 per type) and 2 genuine passes, NORVIA test"),
              "missed": (sorted(fn, key=key), "MISSED tampers (false negatives), all test sets"),
              "false_alarms": (sorted(fp, key=key), "FALSE ALARMS (genuine pairs flagged), all test sets")}
    for name, (chosen, title) in sheets.items():
        # At most 16 pairs per page so the pictures stay large enough to read.
        pages = [chosen[i:i + 16] for i in range(0, len(chosen), 16)]
        for k, page in enumerate(pages, 1):
            suffix = f"_{k}of{len(pages)}" if len(pages) > 1 else ""
            example_sheet(page, scale, thr, out / f"{name}{suffix}.jpg", title + (f" (page {k}/{len(pages)})" if suffix else ""))
    write_csv(out / "examples_index.csv",
              [{"sheet": name, "split": it["split"], "id": it["id"], "tamper": it["tamper"],
                "magnitude": it["magnitude"], "unit": it["unit"], "detail": it["detail"],
                "prob_tampered": it["prob_tampered"], "pred_kind": it["pred_kind"]}
               for name, (chosen, _) in sheets.items() for it in chosen])
    counts = {"threshold": thr, "TP": len(tp), "FN": len(fn), "FP": len(fp), "TN": len(tn),
              "FN_by_type": {t: sum(it["tamper"] == t for it in fn) for t in TAMPER_TYPES[1:]},
              "FN_by_split": {s: sum(it["split"] == s for it in fn) for s, _ in TEST_SPLITS},
              "FP_by_split": {s: sum(it["split"] == s for it in fp) for s, _ in TEST_SPLITS}}
    print("examples:", counts)
    del model
    torch.cuda.empty_cache()
    return counts


def example_sheet(chosen, scale, thr, path, title):
    """One row per pair: candidate | reference | difference map, with the label written above."""
    height = 1.55 * len(chosen) + 0.8
    fig, axes = plt.subplots(len(chosen), 3, figsize=(13, height), squeeze=False)
    for row, it in zip(axes, chosen):
        row[0].imshow(rgb(it["cand"])), clean(row[0])
        row[1].imshow(rgb(it["ref"])), clean(row[1])
        show_heat(row[2], rgb(it["cand"]), it["heat"], scale), clean(row[2])
        what = "genuine" if it["tamper"] == "none" else f"{it['tamper']}  {it['detail']}  (magnitude {float(it['magnitude']):.2f} {it['unit']})"
        row[0].set_title(f"{it['split']}  {it['id']}  TRUE: {what}", fontsize=8, loc="left")
        row[2].set_title(f"p(tampered) = {float(it['prob_tampered']):.3f}  ->  {'FLAGGED' if it['flagged'] else 'passed'}"
                         f"   predicted kind: {it['pred_kind']}", fontsize=8, loc="left")
    fig.suptitle(f"{BEST}: {title}. Columns: candidate | reference | feature difference (one scale for all examples). "
                 f"Flagged if p >= {thr:.3f}.", fontsize=9.5, x=0.01, ha="left")
    fig.subplots_adjust(left=0.005, right=0.995, top=1 - 1.1 / height, bottom=0.005,
                        hspace=0.45, wspace=0.02)
    fig.savefig(path, dpi=100)
    plt.close(fig)


# ---- 5. architecture --------------------------------------------------------------------

def count(module):
    return sum(p.numel() for p in module.parameters())


def trainable_for(run):
    """Trainable parameters of a run, applying --freeze-stem the way train_pairs.py does."""
    m = metrics_of(run)
    cfg = json.loads((RUNS_DIR / run / "config.json").read_text())
    model = build(m["arch"], pretrained=False, **m.get("model_options", {}))
    if cfg.get("freeze_stem"):
        net = model.encoder if hasattr(model, "encoder") else model.net
        for layer in list(net.children())[:5]:
            for p in layer.parameters():
                p.requires_grad = False
    # sd_base predates model_options; it used the default depth 2. Other archs have no options.
    opts = m.get("model_options", {"depth": 2}) if m["arch"] == "siamese_diff" else "-"
    return m["arch"], opts, sum(p.numel() for p in model.parameters() if p.requires_grad), count(model)


@torch.no_grad()
def architecture():
    out = OUT / "architecture"
    out.mkdir(parents=True, exist_ok=True)
    x = torch.zeros(1, 3, INPUT_H, INPUT_W)
    sd = build("siamese_diff", pretrained=False, depth=1).eval()
    ef = build("early_fusion", pretrained=False).eval()
    # Shapes measured from a forward pass, not written by hand.
    a = sd.encoder(x)
    c = sd.compare(torch.cat([a, a], 1))
    ef_feat = ef.net(torch.cat([x, x], 1))

    lines = ["EXP-007 parameter counts, computed from src/models.py (python src/exp007_evidence.py)", "",
             "Per architecture (all parameters; 'trainable' = what the optimiser updates by default):"]
    sd2 = build("siamese_diff", pretrained=False, depth=2)
    sd3 = build("siamese_diff", pretrained=False, depth=3)
    for name, model in (("siamese_diff depth 1", sd), ("siamese_diff depth 2", sd2), ("siamese_diff depth 3", sd3)):
        lines.append(f"  {name:22s} total {count(model):>10,}  = encoder {count(model.encoder):>9,} + compare "
                     f"{count(model.compare):>8,} + heads {count(model.heads):>5,}")
    lines.append(f"  {'early_fusion':22s} total {count(ef):>10,}  = ResNet-18 (6-ch conv1, no fc) {count(ef.net):>10,} "
                 f"+ heads {count(ef.heads):>5,}")
    try:
        dino = build("dino_frozen", pretrained=False)
        dino_train = sum(p.numel() for p in dino.parameters() if p.requires_grad)
        lines.append(f"  {'dino_frozen':22s} total {count(dino):>10,}  = frozen ViT-S/14 {count(dino.vit):>10,} "
                     f"+ compare {count(dino.compare):>8,} + heads {count(dino.heads):>5,}  (trainable {dino_train:,})")
    except Exception as e:   # timm missing: say so instead of guessing a number
        lines.append(f"  dino_frozen: could not be built here ({e})")
    lines += ["", "Measured tensor shapes (batch 1, input 3 x 128 x 448):",
              f"  siamese_diff depth 1: encoder output {tuple(a.shape[1:])}, after compare {tuple(c.shape[1:])}, "
              f"pooled {2 * c.shape[1]} -> heads 1 + 8",
              f"  early_fusion: 6-channel input -> ResNet-18 -> {tuple(ef_feat.shape[1:])} -> heads 1 + 8", "",
              "Trainable parameters per run (from each run's metrics.json / config.json):",
              f"  {'run':20s} {'arch':14s} {'options':28s} {'trainable':>11s} {'total':>11s}"]
    for metrics_file in sorted(RUNS_DIR.glob("*/metrics.json")):
        run = metrics_file.parent.name
        if run == "evidence":
            continue
        arch, opts, n_train, n_total = trainable_for(run)
        lines.append(f"  {run:20s} {arch:14s} {str(opts):28s} {n_train:>11,} {n_total:>11,}")
    (out / "parameter_counts.txt").write_text("\n".join(lines) + "\n")

    diagram(tuple(a.shape[1:]), tuple(c.shape[1:]), tuple(ef_feat.shape[1:]), count(sd), count(ef), out)
    return lines


def diagram(enc_shape, cmp_shape, ef_shape, n_sd, n_ef, out):
    fig, ax = plt.subplots(figsize=(15, 6.6))
    ax.set_xlim(0, 15), ax.set_ylim(0, 6.6)
    clean(ax)

    def box(x, y, w, h, text, colour):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=colour, ec="#52514e", lw=1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops={"arrowstyle": "-|>", "color": "#52514e", "lw": 1})

    inp, enc, cmp_c, head = "#eef3fb", "#d6e6fa", "#fde6d9", "#e2f4ea"
    c, h, w = enc_shape
    # siamese_diff, top half
    ax.text(0.1, 6.35, f"A. siamese_diff, depth 1 (recommended; {n_sd:,} parameters, all trained)",
            fontsize=11, fontweight="bold")
    box(0.2, 5.0, 1.6, 0.8, "reference\n3 x 128 x 448", inp)
    box(0.2, 3.6, 1.6, 0.8, "candidate\n(rectified crop)\n3 x 128 x 448", inp)
    box(2.4, 5.0, 2.6, 0.8, "ResNet-18 stem + stage 1\n(7x7 conv, maxpool, 4 conv)", enc)
    box(2.4, 3.6, 2.6, 0.8, "same encoder\n(SHARED weights)", enc)
    ax.text(3.7, 4.6, "same weights", ha="center", fontsize=7.5, color="#52514e")
    arrow(1.8, 5.4, 2.4, 5.4), arrow(1.8, 4.0, 2.4, 4.0)
    box(5.6, 5.0, 1.5, 0.8, f"a: {c} x {h} x {w}\n(1/4 resolution)", enc)
    box(5.6, 3.6, 1.5, 0.8, f"b: {c} x {h} x {w}", enc)
    arrow(5.0, 5.4, 5.6, 5.4), arrow(5.0, 4.0, 5.6, 4.0)
    box(7.6, 4.3, 1.7, 0.8, f"|a - b|  and  a * b\nstacked: {2 * c} x {h} x {w}", cmp_c)
    arrow(7.1, 5.4, 7.6, 4.9), arrow(7.1, 4.0, 7.6, 4.5)
    box(9.7, 4.3, 2.3, 0.8, f"2 conv 3x3 + BN + ReLU\n(2nd stride 2) -> {cmp_shape[0]} x {cmp_shape[1]} x {cmp_shape[2]}", cmp_c)
    arrow(9.3, 4.7, 9.7, 4.7)
    box(12.2, 4.3, 1.3, 0.8, f"max + mean\npool -> {2 * cmp_shape[0]}", cmp_c)
    arrow(12.0, 4.7, 12.2, 4.7)
    box(13.9, 5.0, 1.0, 0.6, "tampered?\n1 logit", head)
    box(13.9, 3.8, 1.0, 0.6, "which?\n8 classes", head)
    arrow(13.5, 4.8, 13.9, 5.3), arrow(13.5, 4.6, 13.9, 4.1)
    ax.text(5.6, 3.15, "Heatmap shown in the evidence = mean over channels of |a - b| (where the two feature maps "
            "differ), before the learned comparison.", fontsize=8, color="#52514e")
    # early_fusion, bottom half
    ax.text(0.1, 2.55, f"B. early_fusion (comparison baseline; {n_ef:,} parameters, all trained)", fontsize=11,
            fontweight="bold")
    box(0.2, 1.5, 1.6, 0.6, "reference\n3 x 128 x 448", inp)
    box(0.2, 0.6, 1.6, 0.6, "candidate\n3 x 128 x 448", inp)
    box(2.4, 1.0, 1.6, 0.7, "stack channels\n6 x 128 x 448", cmp_c)
    arrow(1.8, 1.8, 2.4, 1.45), arrow(1.8, 0.9, 2.4, 1.25)
    box(4.6, 1.0, 4.6, 0.7, "full ResNet-18 (stem + 4 stages); first conv widened to 6 channels\n"
                            "(pretrained RGB filters copied for both images, halved)", enc)
    arrow(4.0, 1.35, 4.6, 1.35)
    box(9.8, 1.0, 2.0, 0.7, f"global average pool\n-> {ef_shape[0]}", enc)
    arrow(9.2, 1.35, 9.8, 1.35)
    box(13.9, 1.5, 1.0, 0.6, "tampered?\n1 logit", head)
    box(13.9, 0.5, 1.0, 0.6, "which?\n8 classes", head)
    arrow(11.8, 1.45, 13.9, 1.8), arrow(11.8, 1.25, 13.9, 0.8)
    ax.text(4.6, 0.25, "No explicit comparison step: the network must learn to compare the two images itself.",
            fontsize=8, color="#52514e")
    fig.savefig(out / "architecture_diagram.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"device {device}, writing to {OUT}")
    lines = architecture()
    print("\n".join(lines[:9]))
    training_curves()
    test_results()
    counts, max_diff, same = real_photos()
    print("real photo decisions:", json.dumps(counts, indent=1))
    examples()
    print(f"done: {OUT}")


if __name__ == "__main__":
    main()
