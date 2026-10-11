"""
EXP-007: run a list of training configurations one after another and collect
the results in one table.

Each run is a normal call of train_pairs.py, so every run also has its own
TensorBoard log, metrics.json and predictions.csv. A run whose metrics.json
already exists is skipped, so the sweep can be restarted safely.

Model choice uses VALIDATION AUC only. Test numbers are reported next to it
so we can see how well validation predicts test, never to pick a winner.

Run:   python src/run_sweep.py phase1     (or phase2, phase3, all)
Output: outputs/exp007/sweep_summary.csv and sweep_summary.md (all finished runs)
"""
import csv
import json
import subprocess
import sys

from common import OUTPUTS, ROOT

BASE = ["--arch", "siamese_diff", "--epochs", "15", "--batch", "32"]

# name -> (what changes compared with the baseline, WHY we try it, extra arguments)
PHASES = {
    "phase1": {
        "sd_base_s1": ("baseline, seed 1", "measure run-to-run noise before judging any change", ["--seed", "1"]),
        "sd_base_s2": ("baseline, seed 2", "measure run-to-run noise before judging any change", ["--seed", "2"]),
        "sd_aug": ("mild augmentation (brightness, blur, noise)", "baseline over-fits after ~6 epochs; extra camera-like noise may regularise", ["--augment"]),
        "sd_drop_wd": ("dropout 0.3 + weight decay 1e-3", "classic regularisation against the over-fitting seen in the baseline", ["--dropout", "0.3", "--weight-decay", "1e-3"]),
        "sd_freeze": ("frozen first conv + layer1", "fewer trainable weights = less over-fitting; low-level ImageNet filters may be enough", ["--freeze-stem"]),
        "sd_depth1": ("encoder depth 1 (1/4 resolution)", "re-spacing is missed; finer feature maps keep more position detail", ["--depth", "1"]),
        "sd_depth3": ("encoder depth 3 (1/16 resolution)", "opposite check: does more 'meaning' help glyph/font swaps despite coarser positions?", ["--depth", "3"]),
        "sd_focal": ("focal loss (gamma 2)", "obvious tampers are learned fast; focal loss puts weight on the hard, subtle ones", ["--loss", "focal"]),
        "sd_kind0": ("no 'which tamper' loss", "does the auxiliary 8-class task help or distract the yes/no task?", ["--kind-weight", "0"]),
        "sd_kind1": ("'which tamper' loss weight 1.0", "same question from the other side: a stronger auxiliary task", ["--kind-weight", "1.0"]),
        "sd_lr1e-4": ("learning rate 1e-4", "val AUC peaks early then drops; a slower learning rate may peak later and higher", ["--lr", "1e-4"]),
    },
    "phase2": {
        "sd_depth1_s1": ("encoder depth 1, seed 1", "depth 1 was the only setting at the edge of the noise band (0.905): repeat before believing it", ["--depth", "1", "--seed", "1"]),
        "sd_depth1_s2": ("encoder depth 1, seed 2", "as above", ["--depth", "1", "--seed", "2"]),
    },
    "phase3": {
        "ef_base": ("early_fusion architecture (6-channel ResNet-18)", "compare architectures with the same baseline settings", ["--arch", "early_fusion"]),
        "dino_base": ("dino_frozen architecture (frozen DINOv2 + head)", "compare architectures; frozen backbone may over-fit less", ["--arch", "dino_frozen"]),
    },
    # ~4x more training pairs (data/synth_more: new seeds, train split only). Val/test are unchanged.
    "phase4": {
        "sd_more": ("baseline + data/synth_more (~4x training pairs)", "the baseline over-fits after ~6 epochs on 2k pairs; more data is the most direct fix", ["--extra-data", "data/synth_more"]),
        "sd_depth1_more": ("depth 1 + data/synth_more", "best phase-1 setting, with the extra data", ["--depth", "1", "--extra-data", "data/synth_more"]),
    },
    # Candidate: "i was hoping we could also train with just 2 classes - tampered yes and no".
    # kind-weight 0 = only the yes/no loss (BCE). sd_kind0 tested this at depth 2 (no difference);
    # here with the new default depth 1, two seeds because of the noise.
    "phase5": {
        "sd_depth1_bin_s0": ("depth 1, yes/no loss only (no 8-class task), seed 0", " pure 2-class training, with the best setting", ["--depth", "1", "--kind-weight", "0"]),
        "sd_depth1_bin_s1": ("depth 1, yes/no loss only, seed 1", "as above (second seed)", ["--depth", "1", "--kind-weight", "0", "--seed", "1"]),
    },
    # Resolution experiment, agreed with the candidate (D44): NORVIA only, depth 1, batch 16,
    # everything else unchanged. The 448 run is the control: same data, only the resolution differs.
    # NOTE: val/test are NORVIA-only here, so val AUC is not directly comparable with the
    # D1+D2 runs above; compare test_D2, and the two runs with each other.
    "phase6": {
        "norvia_448": ("NORVIA only, 448x128, depth 1, batch 16", "control for the resolution run (and shows whether dropping the font set D1 changes NORVIA results)",
                       ["--sets", "D2", "--extra-data", "data/synth_more", "--depth", "1", "--batch", "16"]),
        "norvia_896": ("NORVIA only, 896x256 (full resolution), depth 1, batch 16", "half-size crops turn 1-2 px tampers into sub-pixel changes; full resolution keeps them",
                       ["--sets", "D2", "--extra-data", "data/synth_more", "--depth", "1", "--batch", "16",
                        "--full-res", "--input-w", "896", "--input-h", "256",
                        # memory only (first attempt ran out of GPU memory in evaluation):
                        "--gpu-cache-mb", "0", "--eval-batch", "16"]),
    },
}
NOISE_BAND = (0.884, 0.896)     # val AUC of 4 identical baseline runs (see EXP-007)

TAMPERS = ["stretch", "recolour", "respace", "glyph_swap", "font_swap", "overlap", "cut"]


def run_one(name, extra):
    out = OUTPUTS / "exp007" / name
    if (out / "metrics.json").exists():
        print(f"skip {name} (already done)", flush=True)
        return
    log = OUTPUTS / "exp007" / "logs" / f"{name}.log"
    cmd = [sys.executable, str(ROOT / "src" / "train_pairs.py"), *BASE, "--run", name, *extra]
    print(f"start {name}: {' '.join(extra)}", flush=True)
    with open(log, "w") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    print(f"end   {name}: exit code {result.returncode}", flush=True)
    summarise()            # keep the summary and tuning log up to date after every run
    write_tuning_log()


def summarise():
    rows = []
    for metrics_file in sorted((OUTPUTS / "exp007").glob("*/metrics.json")):
        m = json.loads(metrics_file.read_text())
        res = m["results"]
        row = {"run": m["run"], "best_epoch": m["best_epoch"], "val_auc": round(m["best_val_auc"], 3)}
        for split in ("test_D2", "test_D1", "test_unseen_font_D1"):
            row[f"{split}_auc"] = round(res[split]["auc"], 3) if split in res else ""
        for t in TAMPERS:   # share caught on the NORVIA test set at 5 % false alarms (val threshold)
            row[f"D2_{t}"] = round(res["test_D2"]["caught_per_tamper"].get(t, {}).get("caught", float("nan")), 2)
        row["D2_false_alarms"] = round(res["test_D2"]["false_alarm_rate_at_threshold"], 3)
        rows.append(row)
    if not rows:
        return
    with open(OUTPUTS / "exp007" / "sweep_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    keys = list(rows[0].keys())
    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    lines += ["| " + " | ".join(str(r[k]) for k in keys) + " |" for r in sorted(rows, key=lambda r: -r["val_auc"])]
    (OUTPUTS / "exp007" / "sweep_summary.md").write_text(
        "Sorted by validation AUC. D2_* = share of each tamper caught on the NORVIA test set "
        "at the threshold that flags 5 % of genuine validation samples.\n\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))


def write_tuning_log():
    """experiments/EXP-007_tuning_log.md: every tuning run with what changed, why, result and verdict."""
    lines = ["# EXP-007 tuning log (generated by src/run_sweep.py)", "",
             f"Baseline: siamese_diff, depth 2, batch 32, AdamW lr 3e-4, cosine, 15 epochs, BCE + 0.3 x CE. "
             f"Noise band (4 identical runs): val AUC {NOISE_BAND[0]}-{NOISE_BAND[1]}. "
             f"Verdict rule: better only if val AUC > {NOISE_BAND[1] + 0.009:.3f}; worse if < {NOISE_BAND[0] - 0.005:.3f}. "
             "Selection uses validation only; test is shown for reference.", "",
             "| phase | run | change | why | val AUC | test D2 AUC | unseen font AUC | verdict |",
             "|---|---|---|---|---|---|---|---|"]
    for ph, runs in PHASES.items():
        for name, (change, why, _) in runs.items():
            f = OUTPUTS / "exp007" / name / "metrics.json"
            if not f.exists():
                lines.append(f"| {ph} | {name} | {change} | {why} | - | - | - | not run yet |")
                continue
            m = json.loads(f.read_text())
            v = m["best_val_auc"]
            verdict = ("better" if v > NOISE_BAND[1] + 0.009 else "worse" if v < NOISE_BAND[0] - 0.005
                       else "no measurable difference")
            if name.startswith("sd_base"):
                verdict = "noise measurement"
            r = m["results"]
            unseen = r.get("test_unseen_font_D1", {}).get("auc")
            unseen = f"{unseen:.3f}" if unseen is not None else "n/a (NORVIA only)"
            lines.append(f"| {ph} | {name} | {change} | {why} | {v:.3f} | {r['test_D2']['auc']:.3f} | "
                         f"{unseen} | {verdict} |")
    (ROOT / "experiments" / "EXP-007_tuning_log.md").write_text("\n".join(lines) + "\n")


def main():
    phase = sys.argv[1] if len(sys.argv) > 1 else None
    phases = [phase] if phase != "all" else list(PHASES)
    if phase:
        for ph in phases:
            for name, (change, why, extra) in PHASES[ph].items():
                run_one(name, extra)
    summarise()
    write_tuning_log()


if __name__ == "__main__":
    main()
