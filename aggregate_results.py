"""
aggregate_results.py
====================
Walks all runs/ subdirectories, reads every results.json,
and produces two outputs:
  1. results_full.csv    — one row per evaluation (all metrics, all runs)
  2. results_summary.csv — best config per model/variety combination
     ranked by weighted-avg F1 on same-variety test set

Usage:
    python aggregate_results.py
    python aggregate_results.py --runs_dir runs --out_dir .
"""

import argparse
import csv
import json
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--out_dir",  default=".")
    return p.parse_args()


CLASSES      = ["black", "broken", "moldy"]
METRIC_COLS  = ["precision", "recall", "f1-score", "support"]
AGG_ROWS     = ["macro avg", "weighted avg"]


def flatten_report(report: dict) -> dict:
    """Turn a sklearn classification_report dict into flat columns."""
    row = {}
    for cls in CLASSES + AGG_ROWS:
        safe = cls.replace(" ", "_")
        for m in METRIC_COLS:
            row[f"{safe}__{m}"] = round(report.get(cls, {}).get(m, 0), 4)
    row["accuracy"] = round(report.get("accuracy", 0), 4)
    return row


def load_all_runs(runs_dir: Path) -> list[dict]:
    rows = []
    for result_file in sorted(runs_dir.glob("*/results.json")):
        with open(result_file) as f:
            try:
                meta = json.load(f)
            except json.JSONDecodeError:
                print(f"[WARN] Could not parse {result_file}, skipping.")
                continue

        base = {
            "run_name":        meta.get("run_name", ""),
            "phase":           "phase3" if meta.get("pretrained_from") else "phase12",
            "train_variety":   meta.get("variety", ""),
            "model":           meta.get("model", ""),
            "optimizer":       meta.get("optimizer", ""),
            "lr":              meta.get("lr", ""),
            "batch_size":      meta.get("batch_size", ""),
            "dropout":         meta.get("dropout", ""),
            "best_epoch":      meta.get("best_epoch", ""),
            "best_val_loss":   meta.get("best_val_loss", ""),
            "pretrained_from": meta.get("pretrained_from") or "",
        }

        for eval_result in meta.get("results", []):
            row = dict(base)
            row["eval_label"] = eval_result.get("label", "")

            # Determine test variety from label
            label = eval_result.get("label", "")
            row["test_variety"] = (
                "same" if "→ tested:" + base["train_variety"] in label
                else "other"
            )

            row.update(flatten_report(eval_result.get("report", {})))
            row["confusion_matrix"] = str(eval_result.get("confusion_matrix", []))
            rows.append(row)

    return rows


def write_full_csv(rows: list[dict], path: Path):
    if not rows:
        print("[WARN] No rows to write.")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] Full results  → {path}  ({len(rows)} rows)")


def write_summary_csv(rows: list[dict], path: Path):
    """
    For each (phase, train_variety, model) group, find the run with the
    highest weighted-avg F1 on the same-variety test set and write one row.
    """
    # Only consider same-variety evaluations for ranking
    same_rows = [r for r in rows if r["test_variety"] == "same"]

    # Group by (phase, train_variety, model)
    groups: dict[tuple, list] = {}
    for r in same_rows:
        key = (r["phase"], r["train_variety"], r["model"])
        groups.setdefault(key, []).append(r)

    summary = []
    for key, group in sorted(groups.items()):
        best = max(group, key=lambda x: float(x.get("weighted_avg__f1-score", 0)))
        summary.append(best)

    if not summary:
        print("[WARN] No summary rows to write.")
        return

    fieldnames = list(summary[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)
    print(f"[OK] Summary       → {path}  ({len(summary)} rows)")


def print_leaderboard(rows: list[dict]):
    """Print a quick terminal leaderboard of best configs."""
    same_rows = [r for r in rows if r["test_variety"] == "same"
                 and r["phase"] == "phase12"]
    if not same_rows:
        return

    print("\n" + "="*75)
    print(f"  {'Model':<15} {'Variety':<10} {'Optimizer':<10} {'LR':<8} "
          f"{'BS':<5} {'DO':<5} {'wF1':>6}")
    print("="*75)

    groups: dict[tuple, list] = {}
    for r in same_rows:
        key = (r["train_variety"], r["model"])
        groups.setdefault(key, []).append(r)

    for key in sorted(groups.keys()):
        group = groups[key]
        best = max(group, key=lambda x: float(x.get("weighted_avg__f1-score", 0)))
        print(f"  {best['model']:<15} {best['train_variety']:<10} "
              f"{best['optimizer']:<10} {str(best['lr']):<8} "
              f"{str(best['batch_size']):<5} {str(best['dropout']):<5} "
              f"{float(best.get('weighted_avg__f1-score', 0)):>6.4f}")
    print("="*75)


def main():
    args   = parse_args()
    runs   = Path(args.runs_dir)
    outdir = Path(args.out_dir)
    outdir.mkdir(exist_ok=True)

    print(f"Scanning {runs} ...")
    rows = load_all_runs(runs)

    if not rows:
        print("[WARN] No results.json files found. "
              "Have any training runs completed?")
        return

    print(f"Found {len(rows)} evaluation rows across "
          f"{len(set(r['run_name'] for r in rows))} runs.\n")

    write_full_csv(rows,    outdir / "results_full.csv")
    write_summary_csv(rows, outdir / "results_summary.csv")
    print_leaderboard(rows)

    print("\nDone. Open results_full.csv in Excel for full analysis.")


if __name__ == "__main__":
    main()