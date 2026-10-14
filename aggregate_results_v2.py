"""
aggregate_results_v2.py
=======================
Aggregate Stage 1 run results into CSV files without including epoch history.

Stage 2 directories named fixed_ft_* are excluded based on their folder name.
The folder name is also retained because older results.json files may contain
an inconsistent run_name.

Usage:
    python aggregate_results_v2.py
    python aggregate_results_v2.py --runs_dir linuxruns --out_dir .
"""

import argparse
import csv
import json
from pathlib import Path


CLASSES = ["black", "broken", "moldy"]
METRIC_COLS = ["precision", "recall", "f1-score", "support"]
AGG_ROWS = ["macro avg", "weighted avg"]
NORLR_PREFIX = "fixed_NORLR_"


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate Stage 1 run results")
    parser.add_argument("--runs_dir", default="linuxruns")
    parser.add_argument("--out_dir", default=".")
    return parser.parse_args()


def flatten_report(report: dict) -> dict:
    row = {}
    for class_name in CLASSES + AGG_ROWS:
        safe_name = class_name.replace(" ", "_")
        for metric in METRIC_COLS:
            value = report.get(class_name, {}).get(metric, "")
            row[f"{safe_name}__{metric}"] = round(value, 4) if isinstance(value, (int, float)) else value
    accuracy = report.get("accuracy", "")
    row["accuracy"] = round(accuracy, 4) if isinstance(accuracy, (int, float)) else accuracy
    return row


def load_stage1_runs(runs_dir: Path) -> list[dict]:
    rows = []
    skipped_stage2 = 0
    skipped_invalid = 0

    for result_file in sorted(runs_dir.glob("*/results.json")):
        folder_name = result_file.parent.name

        if folder_name.startswith(NORLR_PREFIX):
            pass
        elif folder_name.startswith("fixed_ft_"):
            skipped_stage2 += 1
            continue
        else:
            skipped_invalid += 1
            continue

        try:
            with result_file.open(encoding="utf-8") as file:
                meta = json.load(file)
        except (OSError, json.JSONDecodeError) as error:
            print(f"[WARN] Could not read {result_file}: {error}")
            skipped_invalid += 1
            continue

        base = {
            "folder_name": folder_name,
            "run_name": meta.get("run_name") or folder_name,
            "phase": "stage1",
            "train_variety": meta.get("variety", ""),
            "model": meta.get("model", ""),
            "optimizer": meta.get("optimizer", ""),
            "lr": meta.get("lr", ""),
            "batch_size": meta.get("batch_size", ""),
            "dropout": meta.get("dropout", ""),
            "max_epochs": meta.get("max_epochs", ""),
            "best_epoch": meta.get("best_epoch", ""),
            "best_val_loss": meta.get("best_val_loss", ""),
        }

        for evaluation in meta.get("results", []):
            row = dict(base)
            label = evaluation.get("label", "")
            row["eval_label"] = label
            row["test_variety"] = (
                "same" if f"tested:{base['train_variety']}" in label else "other"
            )
            row.update(flatten_report(evaluation.get("report", {})))
            row["confusion_matrix"] = str(evaluation.get("confusion_matrix", []))
            rows.append(row)
    print(
            f"Skipped {skipped_stage2} Stage 2 run(s) and "
            f"{skipped_invalid} non-NORLR/invalid folder(s)."
        )
    return rows


def write_csv(rows: list[dict], path: Path):
    if not rows:
        print(f"[WARN] No rows to write: {path}")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] Wrote {path} ({len(rows)} rows)")


def build_summary(rows: list[dict]) -> list[dict]:
    same_rows = [row for row in rows if row["test_variety"] == "same"]
    groups = {}
    for row in same_rows:
        key = (row["train_variety"], row["model"])
        groups.setdefault(key, []).append(row)

    summary = []
    for key in sorted(groups):
        best = max(
            groups[key],
            key=lambda row: float(row.get("weighted_avg__f1-score") or 0),
        )
        summary.append(best)
    return summary


def main():
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning Stage 1 runs in {runs_dir} ...")
    rows = load_stage1_runs(runs_dir)
    if not rows:
        print("[WARN] No completed Stage 1 evaluation results found.")
        return

    run_count = len({row["folder_name"] for row in rows})
    print(f"Found {len(rows)} evaluation row(s) across {run_count} Stage 1 run(s).")
    write_csv(rows, out_dir / "results_full_v2.csv")
    write_csv(build_summary(rows), out_dir / "results_summary_v2.csv")


if __name__ == "__main__":
    main()
