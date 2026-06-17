import json
import re
import shutil
from pathlib import Path

RUNS_DIR = Path("runs")
DUPE_DIR = RUNS_DIR / "dupe_runs"
DUPE_DIR.mkdir(exist_ok=True)

DRY_RUN = True

timestamp_re = re.compile(r"(.+)_\d{8}_\d{6}$")

groups = {}


BS8_DIR = RUNS_DIR / "removed_bs8"
BS8_DIR.mkdir(exist_ok=True)

for folder in list(RUNS_DIR.iterdir()):
    if not folder.is_dir():
        continue

    if folder.name in ("dupe_runs", "removed_bs8"):
        continue

    if "_bs8_" in folder.name:
        print(f"BS8 MOVE  {folder.name}")

        if not DRY_RUN:
            shutil.move(
                str(folder),
                str(BS8_DIR / folder.name)
            )

for folder in RUNS_DIR.iterdir():
    if not folder.is_dir():
        continue

    if folder.name == "dupe_runs":
        continue

    m = timestamp_re.match(folder.name)
    if not m:
        continue

    config = m.group(1)
    groups.setdefault(config, []).append(folder)

moved = 0

for config, folders in groups.items():

    if len(folders) < 2:
        continue

    candidates = []

    for folder in folders:

        results_file = folder / "results.json"

        if not results_file.exists():
            print(f"[SKIP] Missing results.json: {folder.name}")
            continue

        try:
            with open(results_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            variety = data["variety"]

            score = None

            for result in data["results"]:

                label = result["label"].lower()

                if (
                    f"trained:{variety}" in label
                    and f"tested:{variety}" in label
                ):
                    score = result["report"]["macro avg"]["f1-score"]
                    break

            if score is None:
                print(f"[SKIP] Could not find same-variety score: {folder.name}")
                continue

            candidates.append((score, folder))

        except Exception as e:
            print(f"[ERROR] {folder.name}: {e}")

    if len(candidates) < 2:
        continue

    candidates.sort(key=lambda x: x[0], reverse=True)

    best_score, best_folder = candidates[0]

    print(
        f"\nKEEP  {best_folder.name}"
        f"  (macro_f1={best_score:.6f})"
    )

    for score, folder in candidates[1:]:

        destination = DUPE_DIR / folder.name

        print(
            f"MOVE  {folder.name}"
            f"  (macro_f1={score:.6f})"
        )

        if not DRY_RUN:
            shutil.move(str(folder), str(destination))
            moved += 1

print(f"\nDone. Moved {moved} duplicate runs.")