from pathlib import Path
import json
import re

runs_dir = Path("linuxruns_v2")

for run_dir in sorted(runs_dir.iterdir()):
    if not run_dir.is_dir():
        continue

    results = run_dir / "results.json"

    if not results.exists():
        print(f"SKIP    {run_dir.name} (no results.json)")
        continue

    try:
        data = json.loads(results.read_text())
        lr = data.get("lr")

        if lr is None:
            print(f"SKIP    {run_dir.name} (no lr in results.json)")
            continue

        base_name = run_dir.name.removeprefix("fixed_NORLR_")

        lr_str = f"{float(lr):.0e}"
        lr_str = re.sub(r"e([+-])0+(\d+)", r"e\1\2", lr_str)

        new_name = re.sub(
            r"_lr[^_]+(?=_bs)",
            f"_lr{lr_str}",
            base_name
        )

        if new_name == run_dir.name:
            print(f"OK      {run_dir.name}")
            continue

        new_dir = run_dir.parent / new_name

        if new_dir.exists():
            print(f"ERROR   Cannot rename {run_dir.name}")
            print(f"        Target already exists: {new_name}")
            continue

        run_dir.rename(new_dir)
        print(f"RENAMED {run_dir.name} -> {new_name}")

    except Exception as e:
        print(f"ERROR   {run_dir.name}: {e}")