#!/usr/bin/env python3
"""Create an HTML visual validation report for a saved coffee checkpoint."""

import argparse
import base64
import html
import mimetypes
from pathlib import Path

import torch

from test_saved_checkpoint import (
    CLASSES,
    collect_images,
    get_model,
    get_transforms,
    load_model,
    load_run_config,
    predict,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a visual HTML report for checkpoint predictions"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run_dir", help="Run directory containing best_model.pth and results.json")
    source.add_argument("--checkpoint", help="Path to best_model.pth")
    parser.add_argument("--image", required=True, help="Test image or directory")
    parser.add_argument("--model", choices=["squeezenet", "shufflenet", "mobilenetv1", "mobilenetv2", "mobilenetv3", "efficientnet"])
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--recursive", action="store_true", help="Search image directories recursively")
    parser.add_argument("--output", default="visual_validation_report.html", help="HTML report path")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    return parser.parse_args()


def ground_truth(image_path: Path):
    for parent in image_path.parents:
        if parent.name.lower() in CLASSES:
            return parent.name.lower()
    return None


def image_data_uri(image_path: Path):
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def percent(value):
    return f"{value * 100:.1f}%"


def build_report(rows, checkpoint, model_name, device, image_root):
    known_rows = [row for row in rows if row["actual"] is not None]
    correct = sum(row["actual"] == row["prediction"] for row in known_rows)
    accuracy = correct / len(known_rows) if known_rows else None
    matrix = {actual: {predicted: 0 for predicted in CLASSES} for actual in CLASSES}
    for row in known_rows:
        matrix[row["actual"]][row["prediction"]] += 1

    cards = []
    for row in rows:
        actual = row["actual"] or "unknown"
        result_class = "correct" if row["actual"] == row["prediction"] else "incorrect"
        actual_display = actual if row["actual"] else "ground truth unavailable"
        probabilities = "".join(
            f'<div class="prob"><span>{html.escape(label)}</span>'
            f'<div class="bar"><i style="width:{value * 100:.1f}%"></i></div>'
            f'<b>{percent(value)}</b></div>'
            for label, value in zip(CLASSES, row["probabilities"])
        )
        if row["actual"] is None:
            result_class = "unknown"
        elif row["actual"] == row["prediction"]:
            result_class = "correct"
        else:
            result_class = "incorrect"
        cards.append(
            f'<article class="tile {result_class}" data-result="{result_class}" '
            f'data-label="{html.escape(row["prediction"])}">'
            f'<img src="{row["image_uri"]}" loading="lazy" alt="{html.escape(row["name"])}">'
            f'<div class="tile-body"><div class="file">{html.escape(row["name"])}</div>'
            f'<div class="labels"><span class="prediction">Predicted: <strong>{html.escape(row["prediction"])}</strong></span>'
            f'<span class="actual">Actual: <strong>{html.escape(actual_display)}</strong></span></div>'
            f'<div class="confidence">Confidence {percent(row["confidence"])}</div>{probabilities}</div></article>'
        )

    matrix_header = "".join(f"<th>{html.escape(label)}</th>" for label in CLASSES)
    matrix_rows = "".join(
        f'<tr><th>{html.escape(actual)}</th>'
        + "".join(f"<td>{matrix[actual][predicted]}</td>" for predicted in CLASSES)
        + "</tr>"
        for actual in CLASSES
    )
    accuracy_text = percent(accuracy) if accuracy is not None else "n/a"
    report_title = f"Coffee checkpoint validation | {model_name}"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(report_title)}</title>
<style>
:root {{ --ink:#172321; --muted:#64736e; --paper:#f4f1e8; --panel:#fffdf8; --line:#d9ded5; --green:#216b52; --red:#b44d3c; --gold:#c58a26; }}
* {{ box-sizing:border-box; }} body {{ margin:0; color:var(--ink); background:var(--paper); font:15px/1.4 Georgia, serif; }}
header {{ padding:42px 5vw 30px; background:linear-gradient(135deg,#dfe9db,#f4f1e8 62%,#eadcc4); border-bottom:1px solid var(--line); }}
h1 {{ margin:0 0 8px; font:700 clamp(28px,4vw,48px)/1.05 Georgia, serif; letter-spacing:0; }}
.subtitle {{ color:var(--muted); max-width:900px; overflow-wrap:anywhere; }} main {{ width:min(1500px,90vw); margin:28px auto 60px; }}
.summary {{ display:grid; grid-template-columns:repeat(4,minmax(130px,1fr)); gap:12px; margin-bottom:24px; }}
.metric {{ padding:18px; background:var(--panel); border:1px solid var(--line); border-radius:6px; }} .metric b {{ display:block; font:700 28px Georgia,serif; }} .metric span {{ color:var(--muted); }}
.controls {{ display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-bottom:20px; }} button {{ border:1px solid var(--line); background:var(--panel); color:var(--ink); padding:9px 13px; border-radius:4px; cursor:pointer; font:inherit; }} button.active,button:hover {{ background:var(--green); color:white; border-color:var(--green); }}
.report-grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(265px,1fr)); gap:16px; }} .tile {{ overflow:hidden; background:var(--panel); border:2px solid var(--line); border-radius:6px; }} .tile.correct {{ border-color:#86b89a; }} .tile.incorrect {{ border-color:#d48676; }}
.tile img {{ display:block; width:100%; aspect-ratio:4/3; object-fit:cover; background:#e8e6df; }} .tile-body {{ padding:12px; }} .file {{ white-space:nowrap; overflow:hidden; text-overflow:ellipsis; color:var(--muted); font-size:13px; margin-bottom:9px; }}
.labels {{ display:grid; gap:3px; }} .prediction strong {{ color:var(--green); }} .incorrect .prediction strong {{ color:var(--red); }} .actual {{ color:var(--muted); }} .confidence {{ margin:10px 0 7px; font-weight:bold; }} .prob {{ display:grid; grid-template-columns:48px 1fr 43px; gap:6px; align-items:center; font-size:12px; color:var(--muted); }} .prob b {{ text-align:right; font-weight:normal; }} .bar {{ height:7px; background:#e5e9e2; border-radius:3px; overflow:hidden; }} .bar i {{ display:block; height:100%; background:var(--gold); }}
section.review {{ margin:34px 0; }} h2 {{ font-size:25px; margin:0 0 12px; }} table {{ border-collapse:collapse; background:var(--panel); min-width:430px; }} th,td {{ border:1px solid var(--line); padding:9px 18px; text-align:center; }} th {{ background:#e6eee1; }}
@media (max-width:650px) {{ .summary {{ grid-template-columns:repeat(2,1fr); }} main {{ width:94vw; }} .report-grid {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<header><h1>Visual checkpoint validation</h1><div class="subtitle">{html.escape(model_name)} | {html.escape(str(checkpoint))} | device: {html.escape(str(device))} | image root: {html.escape(str(image_root))}</div></header>
<main>
<div class="summary"><div class="metric"><b>{len(rows)}</b><span>images reviewed</span></div><div class="metric"><b>{accuracy_text}</b><span>accuracy with known labels</span></div><div class="metric"><b>{sum(row["actual"] is None for row in rows)}</b><span>unknown labels</span></div><div class="metric"><b>{sum(row["actual"] != row["prediction"] for row in known_rows)}</b><span>errors to inspect</span></div></div>
<section class="review"><h2>Confusion matrix</h2><table><thead><tr><th>Actual \\ Predicted</th>{matrix_header}</tr></thead><tbody>{matrix_rows}</tbody></table></section>
<div class="controls"><strong>Show:</strong><button class="active" data-filter="all">All</button><button data-filter="correct">Correct</button><button data-filter="incorrect">Errors</button>{''.join(f'<button data-filter="{html.escape(label)}">{html.escape(label.title())}</button>' for label in CLASSES)}</div>
<section class="report-grid">{''.join(cards)}</section>
</main>
<script>
const tiles = [...document.querySelectorAll('.tile')];
document.querySelectorAll('button[data-filter]').forEach(button => button.addEventListener('click', () => {{
  document.querySelectorAll('button[data-filter]').forEach(item => item.classList.remove('active'));
  button.classList.add('active');
  const filter = button.dataset.filter;
  tiles.forEach(tile => {{ tile.hidden = filter !== 'all' && tile.dataset.result !== filter && tile.dataset.label !== filter; }});
}}));
</script>
</body></html>"""


def main():
    args = parse_args()
    checkpoint, model_name, dropout = load_run_config(args)
    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(device_name)
    _, transform = get_transforms(model_name, train=False)
    model = load_model(checkpoint, model_name, dropout, device)
    image_root = Path(args.image)
    images = collect_images(image_root, args.recursive)

    rows = []
    for image_path in images:
        prediction, confidence, probabilities = predict(model, transform, image_path, device)
        rows.append({
            "name": str(image_path.relative_to(image_root)) if image_root.is_dir() else image_path.name,
            "image_uri": image_data_uri(image_path),
            "actual": ground_truth(image_path),
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": probabilities,
        })

    output_path = Path(args.output)
    output_path.write_text(
        build_report(rows, checkpoint, model_name, device, image_root), encoding="utf-8"
    )
    print(f"Reviewed {len(rows)} image(s)")
    print(f"Saved visual report: {output_path}")


if __name__ == "__main__":
    main()
