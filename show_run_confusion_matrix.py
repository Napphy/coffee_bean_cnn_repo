#!/usr/bin/env python3
"""Display confusion matrices saved in a training run's results.json."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


CLASSES = ["black", "broken", "moldy"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Show the saved confusion matrix for a training run"
    )
    parser.add_argument(
        "run_dir",
        help="Run directory containing results.json, or the results.json file itself",
    )
    parser.add_argument(
        "--tested-variety",
        choices=["arabica", "robusta"],
        help="Show only evaluations for this tested variety (default: show all)",
    )
    parser.add_argument(
        "--output",
        default="confusion_matrices.png",
        help="PNG figure output path (default: confusion_matrices.png)",
    )
    return parser.parse_args()


def load_evaluations(run_path: Path, tested_variety: str | None):
    results_path = run_path if run_path.is_file() else run_path / "results.json"
    if not results_path.is_file():
        raise FileNotFoundError(f"Run results not found: {results_path}")

    with results_path.open(encoding="utf-8") as file:
        metadata = json.load(file)

    evaluations = metadata.get("results", [])
    if tested_variety:
        evaluations = [
            evaluation for evaluation in evaluations
            if f"tested:{tested_variety}" in evaluation.get("label", "")
        ]
    if not evaluations:
        suffix = f" for tested variety '{tested_variety}'" if tested_variety else ""
        raise ValueError(f"No saved evaluations{suffix} in {results_path}")

    for evaluation in evaluations:
        matrix = evaluation.get("confusion_matrix")
        if not isinstance(matrix, list) or len(matrix) != len(CLASSES) or any(
            not isinstance(row, list) or len(row) != len(CLASSES) for row in matrix
        ):
            raise ValueError(
                f"Invalid or missing 3-class confusion matrix in: {evaluation.get('label', 'evaluation')}"
            )
    return metadata, results_path, evaluations


def load_font(size, bold=False):
    font_name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(font_name, size)
    except OSError:
        return ImageFont.load_default()


def draw_report(metadata, evaluations):
    width = 1180
    header_height = 145
    section_height = 565
    image = Image.new("RGB", (width, header_height + section_height * len(evaluations)), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(38, bold=True)
    subtitle_font = load_font(19)
    section_font = load_font(25, bold=True)
    label_font = load_font(18, bold=True)
    count_font = load_font(29, bold=True)
    detail_font = load_font(18)

    run_name = metadata.get("run_name", "Saved run")
    run_details = " | ".join(
        str(value) for value in (
            metadata.get("variety"), metadata.get("model"), metadata.get("optimizer")
        ) if value
    )
    draw.text((64, 42), "Confusion Matrix", fill="#172321", font=title_font)
    draw.text((66, 98), str(run_name), fill="#384942", font=subtitle_font)
    draw.text((66, 130), run_details, fill="#64736e", font=detail_font)

    cell_size = 132
    grid_x = 470
    grid_offset_y = 160
    for eval_index, evaluation in enumerate(evaluations):
        section_y = header_height + eval_index * section_height
        if eval_index:
            draw.line((64, section_y, width - 64, section_y), fill="#d9ded5", width=2)
        label = evaluation.get("label", "Evaluation")
        draw.text((66, section_y + 28), label, fill="#172321", font=section_font)
        draw.text((grid_x + cell_size * 1.5, section_y + 100), "Predicted class", fill="#384942", font=label_font, anchor="mm")
        actual_label = Image.new("RGBA", (110, 30), (255, 255, 255, 0))
        ImageDraw.Draw(actual_label).text((55, 15), "Actual class", fill="#384942", font=label_font, anchor="mm")
        actual_label = actual_label.rotate(90, expand=True)
        image.paste(
            actual_label,
            (grid_x - 132, section_y + grid_offset_y + (cell_size * 3 - actual_label.height) // 2),
            actual_label,
        )

        matrix = evaluation["confusion_matrix"]
        maximum = max((max(row) for row in matrix), default=0)
        grid_y = section_y + grid_offset_y
        for column, class_name in enumerate(CLASSES):
            cell_left = grid_x + column * cell_size
            draw.text((cell_left + cell_size / 2, grid_y - 12), class_name.title(), fill="#172321", font=label_font, anchor="ms")
        for row, class_name in enumerate(CLASSES):
            cell_top = grid_y + row * cell_size
            draw.text((grid_x - 22, cell_top + cell_size / 2), class_name.title(), fill="#172321", font=label_font, anchor="rm")
            for column, value in enumerate(matrix[row]):
                ratio = value / maximum if maximum else 0
                green = int(236 - 115 * ratio)
                red_blue = int(229 - 145 * ratio)
                fill = (red_blue, green, red_blue)
                cell_left = grid_x + column * cell_size
                draw.rectangle(
                    (cell_left, cell_top, cell_left + cell_size, cell_top + cell_size),
                    fill=fill,
                    outline="white",
                    width=3,
                )
                text_color = "white" if ratio > 0.62 else "#172321"
                draw.text(
                    (cell_left + cell_size / 2, cell_top + cell_size / 2),
                    str(value), fill=text_color, font=count_font, anchor="mm",
                )

    return image


def main():
    args = parse_args()
    metadata, results_path, evaluations = load_evaluations(
        Path(args.run_dir), args.tested_variety
    )
    for evaluation in evaluations:
        matrix = evaluation["confusion_matrix"]
        print(f"\n{evaluation.get('label', 'Evaluation')}")
        print("Rows = actual; columns = predicted; class order: " + " / ".join(CLASSES))
        print("                  " + "  ".join(f"{name:>7}" for name in CLASSES))
        for label, row in zip(CLASSES, matrix):
            print(f"{label:>12} actual  " + "  ".join(f"{value:7}" for value in row))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    draw_report(metadata, evaluations).save(output_path, format="PNG", dpi=(300, 300), optimize=True)
    print(f"\nSaved PNG confusion matrix: {output_path}")


if __name__ == "__main__":
    main()