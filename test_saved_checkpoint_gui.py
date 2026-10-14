#!/usr/bin/env python3
"""Simple desktop GUI for testing saved coffee-defect checkpoints."""

import csv
import base64
import html
import json
import mimetypes
import re
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from types import SimpleNamespace

import torch
from PIL import Image, ImageTk

from test_saved_checkpoint import (
    CLASSES,
    collect_images,
    get_transforms,
    load_model,
    load_run_config,
    predict,
)


def actual_class(image_path: Path):
    for parent in image_path.parents:
        if parent.name.lower() in CLASSES:
            return parent.name.lower()
    return None


def infer_tested_variety(image_path: Path):
    path_parts = [part.lower() for part in image_path.parts]
    for variety in ("arabica", "robusta"):
        if variety in path_parts:
            return variety
    for split in ("test", "testing", "val", "validation", "train"):
        if split in path_parts:
            index = path_parts.index(split)
            if index > 0:
                return image_path.parts[index - 1]
    return "Unknown"


def image_data_uri(image_path: Path):
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_visual_report(rows, report_title="Coffee checkpoint validation"):
    grouped_rows = {}
    image_uris = {}
    for row in rows:
        group = (
            row["checkpoint"], row["checkpoint_name"], row["tested_variety"]
        )
        grouped_rows.setdefault(group, []).append(row)
        image_path = Path(row["image"])
        if image_path not in image_uris:
            image_uris[image_path] = image_data_uri(image_path)

    sections = []
    for (checkpoint, checkpoint_name, tested_variety), checkpoint_rows in grouped_rows.items():
        cards = []
        for row in checkpoint_rows:
            actual = row["actual"] or "ground truth unavailable"
            result_class = (
                "unknown" if row["actual"] is None
                else "correct" if row["actual"] == row["prediction"]
                else "incorrect"
            )
            probabilities = "".join(
                f'<div class="prob"><span>{html.escape(label)}</span>'
                f'<div class="bar"><i style="width:{value * 100:.1f}%"></i></div>'
                f'<b>{value:.1%}</b></div>'
                for label, value in zip(CLASSES, row["probabilities"])
            )
            cards.append(
                f'<article class="tile {result_class}">'
                f'<img src="{image_uris[Path(row["image"])]}" loading="lazy" '
                f'alt="{html.escape(row["name"])}">'
                f'<div class="tile-body"><div class="file">{html.escape(row["name"])}</div>'
                f'<div class="labels"><span class="prediction">Predicted: '
                f'<strong>{html.escape(row["prediction"])}</strong></span>'
                f'<span class="actual">Actual: <strong>{html.escape(actual)}</strong></span></div>'
                f'<div class="confidence">Confidence {row["confidence"]:.1%}</div>'
                f'{probabilities}</div></article>'
            )
        sections.append(
            f'<section class="checkpoint"><div class="checkpoint-head">'
            f'<div><h2>{html.escape(checkpoint)}</h2>'
            f'<div class="folder">Checkpoint folder: {html.escape(checkpoint_name)}</div></div>'
            f'<div class="variety">Tested variety: <strong>{html.escape(tested_variety)}</strong>'
            f'<br>{len(checkpoint_rows)} image(s)</div></div>'
            f'<div class="report-grid">{"".join(cards)}</div></section>'
        )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(report_title)}</title>
<style>
:root {{ --ink:#172321; --muted:#64736e; --paper:#f4f1e8; --panel:#fffdf8; --line:#d9ded5; --green:#216b52; --red:#b44d3c; --gold:#c58a26; }}
* {{ box-sizing:border-box; }} body {{ margin:0; color:var(--ink); background:var(--paper); font:15px/1.4 Georgia,serif; }}
header {{ padding:34px 5vw 26px; background:linear-gradient(135deg,#dfe9db,#f4f1e8 62%,#eadcc4); border-bottom:1px solid var(--line); }}
h1 {{ margin:0 0 7px; font:700 clamp(27px,4vw,42px)/1.08 Georgia,serif; }} .subtitle {{ color:var(--muted); }}
main {{ width:min(1500px,92vw); margin:26px auto 60px; }} .checkpoint {{ padding:20px 0 30px; border-bottom:1px solid var(--line); }}
.checkpoint-head {{ display:flex; flex-wrap:wrap; align-items:flex-start; justify-content:space-between; gap:12px; margin-bottom:16px; }}
h2 {{ margin:0 0 4px; font-size:23px; overflow-wrap:anywhere; }} .folder {{ color:var(--muted); font-size:13px; overflow-wrap:anywhere; }} .variety {{ color:var(--muted); text-align:right; }}
.variety strong {{ color:var(--ink); }} .report-grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(255px,1fr)); gap:15px; }}
.tile {{ overflow:hidden; background:var(--panel); border:2px solid var(--line); border-radius:6px; }} .tile.correct {{ border-color:#86b89a; }} .tile.incorrect {{ border-color:#d48676; }} .tile.unknown {{ border-color:#c9c5b8; }}
.tile img {{ display:block; width:100%; aspect-ratio:4/3; object-fit:cover; background:#e8e6df; }} .tile-body {{ padding:12px; }}
.file {{ white-space:nowrap; overflow:hidden; text-overflow:ellipsis; color:var(--muted); font-size:13px; margin-bottom:9px; }} .labels {{ display:grid; gap:3px; }}
.prediction strong {{ color:var(--green); }} .incorrect .prediction strong {{ color:var(--red); }} .actual {{ color:var(--muted); }} .confidence {{ margin:10px 0 7px; font-weight:bold; }}
.prob {{ display:grid; grid-template-columns:48px 1fr 43px; gap:6px; align-items:center; color:var(--muted); font-size:12px; }} .prob b {{ text-align:right; font-weight:normal; }}
.bar {{ height:7px; background:#e5e9e2; border-radius:3px; overflow:hidden; }} .bar i {{ display:block; height:100%; background:var(--gold); }}
@media(max-width:650px) {{ main {{ width:94vw; }} .report-grid {{ grid-template-columns:1fr; }} .variety {{ text-align:left; }} }}
</style></head><body><header><h1>Visual checkpoint validation</h1>
<div class="subtitle">{len(rows)} prediction(s) across {len(grouped_rows)} checkpoint evaluation(s)</div></header>
<main>{"".join(sections)}</main></body></html>"""


class CheckpointTesterGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Coffee Bean Checkpoint Tester")
        self.root.geometry("1240x860")
        self.root.minsize(980, 700)

        self.image_path = tk.StringVar()
        self.device = tk.StringVar(value="cuda" if torch.cuda.is_available() else "cpu")
        self.recursive = tk.BooleanVar(value=True)
        self.single_actual = tk.StringVar(value="Unknown")
        self.actual_result = tk.StringVar(value="Unknown")
        self.status = tk.StringVar(value="Choose checkpoint filters, then select an image or folder.")
        self.predict_button = None
        self.report_button = None
        self.output_rows = []
        self.checkpoint_runs = []
        self.visible_run_indices = []
        self.filter_options = {}
        self.filter_buttons = {}
        self.preview_photo = None
        self.selected_result_index = None

        self._build_layout()
        self.refresh_checkpoints()

    def _build_layout(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        controls = ttk.LabelFrame(self.root, text="Test setup")
        controls.grid(row=0, column=0, padx=12, pady=12, sticky="ew")
        controls.columnconfigure(0, weight=1)
        ttk.Label(controls, text="Trained variety").grid(row=0, column=0, padx=8, pady=(8, 2), sticky="w")
        ttk.Label(controls, text="Model").grid(row=0, column=1, padx=8, pady=(8, 2), sticky="w")
        ttk.Label(controls, text="Optimizer").grid(row=0, column=2, padx=8, pady=(8, 2), sticky="w")
        for column, key in enumerate(("variety", "model", "optimizer")):
            button = ttk.Menubutton(controls, text=f"{key.title()}: All")
            button.grid(row=1, column=column, padx=8, pady=(0, 8), sticky="ew")
            button.menu = tk.Menu(button, tearoff=False)
            button["menu"] = button.menu
            self.filter_buttons[key] = button

        run_tools = ttk.Frame(controls)
        run_tools.grid(row=0, column=3, rowspan=2, padx=8, pady=8, sticky="ne")
        ttk.Button(run_tools, text="Refresh", command=self.refresh_checkpoints).pack(side="left", padx=3)
        ttk.Button(run_tools, text="Add run folder...", command=self.add_run).pack(side="left", padx=3)

        ttk.Label(controls, text="Matching checkpoints (Ctrl/Shift-click to select multiple)").grid(
            row=2, column=0, columnspan=4, padx=8, pady=(0, 3), sticky="w"
        )
        run_list_frame = ttk.Frame(controls)
        run_list_frame.grid(row=3, column=0, columnspan=4, padx=8, sticky="ew")
        run_list_frame.columnconfigure(0, weight=1)
        self.run_list = tk.Listbox(run_list_frame, selectmode=tk.EXTENDED, height=4, exportselection=False)
        self.run_list.grid(row=0, column=0, sticky="ew")
        run_scrollbar = ttk.Scrollbar(run_list_frame, orient="vertical", command=self.run_list.yview)
        run_scrollbar.grid(row=0, column=1, sticky="ns")
        self.run_list.configure(yscrollcommand=run_scrollbar.set)
        run_buttons = ttk.Frame(run_list_frame)
        run_buttons.grid(row=0, column=2, padx=(8, 0), sticky="ns")
        ttk.Button(run_buttons, text="Select visible", command=self.select_all_runs).pack(fill="x")
        ttk.Button(run_buttons, text="Clear selection", command=self.clear_run_selection).pack(fill="x", pady=(5, 0))

        ttk.Label(controls, text="Image or folder").grid(row=4, column=0, padx=8, pady=7, sticky="w")
        ttk.Entry(controls, textvariable=self.image_path).grid(row=4, column=1, padx=8, pady=7, sticky="ew")
        ttk.Button(controls, text="Choose image...", command=self.choose_image).grid(row=4, column=2, padx=4, pady=7)
        ttk.Button(controls, text="Choose folder...", command=self.choose_folder).grid(row=4, column=3, padx=4, pady=7)

        options = ttk.Frame(controls)
        options.grid(row=5, column=0, columnspan=4, padx=8, pady=(0, 8), sticky="ew")
        ttk.Checkbutton(options, text="Search recursively", variable=self.recursive).pack(side="left", padx=(0, 18))
        ttk.Label(options, text="Actual label for a single image:").pack(side="left")
        ttk.Combobox(options, textvariable=self.single_actual, values=["Unknown", *CLASSES], state="readonly", width=12).pack(side="left", padx=6)
        ttk.Label(options, text="Device:").pack(side="left", padx=(12, 0))
        ttk.Combobox(options, textvariable=self.device, values=["cpu", "cuda"], state="readonly", width=8).pack(side="left", padx=6)
        self.predict_button = ttk.Button(options, text="Test selected checkpoints", command=self.start_prediction)
        self.predict_button.pack(side="right", padx=4)
        ttk.Button(options, text="Clear results", command=self.clear_results).pack(side="right", padx=4)

        results = ttk.LabelFrame(self.root, text="Bean predictions")
        results.grid(row=1, column=0, padx=12, pady=(0, 12), sticky="nsew")
        results.columnconfigure(0, weight=0, minsize=280)
        results.columnconfigure(1, weight=1)
        results.rowconfigure(0, weight=1)
        list_panel = ttk.Frame(results)
        list_panel.grid(row=0, column=0, padx=8, pady=8, sticky="nsew")
        list_panel.rowconfigure(1, weight=1)
        ttk.Label(list_panel, text="Prediction runs").grid(row=0, column=0, sticky="w", pady=(0, 5))
        self.result_list = tk.Listbox(list_panel, width=38, exportselection=False)
        self.result_list.grid(row=1, column=0, sticky="nsew")
        result_scrollbar = ttk.Scrollbar(list_panel, orient="vertical", command=self.result_list.yview)
        result_scrollbar.grid(row=1, column=1, sticky="ns")
        self.result_list.configure(yscrollcommand=result_scrollbar.set)
        self.result_list.bind("<<ListboxSelect>>", self.show_selected_result)

        detail = ttk.Frame(results)
        detail.grid(row=0, column=1, padx=(4, 10), pady=8, sticky="nsew")
        detail.columnconfigure(0, weight=1)
        detail.columnconfigure(1, weight=1)
        detail.rowconfigure(0, weight=1)
        image_frame = ttk.LabelFrame(detail, text="Bean image")
        image_frame.grid(row=0, column=0, padx=(0, 6), sticky="nsew")
        image_frame.rowconfigure(0, weight=1)
        image_frame.columnconfigure(0, weight=1)
        self.preview = ttk.Label(image_frame, text="Choose an image to preview", anchor="center")
        self.preview.grid(row=0, column=0, padx=8, pady=8, sticky="nsew")

        detail_frame = ttk.LabelFrame(detail, text="Prediction and confidence")
        detail_frame.grid(row=0, column=1, padx=(6, 0), sticky="nsew")
        detail_frame.columnconfigure(0, weight=1)
        self.prediction_title = ttk.Label(detail_frame, text="No prediction yet", font=("TkDefaultFont", 18, "bold"))
        self.prediction_title.grid(row=0, column=0, padx=12, pady=(12, 4), sticky="w")
        self.confidence_title = ttk.Label(detail_frame, text="Confidence: --", font=("TkDefaultFont", 12, "bold"))
        self.confidence_title.grid(row=1, column=0, padx=12, pady=4, sticky="w")
        ttk.Label(detail_frame, text="Actual label").grid(row=2, column=0, padx=12, pady=(12, 2), sticky="w")
        self.actual_combo = ttk.Combobox(detail_frame, textvariable=self.actual_result, values=["Unknown", *CLASSES], state="readonly")
        self.actual_combo.grid(row=3, column=0, padx=12, sticky="ew")
        self.actual_combo.bind("<<ComboboxSelected>>", self.update_actual_label)
        self.checkpoint_title = ttk.Label(detail_frame, text="", wraplength=350, justify="left")
        self.checkpoint_title.grid(row=4, column=0, padx=12, pady=(8, 2), sticky="w")
        self.chart = tk.Canvas(detail_frame, height=210, highlightthickness=0, background="#fbfaf6")
        self.chart.grid(row=5, column=0, padx=8, pady=(4, 10), sticky="ew")
        self.chart.bind("<Configure>", self._redraw_selected_chart)

        footer = ttk.Frame(self.root)
        footer.grid(row=2, column=0, padx=12, pady=(0, 12), sticky="ew")
        footer.columnconfigure(0, weight=1)
        ttk.Label(footer, textvariable=self.status).grid(row=0, column=0, sticky="w")
        self.report_button = ttk.Button(footer, text="Save Visual HTML...", command=self.save_visual_report)
        self.report_button.grid(row=0, column=1, padx=5)
        ttk.Button(footer, text="Save CSV...", command=self.save_csv).grid(row=0, column=2, padx=5)

    @staticmethod
    def short_run_label(run_dir: Path):
        metadata_path = run_dir / "results.json"
        if metadata_path.is_file():
            try:
                with metadata_path.open(encoding="utf-8") as file:
                    metadata = json.load(file)
                variety = metadata.get("variety")
                model = metadata.get("model")
                optimizer = metadata.get("optimizer")
                if variety and model and optimizer:
                    return f"{variety}_{model}_{optimizer}"
            except (OSError, json.JSONDecodeError):
                pass
        return re.sub(r"^fixed_(NORLR_|ft_)?", "", run_dir.name)

    def refresh_checkpoints(self):
        checkpoints_root = Path(__file__).resolve().parent / "linuxcheckpoints"
        discovered = sorted(
            run_dir for run_dir in checkpoints_root.iterdir()
            if run_dir.is_dir()
            and (run_dir / "best_model.pth").is_file()
            and (run_dir / "results.json").is_file()
        ) if checkpoints_root.is_dir() else []
        self._merge_runs(discovered)
        self.status.set(f"Loaded {len(self.checkpoint_runs)} checkpoint run(s) from linuxcheckpoints.")

    def _merge_runs(self, run_dirs):
        known_paths = {run_dir.resolve() for _, run_dir in self.checkpoint_runs}
        for run_dir in run_dirs:
            run_dir = Path(run_dir)
            resolved = run_dir.resolve()
            if resolved in known_paths:
                continue
            if not (run_dir / "best_model.pth").is_file() or not (run_dir / "results.json").is_file():
                continue
            label = self.short_run_label(run_dir)
            existing_labels = {existing_label for existing_label, _ in self.checkpoint_runs}
            if label in existing_labels:
                label = f"{label} ({run_dir.parent.name})"
            self.checkpoint_runs.append((label, run_dir))
            known_paths.add(resolved)
        self._refresh_filter_menus()

    @staticmethod
    def _run_metadata(run_dir):
        try:
            with (run_dir / "results.json").open(encoding="utf-8") as file:
                return json.load(file)
        except (OSError, json.JSONDecodeError):
            return {}

    def _refresh_filter_menus(self):
        for key in ("variety", "model", "optimizer"):
            values = sorted({
                str(self._run_metadata(run_dir).get(key, "")).strip().lower()
                for _, run_dir in self.checkpoint_runs
                if self._run_metadata(run_dir).get(key)
            })
            menu = self.filter_buttons[key].menu
            menu.delete(0, tk.END)
            self.filter_options[key] = {}
            for value in values:
                variable = tk.BooleanVar(value=False)
                self.filter_options[key][value] = variable
                menu.add_checkbutton(
                    label=value,
                    variable=variable,
                    command=self._refresh_run_view,
                )
        self._refresh_run_view()

    def _refresh_run_view(self):
        selected_filters = {
            key: {value for value, variable in options.items() if variable.get()}
            for key, options in self.filter_options.items()
        }
        for key, values in selected_filters.items():
            title = key.title()
            suffix = f" ({len(values)})" if values else " (All)"
            self.filter_buttons[key].configure(text=f"{title}{suffix}")

        self.visible_run_indices = []
        self.run_list.delete(0, tk.END)
        for index, (label, run_dir) in enumerate(self.checkpoint_runs):
            metadata = self._run_metadata(run_dir)
            if any(
                selected and str(metadata.get(key, "")).lower() not in selected
                for key, selected in selected_filters.items()
            ):
                continue
            self.visible_run_indices.append(index)
            self.run_list.insert(tk.END, label)

    def add_run(self):
        selected = filedialog.askdirectory(title="Select a checkpoint run folder")
        if selected:
            run_dir = Path(selected)
            if not (run_dir / "best_model.pth").is_file() or not (run_dir / "results.json").is_file():
                messagebox.showerror(
                    "Invalid checkpoint",
                    "The selected folder must contain best_model.pth and results.json.",
                )
                return
            self._merge_runs([run_dir])
            self.status.set(f"Added checkpoint: {self.short_run_label(run_dir)}")

    def select_all_runs(self):
        self.run_list.selection_set(0, tk.END)

    def clear_run_selection(self):
        self.run_list.selection_clear(0, tk.END)

    def choose_image(self):
        selected = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff"), ("All files", "*")],
        )
        if selected:
            self.image_path.set(selected)
            self.single_actual.set("Unknown")
            self._set_preview_image(Path(selected))

    def choose_folder(self):
        selected = filedialog.askdirectory(title="Select image folder")
        if selected:
            self.image_path.set(selected)
            try:
                images = collect_images(Path(selected), self.recursive.get())
                self._set_preview_image(images[0])
            except (OSError, ValueError) as error:
                self.preview.configure(image="", text=str(error))
                self.preview_photo = None

    def clear_results(self):
        self.result_list.delete(0, tk.END)
        self.output_rows = []
        self.selected_result_index = None
        self.prediction_title.configure(text="No prediction yet")
        self.confidence_title.configure(text="Confidence: --")
        self.checkpoint_title.configure(text="")
        self.actual_result.set("Unknown")
        self.chart.delete("all")
        self.preview.configure(image="", text="Choose an image to preview")
        self.preview_photo = None
        self.status.set("Results cleared.")

    def _set_preview_image(self, image_path):
        try:
            with Image.open(image_path) as image:
                preview = image.convert("RGB")
            preview.thumbnail((620, 460), Image.Resampling.LANCZOS)
            self.preview_photo = ImageTk.PhotoImage(preview)
            self.preview.configure(image=self.preview_photo, text="")
        except (OSError, tk.TclError) as error:
            self.preview_photo = None
            self.preview.configure(image="", text=f"Could not preview image: {error}")

    def show_selected_result(self, _event=None):
        selection = self.result_list.curselection()
        if not selection:
            return
        index = selection[0]
        if index >= len(self.output_rows):
            return
        row = self.output_rows[index]
        self.selected_result_index = index
        self._set_preview_image(Path(row["image"]))
        self.prediction_title.configure(text=f"Predicted: {row['prediction'].title()}")
        self.confidence_title.configure(text=f"Confidence: {row['confidence']:.1%}")
        actual = row.get("actual") or "Unknown"
        self.actual_result.set(actual)
        self.checkpoint_title.configure(
            text=f"{row['checkpoint']}\nTested variety: {row['tested_variety']}"
        )
        self._draw_confidence_chart(row["probabilities"])

    def _draw_confidence_chart(self, probabilities):
        self.chart.delete("all")
        width = max(self.chart.winfo_width(), 360)
        left, right = 78, width - 58
        bar_width = max(right - left, 120)
        colors = {"black": "#256b50", "broken": "#c58a26", "moldy": "#b44d3c"}
        for index, (label, value) in enumerate(zip(CLASSES, probabilities)):
            y = 34 + index * 57
            self.chart.create_text(8, y + 9, text=label.title(), anchor="w", fill="#35413c")
            self.chart.create_rectangle(left, y, right, y + 19, fill="#e5e9e2", outline="")
            self.chart.create_rectangle(
                left, y, left + bar_width * max(0, min(value, 1)), y + 19,
                fill=colors.get(label, "#256b50"), outline="",
            )
            self.chart.create_text(
                width - 8, y + 9, text=f"{value:.1%}", anchor="e", fill="#202b28",
            )

    def _redraw_selected_chart(self, _event=None):
        if self.selected_result_index is not None and self.selected_result_index < len(self.output_rows):
            self._draw_confidence_chart(self.output_rows[self.selected_result_index]["probabilities"])

    def update_actual_label(self, _event=None):
        if self.selected_result_index is None:
            return
        actual = self.actual_result.get()
        self.output_rows[self.selected_result_index]["actual"] = (
            actual.lower() if actual != "Unknown" else None
        )

    def start_prediction(self):
        selected_indices = self.run_list.curselection()
        if not selected_indices or not self.image_path.get():
            messagebox.showerror("Missing input", "Select one or more checkpoint runs and an image or folder first.")
            return
        if self.device.get() == "cuda" and not torch.cuda.is_available():
            messagebox.showerror("CUDA unavailable", "CUDA is not available in this Python environment. Choose CPU.")
            return

        self.predict_button.configure(state="disabled")
        run_selection = [self.checkpoint_runs[self.visible_run_indices[index]] for index in selected_indices]
        image_path = Path(self.image_path.get())
        recursive = self.recursive.get()
        device_name = self.device.get()
        tested_variety = infer_tested_variety(image_path)
        single_actual = self.single_actual.get()
        self.status.set(f"Testing {len(run_selection)} checkpoint(s)...")
        threading.Thread(
            target=self._predict_worker,
            args=(run_selection, image_path, recursive, device_name, tested_variety, single_actual),
            daemon=True,
        ).start()

    def _predict_worker(self, run_selection, image_path, recursive, device_name, tested_variety, single_actual):
        try:
            images = collect_images(image_path, recursive)
            device = torch.device(device_name)
            rows = []
            failures = []
            for run_number, (run_label, run_dir) in enumerate(run_selection, start=1):
                self.root.after(
                    0,
                    self.status.set,
                    f"Checkpoint {run_number}/{len(run_selection)}: {run_label}",
                )
                config = SimpleNamespace(
                    run_dir=str(run_dir), checkpoint=None, model=None, dropout=None
                )
                try:
                    checkpoint, model_name, dropout = load_run_config(config)
                    _, transform = get_transforms(model_name, train=False)
                    model = load_model(checkpoint, model_name, dropout, device)
                    for image_file in images:
                        predicted, confidence, probabilities = predict(
                            model, transform, image_file, device
                        )
                        rows.append({
                            "checkpoint": run_label,
                            "checkpoint_filename_label": self.short_run_label(run_dir),
                            "checkpoint_name": run_dir.name,
                            "checkpoint_path": str(checkpoint),
                            "tested_variety": tested_variety,
                            "image": str(image_file),
                            "name": str(image_file.relative_to(image_path))
                            if image_path.is_dir() else image_file.name,
                            "actual": actual_class(image_file) or (
                                single_actual.lower() if image_path.is_file() and single_actual != "Unknown" else None
                            ),
                            "prediction": predicted,
                            "confidence": confidence,
                            "black_probability": probabilities[0],
                            "broken_probability": probabilities[1],
                            "moldy_probability": probabilities[2],
                            "probabilities": probabilities,
                        })
                    del model
                except Exception as error:
                    failures.append(f"{run_label}: {error}")
            self.root.after(0, self._show_results, rows, len(run_selection), len(images), failures)
        except Exception as error:
            self.root.after(0, self._show_error, str(error))

    def _show_results(self, rows, checkpoint_count, image_count, failures):
        self.output_rows = rows
        self.result_list.delete(0, tk.END)
        for row in rows:
            image_name = Path(row["image"]).name
            self.result_list.insert(
                tk.END,
                f"{image_name}  |  {row['prediction'].title()} {row['confidence']:.1%}\n    {row['checkpoint']}",
            )
        self.predict_button.configure(state="normal")
        if rows:
            self.result_list.selection_set(0)
            self.show_selected_result()
        self.status.set(
            f"Completed {len(rows)} prediction(s): {image_count} image(s), "
            f"{checkpoint_count - len(failures)}/{checkpoint_count} checkpoint(s) succeeded."
        )
        if failures:
            messagebox.showwarning("Some checkpoints failed", "\n".join(failures))

    def _show_error(self, message):
        self.predict_button.configure(state="normal")
        self.status.set("Classification failed.")
        messagebox.showerror("Classification error", message)

    def save_visual_report(self):
        if not self.output_rows:
            messagebox.showinfo("No results", "Run a checkpoint classification first.")
            return
        output_dir = filedialog.askdirectory(title="Choose a folder for per-checkpoint HTML reports")
        if not output_dir:
            return
        report_groups = {}
        for row in self.output_rows:
            key = (row["checkpoint"], row["checkpoint_name"], row["tested_variety"])
            report_groups.setdefault(key, []).append(row)
        self.report_button.configure(state="disabled")
        self.status.set(f"Building {len(report_groups)} checkpoint HTML report(s)...")
        threading.Thread(
            target=self._write_visual_reports,
            args=(Path(output_dir), report_groups),
            daemon=True,
        ).start()

    def _write_visual_reports(self, output_dir, report_groups):
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            report_paths = []
            for (checkpoint, checkpoint_name, tested_variety), rows in report_groups.items():
                checkpoint_filename_label = rows[0]["checkpoint_filename_label"]
                safe_checkpoint = re.sub(
                    r"[^A-Za-z0-9._-]+", "_", checkpoint_filename_label
                ).strip("._")
                safe_variety = re.sub(r"[^A-Za-z0-9._-]+", "_", tested_variety).strip("._")
                filename = f"{safe_checkpoint}_{safe_variety}.html"
                destination = output_dir / filename
                title = f"Visual validation | {checkpoint} | tested {tested_variety}"
                destination.write_text(
                    build_visual_report(rows, report_title=title), encoding="utf-8"
                )
                report_paths.append(destination)
            self.root.after(0, self._visual_reports_saved, output_dir, report_paths)
        except Exception as error:
            self.root.after(0, self._visual_reports_failed, str(error))

    def _visual_reports_saved(self, output_dir, report_paths):
        self.report_button.configure(state="normal")
        self.status.set(f"Saved {len(report_paths)} per-checkpoint HTML report(s) to {output_dir}")
        messagebox.showinfo(
            "Reports saved",
            f"Saved {len(report_paths)} HTML report(s) to:\n{output_dir}",
        )

    def _visual_reports_failed(self, message):
        self.report_button.configure(state="normal")
        self.status.set("Could not create visual report.")
        messagebox.showerror("Report error", message)

    def save_csv(self):
        if not self.output_rows:
            messagebox.showinfo("No results", "Run a classification first.")
            return
        destination = filedialog.asksaveasfilename(
            title="Save predictions",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
        )
        if not destination:
            return
        with open(destination, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(self.output_rows[0]))
            writer.writeheader()
            writer.writerows(self.output_rows)
        self.status.set(f"Saved predictions to {destination}")


def main():
    root = tk.Tk()
    CheckpointTesterGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
