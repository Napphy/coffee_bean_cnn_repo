#!/usr/bin/env python3
"""Native PySide6 visual tester for saved coffee-defect checkpoints."""

import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import torch
from PySide6.QtCore import QEvent, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QFont, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from test_saved_checkpoint import (
    CLASSES,
    collect_images,
    get_transforms,
    load_model,
    load_run_config,
    predict,
)


ROOT = Path(__file__).resolve().parent
CHECKPOINT_ROOT = ROOT / "linuxcheckpoints"
FILTERS = ("variety", "model", "optimizer")
IMAGE_FILTER = "Images (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff)"
BAR_COLORS = {
    "black": QColor("#267452"),
    "broken": QColor("#d29a2e"),
    "moldy": QColor("#b44d3c"),
}


class PersistentCheckMenu(QMenu):
    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        QApplication.instance().installEventFilter(self)

    def eventFilter(self, watched, event):
        if self.isVisible() and event.type() == QEvent.Type.MouseButtonPress:
            global_position = event.globalPosition().toPoint()
            if not self.geometry().contains(global_position):
                self.hide()
        return super().eventFilter(watched, event)

    def mouseReleaseEvent(self, event):
        action = self.actionAt(event.position().toPoint())
        should_stay_open = action is not None and action.isCheckable()
        super().mouseReleaseEvent(event)
        if should_stay_open:
            QTimer.singleShot(0, self._restore_popup)

    def _restore_popup(self):
        if self.owner.isVisible():
            self.popup(self.owner.mapToGlobal(self.owner.rect().bottomLeft()))


def read_metadata(run_dir):
    try:
        return json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def discover_runs():
    runs = []
    if not CHECKPOINT_ROOT.is_dir():
        return runs
    for run_dir in sorted(CHECKPOINT_ROOT.iterdir()):
        if not run_dir.is_dir() or not (run_dir / "best_model.pth").is_file():
            continue
        metadata = read_metadata(run_dir)
        if all(metadata.get(key) for key in FILTERS):
            runs.append({
                "path": run_dir,
                "metadata": metadata,
                **{key: str(metadata[key]).strip().lower() for key in FILTERS},
            })
    return runs


def actual_class(image_path):
    for parent in image_path.parents:
        if parent.name.lower() in CLASSES:
            return parent.name.lower()
    return None


def tested_variety(image_path):
    parts = [part.lower() for part in image_path.parts]
    for variety in ("arabica", "robusta"):
        if variety in parts:
            return variety
    return "Unknown"


class CheckMenuButton(QToolButton):
    changed = Signal()

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.title = title
        self.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.setMinimumHeight(38)
        self.menu = self.add_menu()
        self.actions_by_value = {}
        self.update_title()

    def add_menu(self):
        menu = PersistentCheckMenu(self)
        self.setMenu(menu)
        return menu

    def set_values(self, values):
        self.menu.clear()
        self.actions_by_value.clear()
        for value in values:
            action = QAction(value, self.menu)
            action.setCheckable(True)
            action.toggled.connect(self._selection_changed)
            self.menu.addAction(action)
            self.actions_by_value[value] = action
        self.update_title()

    def _selection_changed(self, _checked):
        self.update_title()
        self.changed.emit()

    def selected_values(self):
        return {value for value, action in self.actions_by_value.items() if action.isChecked()}

    def update_title(self):
        selected = sorted(self.selected_values()) if hasattr(self, "actions_by_value") else []
        value = ", ".join(selected) if selected else "All"
        self.setText(f"{self.title}: {value}  ▾")
        self.setToolTip(f"{self.title}: {value}")


class ConfidenceChart(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.values = [0.0] * len(CLASSES)
        self.setMinimumHeight(124)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_values(self, values):
        self.values = list(values)
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        left = 74
        right = self.width() - 58
        bar_width = max(80, right - left)
        painter.setFont(QFont("Sans Serif", 9))
        for index, (label, value) in enumerate(zip(CLASSES, self.values)):
            y = 18 + index * 36
            painter.setPen(QColor("#35413c"))
            painter.drawText(4, y + 12, label.title())
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#e5e9e2"))
            painter.drawRoundedRect(left, y, bar_width, 15, 3, 3)
            painter.setBrush(BAR_COLORS.get(label, QColor("#267452")))
            painter.drawRoundedRect(left, y, int(bar_width * max(0.0, min(value, 1.0))), 15, 3, 3)
            painter.setPen(QColor("#202b28"))
            painter.drawText(self.width() - 52, y + 12, f"{value:.1%}")
        painter.end()


class PredictionWorker(QThread):
    completed = Signal(object, int, int, object)

    def __init__(self, selected_runs, image_root, recursive, device_name, parent=None):
        super().__init__(parent)
        self.selected_runs = selected_runs
        self.image_root = image_root
        self.recursive = recursive
        self.device_name = device_name

    def run(self):
        rows = []
        failures = []
        try:
            images = collect_images(self.image_root, self.recursive)
            device = torch.device(self.device_name)
            for run in self.selected_runs:
                metadata = run["metadata"]
                try:
                    config = SimpleNamespace(
                        run_dir=str(run["path"]), checkpoint=None, model=None, dropout=None
                    )
                    checkpoint, model_name, dropout = load_run_config(config)
                    _, transform = get_transforms(model_name, train=False)
                    model = load_model(checkpoint, model_name, dropout, device)
                    for image_path in images:
                        predicted, confidence, probabilities = predict(
                            model, transform, image_path, device
                        )
                        rows.append({
                            "image": str(image_path),
                            "name": str(image_path.relative_to(self.image_root))
                            if self.image_root.is_dir() else image_path.name,
                            "actual": actual_class(image_path),
                            "tested_variety": tested_variety(image_path),
                            "checkpoint": f"{run['variety']} / {model_name} / {run['optimizer']}",
                            "trained_variety": run["variety"],
                            "model": model_name,
                            "optimizer": run["optimizer"],
                            "checkpoint_path": str(checkpoint),
                            "checkpoint_name": run["path"].name,
                            "prediction": predicted,
                            "confidence": confidence,
                            "probabilities": probabilities,
                            "black_probability": probabilities[0],
                            "broken_probability": probabilities[1],
                            "moldy_probability": probabilities[2],
                            "checkpoint_filename_label": self._short_label(run),
                        })
                    del model
                except Exception as error:
                    failures.append(f"{run['path'].name}: {error}")
            self.completed.emit(rows, len(self.selected_runs), len(images), failures)
        except Exception as error:
            self.completed.emit([], len(self.selected_runs), 0, [str(error)])

    @staticmethod
    def _short_label(run):
        return f"{run['variety']}_{run['model']}_{run['optimizer']}"


class CheckpointTesterWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Coffee Bean Checkpoint POC")
        self.resize(1320, 900)
        self.setMinimumSize(1040, 720)
        self.runs = discover_runs()
        self.rows = []
        self.preview_source = QPixmap()
        self.worker = None
        self._build_ui()
        self._populate_filters()
        self._refresh_matches()

    def _build_ui(self):
        self.setStyleSheet("""
            QWidget { color: #202b28; background: #f4f3ed; font: 10pt 'Sans Serif'; }
            QGroupBox { border: 1px solid #d9ddd5; border-radius: 5px; margin-top: 10px; padding: 10px; background: #fffefa; font-weight: 600; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QLineEdit, QComboBox, QToolButton { border: 1px solid #d9ddd5; border-radius: 4px; padding: 7px; background: #fffefa; }
            QPushButton { border: 1px solid #17664f; border-radius: 4px; padding: 8px 14px; color: white; background: #17664f; font-weight: 600; }
            QPushButton:hover { background: #10563f; }
            QPushButton:disabled { background: #8aa59a; }
            QListWidget { border: 1px solid #d9ddd5; border-radius: 4px; background: #fffefa; padding: 4px; }
            QListWidget::item { padding: 8px; border-bottom: 1px solid #eceee9; }
            QListWidget::item:selected { color: #fff; background: #17664f; }
        """)
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(18, 16, 18, 14)
        layout.setSpacing(12)

        title = QLabel("Coffee bean checkpoint tester")
        title.setStyleSheet("font: 700 23pt 'Serif'; color: #174e3e;")
        layout.addWidget(title)
        subtitle = QLabel("Test bean images, or leave the input empty to inspect saved model results.")
        subtitle.setStyleSheet("color: #66736e;")
        layout.addWidget(subtitle)

        setup = QGroupBox("Test setup")
        setup_layout = QVBoxLayout(setup)
        filter_row = QHBoxLayout()
        self.filters = {}
        for key, title_text in (("variety", "Trained variety"), ("model", "Model"), ("optimizer", "Optimizer")):
            selector = CheckMenuButton(title_text)
            selector.changed.connect(self._refresh_matches)
            self.filters[key] = selector
            filter_row.addWidget(selector, 1)
        refresh_button = QPushButton("Refresh checkpoints")
        refresh_button.clicked.connect(self._refresh_runs)
        filter_row.addWidget(refresh_button)
        setup_layout.addLayout(filter_row)

        self.match_count = QLabel()
        self.match_count.setStyleSheet("color: #66736e;")
        setup_layout.addWidget(self.match_count)

        input_row = QHBoxLayout()
        self.path_label = QLabel("Choose an image or folder…")
        self.path_label.setMinimumWidth(280)
        self.path_label.setStyleSheet("padding: 8px; border: 1px solid #d9ddd5; background: #fffefa;")
        input_row.addWidget(self.path_label, 1)
        choose_image_button = QPushButton("Choose image…")
        choose_image_button.clicked.connect(self._choose_image)
        input_row.addWidget(choose_image_button)
        choose_folder_button = QPushButton("Choose folder…")
        choose_folder_button.clicked.connect(self._choose_folder)
        input_row.addWidget(choose_folder_button)
        self.clear_input_button = QPushButton("Clear input")
        self.clear_input_button.clicked.connect(self._clear_input)
        self.clear_input_button.setEnabled(False)
        input_row.addWidget(self.clear_input_button)
        setup_layout.addLayout(input_row)

        action_row = QHBoxLayout()
        self.recursive = QCheckBox("Search folders recursively")
        self.recursive.setChecked(True)
        action_row.addWidget(self.recursive)
        action_row.addSpacing(12)
        action_row.addWidget(QLabel("Device"))
        self.device = QComboBox()
        self.device.addItems(["cpu", "cuda"])
        self.device.setCurrentText("cuda" if torch.cuda.is_available() else "cpu")
        action_row.addWidget(self.device)
        action_row.addStretch(1)
        self.test_button = QPushButton("Test / view matching checkpoints")
        self.test_button.clicked.connect(self._start_prediction)
        action_row.addWidget(self.test_button)
        clear_button = QPushButton("Clear results")
        clear_button.clicked.connect(self._clear_results)
        action_row.addWidget(clear_button)
        setup_layout.addLayout(action_row)
        layout.addWidget(setup)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        results_panel = QWidget()
        results_layout = QVBoxLayout(results_panel)
        results_layout.setContentsMargins(0, 0, 0, 0)
        results_filter_row = QHBoxLayout()
        results_filter_row.addWidget(QLabel("Results model"))
        self.results_model = QComboBox()
        self.results_model.addItem("All models")
        self.results_model.currentTextChanged.connect(self._render_result_list)
        results_filter_row.addWidget(self.results_model, 1)
        results_layout.addLayout(results_filter_row)
        self.result_list = QListWidget()
        self.result_list.setMinimumWidth(310)
        self.result_list.currentRowChanged.connect(self._show_result)
        results_layout.addWidget(self.result_list, 1)
        splitter.addWidget(results_panel)

        detail = QWidget()
        detail_layout = QVBoxLayout(detail)
        detail_layout.setContentsMargins(12, 4, 4, 4)
        preview_card = QFrame()
        preview_card.setStyleSheet("QFrame { background: #ebece5; border: 1px solid #d9ddd5; border-radius: 4px; }")
        preview_layout = QVBoxLayout(preview_card)
        self.preview = QLabel("Bean image preview")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(460, 360)
        self.preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        preview_layout.addWidget(self.preview)
        detail_layout.addWidget(preview_card, 5)

        self.prediction = QLabel("Choose an image and test matching checkpoints")
        self.prediction.setStyleSheet("font: 700 15pt 'Serif'; color: #17664f;")
        detail_layout.addWidget(self.prediction)
        self.actual = QLabel("Actual label: unavailable")
        self.actual.setStyleSheet("color: #35413c; font-size: 11pt; font-weight: 700;")
        self.actual.setWordWrap(True)
        detail_layout.addWidget(self.actual)
        self.checkpoint = QLabel("")
        self.checkpoint.setStyleSheet("color: #202b28; font-size: 11pt; font-weight: 700;")
        self.checkpoint.setWordWrap(True)
        detail_layout.addWidget(self.checkpoint)
        self.chart_box = QGroupBox("Class confidence")
        chart_layout = QVBoxLayout(self.chart_box)
        self.chart = ConfidenceChart()
        chart_layout.addWidget(self.chart)
        detail_layout.addWidget(self.chart_box, 1)
        splitter.addWidget(detail)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, 1)

        self.status = QLabel(f"{len(self.runs)} saved checkpoints discovered.")
        self.status.setStyleSheet("color: #66736e; padding: 3px;")
        layout.addWidget(self.status)

    def _populate_filters(self):
        for key, selector in self.filters.items():
            values = sorted({run[key] for run in self.runs})
            selector.set_values(values)

    def _matching_runs(self):
        selected = {key: selector.selected_values() for key, selector in self.filters.items()}
        return [
            run for run in self.runs
            if all(not selected[key] or run[key] in selected[key] for key in FILTERS)
        ]

    def _refresh_matches(self):
        if not hasattr(self, "match_count"):
            return
        count = len(self._matching_runs())
        self.match_count.setText(
            f"{count} checkpoint(s) will run. A filter set to All includes every value."
        )

    def _refresh_runs(self):
        self.runs = discover_runs()
        self._populate_filters()
        self._refresh_matches()
        self.status.setText(f"Refreshed: {len(self.runs)} checkpoint(s) discovered.")

    def _choose_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose bean image", str(ROOT), IMAGE_FILTER)
        if path:
            self._set_input_path(Path(path))

    def _choose_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Choose image folder", str(ROOT))
        if path:
            self._set_input_path(Path(path))

    def _set_input_path(self, path):
        self.input_path = path
        self.path_label.setText(str(path))
        self.path_label.setToolTip(str(path))
        self.clear_input_button.setEnabled(True)
        preview_path = path
        if path.is_dir():
            try:
                preview_path = collect_images(path, self.recursive.isChecked())[0]
            except (OSError, ValueError, IndexError):
                self.preview.setText("No supported images found in this folder")
                self.preview_source = QPixmap()
                return
        self.preview_source = QPixmap(str(preview_path))
        self._update_preview()

    def _clear_input(self):
        self.input_path = None
        self.path_label.setText("Choose an image or folder…")
        self.path_label.setToolTip("")
        self.clear_input_button.setEnabled(False)
        self.preview_source = QPixmap()
        self.preview.setPixmap(QPixmap())
        self.preview.setText("Bean image preview")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_preview()

    def _update_preview(self):
        if not self.preview_source.isNull():
            self.preview.setPixmap(self.preview_source.scaled(
                self.preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            ))

    def _start_prediction(self):
        matching = self._matching_runs()
        image_root = getattr(self, "input_path", None)
        if not matching:
            QMessageBox.warning(self, "No matching checkpoints", "Adjust the filters to select at least one checkpoint.")
            return
        if image_root is None:
            self._show_saved_results(matching)
            return
        if self.device.currentText() == "cuda" and not torch.cuda.is_available():
            QMessageBox.warning(self, "CUDA unavailable", "CUDA is not available. Choose CPU.")
            return
        if len(matching) > 12:
            answer = QMessageBox.question(
                self,
                "Run many checkpoints?",
                f"The selected filters match {len(matching)} checkpoints. This may take a while. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.test_button.setEnabled(False)
        self.status.setText(f"Testing {len(matching)} matching checkpoint(s)…")
        self.worker = PredictionWorker(
            matching,
            image_root,
            self.recursive.isChecked(),
            self.device.currentText(),
            self,
        )
        self.worker.completed.connect(self._show_results)
        self.worker.start()

    def _show_saved_results(self, selected_runs):
        rows = []
        for run in selected_runs:
            metadata = run["metadata"]
            for evaluation in metadata.get("results", []):
                rows.append({
                    "kind": "metrics",
                    "model": run["model"],
                    "trained_variety": run["variety"],
                    "optimizer": run["optimizer"],
                    "checkpoint": run["path"].name,
                    "evaluation_label": evaluation.get("label", "Saved evaluation"),
                    "report": evaluation.get("report", {}),
                    "best_epoch": metadata.get("best_epoch"),
                    "best_val_loss": metadata.get("best_val_loss"),
                })
        self._show_results(rows, len(selected_runs), 0, [], saved_results=True)

    def _show_results(self, rows, checkpoint_count, image_count, failures, saved_results=False):
        self.rows = rows
        previous_model = self.results_model.currentText()
        self.results_model.blockSignals(True)
        self.results_model.clear()
        self.results_model.addItem("All models")
        self.results_model.addItems(sorted({row["model"] for row in rows}))
        if self.results_model.findText(previous_model) >= 0:
            self.results_model.setCurrentText(previous_model)
        self.results_model.blockSignals(False)
        self._render_result_list()
        succeeded = checkpoint_count - len(failures)
        if saved_results:
            self.status.setText(
                f"Showing {len(rows)} saved evaluation(s) from {succeeded}/{checkpoint_count} checkpoint(s)."
            )
        else:
            self.status.setText(
                f"Completed {len(rows)} prediction(s) from {succeeded}/{checkpoint_count} checkpoint(s); "
                f"{image_count} image(s)."
            )
        self.test_button.setEnabled(True)
        if failures:
            QMessageBox.warning(self, "Checkpoint errors", "\n".join(failures))

    def _render_result_list(self, _selected_model=None):
        selected_model = self.results_model.currentText()
        self.result_list.clear()
        for row_index, row in enumerate(self.rows):
            if selected_model != "All models" and row["model"] != selected_model:
                continue
            if row.get("kind") == "metrics":
                report = row["report"]
                accuracy = report.get("accuracy")
                macro_f1 = report.get("macro avg", {}).get("f1-score")
                accuracy_text = f"{accuracy:.1%}" if isinstance(accuracy, (int, float)) else "n/a"
                f1_text = f"{macro_f1:.1%}" if isinstance(macro_f1, (int, float)) else "n/a"
                text = f"{row['evaluation_label']}\nAccuracy {accuracy_text} · Macro F1 {f1_text}\n{row['checkpoint']}"
            else:
                actual = row["actual"] or "actual unknown"
                text = (
                    f"{Path(row['image']).name}\n"
                    f"{row['prediction'].title()} · {row['confidence']:.1%} · {actual}\n"
                    f"{row['checkpoint']}"
                )
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, row_index)
            self.result_list.addItem(item)
        if self.result_list.count():
            self.result_list.setCurrentRow(0)

    def _show_result(self, index):
        item = self.result_list.item(index) if index >= 0 else None
        if item is None:
            return
        row = self.rows[item.data(Qt.ItemDataRole.UserRole)]
        if row.get("kind") == "metrics":
            report = row["report"]
            macro = report.get("macro avg", {})
            accuracy = report.get("accuracy")
            accuracy_text = f"{accuracy:.1%}" if isinstance(accuracy, (int, float)) else "n/a"
            macro_f1 = macro.get("f1-score")
            macro_f1_text = f"{macro_f1:.1%}" if isinstance(macro_f1, (int, float)) else "n/a"
            self.preview_source = QPixmap()
            self.preview.setPixmap(QPixmap())
            self.preview.setText("Saved evaluation results")
            self.prediction.setText(f"Accuracy: {accuracy_text} · Macro F1: {macro_f1_text}")
            self.actual.setText(
                f"{row['evaluation_label']} · Macro precision: "
                f"{macro.get('precision', 0):.1%} · Macro recall: {macro.get('recall', 0):.1%}"
            )
            best_epoch = row["best_epoch"] if row["best_epoch"] is not None else "n/a"
            best_val_loss = row["best_val_loss"]
            loss_text = f"{best_val_loss:.4f}" if isinstance(best_val_loss, (int, float)) else "n/a"
            self.checkpoint.setText(
                f"Trained variety: {row['trained_variety']}    Model: {row['model']}    "
                f"Optimizer: {row['optimizer']}\nBest epoch: {best_epoch}    Best validation loss: {loss_text}"
            )
            self.chart_box.setTitle("Class F1 score")
            self.chart.set_values([
                report.get(class_name, {}).get("f1-score", 0.0) for class_name in CLASSES
            ])
            return
        self.preview_source = QPixmap(row["image"])
        self._update_preview()
        self.prediction.setText(f"Predicted: {row['prediction'].title()} · {row['confidence']:.1%}")
        self.actual.setText(f"Actual label: {row['actual'] or 'unavailable'} · Tested variety: {row['tested_variety']}")
        self.checkpoint.setText(
            f"Trained variety: {row['trained_variety']}    Model: {row['model']}    Optimizer: {row['optimizer']}"
        )
        self.chart_box.setTitle("Class confidence")
        self.chart.set_values(row["probabilities"])

    def _clear_results(self):
        self.rows.clear()
        self.result_list.clear()
        self.prediction.setText("Choose an image and test matching checkpoints")
        self.actual.setText("Actual label: unavailable")
        self.checkpoint.clear()
        self.chart_box.setTitle("Class confidence")
        self.chart.set_values([0.0] * len(CLASSES))
        if getattr(self, "input_path", None) is not None:
            self._set_input_path(self.input_path)
        else:
            self._clear_input()


def main():
    app = QApplication(sys.argv)
    window = CheckpointTesterWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()