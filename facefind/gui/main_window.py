"""Main application window for the FaceFind GUI."""

# ruff: noqa: E701, E702
import sys
from pathlib import Path

try:  # pragma: no cover - optional dependency
    from PyQt6.QtGui import QAction, QTextCursor
    from PyQt6.QtWidgets import (
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSpinBox,
        QSplitter,
        QTabWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ModuleNotFoundError as e:  # pragma: no cover
    raise ImportError(
        "PyQt6 is required for facefind.gui.main_window."
        " Install the 'PyQt6' package to use this module."
    ) from e

from facefind.utils import ensure_dir

from .image_grid import ImageGrid
from .process_worker import ProcessWorker
from .utils import link_or_copy


# -------------------------
# Base class used by tabs
# -------------------------
class BaseRunnerTab(QWidget):
    """Runs a script via QProcess and streams logs into a QTextEdit."""

    def __init__(self, repo_root: Path):
        super().__init__()
        self.repo_root = Path(repo_root)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.run_btn = QPushButton("Run")
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)

        self.worker: ProcessWorker | None = None

    def _start(self, argv, mps_fallback: bool = False):
        if self.worker:
            QMessageBox.warning(self, "Busy", "Another task is already running.")
            return
        env = {}
        if mps_fallback:
            # Allow CPU fallback for missing MPS ops when training/predicting
            env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

        self.worker = ProcessWorker(self.repo_root, env=env)
        self.worker.started.connect(lambda args: self._on_started(args))
        self.worker.output.connect(self._on_output)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(lambda e: self._append(f"[ERROR] {e}\n"))
        self.worker.run(argv)

    def _on_started(self, args):
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.log.clear()
        self._append("$ " + " ".join(args) + "\n")

    def _on_output(self, text: str):
        self._append(text)

    def _append(self, text: str):
        """PyQt6-safe append that won’t crash the app on errors."""
        try:
            cursor = self.log.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self.log.setTextCursor(cursor)
            self.log.insertPlainText(text)
            self.log.ensureCursorVisible()
        except Exception as e:
            # Don’t let UI logging exceptions abort the process.
            self.log.append(f"\n[UI-LOG ERROR] {e}\n{text}")

    def _on_finished(self, code: int):
        self._append(f"\n[EXIT] {code}\n")
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.worker = None


# -------------------------
# Detect tab
# -------------------------
class DetectTab(BaseRunnerTab):
    def __init__(self, repo_root: Path):
        super().__init__(repo_root)
        self.input_edit = QLineEdit(str(Path.home() / "Documents/July22/Temp"))
        self.output_edit = QLineEdit("outputs")
        self.device_combo = QComboBox()
        self.device_combo.addItems(["mps", "cuda", "cpu"])
        self.strict_combo = QComboBox()
        self.strict_combo.addItems(["strict", "normal", "loose"])
        self.video_step = QSpinBox()
        self.video_step.setRange(1, 60)
        self.video_step.setValue(5)
        self.max_per = QSpinBox()
        self.max_per.setRange(1, 1000)
        self.max_per.setValue(50)
        self.progress_every = QSpinBox()
        self.progress_every.setRange(1, 10000)
        self.progress_every.setValue(100)
        self.log_no_face = QCheckBox("Log no-face files")

        browse_in = QPushButton("Browse…")
        browse_in.clicked.connect(self._pick_input)
        browse_out = QPushButton("Browse…")
        browse_out.clicked.connect(self._pick_output)

        form = QFormLayout()
        form.addRow("Input directory", self._h(self.input_edit, browse_in))
        form.addRow("Output directory", self._h(self.output_edit, browse_out))
        form.addRow("Device", self.device_combo)
        form.addRow("Strictness", self.strict_combo)
        form.addRow("Video step", self.video_step)
        form.addRow("Max faces per media", self.max_per)
        form.addRow("Progress every", self.progress_every)
        form.addRow("", self.log_no_face)

        btns = self._h(self.run_btn, self.cancel_btn)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.log)
        layout.addLayout(btns)

        self.run_btn.clicked.connect(self._run)
        self.cancel_btn.clicked.connect(self._cancel)

    def _h(self, *w):
        box = QHBoxLayout()
        for x in w:
            box.addWidget(x)
        box.addStretch(1)
        return box

    def _pick_input(self):
        d = QFileDialog.getExistingDirectory(self, "Select input directory", self.input_edit.text())
        if d:
            self.input_edit.setText(d)

    def _pick_output(self):
        d = QFileDialog.getExistingDirectory(
            self, "Select output directory", self.output_edit.text()
        )
        if d:
            self.output_edit.setText(d)

    def _run(self):
        py = sys.executable
        args = [
            py,
            "main.py",
            "--input",
            self.input_edit.text(),
            "--output",
            self.output_edit.text(),
            "--video-step",
            str(self.video_step.value()),
            "--strictness",
            self.strict_combo.currentText(),
            "--device",
            self.device_combo.currentText(),
            "--max-per-media",
            str(self.max_per.value()),
            "--progress-every",
            str(self.progress_every.value()),
        ]
        if self.log_no_face.isChecked():
            args.append("--log-no-face")
        # mtcnn device fallback is handled inside your main.py
        self._start(args)

    def _cancel(self):
        if self.worker:
            self.worker.kill()


# -------------------------
# Verify tab
# -------------------------
class VerifyTab(BaseRunnerTab):
    def __init__(self, repo_root: Path):
        super().__init__(repo_root)
        self.crops_edit = QLineEdit("outputs/crops/pending")
        self.reject_edit = QLineEdit("outputs/crops/rejects")
        self.device_combo = QComboBox()
        self.device_combo.addItems(["cpu", "mps", "cuda"])  # CPU default
        self.strict_combo = QComboBox()
        self.strict_combo.addItems(["strict", "normal", "loose"])
        browse_crops = QPushButton("Browse…")
        browse_crops.clicked.connect(self._pick_crops)
        browse_rej = QPushButton("Browse…")
        browse_rej.clicked.connect(self._pick_rejects)

        form = QFormLayout()
        form.addRow("Crops dir", self._h(self.crops_edit, browse_crops))
        form.addRow("Reject dir", self._h(self.reject_edit, browse_rej))
        form.addRow("Device", self.device_combo)
        form.addRow("Strictness", self.strict_combo)

        btns = self._h(self.run_btn, self.cancel_btn)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.log)
        layout.addLayout(btns)

        self.run_btn.clicked.connect(self._run)
        self.cancel_btn.clicked.connect(self._cancel)

    def _h(self, *w):
        box = QHBoxLayout()
        for x in w:
            box.addWidget(x)
        box.addStretch(1)
        return box

    def _pick_crops(self):
        d = QFileDialog.getExistingDirectory(self, "Select crops directory", self.crops_edit.text())
        if d:
            self.crops_edit.setText(d)

    def _pick_rejects(self):
        d = QFileDialog.getExistingDirectory(
            self, "Select rejects directory", self.reject_edit.text()
        )
        if d:
            self.reject_edit.setText(d)

    def _run(self):
        py = sys.executable
        args = [
            py,
            "verify_crops.py",
            self.crops_edit.text(),
            "--reject-dir",
            self.reject_edit.text(),
            "--strictness",
            self.strict_combo.currentText(),
            "--device",
            self.device_combo.currentText(),
        ]
        self._start(args)

    def _cancel(self):
        if self.worker:
            self.worker.kill()


# -------------------------
# Label tab
# -------------------------
class LabelTab(QWidget):
    def __init__(self, repo_root: Path):
        super().__init__()
        self.repo_root = Path(repo_root)
        self.verified_dir_edit = QLineEdit("outputs/crops/verified")
        self.people_dir_edit = QLineEdit("outputs/people_by_cluster")
        self.copy_check = QCheckBox("Copy instead of hard-link")
        self.refresh_btn = QPushButton("Refresh")
        self.add_person_btn = QPushButton("Add Person…")
        self.rename_person_btn = QPushButton("Rename…")
        self.delete_person_btn = QPushButton("Delete…")
        self.move_selected_btn = QPushButton("Move selected →")

        self.grid = ImageGrid()
        self.people_list = QListWidget()
        self.people_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)

        top = QFormLayout()
        b_ver = QPushButton("Browse…")
        b_ver.clicked.connect(self._pick_verified)
        b_people = QPushButton("Browse…")
        b_people.clicked.connect(self._pick_people)
        top.addRow("Verified crops", self._h(self.verified_dir_edit, b_ver, self.refresh_btn))
        top.addRow("People dir", self._h(self.people_dir_edit, b_people, self.copy_check))

        left = QVBoxLayout()
        left.addWidget(QLabel("People"))
        left.addWidget(self.people_list)
        left_btns = QHBoxLayout()
        left_btns.addWidget(self.add_person_btn)
        left_btns.addWidget(self.rename_person_btn)
        left_btns.addWidget(self.delete_person_btn)
        left.addLayout(left_btns)

        right = QVBoxLayout()
        right.addWidget(QLabel("Verified crops"))
        right.addWidget(self.grid)
        right.addWidget(self.move_selected_btn)

        split = QSplitter()
        lw = QWidget()
        lw.setLayout(left)
        rw = QWidget()
        rw.setLayout(right)
        split.addWidget(lw)
        split.addWidget(rw)
        split.setStretchFactor(1, 1)

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(split)

        self.refresh_btn.clicked.connect(self.refresh)
        self.add_person_btn.clicked.connect(self.add_person)
        self.rename_person_btn.clicked.connect(self.rename_person)
        self.delete_person_btn.clicked.connect(self.delete_person)
        self.move_selected_btn.clicked.connect(self.move_selected)

        self.refresh()

    def _h(self, *w):
        box = QHBoxLayout()
        for x in w:
            box.addWidget(x)
        box.addStretch(1)
        return box

    def _pick_verified(self):
        d = QFileDialog.getExistingDirectory(
            self, "Select verified crops directory", self.verified_dir_edit.text()
        )
        if d:
            self.verified_dir_edit.setText(d)
            self.refresh()

    def _pick_people(self):
        d = QFileDialog.getExistingDirectory(
            self, "Select people directory", self.people_dir_edit.text()
        )
        if d:
            self.people_dir_edit.setText(d)
            self.refresh_people()

    def refresh(self):
        self.grid.load_dir(Path(self.verified_dir_edit.text()))
        self.refresh_people()

    def refresh_people(self):
        self.people_list.clear()
        pdir = Path(self.people_dir_edit.text())
        pdir.mkdir(parents=True, exist_ok=True)
        for d in sorted(pdir.glob("*")):
            if d.is_dir():
                self.people_list.addItem(QListWidgetItem(d.name))

    def _current_person(self) -> str | None:
        it = self.people_list.currentItem()
        return it.text() if it else None

    def add_person(self):
        from PyQt6.QtWidgets import QInputDialog

        name, ok = QInputDialog.getText(self, "New person", "Name:")
        if ok and name.strip():
            (Path(self.people_dir_edit.text()) / name.strip()).mkdir(parents=True, exist_ok=True)
            self.refresh_people()

    def rename_person(self):
        from PyQt6.QtWidgets import QInputDialog

        curr = self._current_person()
        if not curr:
            QMessageBox.information(self, "Rename", "Select a person first.")
            return
        name, ok = QInputDialog.getText(self, "Rename person", "New name:", text=curr)
        if ok and name.strip() and name != curr:
            src = Path(self.people_dir_edit.text()) / curr
            dst = Path(self.people_dir_edit.text()) / name.strip()
            try:
                src.rename(dst)
                self.refresh_people()
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def delete_person(self):
        curr = self._current_person()
        if not curr:
            QMessageBox.information(self, "Delete", "Select a person first.")
            return
        reply = QMessageBox.question(
            self, "Delete", f"Delete folder '{curr}'? This will remove files."
        )
        if reply == QMessageBox.StandardButton.Yes:
            import shutil

            try:
                shutil.rmtree(Path(self.people_dir_edit.text()) / curr)
                self.refresh_people()
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def move_selected(self):
        curr = self._current_person()
        if not curr:
            QMessageBox.information(self, "Move", "Select a person on the left.")
            return
        target_dir = Path(self.people_dir_edit.text()) / curr
        ensure_dir(target_dir)
        copy = self.copy_check.isChecked()
        count = 0
        for p in self.grid.selected_paths():
            try:
                link_or_copy(p, target_dir / p.name, copy=copy)
                count += 1
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed: {p}\n{e}")
        QMessageBox.information(self, "Moved", f"Placed {count} file(s) into '{curr}'.")


# -------------------------
# Train / Predict tab
# -------------------------
class TrainPredictTab(BaseRunnerTab):
    def __init__(self, repo_root: Path):
        super().__init__(repo_root)
        # Train
        self.people_dir_edit = QLineEdit("outputs/people_by_cluster")
        self.model_dir_edit = QLineEdit("models")
        self.device_combo = QComboBox()
        self.device_combo.addItems(["mps", "cuda", "cpu"])  # MPS + fallback
        self.train_btn = QPushButton("Train")

        # Predict
        self.images_dir_edit = QLineEdit("outputs/crops/pending")
        self.pred_csv_edit = QLineEdit("outputs/predictions.csv")
        self.predict_btn = QPushButton("Predict")

        # Autosort
        self.autosort_out_edit = QLineEdit("outputs/autosort")
        self.accept_spin = QDoubleSpinBox()
        self.accept_spin.setRange(0.0, 1.0)
        self.accept_spin.setSingleStep(0.01)
        self.accept_spin.setValue(0.80)
        self.review_spin = QDoubleSpinBox()
        self.review_spin.setRange(0.0, 1.0)
        self.review_spin.setSingleStep(0.01)
        self.review_spin.setValue(0.50)
        self.copy_check = QCheckBox("Copy instead of hard-link")
        self.autosort_btn = QPushButton("Auto-sort")

        form = QFormLayout()
        form.addRow(QLabel("▲ Training"))
        b_people = QPushButton("Browse…")
        b_people.clicked.connect(lambda: self._pick_dir(self.people_dir_edit))
        b_model = QPushButton("Browse…")
        b_model.clicked.connect(lambda: self._pick_dir(self.model_dir_edit))
        form.addRow("People dir", self._h(self.people_dir_edit, b_people))
        form.addRow("Model dir", self._h(self.model_dir_edit, b_model))
        form.addRow("Device", self.device_combo)
        form.addRow("", self.train_btn)

        form.addRow(QLabel("\n▲ Prediction"))
        b_images = QPushButton("Browse…")
        b_images.clicked.connect(lambda: self._pick_dir(self.images_dir_edit))
        b_pred = QPushButton("Browse…")
        b_pred.clicked.connect(lambda: self._pick_file(self.pred_csv_edit))
        form.addRow("Images dir", self._h(self.images_dir_edit, b_images))
        form.addRow("Predictions CSV", self._h(self.pred_csv_edit, b_pred))
        form.addRow("", self.predict_btn)

        form.addRow(QLabel("\n▲ Autosort"))
        b_auto = QPushButton("Browse…")
        b_auto.clicked.connect(lambda: self._pick_dir(self.autosort_out_edit))
        form.addRow("Out dir", self._h(self.autosort_out_edit, b_auto))
        form.addRow("Accept threshold", self.accept_spin)
        form.addRow("Review threshold", self.review_spin)
        form.addRow("", self.copy_check)
        form.addRow("", self.autosort_btn)

        btns = self._h(self.run_btn, self.cancel_btn)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.log)
        layout.addLayout(btns)

        self.train_btn.clicked.connect(self._train)
        self.predict_btn.clicked.connect(self._predict)
        self.autosort_btn.clicked.connect(self._autosort)
        self.cancel_btn.clicked.connect(self._cancel)

    def _h(self, *w):
        box = QHBoxLayout()
        for x in w:
            box.addWidget(x)
        box.addStretch(1)
        return box

    def _pick_dir(self, line: QLineEdit):
        d = QFileDialog.getExistingDirectory(self, "Select directory", line.text())
        if d:
            line.setText(d)

    def _pick_file(self, line: QLineEdit):
        fn, _ = QFileDialog.getSaveFileName(self, "Select file", line.text(), "CSV Files (*.csv)")
        if fn:
            line.setText(fn)

    def _train(self):
        py = sys.executable
        args = [
            py,
            "train_face_classifier.py",
            "--data",
            self.people_dir_edit.text(),
            "--out",
            self.model_dir_edit.text(),
            "--device",
            self.device_combo.currentText(),
        ]
        # Allow CPU fallback for rare missing ops when device=mps
        self._start(args, mps_fallback=self.device_combo.currentText() == "mps")

    def _predict(self):
        py = sys.executable
        args = [
            py,
            "predict_face.py",
            self.images_dir_edit.text(),
            "--model-dir",
            self.model_dir_edit.text(),
            "--out",
            self.pred_csv_edit.text(),
            "--device",
            self.device_combo.currentText(),
        ]
        self._start(args, mps_fallback=self.device_combo.currentText() == "mps")

    def _autosort(self):
        py = sys.executable
        args = [
            py,
            "apply_predictions.py",
            self.pred_csv_edit.text(),
            "--people-dir",
            self.people_dir_edit.text(),
            "--out-dir",
            self.autosort_out_edit.text(),
            "--accept-threshold",
            f"{self.accept_spin.value():.2f}",
            "--review-threshold",
            f"{self.review_spin.value():.2f}",
        ]
        if self.copy_check.isChecked():
            args.append("--copy")
        self._start(args)

    def _cancel(self):
        if self.worker:
            self.worker.kill()


# -------------------------
# Main window
# -------------------------
class MainWindow(QMainWindow):
    def __init__(self, repo_root: Path):
        super().__init__()
        self.repo_root = Path(repo_root)
        self.setWindowTitle("FaceFind GUI")
        self.resize(1200, 800)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.detect_tab = DetectTab(self.repo_root)
        self.verify_tab = VerifyTab(self.repo_root)
        self.label_tab = LabelTab(self.repo_root)
        self.train_tab = TrainPredictTab(self.repo_root)

        self.tabs.addTab(self.detect_tab, "Detect")
        self.tabs.addTab(self.verify_tab, "Verify")
        self.tabs.addTab(self.label_tab, "Label")
        self.tabs.addTab(self.train_tab, "Train / Predict")

        self._build_menu()

    def _build_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        act_quit = QAction("Quit", self)
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)
