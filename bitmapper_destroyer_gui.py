#!/usr/bin/env python3
"""
BitMapper Destroyer GUI (PySide6)
Convert RGB images/GIFs/videos to corrupted black & white versions.
"""

import glob
import os
import random
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np

from PySide6.QtCore import Qt, QThread, Signal, QUrl
from PySide6.QtGui import QPixmap, QMovie, QDragEnterEvent, QDragMoveEvent, QDropEvent, QResizeEvent
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QComboBox, QPushButton,
    QCheckBox, QProgressBar, QFileDialog, QStackedWidget,
    QFrame, QMessageBox, QScrollArea,
)
from PIL import Image
import cv2

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

SUPPORTED_IMAGE_FORMATS = {".jpg", ".jpeg", ".png"}
SUPPORTED_VIDEO_FORMATS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".3gp", ".3g2"}
GIF_EXTENSION = ".gif"
ALL_SUPPORTED = SUPPORTED_IMAGE_FORMATS | SUPPORTED_VIDEO_FORMATS | {GIF_EXTENSION}

RESAMPLING_FILTERS = {
    "LANCZOS": Image.Resampling.LANCZOS,
    "BICUBIC": Image.Resampling.BICUBIC,
    "BILINEAR": Image.Resampling.BILINEAR,
    "NEAREST": Image.Resampling.NEAREST,
    "BOX": Image.Resampling.BOX,
    "HAMMING": Image.Resampling.HAMMING,
}

DEFAULTS = {
    "threshold": 128,
    "downscale": 1,
    "resampling_filter": "NEAREST",
    "frame_prefix_image": "image_frame",
    "frame_prefix_video": "video_frame",
}

GREY_LEVELS_ARRAY = [0 if random.randint(0, 255) < i else 255 for i in range(256)]

# ============================================================================
# CORE IMAGE PROCESSING (unchanged from original)
# ============================================================================


def apply_threshold_or_dither(image, threshold, dither, randomize):
    if dither is not None:
        return image.convert("1", dither=dither)

    def threshold_fn(pixel):
        if randomize:
            return 0 if pixel < threshold or GREY_LEVELS_ARRAY[pixel] > pixel else 255
        return 0 if pixel < threshold else 255

    return image.point(threshold_fn, "1")


def apply_scaling(image, downscale, resampling_filter, resize_after=True):
    if downscale == DEFAULTS["downscale"]:
        return image

    original_size = image.size
    scale_factor = 1 / abs(downscale) if downscale < 0 else downscale
    new_width = round(original_size[0] * scale_factor)
    new_height = round(original_size[1] * scale_factor)

    if downscale < 0:
        image = image.resize((new_width, new_height), resampling_filter)
        if resize_after:
            return image.resize(original_size, resampling_filter)
        return image
    return image.resize((new_width, new_height), resampling_filter)


def apply_glitch_shift(image, probability, amplitude):
    if probability <= 0 or amplitude <= 0:
        return image
    arr = np.array(image)
    h, w = arr.shape
    rows = np.where(np.random.random(h) < (probability / 100))[0]
    if len(rows) == 0:
        return image
    shifts = np.random.randint(-amplitude, amplitude + 1, len(rows))
    for y, s in zip(rows, shifts):
        if s != 0:
            arr[y] = np.roll(arr[y], s)
    return Image.fromarray(arr)


def apply_pixel_scatter(image, probability, radius):
    if probability <= 0 or radius <= 0:
        return image
    arr = np.array(image)
    h, w = arr.shape
    mask = np.random.random((h, w)) < (probability / 100)
    count = mask.sum()
    if count == 0:
        return image
    dy = np.random.randint(-radius, radius + 1, count)
    dx = np.random.randint(-radius, radius + 1, count)
    y_idx, x_idx = np.where(mask)
    ny = (y_idx + dy).clip(0, h - 1)
    nx = (x_idx + dx).clip(0, w - 1)
    arr[y_idx, x_idx] = arr[ny, nx]
    return Image.fromarray(arr)


def apply_bloom(image, radius):
    if radius <= 0:
        return image
    arr = np.array(image)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    arr = cv2.dilate(arr, kernel)
    arr = cv2.erode(arr, kernel)
    return Image.fromarray(arr)


def apply_block_shift(image, probability, block_size, amplitude):
    if probability <= 0 or amplitude <= 0 or block_size <= 0:
        return image
    arr = np.array(image)
    h, w = arr.shape
    bh = max(1, int(block_size / 2))
    for y in range(0, h - bh, bh):
        if random.random() < probability / 100:
            shift = random.randint(-amplitude, amplitude)
            if shift != 0:
                y0, y1 = y, min(y + bh, h)
                arr[y0:y1] = np.roll(arr[y0:y1], shift, axis=1)
    return Image.fromarray(arr)


def apply_data_corruption(image, probability, target='all'):
    if probability <= 0:
        return image
    arr = np.array(image)
    h, w = arr.shape
    cand = np.random.random((h, w)) < (probability / 100)
    if target == 'white':
        cand &= arr > 128
    elif target == 'black':
        cand &= arr < 128
    count = cand.sum()
    if count == 0:
        return image
    arr[cand] = np.where(np.random.random(count) < 0.5, 0, 255)
    return Image.fromarray(arr)


def apply_mirror_segments(image, probability, max_size, direction):
    if probability <= 0 or max_size <= 0:
        return image
    arr = np.array(image)
    h, w = arr.shape
    i = 0
    while i < h:
        if random.random() < probability / 100 and h - i >= 2:
            seg_h = random.randint(2, min(max_size, h - i))
            if direction == 'random':
                d = random.choice(['h', 'v'])
            else:
                d = direction
            if d == 'h':
                arr[i:i + seg_h] = arr[i:i + seg_h, ::-1]
            else:
                arr[i:i + seg_h] = arr[i:i + seg_h][::-1]
            i += seg_h
        else:
            i += 1
    return Image.fromarray(arr)


def apply_pixel_sort(image, probability, direction):
    if probability <= 0 or direction == 'off':
        return image
    arr = np.array(image)
    h, w = arr.shape
    rows = np.where(np.random.random(h) < (probability / 100))[0]
    for y in rows:
        if direction == 'random':
            d = 'asc' if random.random() < 0.5 else 'desc'
        else:
            d = direction
        if d == 'asc':
            arr[y] = np.sort(arr[y])
        else:
            arr[y] = np.sort(arr[y])[::-1]
    return Image.fromarray(arr)


def percent_to_threshold(image, percent):
    """Compute threshold so that ~percent% of pixels end up white."""
    arr = np.array(image)
    hist = np.bincount(arr.ravel(), minlength=256)
    target = int(arr.size * (100 - percent) / 100)
    cumsum = np.cumsum(hist)
    return int(np.searchsorted(cumsum, target))


def apply_bw_reverse(image, enabled):
    if not enabled:
        return image
    arr = np.array(image)
    arr = np.where(arr == 0, 255, 0).astype(np.uint8)
    return Image.fromarray(arr, 'L')


def process_image_to_bitmap(image, threshold, dither, downscale,
                            resampling_filter, randomize,
                            percent=50,
                            line_shift_prob=0, line_shift_amp=0,
                            sort_prob=0, sort_dir='off',
                            scatter_prob=0, scatter_radius=0,
                            bloom_radius=0,
                            block_prob=0, block_size=0, block_amp=0,
                            corrupt_prob=0, corrupt_target='all',
                            mirror_prob=0, mirror_size=0, mirror_dir='h',
                            bw_reverse=False,
                            resize_after=True):
    bitmap = apply_bloom(image, bloom_radius)
    bitmap = apply_data_corruption(bitmap, corrupt_prob, corrupt_target)
    bitmap = apply_block_shift(bitmap, block_prob, block_size, block_amp)
    bitmap = apply_pixel_sort(bitmap, sort_prob, sort_dir)
    bitmap = apply_glitch_shift(bitmap, line_shift_prob, line_shift_amp)
    bitmap = apply_pixel_scatter(bitmap, scatter_prob, scatter_radius)
    is_percent = dither == 'percent'
    bitmap = apply_mirror_segments(bitmap, mirror_prob, mirror_size, mirror_dir)
    if is_percent:
        threshold = percent_to_threshold(bitmap, percent)
        dither = None
    bitmap = apply_threshold_or_dither(bitmap, threshold, dither, randomize)
    bitmap = apply_bw_reverse(bitmap, bw_reverse)
    bitmap = apply_scaling(bitmap, downscale, resampling_filter, resize_after)
    if downscale != DEFAULTS["downscale"]:
        if is_percent:
            threshold = percent_to_threshold(bitmap, percent)
        bitmap = apply_threshold_or_dither(bitmap, threshold, dither, randomize)
        bitmap = apply_bw_reverse(bitmap, bw_reverse)
    return bitmap


def _process_one_frame(args):
    fp = args['fp']
    img = Image.open(fp).convert("L")
    bitmap = process_image_to_bitmap(img, **args['params'])
    bitmap.save(fp)
    img.close()


# ============================================================================
# PROCESSING WORKER THREAD
# ============================================================================

class ProcessWorker(QThread):
    progress = Signal(int)
    status = Signal(str)
    done = Signal(str, str)  # output_path, ext
    error = Signal(str)

    def __init__(self, source, ext, output_path, params):
        super().__init__()
        self.source = source
        self.ext = ext
        self.output_path = output_path
        self.params = params

    def run(self):
        try:
            if self.ext in SUPPORTED_IMAGE_FORMATS:
                self._process_image(self.output_path)
            elif self.ext == GIF_EXTENSION:
                self._process_gif(self.output_path)
            elif self.ext in SUPPORTED_VIDEO_FORMATS:
                self._process_video(self.output_path)

            self.done.emit(str(self.output_path), self.ext)
        except Exception as e:
            self.error.emit(str(e))

    def _process_image(self, output):
        self.status.emit("Processing image...")
        image = Image.open(self.source).convert("L")
        bitmap = process_image_to_bitmap(image, **self.params)
        bitmap.save(str(output))
        image.close()
        self.progress.emit(100)

    def _process_gif(self, output):
        gif = Image.open(self.source)
        frame_count = gif.n_frames
        frames = []
        durations = []

        for i in range(frame_count):
            gif.seek(i)
            durations.append(gif.info.get("duration", 100))
            frame_image = gif.copy().convert("L")
            bitmap = process_image_to_bitmap(frame_image, **self.params)
            frames.append(bitmap)
            pct = int((i + 1) / frame_count * 100)
            self.progress.emit(pct)
            self.status.emit(f"GIF frame {i + 1}/{frame_count}...")

        self.status.emit("Saving GIF...")
        self.progress.emit(95)
        frames[0].save(
            str(output), save_all=True, append_images=frames[1:],
            duration=durations, loop=0, optimize=False, disposal=2,
        )
        self.progress.emit(100)
        gif.close()

    def _process_video(self, output):
        video = cv2.VideoCapture(str(self.source))
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        self.status.emit(f"Video: {frame_count} frames @ {fps:.1f} fps")
        if frame_count > 1000:
            self.status.emit("This may take several minutes...")
        num_digits = len(str(frame_count))

        with tempfile.TemporaryDirectory(prefix=f"{DEFAULTS['frame_prefix_video']}_") as tmp:
            self.status.emit("Extracting frames...")
            idx = 0
            while True:
                ok, bgr = video.read()
                if not ok:
                    break
                idx += 1
                cv2.imwrite(str(Path(tmp) / f"frame_{idx:0{num_digits}d}.png"), bgr)
            video.release()

            frames = sorted(glob.glob(str(Path(tmp) / "*.png")))
            total = len(frames)
            args = [{'fp': fp, 'params': self.params} for fp in frames]
            workers = min(os.cpu_count() or 4, 8)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futs = {executor.submit(_process_one_frame, a): a for a in args}
                done = 0
                for _ in as_completed(futs):
                    _.result()
                    done += 1
                    pct = int(done / total * 100)
                    self.progress.emit(pct)
                    if done % max(1, total // 10) == 0:
                        self.status.emit(f"Frame {done}/{total}...")

            self.status.emit("Reconstructing video...")
            first = cv2.imread(frames[0])
            h, w = first.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output), fourcc, fps, (w, h))
            for fp in frames:
                writer.write(cv2.imread(fp))
            writer.release()


def _hsep(style=""):
    sep = QFrame()
    sep.setFrameShape(QFrame.Shape.HLine)
    sep.setStyleSheet(style)
    return sep


# ============================================================================
# MAIN WINDOW
# ============================================================================

class BitMapperDestroyerApp(QMainWindow):

    DITHER_OPTIONS = {
        "None": Image.NONE,
        "Floyd-Steinberg": Image.FLOYDSTEINBERG,
        "Custom Threshold": None,
        "Percent Threshold": "percent",
    }

    def __init__(self):
        super().__init__()
        self.setWindowTitle("BitMapper Destroyer")
        self.setMinimumSize(1000, 700)
        self.resize(1200, 800)

        self.source_path: Path | None = None
        self._worker: ProcessWorker | None = None
        self._last_output: str | None = None

        # Preview state
        self._movie: QMovie | None = None
        self._player: QMediaPlayer | None = None
        self._audio: QAudioOutput | None = None

        self._build_ui()
        QApplication.instance().installEventFilter(self)

    def eventFilter(self, obj, event):
        t = event.type()
        if t == QDragEnterEvent.DragEnter:
            if event.mimeData().hasUrls():
                event.acceptProposedAction()
                return True
        elif t == QDragMoveEvent.DragMove:
            if event.mimeData().hasUrls():
                event.acceptProposedAction()
                return True
        elif t == QDropEvent.Drop:
            urls = event.mimeData().urls()
            if urls:
                self._load_file(urls[0].toLocalFile())
                return True
        return super().eventFilter(obj, event)

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        layout.addWidget(self._build_preview_panel(), 1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(355)
        scroll.setWidget(self._build_left_panel())
        layout.addWidget(scroll, 0)

    def _build_left_panel(self):
        panel = QFrame()
        panel.setFrameStyle(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(panel)
        layout.setSpacing(4)

        title = QLabel("Parameters")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(title)
        layout.addSpacing(4)

        sep_style = "QFrame { background-color: #3b3b3b; max-height: 1px; }"

        # ================================================================
        # File
        # ================================================================
        layout.addWidget(QLabel("<b>File</b>"))
        file_row = QHBoxLayout()
        self.file_label = QLabel("No file selected")
        self.file_label.setStyleSheet("color: gray;")
        self.file_label.setMinimumWidth(50)
        file_row.addWidget(self.file_label, 1)
        browse_btn = QPushButton("Browse")
        browse_btn.setFixedWidth(80)
        browse_btn.clicked.connect(self._browse_file)
        file_row.addWidget(browse_btn)
        layout.addLayout(file_row)

        self.output_label = QLabel("")
        self.output_label.setStyleSheet("color: #666; font-size: 11px;")
        self.output_label.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.output_label)

        layout.addSpacing(6)
        layout.addWidget(_hsep(sep_style))
        layout.addSpacing(4)

        # ================================================================
        # Dither
        # ================================================================
        layout.addWidget(QLabel("<b>Dither</b>"))
        self.dither_combo = QComboBox()
        self.dither_combo.addItems(list(self.DITHER_OPTIONS.keys()))
        self.dither_combo.setCurrentText("None")
        self.dither_combo.currentTextChanged.connect(self._on_dither_changed)
        layout.addWidget(self.dither_combo)

        self._threshold_container = QWidget()
        thr_inner = QVBoxLayout(self._threshold_container)
        thr_inner.setContentsMargins(0, 6, 0, 0)
        thr_inner.setSpacing(4)
        thr_inner.addWidget(QLabel("Threshold"))
        thr_row = QHBoxLayout()
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(1, 255)
        self.threshold_slider.setValue(DEFAULTS["threshold"])
        self.threshold_slider.valueChanged.connect(self._on_threshold_change)
        thr_row.addWidget(self.threshold_slider, 1)
        self.threshold_label = QLabel(str(DEFAULTS["threshold"]))
        self.threshold_label.setFixedWidth(35)
        self.threshold_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        thr_row.addWidget(self.threshold_label)
        thr_inner.addLayout(thr_row)
        layout.addWidget(self._threshold_container)

        # Percent threshold slider (only visible for Percent Threshold)
        self._percent_container = QWidget()
        pct_inner = QVBoxLayout(self._percent_container)
        pct_inner.setContentsMargins(0, 6, 0, 0)
        pct_inner.setSpacing(4)
        pct_inner.addWidget(QLabel("White %"))
        pct_row = QHBoxLayout()
        self.percent_slider = QSlider(Qt.Orientation.Horizontal)
        self.percent_slider.setRange(1, 99)
        self.percent_slider.setValue(50)
        pct_row.addWidget(self.percent_slider, 1)
        self.percent_label = QLabel("50%")
        self.percent_label.setFixedWidth(35)
        self.percent_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        pct_row.addWidget(self.percent_label)
        self.percent_slider.valueChanged.connect(
            lambda v: self.percent_label.setText(f"{v}%"))
        pct_inner.addLayout(pct_row)
        self._percent_container.setVisible(False)
        layout.addWidget(self._percent_container)

        self.randomize_check = QCheckBox("Stochastic Noise")
        layout.addWidget(self.randomize_check)
        self._on_dither_changed("None")

        layout.addSpacing(6)
        layout.addWidget(_hsep(sep_style))
        layout.addSpacing(4)

        # ================================================================
        # Scaling
        # ================================================================
        layout.addWidget(QLabel("<b>Scaling</b>"))
        ds_row = QHBoxLayout()
        self._downscale_value = 1
        self._downscale_buttons: list[QPushButton] = []
        for val, lbl in [(1, "1"), (-2, "1/2"),
                          (-3, "1/3"), (-4, "1/4"), (-8, "1/8"),
                          (-16, "1/16")]:
            btn = QPushButton(lbl)
            btn.setFixedWidth(42)
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, v=val, b=btn: self._on_downscale_preset(v, b))
            ds_row.addWidget(btn)
            self._downscale_buttons.append(btn)
        self._downscale_buttons[0].setChecked(True)
        ds_row.addStretch()
        layout.addLayout(ds_row)

        self._resampling_container = QWidget()
        rf_inner = QVBoxLayout(self._resampling_container)
        rf_inner.setContentsMargins(0, 4, 0, 0)
        rf_inner.setSpacing(2)
        rf_inner.addWidget(QLabel("Resampling Filter"))
        self.resampling_combo = QComboBox()
        self.resampling_combo.addItems(list(RESAMPLING_FILTERS.keys()))
        self.resampling_combo.setCurrentText(DEFAULTS["resampling_filter"])
        rf_inner.addWidget(self.resampling_combo)
        self.resize_after_check = QCheckBox("Resize after downscale")
        self.resize_after_check.setChecked(True)
        rf_inner.addWidget(self.resize_after_check)
        self._resampling_container.setVisible(False)
        layout.addWidget(self._resampling_container)

        layout.addSpacing(6)
        layout.addWidget(_hsep(sep_style))
        layout.addSpacing(4)

        # ================================================================
        # Glitch
        # ================================================================
        layout.addWidget(QLabel("<b>Glitch</b>"))

        # --- Line Shift ---
        self.glitch_check = QCheckBox("Line Shift")
        self.glitch_check.toggled.connect(self._on_glitch_toggled)
        layout.addWidget(self.glitch_check)

        self._glitch_container = QWidget()
        gl_inner = QVBoxLayout(self._glitch_container)
        gl_inner.setContentsMargins(0, 4, 0, 0)
        gl_inner.setSpacing(4)

        gp_row = QHBoxLayout()
        gp_row.addWidget(QLabel("Probability"))
        self.glitch_prob_slider = QSlider(Qt.Orientation.Horizontal)
        self.glitch_prob_slider.setRange(1, 100)
        self.glitch_prob_slider.setValue(10)
        gp_row.addWidget(self.glitch_prob_slider, 1)
        self.glitch_prob_label = QLabel("10%")
        self.glitch_prob_label.setFixedWidth(30)
        self.glitch_prob_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        gp_row.addWidget(self.glitch_prob_label)
        self.glitch_prob_slider.valueChanged.connect(
            lambda v: self.glitch_prob_label.setText(f"{v}%")
        )
        gl_inner.addLayout(gp_row)

        ga_row = QHBoxLayout()
        ga_row.addWidget(QLabel("Amplitude"))
        self.glitch_amp_slider = QSlider(Qt.Orientation.Horizontal)
        self.glitch_amp_slider.setRange(1, 200)
        self.glitch_amp_slider.setValue(10)
        ga_row.addWidget(self.glitch_amp_slider, 1)
        self.glitch_amp_label = QLabel("10px")
        self.glitch_amp_label.setFixedWidth(35)
        self.glitch_amp_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        ga_row.addWidget(self.glitch_amp_label)
        self.glitch_amp_slider.valueChanged.connect(
            lambda v: self.glitch_amp_label.setText(f"{v}px")
        )
        gl_inner.addLayout(ga_row)

        self._glitch_container.setVisible(False)
        layout.addWidget(self._glitch_container)

        # --- Pixel Sorting ---
        self.sort_check = QCheckBox("Pixel Sorting")
        self.sort_check.toggled.connect(self._on_sort_toggled)
        layout.addWidget(self.sort_check)

        self._sort_container = QWidget()
        sort_inner = QVBoxLayout(self._sort_container)
        sort_inner.setContentsMargins(0, 4, 0, 0)
        sort_inner.setSpacing(4)

        sp_row = QHBoxLayout()
        sp_row.addWidget(QLabel("Probability"))
        self.sort_prob_slider = QSlider(Qt.Orientation.Horizontal)
        self.sort_prob_slider.setRange(1, 100)
        self.sort_prob_slider.setValue(10)
        sp_row.addWidget(self.sort_prob_slider, 1)
        self.sort_prob_label = QLabel("10%")
        self.sort_prob_label.setFixedWidth(30)
        self.sort_prob_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        sp_row.addWidget(self.sort_prob_label)
        self.sort_prob_slider.valueChanged.connect(
            lambda v: self.sort_prob_label.setText(f"{v}%")
        )
        sort_inner.addLayout(sp_row)

        sd_row = QHBoxLayout()
        sd_row.addWidget(QLabel("Direction"))
        self.sort_dir_combo = QComboBox()
        self.sort_dir_combo.addItems(["Ascending", "Descending", "Random"])
        sd_row.addWidget(self.sort_dir_combo, 1)
        sort_inner.addLayout(sd_row)

        self._sort_container.setVisible(False)
        layout.addWidget(self._sort_container)

        # --- Pixel Scatter ---
        self.scatter_check = QCheckBox("Pixel Scatter")
        self.scatter_check.toggled.connect(self._on_scatter_toggled)
        layout.addWidget(self.scatter_check)

        self._scatter_container = QWidget()
        sc_inner = QVBoxLayout(self._scatter_container)
        sc_inner.setContentsMargins(0, 4, 0, 0)
        sc_inner.setSpacing(4)

        scp_row = QHBoxLayout()
        scp_row.addWidget(QLabel("Probability"))
        self.scatter_prob_slider = QSlider(Qt.Orientation.Horizontal)
        self.scatter_prob_slider.setRange(1, 100)
        self.scatter_prob_slider.setValue(5)
        scp_row.addWidget(self.scatter_prob_slider, 1)
        self.scatter_prob_label = QLabel("5%")
        self.scatter_prob_label.setFixedWidth(30)
        self.scatter_prob_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        scp_row.addWidget(self.scatter_prob_label)
        self.scatter_prob_slider.valueChanged.connect(
            lambda v: self.scatter_prob_label.setText(f"{v}%")
        )
        sc_inner.addLayout(scp_row)

        scr_row = QHBoxLayout()
        scr_row.addWidget(QLabel("Radius"))
        self.scatter_radius_slider = QSlider(Qt.Orientation.Horizontal)
        self.scatter_radius_slider.setRange(1, 50)
        self.scatter_radius_slider.setValue(3)
        scr_row.addWidget(self.scatter_radius_slider, 1)
        self.scatter_radius_label = QLabel("3px")
        self.scatter_radius_label.setFixedWidth(30)
        self.scatter_radius_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        scr_row.addWidget(self.scatter_radius_label)
        self.scatter_radius_slider.valueChanged.connect(
            lambda v: self.scatter_radius_label.setText(f"{v}px")
        )
        sc_inner.addLayout(scr_row)

        self._scatter_container.setVisible(False)
        layout.addWidget(self._scatter_container)

        # --- Bloom ---
        self.bloom_check = QCheckBox("Bloom")
        self.bloom_check.toggled.connect(self._on_bloom_toggled)
        layout.addWidget(self.bloom_check)

        self._bloom_container = QWidget()
        bl_inner = QVBoxLayout(self._bloom_container)
        bl_inner.setContentsMargins(0, 4, 0, 0)
        bl_inner.setSpacing(4)

        bl_row = QHBoxLayout()
        bl_row.addWidget(QLabel("Radius"))
        self.bloom_radius_slider = QSlider(Qt.Orientation.Horizontal)
        self.bloom_radius_slider.setRange(1, 10)
        self.bloom_radius_slider.setValue(2)
        bl_row.addWidget(self.bloom_radius_slider, 1)
        self.bloom_radius_label = QLabel("2px")
        self.bloom_radius_label.setFixedWidth(30)
        self.bloom_radius_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        bl_row.addWidget(self.bloom_radius_label)
        self.bloom_radius_slider.valueChanged.connect(
            lambda v: self.bloom_radius_label.setText(f"{v}px")
        )
        bl_inner.addLayout(bl_row)

        self._bloom_container.setVisible(False)
        layout.addWidget(self._bloom_container)

        # --- Block Shift ---
        self.block_check = QCheckBox("Block Shift")
        self.block_check.toggled.connect(self._on_block_toggled)
        layout.addWidget(self.block_check)

        self._block_container = QWidget()
        bk_inner = QVBoxLayout(self._block_container)
        bk_inner.setContentsMargins(0, 4, 0, 0)
        bk_inner.setSpacing(4)

        bkp_row = QHBoxLayout()
        bkp_row.addWidget(QLabel("Probability"))
        self.block_prob_slider = QSlider(Qt.Orientation.Horizontal)
        self.block_prob_slider.setRange(1, 100)
        self.block_prob_slider.setValue(10)
        bkp_row.addWidget(self.block_prob_slider, 1)
        self.block_prob_label = QLabel("10%")
        self.block_prob_label.setFixedWidth(30)
        self.block_prob_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        bkp_row.addWidget(self.block_prob_label)
        self.block_prob_slider.valueChanged.connect(
            lambda v: self.block_prob_label.setText(f"{v}%"))
        bk_inner.addLayout(bkp_row)

        bks_row = QHBoxLayout()
        bks_row.addWidget(QLabel("Block Size"))
        self.block_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.block_size_slider.setRange(2, 100)
        self.block_size_slider.setValue(10)
        bks_row.addWidget(self.block_size_slider, 1)
        self.block_size_label = QLabel("10px")
        self.block_size_label.setFixedWidth(30)
        self.block_size_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        bks_row.addWidget(self.block_size_label)
        self.block_size_slider.valueChanged.connect(
            lambda v: self.block_size_label.setText(f"{v}px"))
        bk_inner.addLayout(bks_row)

        bka_row = QHBoxLayout()
        bka_row.addWidget(QLabel("Amplitude"))
        self.block_amp_slider = QSlider(Qt.Orientation.Horizontal)
        self.block_amp_slider.setRange(1, 200)
        self.block_amp_slider.setValue(20)
        bka_row.addWidget(self.block_amp_slider, 1)
        self.block_amp_label = QLabel("20px")
        self.block_amp_label.setFixedWidth(30)
        self.block_amp_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        bka_row.addWidget(self.block_amp_label)
        self.block_amp_slider.valueChanged.connect(
            lambda v: self.block_amp_label.setText(f"{v}px"))
        bk_inner.addLayout(bka_row)

        self._block_container.setVisible(False)
        layout.addWidget(self._block_container)

        # --- Data Corruption ---
        self.corrupt_check = QCheckBox("Data Corruption")
        self.corrupt_check.toggled.connect(self._on_corrupt_toggled)
        layout.addWidget(self.corrupt_check)

        self._corrupt_container = QWidget()
        co_inner = QVBoxLayout(self._corrupt_container)
        co_inner.setContentsMargins(0, 4, 0, 0)
        co_inner.setSpacing(4)

        cop_row = QHBoxLayout()
        cop_row.addWidget(QLabel("Probability"))
        self.corrupt_prob_slider = QSlider(Qt.Orientation.Horizontal)
        self.corrupt_prob_slider.setRange(1, 100)
        self.corrupt_prob_slider.setValue(2)
        cop_row.addWidget(self.corrupt_prob_slider, 1)
        self.corrupt_prob_label = QLabel("2%")
        self.corrupt_prob_label.setFixedWidth(30)
        self.corrupt_prob_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        cop_row.addWidget(self.corrupt_prob_label)
        self.corrupt_prob_slider.valueChanged.connect(
            lambda v: self.corrupt_prob_label.setText(f"{v}%"))
        co_inner.addLayout(cop_row)

        cot_row = QHBoxLayout()
        cot_row.addWidget(QLabel("Target"))
        self.corrupt_target_combo = QComboBox()
        self.corrupt_target_combo.addItems(["All", "White", "Black"])
        cot_row.addWidget(self.corrupt_target_combo, 1)
        co_inner.addLayout(cot_row)

        self._corrupt_container.setVisible(False)
        layout.addWidget(self._corrupt_container)

        # --- Mirror Segments ---
        self.mirror_check = QCheckBox("Mirror Segments")
        self.mirror_check.toggled.connect(self._on_mirror_toggled)
        layout.addWidget(self.mirror_check)

        self._mirror_container = QWidget()
        mr_inner = QVBoxLayout(self._mirror_container)
        mr_inner.setContentsMargins(0, 4, 0, 0)
        mr_inner.setSpacing(4)

        mrp_row = QHBoxLayout()
        mrp_row.addWidget(QLabel("Probability"))
        self.mirror_prob_slider = QSlider(Qt.Orientation.Horizontal)
        self.mirror_prob_slider.setRange(1, 100)
        self.mirror_prob_slider.setValue(10)
        mrp_row.addWidget(self.mirror_prob_slider, 1)
        self.mirror_prob_label = QLabel("10%")
        self.mirror_prob_label.setFixedWidth(30)
        self.mirror_prob_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        mrp_row.addWidget(self.mirror_prob_label)
        self.mirror_prob_slider.valueChanged.connect(
            lambda v: self.mirror_prob_label.setText(f"{v}%"))
        mr_inner.addLayout(mrp_row)

        mrs_row = QHBoxLayout()
        mrs_row.addWidget(QLabel("Max Size"))
        self.mirror_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.mirror_size_slider.setRange(5, 200)
        self.mirror_size_slider.setValue(50)
        mrs_row.addWidget(self.mirror_size_slider, 1)
        self.mirror_size_label = QLabel("50px")
        self.mirror_size_label.setFixedWidth(30)
        self.mirror_size_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        mrs_row.addWidget(self.mirror_size_label)
        self.mirror_size_slider.valueChanged.connect(
            lambda v: self.mirror_size_label.setText(f"{v}px"))
        mr_inner.addLayout(mrs_row)

        mrd_row = QHBoxLayout()
        mrd_row.addWidget(QLabel("Direction"))
        self.mirror_dir_combo = QComboBox()
        self.mirror_dir_combo.addItems(["Horizontal", "Vertical", "Random"])
        mrd_row.addWidget(self.mirror_dir_combo, 1)
        mr_inner.addLayout(mrd_row)

        self._mirror_container.setVisible(False)
        layout.addWidget(self._mirror_container)

        # --- BW Reverse ---
        self.bw_reverse_check = QCheckBox("BW Reverse")
        layout.addWidget(self.bw_reverse_check)

        layout.addSpacing(8)

        # ================================================================
        # Advanced Options
        # ================================================================
        layout.addSpacing(6)
        layout.addWidget(_hsep(sep_style))
        layout.addSpacing(4)
        layout.addWidget(QLabel("<b>Advanced Options</b>"))
        self.auto_save_check = QCheckBox("Auto Save")
        self.auto_save_check.setChecked(False)
        layout.addWidget(self.auto_save_check)

        layout.addSpacing(6)
        layout.addWidget(_hsep(sep_style))
        layout.addSpacing(4)

        # ================================================================
        # Process
        # ================================================================
        layout.addStretch()
        self.process_btn = QPushButton("PROCESS")
        self.process_btn.setFixedHeight(36)
        self.process_btn.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.process_btn.setEnabled(False)
        self.process_btn.clicked.connect(self._process_file)
        layout.addWidget(self.process_btn)

        self.save_btn = QPushButton("SAVE")
        self.save_btn.setFixedHeight(36)
        self.save_btn.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.save_btn.setVisible(False)
        self.save_btn.clicked.connect(self._save_file)
        layout.addWidget(self.save_btn)

        self.auto_save_check.toggled.connect(self._on_auto_save_toggled)
        self._on_auto_save_toggled(False)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: gray;")
        layout.addWidget(self.status_label)

        return panel

    def _on_dither_changed(self, text):
        is_custom = text == "Custom Threshold"
        is_percent = text == "Percent Threshold"
        self._threshold_container.setVisible(is_custom)
        self._percent_container.setVisible(is_percent)
        self.randomize_check.setVisible(is_custom or is_percent)

    def _on_downscale_preset(self, value, btn):
        for b in self._downscale_buttons:
            b.setChecked(b is btn)
        self._downscale_value = value
        self._resampling_container.setVisible(value != 1)

    def _on_glitch_toggled(self, checked):
        self._glitch_container.setVisible(checked)

    def _on_sort_toggled(self, checked):
        self._sort_container.setVisible(checked)

    def _on_scatter_toggled(self, checked):
        self._scatter_container.setVisible(checked)

    def _on_bloom_toggled(self, checked):
        self._bloom_container.setVisible(checked)

    def _on_block_toggled(self, checked):
        self._block_container.setVisible(checked)

    def _on_corrupt_toggled(self, checked):
        self._corrupt_container.setVisible(checked)

    def _on_mirror_toggled(self, checked):
        self._mirror_container.setVisible(checked)

    def _on_auto_save_toggled(self, checked):
        self.save_btn.setVisible(not checked)
        self.save_btn.setEnabled(False)

    def _build_preview_panel(self):
        panel = QFrame()
        panel.setFrameStyle(QFrame.Shape.StyledPanel)
        panel.setStyleSheet("background-color: #2b2b2b;")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        self.preview_stack = QStackedWidget()

        # Page 0: placeholder
        placeholder = QLabel("Drag & Drop a file here\nor click to browse")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder.setStyleSheet("color: #888; font-size: 18px;")
        self.preview_stack.addWidget(placeholder)

        # Page 1: static image / GIF via QMovie
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background-color: #2b2b2b;")
        self.preview_stack.addWidget(self.image_label)

        # Page 2: video
        self.video_widget = QVideoWidget()
        self.video_widget.setStyleSheet("background-color: #000;")
        self.preview_stack.addWidget(self.video_widget)

        # Page 3: processing status
        self._processing_status_label = QLabel()
        self._processing_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._processing_status_label.setStyleSheet(
            "color: white; font-size: 16px; background-color: #2b2b2b;"
        )
        self.preview_stack.addWidget(self._processing_status_label)

        self.preview_stack.setCurrentIndex(0)
        layout.addWidget(self.preview_stack)

        return panel

    # ------------------------------------------------------------------
    # Drag & Drop
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # File Loading & Preview
    # ------------------------------------------------------------------

    def _browse_file(self):
        filters = "All supported (*.jpg *.jpeg *.png *.gif *.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm);;Images (*.jpg *.jpeg *.png);;GIF (*.gif);;Videos (*.mp4 *.avi *.mov *.mkv)"
        filepath, _ = QFileDialog.getOpenFileName(self, "Select file", "", filters)
        if filepath:
            self._load_file(filepath)

    def _load_file(self, filepath):
        path = Path(filepath)
        ext = path.suffix.lower()

        if ext not in ALL_SUPPORTED:
            QMessageBox.critical(
                self, "Unsupported Format",
                f"Format '{ext}' is not supported.\n"
                f"Supported: {', '.join(sorted(ALL_SUPPORTED))}",
            )
            return

        self._stop_preview()
        self.source_path = path
        self.process_btn.setEnabled(True)
        self.save_btn.setEnabled(False)

        self._elide_label(self.file_label, path.name)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        out_text = f"Output: {path.stem}destroy{timestamp}{ext}"
        self._elide_label(self.output_label, out_text)

        try:
            if ext in SUPPORTED_IMAGE_FORMATS:
                self._show_image_preview(path)
            elif ext == GIF_EXTENSION:
                self._show_gif_preview(path)
            elif ext in SUPPORTED_VIDEO_FORMATS:
                self._show_video_preview(path)
        except Exception as e:
            self.status_label.setText("Preview failed")
            self.status_label.setStyleSheet("color: #F44336;")
            QMessageBox.critical(self, "Preview Error", str(e))

    def _show_image_preview(self, path):
        pixmap = QPixmap(str(path))
        scaled = pixmap.scaled(
            self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)
        self.preview_stack.setCurrentIndex(1)

    def _show_gif_preview(self, path):
        self._movie = QMovie(str(path))
        self._movie.setCacheMode(QMovie.CacheMode.CacheAll)
        self.image_label.setMovie(self._movie)
        self._movie.start()
        self.preview_stack.setCurrentIndex(1)

    def _show_video_preview(self, path):
        self._player = QMediaPlayer()
        self._audio = QAudioOutput()
        self._player.setAudioOutput(self._audio)
        self._player.setVideoOutput(self.video_widget)
        self._player.setSource(QUrl.fromLocalFile(str(path)))
        self._player.setLoops(QMediaPlayer.Loops.Infinite)
        self._player.play()
        self.preview_stack.setCurrentIndex(2)

    def _stop_preview(self):
        if self._movie is not None:
            self._movie.stop()
            self._movie = None
        if self._player is not None:
            self._player.stop()
            self._player = None
            self._audio = None
        self.image_label.clear()
        self.image_label.setMovie(None)
        self.preview_stack.setCurrentIndex(0)

    def _preview_output(self, filepath, ext):
        """Show processed file in preview without changing source."""
        try:
            if ext in SUPPORTED_IMAGE_FORMATS:
                self._show_image_preview(Path(filepath))
            elif ext == GIF_EXTENSION:
                self._show_gif_preview(Path(filepath))
            elif ext in SUPPORTED_VIDEO_FORMATS:
                self._show_video_preview(Path(filepath))
        except Exception:
            pass  # preview failure is non-critical

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def _process_file(self):
        if self._worker is not None or not self.source_path:
            return

        self._stop_preview()
        self.preview_stack.setCurrentIndex(3)
        self.save_btn.setEnabled(False)

        source = self.source_path
        ext = source.suffix.lower()
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        name = f"{source.stem}destroy{timestamp}{ext}"

        if self.auto_save_check.isChecked():
            output_path = source.with_name(name)
        else:
            tmp_root = Path(tempfile.gettempdir()) / "bitmapper_destroyer" / source.stem
            if tmp_root.exists():
                shutil.rmtree(tmp_root)
            tmp_root.mkdir(parents=True, exist_ok=True)
            output_path = tmp_root / name

        threshold = self.threshold_slider.value()
        dither = self.DITHER_OPTIONS[self.dither_combo.currentText()]
        downscale = self._downscale_value
        resampling_filter = RESAMPLING_FILTERS[self.resampling_combo.currentText()]
        randomize = self.randomize_check.isChecked()
        shift_on = self.glitch_check.isChecked()
        sort_on = self.sort_check.isChecked()
        scatter_on = self.scatter_check.isChecked()
        bloom_on = self.bloom_check.isChecked()
        block_on = self.block_check.isChecked()
        corrupt_on = self.corrupt_check.isChecked()
        mirror_on = self.mirror_check.isChecked()

        params = {
            'threshold': threshold,
            'dither': dither,
            'percent': self.percent_slider.value(),
            'downscale': downscale,
            'resampling_filter': resampling_filter,
            'resize_after': self.resize_after_check.isChecked(),
            'randomize': randomize,
            'line_shift_prob': self.glitch_prob_slider.value() if shift_on else 0,
            'line_shift_amp': self.glitch_amp_slider.value() if shift_on else 0,
            'sort_prob': self.sort_prob_slider.value() if sort_on else 0,
            'sort_dir': {'Ascending': 'asc', 'Descending': 'desc',
                         'Random': 'random'}[self.sort_dir_combo.currentText()] if sort_on else 'off',
            'scatter_prob': self.scatter_prob_slider.value() if scatter_on else 0,
            'scatter_radius': self.scatter_radius_slider.value() if scatter_on else 0,
            'bloom_radius': self.bloom_radius_slider.value() if bloom_on else 0,
            'block_prob': self.block_prob_slider.value() if block_on else 0,
            'block_size': self.block_size_slider.value() if block_on else 0,
            'block_amp': self.block_amp_slider.value() if block_on else 0,
            'corrupt_prob': self.corrupt_prob_slider.value() if corrupt_on else 0,
            'corrupt_target': self.corrupt_target_combo.currentText().lower() if corrupt_on else 'all',
            'mirror_prob': self.mirror_prob_slider.value() if mirror_on else 0,
            'mirror_size': self.mirror_size_slider.value() if mirror_on else 0,
            'mirror_dir': {'Horizontal': 'h', 'Vertical': 'v',
                           'Random': 'random'}[self.mirror_dir_combo.currentText()] if mirror_on else 'h',
            'bw_reverse': self.bw_reverse_check.isChecked(),
        }

        if self._worker is not None:
            self._worker.status.disconnect(self._on_status)
            self._worker.progress.disconnect(self._on_progress)
            self._worker.done.disconnect(self._on_done)
            self._worker.error.disconnect(self._on_error)
            self._worker.finished.disconnect(self._on_worker_finished)
        self._worker = ProcessWorker(source, ext, output_path, params)
        self._worker.status.connect(self._on_status)
        self._worker.progress.connect(self._on_progress)
        self._worker.done.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._on_worker_finished)

        self.process_btn.setEnabled(False)
        self.process_btn.setText("Processing...")
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting...")
        self.status_label.setStyleSheet("color: white;")

        self._worker.start()

    def _on_status(self, msg):
        self.status_label.setText(msg)
        self._processing_status_label.setText(msg)

    def _on_progress(self, pct):
        self.progress_bar.setValue(pct)

    def _on_done(self, output_path, ext):
        self.progress_bar.setValue(100)
        out_name = Path(output_path).name
        self._elide_label(self.status_label, f"Done: {out_name}")
        self.status_label.setStyleSheet("color: #4CAF50;")
        self._preview_output(output_path, ext)
        self._last_output = output_path
        if not self.auto_save_check.isChecked():
            self.save_btn.setEnabled(True)

    def _save_file(self):
        if not self._last_output or not self.source_path:
            return
        dest = self.source_path.parent / Path(self._last_output).name
        shutil.copy2(self._last_output, dest)
        self._elide_label(self.status_label, f"Saved: {dest.name}")
        self.status_label.setStyleSheet("color: #4CAF50;")
        self.save_btn.setEnabled(False)

    def _on_error(self, msg):
        self.status_label.setText("Error!")
        self.status_label.setStyleSheet("color: #F44336;")
        QMessageBox.critical(self, "Error", f"Processing failed:\n{msg}")

    def _on_worker_finished(self):
        self.process_btn.setEnabled(True)
        self.process_btn.setText("PROCESS")
        self._worker = None

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _elide_label(self, label, text):
        if len(text) > 20:
            label.setText(text[:19] + "…")
            label.setToolTip(text)
        else:
            label.setText(text)
            label.setToolTip("")

    def _on_threshold_change(self, value):
        self.threshold_label.setText(str(value))

    def resizeEvent(self, event: QResizeEvent):
        super().resizeEvent(event)
        # Re-scale static image if showing
        if self.preview_stack.currentIndex() == 1 and self._movie is None:
            pixmap = self.image_label.pixmap()
            if pixmap is not None and not pixmap.isNull():
                # re-load from source
                if self.source_path and self.source_path.suffix.lower() in SUPPORTED_IMAGE_FORMATS:
                    p = QPixmap(str(self.source_path))
                    scaled = p.scaled(
                        self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                    self.image_label.setPixmap(scaled)


def main():
    app = QApplication([])
    app.setStyle("Fusion")
    font = app.font()
    font.setFamily("Menlo")
    app.setFont(font)
    window = BitMapperDestroyerApp()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
