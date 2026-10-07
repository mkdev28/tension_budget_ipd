"""
TensionBudget — Stage 3 Real-Time & Offline Bilateral Monitoring App (Phase 8).

A standalone graphical interface built with PyQt5 and pyqtgraph for upper-trapezius sEMG monitoring.
Provides real-time bilateral signal visualization, cumulative load scoring (EIndex), spectral fatigue
trend tracking, interactive 5-second calibration holds, and seamless switching between live USB serial 
(LSL) streaming and offline CSV file replay.
"""

import os
import sys

# Add parent workspace directory to sys.path for standalone script execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import json
import math
from pathlib import Path
import numpy as np
from scipy.signal import butter, filtfilt

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
    QPushButton, QLabel, QComboBox, QTabWidget, QFileDialog, QGroupBox,
    QGridLayout, QProgressBar, QMessageBox, QHeaderView, QTableWidget, QTableWidgetItem,
    QLineEdit, QDialog, QRadioButton, QButtonGroup
)
from PyQt5.QtCore import QTimer, Qt
import pyqtgraph as pg

try:
    import pylsl
except ImportError:
    pylsl = None

from chordspy.tensionbudget.config import TBConfig
from chordspy.tensionbudget.local_logger import LocalSessionLogger
from chordspy.tensionbudget.preprocessing import load_tb_csv, preprocess_channel, StreamingChannelProcessor
from chordspy.tensionbudget.calibration import compute_reference_rms
from chordspy.tensionbudget.scoring import score_bilateral_window, EIndexAccumulator, map_composite_to_display_scale
from chordspy.tensionbudget.spectral import analyze_bilateral_spectral_fatigue
from chordspy.tensionbudget.analytics import analyze_channel, compute_asymmetry


class BorgStrainPopupDialog(QDialog):
    def __init__(self, epoch_idx, elapsed_min, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"🎯 Epoch #{epoch_idx} Completed ({elapsed_min:.1f} min elapsed)")
        self.setModal(True)
        self.resize(520, 500)
        self.selected_rating = None

        layout = QVBoxLayout(self)
        title_lbl = QLabel(
            f"<b>Epoch #{epoch_idx} Complete ({elapsed_min:.1f} minutes)!</b><br>"
            "Please rate your average perceived muscular strain over the past 5 minutes on the Borg CR-10 Scale:"
        )
        title_lbl.setStyleSheet("font-size: 13px; margin-bottom: 8px;")
        layout.addWidget(title_lbl)

        self.btn_group = QButtonGroup(self)
        box = QGroupBox("0–10 Borg CR-10 Continuous Domain")
        box_layout = QVBoxLayout(box)
        
        anchors = [
            (0, "0 — Complete Rest (Nothing at all)"),
            (1, "1 — Very weak (Just noticeable effort)"),
            (2, "2 — Weak (Light effort)"),
            (3, "3 — Moderate (Comfortable working level)"),
            (4, "4 — Somewhat strong"),
            (5, "5 — Strong (Heavy working fatigue)"),
            (6, "6 — Very noticeable fatigue"),
            (7, "7 — Very strong (Severe strain)"),
            (8, "8 — Extremely strong (Near failure)"),
            (9, "9 — Approaching maximum tolerance"),
            (10, "10 — Absolute maximum (Intolerable pain/fatigue)")
        ]
        for idx, (val, text) in enumerate(anchors):
            rb = QRadioButton(text)
            if val == 0:
                rb.setChecked(True)
                self.selected_rating = 0.0
            self.btn_group.addButton(rb, val)
            box_layout.addWidget(rb)
        layout.addWidget(box)

        self.btn_group.idClicked.connect(lambda val_id: setattr(self, "selected_rating", float(val_id)))

        btn_layout = QHBoxLayout()
        btn_submit = QPushButton("✅ Submit Score ($y$)")
        btn_submit.setStyleSheet("background-color: #2E7D32; color: white; font-weight: bold; padding: 8px 16px; font-size: 13px;")
        btn_submit.clicked.connect(self.accept)

        btn_skip = QPushButton("⏭️ Skip / Unreported (NULL)")
        btn_skip.setStyleSheet("padding: 8px 16px; font-size: 13px;")
        btn_skip.clicked.connect(self.skip_submission)

        btn_layout.addWidget(btn_submit)
        btn_layout.addWidget(btn_skip)
        layout.addLayout(btn_layout)

    def skip_submission(self):
        self.selected_rating = None
        self.reject()


class TensionBudgetApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TensionBudget Monitor — Bilateral Upper Trapezius EMG (Phase 8)")
        self.setGeometry(100, 100, 1050, 750)

        self.cfg = TBConfig()
        self.channel_map = self.cfg.CHANNEL_MAP or {"left": 0, "right": 1}
        
        # Stream / Playback state
        self.inlet = None
        self.stream_active = False
        self.mode = "offline"  # "live" or "offline"
        self.offline_data = None
        self.offline_timestamps = None
        self.offline_idx = 0
        self.sampling_rate = self.cfg.TARGET_SAMPLING_RATE  # default 500 Hz
        
        # Buffers: 10-second rolling window for live plotting (5000 samples @ 500 Hz)
        self.buffer_size = self.sampling_rate * 10
        self.raw_left = np.zeros(self.buffer_size)
        self.raw_right = np.zeros(self.buffer_size)
        self.time_data = np.linspace(0, 10, self.buffer_size)
        self.current_index = 0
        
        # Preprocessing filters
        self.b_band, self.a_band = butter(
            self.cfg.BANDPASS_ORDER,
            [self.cfg.BANDPASS_LOW_HZ / (0.5 * self.sampling_rate),
             self.cfg.BANDPASS_HIGH_HZ / (0.5 * self.sampling_rate)],
            btype="bandpass"
        )
        self.rms_window_size = int(self.cfg.RMS_WINDOW_MS * self.sampling_rate / 1000)
        
        # Calibration state (defaulting to 1.0 until calibrated or loaded from file)
        self.ref_rms_left = 1.0
        self.ref_rms_right = 1.0
        self._try_load_existing_calibration()

        # Calibration recording state
        self.is_calibrating = False
        self.calib_buffer_l = []
        self.calib_buffer_r = []
        self.calib_target_samples = self.sampling_rate * 5  # 5 seconds
        
        # Scoring accumulator & session logger state
        self.accumulator_l = EIndexAccumulator()
        self.accumulator_r = EIndexAccumulator()
        self.tick_counter = 0
        
        self.local_logger = None
        self.session_sample_count = 0
        self.last_logged_window = 0
        self.loaded_csv_name = "Offline_CSV"
        self.current_strain_rating = None

        # Phase 12 Full-Epoch Spectral Accumulation (Pre-rectification Bandpass + Notch)
        self.stream_proc_l = StreamingChannelProcessor(fs=self.sampling_rate, config=self.cfg)
        self.stream_proc_r = StreamingChannelProcessor(fs=self.sampling_rate, config=self.cfg)
        self.epoch_filt_l = []
        self.epoch_filt_r = []

        # ML Model Weights & Live Predictor State
        self.prev_mdf_l = None
        self.prev_mdf_r = None
        self.prev_gap_freq_l = None
        self.prev_gap_freq_r = None
        self.prev_eindex_l = None
        self.prev_eindex_r = None
        self._load_ml_model_weights()

        # UI Setup
        self._init_ui()
        
        # Main updates timer (every 20 ms = 50 Hz refresh, matching RMS_STEP_MS)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_loop)
        self.timer.start(20)

    def _load_ml_model_weights(self):
        self.ml_weights = None
        self.ml_loaded = False
        weights_file = Path("tensionbudget_model_weights.json")
        if not weights_file.exists():
            weights_file = Path(__file__).resolve().parent.parent / "tensionbudget_model_weights.json"
        if weights_file.exists():
            try:
                with open(weights_file, "r", encoding="utf-8") as f:
                    self.ml_weights = json.load(f)
                    self.ml_loaded = True
                    print(f"Successfully loaded ML model weights from {weights_file}")
            except Exception as e:
                print(f"Notice: Failed to load ML model weights: {e}")

    def _init_ui(self):
        central_widget = QWidget()
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # ── User Profile & Subject Metadata Bar (Required for Session Logging) ──
        user_box = QGroupBox("Subject Metadata & Personalization Vault (Required for Session & ML Logging)")
        user_layout = QHBoxLayout()
        
        user_layout.addWidget(QLabel("Subject ID / Name:"))
        self.in_user_name = QLineEdit("Default_User")
        self.in_user_name.setPlaceholderText("e.g. Mohit_Kumar")
        user_layout.addWidget(self.in_user_name)
        
        user_layout.addWidget(QLabel("DOB:"))
        self.in_dob = QLineEdit("1995-01-01")
        self.in_dob.setPlaceholderText("YYYY-MM-DD")
        self.in_dob.setMaximumWidth(90)
        user_layout.addWidget(self.in_dob)
        
        user_layout.addWidget(QLabel("Sex:"))
        self.in_sex = QComboBox()
        self.in_sex.addItems(["Unspecified", "M", "F", "Other"])
        self.in_sex.setMaximumWidth(95)
        user_layout.addWidget(self.in_sex)
        
        user_layout.addWidget(QLabel("Wt (kg):"))
        self.in_weight = QLineEdit("70.0")
        self.in_weight.setMaximumWidth(65)
        user_layout.addWidget(self.in_weight)
        
        user_layout.addWidget(QLabel("Ht (cm):"))
        self.in_height = QLineEdit("175.0")
        self.in_height.setMaximumWidth(65)
        user_layout.addWidget(self.in_height)
        
        user_layout.addWidget(QLabel("Password:"))
        self.in_password = QLineEdit()
        self.in_password.setPlaceholderText("Cloud sync password")
        self.in_password.setEchoMode(QLineEdit.Password)
        self.in_password.setMaximumWidth(130)
        user_layout.addWidget(self.in_password)
        
        self.lbl_profile_info = QLabel("Status: Enter details before starting session.")
        self.lbl_profile_info.setStyleSheet("color: #2a6f3a; font-weight: bold;")
        user_layout.addWidget(self.lbl_profile_info, 1)
        
        user_box.setLayout(user_layout)
        main_layout.addWidget(user_box)

        # ── Top Control Bar ──
        ctrl_box = QGroupBox("Stream Controls & Mode Selection")
        ctrl_layout = QHBoxLayout()
        
        ctrl_layout.addWidget(QLabel("Data Source:"))
        self.source_combo = QComboBox()
        self.source_combo.addItems(["Offline CSV Replay Mode", "Live LSL Stream (Uno R4)"])
        self.source_combo.currentIndexChanged.connect(self.on_source_changed)
        ctrl_layout.addWidget(self.source_combo)
        
        self.btn_load_csv = QPushButton("Select CSV File...")
        self.btn_load_csv.clicked.connect(self.on_load_csv)
        ctrl_layout.addWidget(self.btn_load_csv)

        self.btn_start_stop = QPushButton("Start Replay")
        self.btn_start_stop.setStyleSheet("background-color: #2b5c8f; color: white; font-weight: bold; padding: 5px;")
        self.btn_start_stop.clicked.connect(self.on_toggle_stream)
        ctrl_layout.addWidget(self.btn_start_stop)

        self.status_label = QLabel("Status: Idle (Select CSV or Live LSL to begin)")
        self.status_label.setStyleSheet("color: #555; font-weight: bold;")
        ctrl_layout.addWidget(self.status_label, 1)
        
        ctrl_box.setLayout(ctrl_layout)
        main_layout.addWidget(ctrl_box)

        # ── Top Live Alert Banner (Visible Across All Tabs) ──
        self.lbl_top_alert_banner = QLabel("🟢 NORMAL MONITORING STATE: Trapezius Load Within Safe Range.")
        self.lbl_top_alert_banner.setStyleSheet("background-color: #2b6f39; color: white; font-weight: bold; font-size: 14px; padding: 8px; border-radius: 4px;")
        self.lbl_top_alert_banner.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.lbl_top_alert_banner)

        # ── Subjective Strain (Borg CR-10) Target Variable Bar for Bayesian Hierarchy Models ──
        strain_box = QGroupBox("Subjective Self-Report (Borg CR-10 / Bayesian Target Label $y$)")
        strain_layout = QHBoxLayout()
        strain_layout.addWidget(QLabel("Perceived Strain ($y$):"))
        self.strain_combo = QComboBox()
        self.strain_combo.addItem("[NULL] Unreported / Skip (strain_reported = 0)", None)
        for val in range(int(TBConfig.STRAIN_SCALE_MIN), int(TBConfig.STRAIN_SCALE_MAX) + 1):
            desc = TBConfig.STRAIN_TARGET_LABELS.get(val, str(val))
            self.strain_combo.addItem(f"{val} — {desc.split('—')[-1].strip()}", float(val))
        self.strain_combo.setStyleSheet("font-weight: bold; padding: 4px; border: 1px solid #777; border-radius: 4px;")
        self.strain_combo.currentIndexChanged.connect(self.on_strain_changed)
        strain_layout.addWidget(self.strain_combo, 1)

        self.btn_log_strain_now = QPushButton("Log Strain Marker")
        self.btn_log_strain_now.setStyleSheet("background-color: #5a3d77; color: white; font-weight: bold; padding: 4px 10px; border-radius: 4px;")
        self.btn_log_strain_now.clicked.connect(self.on_manual_strain_log)
        strain_layout.addWidget(self.btn_log_strain_now)

        self.lbl_strain_status = QLabel("Target: NULL | strain_reported = 0")
        self.lbl_strain_status.setStyleSheet("color: #4a235a; font-weight: bold; padding-left: 8px;")
        strain_layout.addWidget(self.lbl_strain_status, 1)
        
        strain_box.setLayout(strain_layout)
        main_layout.addWidget(strain_box)

        # ── Tabbed Views ──
        self.tabs = QTabWidget()
        
        # Tab 1: Live Signal View
        tab_signal = QWidget()
        sig_layout = QVBoxLayout()
        
        self.plot_left = pg.PlotWidget(title="Left Upper Trapezius (Ch 0 / A0) — Filtered (Blue) & RMS Envelope (Red)")
        self.plot_left.setBackground('w')
        self.plot_left.showGrid(x=True, y=True)
        self.plot_left.setYRange(-300, 300)
        self.curve_l_filt = self.plot_left.plot(pen=pg.mkPen('b', width=1))
        self.curve_l_rms = self.plot_left.plot(pen=pg.mkPen('r', width=2))
        sig_layout.addWidget(self.plot_left)
        
        self.plot_right = pg.PlotWidget(title="Right Upper Trapezius (Ch 1 / A1) — Filtered (Blue) & RMS Envelope (Red)")
        self.plot_right.setBackground('w')
        self.plot_right.showGrid(x=True, y=True)
        self.plot_right.setYRange(-300, 300)
        self.curve_r_filt = self.plot_right.plot(pen=pg.mkPen('b', width=1))
        self.curve_r_rms = self.plot_right.plot(pen=pg.mkPen('r', width=2))
        sig_layout.addWidget(self.plot_right)
        
        tab_signal.setLayout(sig_layout)
        self.tabs.addTab(tab_signal, "1. Live Bilateral Signals")

        # Tab 2: Analytics & Scoring Dashboard
        tab_dashboard = QWidget()
        dash_layout = QVBoxLayout()

        # ── ML Pain Forecast & Real-Time Alert Engine Card ──
        ml_box = QGroupBox("🧠 Real-Time Bayesian ML 5-Min Pain Forecast & Ergonomic Feedback Engine")
        ml_layout = QVBoxLayout()
        
        self.lbl_ml_forecast = QLabel("Predicted Pain Score (t + 5 min): 0.00 / 10.0 (Borg Scale)")
        self.lbl_ml_forecast.setStyleSheet("font-size: 16px; font-weight: bold; color: #4a235a;")
        
        self.lbl_ml_ci = QLabel("90% Credible Interval: [0.00, 0.00] | residual σ = 0.85")
        self.lbl_ml_ci.setStyleSheet("font-size: 13px; font-weight: bold; color: #555;")
        
        self.lbl_ml_prob = QLabel("P(Pain ≥ 4.0 in 5 min): 0.0%")
        self.lbl_ml_prob.setStyleSheet("font-size: 14px; font-weight: bold; color: #154360;")
        
        feedback_title = QLabel("<b>Instant Ergonomic Feedback & Actionable Guidance:</b>")
        feedback_title.setStyleSheet("margin-top: 6px; font-size: 13px;")
        
        self.lbl_instant_feedback = QLabel("• Monitoring live muscular dynamics...")
        self.lbl_instant_feedback.setStyleSheet("font-size: 13px; color: #1e8449; background-color: #eafaf1; padding: 8px; border: 1px solid #a9dfbf; border-radius: 4px;")
        self.lbl_instant_feedback.setWordWrap(True)
        
        ml_layout.addWidget(self.lbl_ml_forecast)
        ml_layout.addWidget(self.lbl_ml_ci)
        ml_layout.addWidget(self.lbl_ml_prob)
        ml_layout.addWidget(feedback_title)
        ml_layout.addWidget(self.lbl_instant_feedback)
        ml_box.setLayout(ml_layout)
        dash_layout.addWidget(ml_box)
        
        grid_box = QGroupBox("Real-Time TensionBudget Analytics (Stage 3 / Phase 6 & 7)")
        grid_layout = QGridLayout()
        
        # Big metric labels
        self.lbl_score_l = QLabel("Left Score: 0.00 (EIndex: 0.00)")
        self.lbl_score_r = QLabel("Right Score: 0.00 (EIndex: 0.00)")
        self.lbl_composite = QLabel("Composite Score: 0.00")
        self.lbl_composite.setStyleSheet("font-size: 16px; font-weight: bold; color: #1e3d59;")
        
        self.lbl_asymmetry = QLabel("Asymmetry (Laterality Index): 0.00 (Symmetric)")
        self.lbl_gaps = QLabel("Relaxation Gaps: L=0 (0.0s), R=0 (0.0s)")
        self.lbl_suma = QLabel("Short-SUMA Penalty: L=0.00, R=0.00")
        self.lbl_fatigue = QLabel("Spectral Fatigue: L Slope = 0.00 Hz/min | R Slope = 0.00 Hz/min")
        self.lbl_fatigue.setStyleSheet("font-size: 14px; font-weight: bold; color: #8a3033;")
        
        grid_layout.addWidget(self.lbl_composite, 0, 0, 1, 2)
        grid_layout.addWidget(self.lbl_score_l, 1, 0)
        grid_layout.addWidget(self.lbl_score_r, 1, 1)
        grid_layout.addWidget(self.lbl_asymmetry, 2, 0, 1, 2)
        grid_layout.addWidget(self.lbl_gaps, 3, 0)
        grid_layout.addWidget(self.lbl_suma, 3, 1)
        grid_layout.addWidget(self.lbl_fatigue, 4, 0, 1, 2)
        
        # Mandatory passband caveat banner
        caveat_lbl = QLabel(
            "<b>Methodological Caveat:</b> Because our 20–240 Hz band is truncated relative to standard 20–450 Hz conventions "
            "(due to the fixed 500 Hz sample rate and 250 Hz Nyquist ceiling), absolute MDF/MNF values are lower than published norms "
            "and must be interpreted strictly as internal time-trend/slope indicators per channel."
        )
        caveat_lbl.setWordWrap(True)
        caveat_lbl.setStyleSheet("background-color: #fff3cd; color: #856404; padding: 10px; border: 1px solid #ffeeba; border-radius: 4px;")
        grid_layout.addWidget(caveat_lbl, 5, 0, 1, 2)
        
        grid_box.setLayout(grid_layout)
        dash_layout.addWidget(grid_box)
        dash_layout.addStretch(1)
        tab_dashboard.setLayout(dash_layout)
        self.tabs.addTab(tab_dashboard, "2. Analytics Dashboard")

        # Tab 3: Interactive Calibration
        tab_calib = QWidget()
        calib_layout = QVBoxLayout()
        
        calib_box = QGroupBox("Submaximal Shrug Calibration (Phase 4)")
        c_vbox = QVBoxLayout()
        c_vbox.addWidget(QLabel(
            "<b>Instructions:</b> To accurately compute your personalized tension load (%RVE), sit upright in a comfortable resting position. "
            "When you click the start button below, you will perform a <b>5-second submaximal shoulder shrug</b>: "
            "<br><br>1. <b>Ramp Up</b> smoothly during second 1.<br>2. <b>Hold Steady</b> at moderate effort for 3 seconds.<br>3. <b>Relax</b> down during second 5."
        ))
        
        self.btn_start_calib = QPushButton("Start 5-Second Shrug Calibration Hold")
        self.btn_start_calib.setStyleSheet("background-color: #28a745; color: white; font-weight: bold; padding: 10px; font-size: 14px;")
        self.btn_start_calib.clicked.connect(self.on_start_calibration)
        c_vbox.addWidget(self.btn_start_calib)
        
        self.calib_progress = QProgressBar()
        self.calib_progress.setValue(0)
        c_vbox.addWidget(self.calib_progress)
        
        self.lbl_calib_status = QLabel(f"Current Calibration References — Left: {self.ref_rms_left:.4f} | Right: {self.ref_rms_right:.4f} (Uncalibrated / Default)")
        self.lbl_calib_status.setStyleSheet("font-weight: bold; font-size: 13px; color: #333; margin-top: 10px;")
        c_vbox.addWidget(self.lbl_calib_status)
        
        calib_box.setLayout(c_vbox)
        calib_layout.addWidget(calib_box)
        calib_layout.addStretch(1)
        tab_calib.setLayout(calib_layout)
        self.tabs.addTab(tab_calib, "3. Interactive Calibration Mode")

        main_layout.addWidget(self.tabs)

    def _try_load_existing_calibration(self):
        calib_file = Path("tensionbudget_calibration.json")
        if calib_file.exists():
            try:
                with open(calib_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.ref_rms_left = float(data.get("left", {}).get("reference_rms", 1.0))
                    self.ref_rms_right = float(data.get("right", {}).get("reference_rms", 1.0))
                    if hasattr(self, 'lbl_calib_status'):
                        self.lbl_calib_status.setText(
                            f"Loaded Calibration References — Left: {self.ref_rms_left:.4f} | Right: {self.ref_rms_right:.4f}"
                        )
            except Exception as e:
                print(f"Notice: Failed to read existing calibration JSON: {e}")

    def on_strain_changed(self, index):
        val = self.strain_combo.currentData()
        if val is not None and not (isinstance(val, float) and (math.isnan(val) or val < 0)):
            self.current_strain_rating = float(val)
            self.lbl_strain_status.setText(f"Target ($y$): {self.current_strain_rating:.1f} | strain_reported = 1 (Active)")
        else:
            self.current_strain_rating = None
            self.lbl_strain_status.setText("Target ($y$): NULL | strain_reported = 0 (Unreported)")

    def on_manual_strain_log(self):
        if self.local_logger and self.local_logger.is_active:
            self._log_current_epoch(epoch_idx=None, is_final=False)
            self.lbl_strain_status.setText(f"✓ Saved Target ($y$ = {self.current_strain_rating}) to {self.local_logger.csv_path.name}")
        else:
            QMessageBox.information(self, "Session Offline", "Please start a recording or replay session to log target labels to CSV.")

    def on_source_changed(self, index):
        if self.stream_active:
            self.on_toggle_stream()  # Stop current stream
        if index == 0:
            self.mode = "offline"
            self.btn_load_csv.setEnabled(True)
            self.btn_start_stop.setText("Start Replay")
            self.status_label.setText("Status: Offline mode selected. Choose a CSV file.")
        else:
            self.mode = "live"
            self.btn_load_csv.setEnabled(False)
            self.btn_start_stop.setText("Connect LSL Stream")
            self.status_label.setText("Status: Live LSL mode selected. Make sure Uno R4 is plugged in & streaming.")

    def on_load_csv(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Select TensionBudget Raw CSV", "", "CSV Files (*.csv);;All Files (*)")
        if not fname:
            return
        try:
            data, meta = load_tb_csv(fname)

            # Always resolve left/right from CHANNEL_MAP.
            # CHANNEL_MAP uses 0-based indices; CSV column names use 1-based:
            #   index 2 → "Channel3"  (A2 — Left electrode)
            #   index 3 → "Channel4"  (A3 — Right electrode)
            l_idx = self.channel_map.get("left", 2)
            r_idx = self.channel_map.get("right", 3)
            l_col = f"Channel{l_idx + 1}"   # e.g. index 2 → "Channel3"
            r_col = f"Channel{r_idx + 1}"   # e.g. index 3 → "Channel4"

            if l_col in data and r_col in data:
                self.offline_data = {"left": data[l_col], "right": data[r_col]}
                ch_info = f"({l_col}=Left / {r_col}=Right  per CHANNEL_MAP)"
            elif "left_raw" in data and "right_raw" in data:
                # Pre-processed CSV already has named columns
                self.offline_data = {"left": data["left_raw"], "right": data["right_raw"]}
                ch_info = "(left_raw / right_raw columns)"
            else:
                # Last resort: use whatever first two numeric columns exist
                avail = [k for k in data if k.startswith("Channel")]
                if len(avail) >= 2:
                    avail.sort()
                    self.offline_data = {"left": data[avail[0]], "right": data[avail[1]]}
                    ch_info = f"(fallback: {avail[0]}=Left / {avail[1]}=Right)"
                else:
                    raise ValueError(f"No usable channel columns found. Available keys: {list(data.keys())}")

            self.offline_timestamps = data.get(
                "Timestamp_s",
                np.arange(len(self.offline_data["left"])) / self.sampling_rate
            )
            self.offline_idx = 0
            n = len(self.offline_data["left"])
            self.loaded_csv_name = Path(fname).name
            self.status_label.setText(
                f"Status: Loaded {self.loaded_csv_name} — {n:,} samples  {ch_info}. Press Start Replay."
            )
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load CSV:\n{e}")

    def on_toggle_stream(self):
        if not self.stream_active:
            # ── Pre-flight validation ─────────────────────────────────────────
            subject_name = self.in_user_name.text().strip()
            if not subject_name or subject_name == "Default_User":
                QMessageBox.warning(
                    self, "Subject ID Required",
                    "Please enter a real Subject ID / Name before starting.\n\n"
                    "'Default_User' is a placeholder — data recorded under this name\n"
                    "cannot be attributed to a specific participant."
                )
                return
            if not self.in_password.text().strip():
                reply = QMessageBox.question(
                    self, "No Password Entered",
                    "The Password field is empty.\n\n"
                    "Without a password the cloud sync will be REJECTED (HTTP 400).\n"
                    "Local files will still be saved.\n\n"
                    "Continue without a password?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if reply == QMessageBox.No:
                    return
            # ─────────────────────────────────────────────────────────────────
            if self.mode == "live":
                if pylsl is None:
                    QMessageBox.warning(self, "LSL Unavailable", "pylsl module is not installed. Switching to Offline mode.")
                    self.source_combo.setCurrentIndex(0)
                    return
                self.status_label.setText("Status: Searching for available LSL streams...")
                QApplication.processEvents()
                streams = pylsl.resolve_streams()
                if not streams:
                    QMessageBox.warning(self, "Stream Not Found", "No active LSL streams detected. Ensure Connection/USB streaming is started.")
                    self.status_label.setText("Status: No LSL stream found.")
                    return
                self.inlet = pylsl.StreamInlet(streams[0])
                self.sampling_rate = int(self.inlet.info().nominal_srate()) or 500
                self.stream_active = True
                self.btn_start_stop.setText("Disconnect")
                self.status_label.setText(f"Status: Connected to live LSL stream ({streams[0].name()} @ {self.sampling_rate} Hz)")
                self._start_or_resume_logger("Live LSL")
            else:
                if self.offline_data is None:
                    QMessageBox.warning(self, "No CSV Loaded", "Please select a raw CSV file first.")
                    return
                self.stream_active = True
                self.btn_start_stop.setText("Pause Replay")
                self.status_label.setText("Status: Replaying offline CSV data...")
                self._start_or_resume_logger(self.loaded_csv_name)
        else:
            self.stream_active = False
            self.inlet = None
            if self.mode == "live":
                self.btn_start_stop.setText("Connect LSL Stream")
                self.status_label.setText("Status: Disconnected.")
                if self.local_logger and self.local_logger.is_active:
                    # Prompt for final strain rating on disconnect before completing session
                    if self.isVisible() and not os.getenv("TB_HEADLESS"):
                        elapsed_min = (getattr(self, "session_sample_count", 0) / max(1, getattr(self, "sampling_rate", 500))) / 60.0
                        dialog = BorgStrainPopupDialog(
                            epoch_idx=getattr(self, "last_logged_window", 0) + 1,
                            elapsed_min=elapsed_min,
                            parent=self
                        )
                        dialog.exec_()  # Synchronous modal wait since acquisition has stopped
                        self.current_strain_rating = dialog.selected_rating
                    self._log_current_epoch(is_final=True)
                    self.local_logger.end_session(password=getattr(self, "_session_password", None))
                    self.lbl_profile_info.setText("Session Completed & Saved to output_logs/")
            else:
                self.btn_start_stop.setText("Resume Replay")
                self.status_label.setText("Status: Replay paused.")

    def _start_or_resume_logger(self, source_info):
        if self.local_logger is None or not self.local_logger.is_active:
            try:
                wt = float(self.in_weight.text()) if self.in_weight.text() else None
            except ValueError:
                wt = None
            try:
                ht = float(self.in_height.text()) if self.in_height.text() else None
            except ValueError:
                ht = None
            # Capture password at session start so it can be forwarded to cloud sync at end_session
            self._session_password = self.in_password.text().strip() or None
            self.local_logger = LocalSessionLogger(
                user_name=self.in_user_name.text(),
                birth_date_str=self.in_dob.text(),
                gender_sex=self.in_sex.currentText(),
                weight_kg=wt,
                height_cm=ht
            )
            self.local_logger.start_session(
                mode=self.mode,
                calibration_left=self.ref_rms_left,
                calibration_right=self.ref_rms_right,
                source_info=source_info
            )
            self.session_sample_count = 0
            self.last_logged_window = 0
            self.stream_proc_l = StreamingChannelProcessor(fs=self.sampling_rate, config=self.cfg)
            self.stream_proc_r = StreamingChannelProcessor(fs=self.sampling_rate, config=self.cfg)
            self.epoch_filt_l = []
            self.epoch_filt_r = []
            age_txt = f"{self.local_logger.age_years} yrs" if self.local_logger.age_years else "N/A"
            bmi_txt = f"{self.local_logger.bmi_value}" if self.local_logger.bmi_value else "N/A"
            self.lbl_profile_info.setText(f"Active Session — Subject: {self.local_logger.user_name} (Age: {age_txt}, BMI: {bmi_txt})")

    def on_start_calibration(self):
        if not self.stream_active:
            QMessageBox.warning(self, "Stream Inactive", "Please start live streaming or CSV playback before initiating calibration.")
            return
        self.is_calibrating = True
        self.calib_buffer_l = []
        self.calib_buffer_r = []
        self.btn_start_calib.setEnabled(False)
        self.btn_start_calib.setText("Calibrating... Perform Shrug Hold NOW!")
        self.calib_progress.setValue(0)
        self.status_label.setText("Status: Recording 5-second calibration hold...")

    def calculate_moving_rms(self, signal_arr, window_size):
        if len(signal_arr) < window_size or window_size <= 0:
            return np.abs(signal_arr)
        kernel = np.ones(window_size) / window_size
        rms = np.sqrt(np.convolve(signal_arr**2, kernel, mode='valid'))
        return np.pad(rms, (len(signal_arr) - len(rms), 0), 'edge')

    def update_loop(self):
        if not self.stream_active:
            return

        chunk_l = []
        chunk_r = []
        
        # 1. Acquire sample chunk (up to 25 samples per 20ms tick at 500 Hz)
        if self.mode == "live":
            samples, _ = self.inlet.pull_chunk(timeout=0.0, max_samples=40)
            if samples:
                l_idx = self.channel_map["left"]
                r_idx = self.channel_map["right"]
                for s in samples:
                    if len(s) > max(l_idx, r_idx):
                        chunk_l.append(s[l_idx])
                        chunk_r.append(s[r_idx])
        else:
            # Offline replay: fetch next 10 samples per tick (20 ms at 500 Hz = 10 samples)
            end_idx = min(self.offline_idx + 10, len(self.offline_data["left"]))
            if end_idx > self.offline_idx:
                chunk_l = list(self.offline_data["left"][self.offline_idx:end_idx])
                chunk_r = list(self.offline_data["right"][self.offline_idx:end_idx])
                self.offline_idx = end_idx
            else:
                self.stream_active = False
                self.btn_start_stop.setText("Restart Replay")
                self.status_label.setText("Status: Replay reached end of CSV file.")
                self.offline_idx = 0
                if self.local_logger and self.local_logger.is_active:
                    self._log_current_epoch(is_final=True)
                    self.local_logger.end_session()
                    self.lbl_profile_info.setText("Session Completed & Saved to output_logs/")
                return

        if not chunk_l:
            return

        # Phase 12: Accumulate pre-rectification bandpass & notch filtered signal for epoch spectral regression
        out_l = self.stream_proc_l.process_chunk(np.array(chunk_l))
        out_r = self.stream_proc_r.process_chunk(np.array(chunk_r))
        if "filtered" in out_l and len(out_l["filtered"]) > 0:
            self.epoch_filt_l.extend(out_l["filtered"])
        if "filtered" in out_r and len(out_r["filtered"]) > 0:
            self.epoch_filt_r.extend(out_r["filtered"])
        max_epoch_samples = int(self.sampling_rate * 300)
        if len(self.epoch_filt_l) > max_epoch_samples:
            del self.epoch_filt_l[:-max_epoch_samples]
        if len(self.epoch_filt_r) > max_epoch_samples:
            del self.epoch_filt_r[:-max_epoch_samples]

        # Stream raw telemetry directly to disk buffer before incrementing counter
        if self.local_logger and self.local_logger.is_active and hasattr(self, "session_sample_count"):
            curr_epoch_idx = int(self.session_sample_count // (self.sampling_rate * 300)) + 1
            self.local_logger.log_raw_chunk(chunk_l, chunk_r, self.session_sample_count, self.sampling_rate, curr_epoch_idx)

        # Advance sample counter and check 5-minute epoch boundaries (300 seconds)
        self.session_sample_count += len(chunk_l)
        if self.local_logger and self.local_logger.is_active:
            epoch_idx = int(self.session_sample_count // (self.sampling_rate * 300))
            if epoch_idx > self.last_logged_window and hasattr(self, "last_score_res"):
                self.last_logged_window = epoch_idx
                self._trigger_epoch_logging(epoch_idx=epoch_idx, is_final=False)

        # 2. Update Calibration buffer if active
        if self.is_calibrating:
            self.calib_buffer_l.extend(chunk_l)
            self.calib_buffer_r.extend(chunk_r)
            progress = int(100 * len(self.calib_buffer_l) / self.calib_target_samples)
            self.calib_progress.setValue(min(100, progress))
            
            if len(self.calib_buffer_l) >= self.calib_target_samples:
                self.is_calibrating = False
                self.btn_start_calib.setEnabled(True)
                self.btn_start_calib.setText("Start 5-Second Shrug Calibration Hold")
                
                arr_l = np.array(self.calib_buffer_l)
                arr_r = np.array(self.calib_buffer_r)
                
                ref_l = compute_reference_rms(arr_l, fs=self.sampling_rate, config=self.cfg)
                ref_r = compute_reference_rms(arr_r, fs=self.sampling_rate, config=self.cfg)
                
                if ref_l > 0 and not np.isnan(ref_l):
                    self.ref_rms_left = ref_l
                if ref_r > 0 and not np.isnan(ref_r):
                    self.ref_rms_right = ref_r
                    
                self.lbl_calib_status.setText(
                    f"Saved Calibration References — Left: {self.ref_rms_left:.4f} | Right: {self.ref_rms_right:.4f}"
                )
                self.status_label.setText("Status: Calibration completed successfully!")
                
                # Save to disk
                try:
                    out_dict = {
                        "left": {"reference_rms": self.ref_rms_left},
                        "right": {"reference_rms": self.ref_rms_right},
                        "timestamp": time.time()
                    }
                    with open("tensionbudget_calibration.json", "w", encoding="utf-8") as f:
                        json.dump(out_dict, f, indent=4)
                    if self.local_logger and self.local_logger.is_active:
                        self.local_logger.update_calibration(self.ref_rms_left, self.ref_rms_right)
                except Exception as e:
                    print(f"Notice: Could not save calibration JSON: {e}")

        # 3. Update rolling circular buffer for live plots
        n_new = len(chunk_l)
        if n_new >= self.buffer_size:
            self.raw_left[:] = chunk_l[-self.buffer_size:]
            self.raw_right[:] = chunk_r[-self.buffer_size:]
        else:
            self.raw_left[:-n_new] = self.raw_left[n_new:]
            self.raw_left[-n_new:] = chunk_l
            self.raw_right[:-n_new] = self.raw_right[n_new:]
            self.raw_right[-n_new:] = chunk_r

        # 4. Stage 2 Preprocessing: bandpass filter (20-240 Hz) and moving RMS
        filt_l = filtfilt(self.b_band, self.a_band, self.raw_left)
        filt_r = filtfilt(self.b_band, self.a_band, self.raw_right)
        
        rms_l = self.calculate_moving_rms(np.abs(filt_l), self.rms_window_size)
        rms_r = self.calculate_moving_rms(np.abs(filt_r), self.rms_window_size)

        # Update curves
        self.curve_l_filt.setData(self.time_data, filt_l)
        self.curve_l_rms.setData(self.time_data, rms_l)
        self.curve_r_filt.setData(self.time_data, filt_r)
        self.curve_r_rms.setData(self.time_data, rms_r)

        # 5. Stage 3 Real-time Analytics (update dashboard every ~200ms or 10 ticks)
        self.tick_counter += 1
        if self.tick_counter % 10 == 0:
            norm_l = (rms_l / self.ref_rms_left) * 100.0
            norm_r = (rms_r / self.ref_rms_right) * 100.0
            
            # Subsample normalized array to RMS step rate (every 10 samples @ 500 Hz = 20ms step)
            step_samples = int(self.cfg.RMS_STEP_MS * self.sampling_rate / 1000)
            sub_norm_l = norm_l[::step_samples]
            sub_norm_r = norm_r[::step_samples]
            
            # Phase 5 Analytics (gaps, SUMA)
            res_l = analyze_channel(sub_norm_l, config=self.cfg, is_live_buffer=True)
            res_r = analyze_channel(sub_norm_r, config=self.cfg, is_live_buffer=True)
            
            # Phase 6 Composite Score
            score_res = score_bilateral_window(
                sub_norm_l, sub_norm_r,
                res_l.get("suma_bins", {}), res_r.get("suma_bins", {}),
                res_l.get("active_apdf", {}).get(50, np.nan), res_r.get("active_apdf", {}).get(50, np.nan),
                self.accumulator_l, self.accumulator_r,
                config=self.cfg
            )
            
            # Phase 7 Spectral Fatigue (on pre-rectified bandpass filtered signal)
            spec_res = analyze_bilateral_spectral_fatigue(filt_l, filt_r, fs=self.sampling_rate, config=self.cfg)
            
            self.last_score_res = score_res
            self.last_res_l = res_l
            self.last_res_r = res_r
            self.last_spec_res = spec_res
            self.last_sub_norm_l = sub_norm_l
            self.last_sub_norm_r = sub_norm_r

            # Update labels
            c_score = score_res.get('composite', {}).get('composite_score', 0.0)
            s_left = score_res.get('score_left', 0.0)
            s_right = score_res.get('score_right', 0.0)
            e_left = score_res.get('eindex_live_left', 0.0)
            e_right = score_res.get('eindex_live_right', 0.0)
            ai = score_res.get('asymmetry_index', 0.0)
            if ai is None or not isinstance(ai, (int, float)) or np.isnan(ai):
                ai = 0.0

            self.lbl_composite.setText(f"Composite Score: {map_composite_to_display_scale(c_score):.2f} (Worst-Side Driven)")
            self.lbl_score_l.setText(f"Left Side Score: {s_left:.2f} (Live EIndex: {e_left:.2f})")
            self.lbl_score_r.setText(f"Right Side Score: {s_right:.2f} (Live EIndex: {e_right:.2f})")

            interp = "Symmetric"
            if ai < -0.2:
                interp = "Left Dominant"
            elif ai > 0.2:
                interp = "Right Dominant"
            self.lbl_asymmetry.setText(f"Asymmetry (Laterality Index): {ai:.2f} ({interp})")

            self.lbl_gaps.setText(
                f"Relaxation Gaps: L={res_l['gaps_count']} ({res_l['gaps_total_time_s']:.1f}s) | "
                f"R={res_r['gaps_count']} ({res_r['gaps_total_time_s']:.1f}s)"
            )
            self.lbl_suma.setText(
                f"Short-SUMA Penalty: L={score_res.get('short_suma_penalty_left', 0.0):.2f} | "
                f"R={score_res.get('short_suma_penalty_right', 0.0):.2f}"
            )
            
            slope_l = spec_res.get('left', {}).get('slope_hz_per_min', 0.0)
            slope_r = spec_res.get('right', {}).get('slope_hz_per_min', 0.0)
            self.lbl_fatigue.setText(f"Spectral Fatigue Trend: L Slope = {slope_l:.2f} Hz/min | R Slope = {slope_r:.2f} Hz/min")

            # Update Live ML Pain Forecast ($t + 5\text{ min}$), Credible Intervals, Alert Banners & Feedback
            self._update_ml_predictions_and_feedback(score_res, res_l, res_r, spec_res)

    def _update_ml_predictions_and_feedback(self, score_res, res_l, res_r, spec_res):
        eindex_l = float(score_res.get("eindex_live_left", 0.0))
        eindex_r = float(score_res.get("eindex_live_right", 0.0))
        ai = score_res.get("asymmetry_index", 0.0)
        if ai is None or not isinstance(ai, (int, float)) or np.isnan(ai):
            ai = 0.0
        ai = float(ai)

        apdf_l = res_l.get("active_apdf", {}) if isinstance(res_l.get("active_apdf"), dict) else {}
        apdf_r = res_r.get("active_apdf", {}) if isinstance(res_r.get("active_apdf"), dict) else {}
        apdf_10_l = float(apdf_l.get(10, 10.0)) if isinstance(apdf_l.get(10), (int, float)) else 10.0
        apdf_10_r = float(apdf_r.get(10, 10.0)) if isinstance(apdf_r.get(10), (int, float)) else 10.0
        apdf_50_l = float(apdf_l.get(50, 25.0)) if isinstance(apdf_l.get(50), (int, float)) else 25.0
        apdf_50_r = float(apdf_r.get(50, 25.0)) if isinstance(apdf_r.get(50), (int, float)) else 25.0
        apdf_90_l = float(apdf_l.get(90, 50.0)) if isinstance(apdf_l.get(90), (int, float)) else 50.0
        apdf_90_r = float(apdf_r.get(90, 50.0)) if isinstance(apdf_r.get(90), (int, float)) else 50.0

        gaps_count_l = float(res_l.get("gaps_count", 0))
        gaps_count_r = float(res_r.get("gaps_count", 0))
        elapsed_min = max(0.1, (self.session_sample_count / max(1, self.sampling_rate)) / 60.0)
        gap_freq_l = gaps_count_l / elapsed_min
        gap_freq_r = gaps_count_r / elapsed_min

        short_suma_l = float(score_res.get("short_suma_penalty_left", 0.0))
        short_suma_r = float(score_res.get("short_suma_penalty_right", 0.0))
        suma_bins_l = res_l.get("suma_bins", {})
        suma_bins_r = res_r.get("suma_bins", {})
        total_suma_l = float(sum(suma_bins_l.values())) if isinstance(suma_bins_l, dict) else 0.0
        total_suma_r = float(sum(suma_bins_r.values())) if isinstance(suma_bins_r, dict) else 0.0

        mdf_l = float(spec_res.get("left", {}).get("mdf_hz", 45.0))
        mdf_r = float(spec_res.get("right", {}).get("mdf_hz", 45.0))
        slope_l = float(spec_res.get("left", {}).get("slope_hz_per_min", 0.0))
        slope_r = float(spec_res.get("right", {}).get("slope_hz_per_min", 0.0))

        mdf_trend_l = mdf_l - self.prev_mdf_l if self.prev_mdf_l is not None else 0.0
        mdf_trend_r = mdf_r - self.prev_mdf_r if self.prev_mdf_r is not None else 0.0
        gap_trend_l = gap_freq_l - self.prev_gap_freq_l if self.prev_gap_freq_l is not None else 0.0
        gap_trend_r = gap_freq_r - self.prev_gap_freq_r if self.prev_gap_freq_r is not None else 0.0
        eindex_trend_l = eindex_l - self.prev_eindex_l if self.prev_eindex_l is not None else 0.0
        eindex_trend_r = eindex_r - self.prev_eindex_r if self.prev_eindex_r is not None else 0.0

        self.prev_mdf_l = mdf_l
        self.prev_mdf_r = mdf_r
        self.prev_gap_freq_l = gap_freq_l
        self.prev_gap_freq_r = gap_freq_r
        self.prev_eindex_l = eindex_l
        self.prev_eindex_r = eindex_r

        feat_dict = {
            "eindex_live_left": eindex_l, "eindex_live_right": eindex_r,
            "apdf_10_left": apdf_10_l, "apdf_10_right": apdf_10_r,
            "apdf_50_left": apdf_50_l, "apdf_50_right": apdf_50_r,
            "apdf_90_left": apdf_90_l, "apdf_90_right": apdf_90_r,
            "gap_frequency_left": gap_freq_l, "gap_frequency_right": gap_freq_r,
            "short_suma_penalty_left": short_suma_l, "short_suma_penalty_right": short_suma_r,
            "total_suma_bursts_left": total_suma_l, "total_suma_bursts_right": total_suma_r,
            "mdf_hz_left": mdf_l, "mdf_hz_right": mdf_r,
            "fatigue_slope_left": slope_l, "fatigue_slope_right": slope_r,
            "asymmetry_index_ai": ai,
            "calibration_rms_left": float(self.ref_rms_left), "calibration_rms_right": float(self.ref_rms_right),
            "mdf_trend_left": mdf_trend_l, "mdf_trend_right": mdf_trend_r,
            "gap_freq_trend_left": gap_trend_l, "gap_freq_trend_right": gap_trend_r,
            "eindex_trend_left": eindex_trend_l, "eindex_trend_right": eindex_trend_r
        }

        if self.ml_loaded and self.ml_weights:
            means = self.ml_weights.get("scaler_means", {})
            stds = self.ml_weights.get("scaler_stds", {})
            coefs = self.ml_weights.get("population_coefficients", {})
            pop_int = float(self.ml_weights.get("population_intercept", 2.5))
            user_offsets = self.ml_weights.get("user_intercept_offsets", {})
            current_user = self.in_user_name.text().strip()
            u_offset = float(user_offsets.get(current_user, 0.0))
            sigma = float(self.ml_weights.get("residual_sigma", 0.85))

            y_pred = pop_int + u_offset
            for k, val in feat_dict.items():
                m = float(means.get(k, 0.0))
                s = float(stds.get(k, 1.0))
                if s == 0: s = 1.0
                std_val = (val - m) / s
                y_pred += float(coefs.get(k, 0.0)) * std_val

            y_pred_clamped = max(0.0, min(10.0, y_pred))

            ci_low = max(0.0, y_pred_clamped - 1.645 * sigma)
            ci_high = min(10.0, y_pred_clamped + 1.645 * sigma)
            z = (4.0 - y_pred_clamped) / max(1e-6, sigma)
            prob_high_strain = 1.0 - (0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))

            alert_policy = self.ml_weights.get("alert_policy", {})
            req_prob = float(alert_policy.get("confidence_required", 0.85))
            hard_ceil = float(alert_policy.get("hard_safety_ceiling", 5.5))

            is_alert = (prob_high_strain >= req_prob) or (ci_high >= hard_ceil)

            self.lbl_ml_forecast.setText(
                f"Predicted Pain Score (t + 5 min): <b>{y_pred_clamped:.2f} / 10.0</b> (Borg Scale)"
            )
            self.lbl_ml_ci.setText(
                f"90% Credible Interval: <b>[{ci_low:.2f}, {ci_high:.2f}]</b> | residual σ = {sigma:.2f}"
            )
            self.lbl_ml_prob.setText(
                f"P(Pain ≥ 4.0 in 5 min): <b>{prob_high_strain * 100.0:.1f}%</b>"
            )

            if is_alert:
                self.lbl_top_alert_banner.setText(
                    f"🚨 EARLY WARNING ALERT (t + 5 min): Muscular Strain Predicted! (Borg {y_pred_clamped:.1f}, P={prob_high_strain*100:.0f}%) — Take an Ergonomic Break Now!"
                )
                self.lbl_top_alert_banner.setStyleSheet(
                    "background-color: #d9534f; color: white; font-weight: bold; font-size: 14px; padding: 8px; border-radius: 4px;"
                )
            elif y_pred_clamped >= 2.5:
                self.lbl_top_alert_banner.setText(
                    f"⚠️ MODERATE LOAD WARNING: Muscle Fatigue Accumulating (Predicted Strain: {y_pred_clamped:.1f} in 5 min)."
                )
                self.lbl_top_alert_banner.setStyleSheet(
                    "background-color: #f0ad4e; color: black; font-weight: bold; font-size: 14px; padding: 8px; border-radius: 4px;"
                )
            else:
                self.lbl_top_alert_banner.setText(
                    f"🟢 NORMAL MONITORING STATE: Trapezius Load Within Safe Range (Predicted Strain: {y_pred_clamped:.1f})."
                )
                self.lbl_top_alert_banner.setStyleSheet(
                    "background-color: #2b6f39; color: white; font-weight: bold; font-size: 14px; padding: 8px; border-radius: 4px;"
                )

        feedback_bullets = []
        if abs(ai) > 0.25:
            side_str = "RIGHT" if ai > 0 else "LEFT"
            feedback_bullets.append(f"• <b>Asymmetry Detected ({ai:+.2f}):</b> Excessive load on {side_str} trapezius. Check mouse/screen position.")
        if (gap_freq_l < 5.0 and gap_freq_l > 0) or (gap_freq_r < 5.0 and gap_freq_r > 0):
            feedback_bullets.append("• <b>Low Micro-Rest Gaps (< 5 gaps/min):</b> Perform a 10-second shoulder drop to rest motor units.")
        if slope_l < -0.3 or slope_r < -0.3:
            feedback_bullets.append("• <b>Downward Spectral Fatigue (-MDF Slope):</b> Muscle fiber conduction velocity slowing down.")
        if short_suma_l > 0.3 or short_suma_r > 0.3:
            feedback_bullets.append("• <b>Rapid Contraction Bursts (SUMA Penalty Active):</b> High short-duration effort cycles detected.")
        if not feedback_bullets:
            feedback_bullets.append("• <b>Good Ergonomic Posture:</b> Bilateral muscle loading and rest gaps are within optimal baseline limits.")

        self.lbl_instant_feedback.setText("<br>".join(feedback_bullets))

    def _log_current_epoch(self, epoch_idx=None, is_final=False):
        if not self.local_logger or not self.local_logger.is_active or not hasattr(self, "last_score_res"):
            return
        elapsed_min = (self.session_sample_count / max(1, self.sampling_rate)) / 60.0
        if epoch_idx is None:
            epoch_idx = int(self.last_logged_window + 1) if is_final and elapsed_min > (self.last_logged_window * 5) else self.last_logged_window
            if epoch_idx == 0:
                epoch_idx = 1

        if hasattr(self, "epoch_filt_l") and len(self.epoch_filt_l) >= 256 and hasattr(self, "epoch_filt_r") and len(self.epoch_filt_r) >= 256:
            arr_l = np.array(self.epoch_filt_l)
            arr_r = np.array(self.epoch_filt_r)
            spec_res = analyze_bilateral_spectral_fatigue(
                arr_l, arr_r,
                fs=self.sampling_rate, config=self.cfg
            )
            # Compute full-epoch RMS envelopes & normalized %RVE for true 5-minute SUMA duration & Gap exposure analytics
            rms_l = self.calculate_moving_rms(np.abs(arr_l), getattr(self, "rms_window_size", 50))
            rms_r = self.calculate_moving_rms(np.abs(arr_r), getattr(self, "rms_window_size", 50))
            norm_l = (rms_l / max(1e-6, getattr(self, "ref_rms_left", 1.0))) * 100.0
            norm_r = (rms_r / max(1e-6, getattr(self, "ref_rms_right", 1.0))) * 100.0
            
            step_samples = max(1, int(self.cfg.RMS_STEP_MS * getattr(self, "sampling_rate", 500) / 1000))
            sub_l = norm_l[::step_samples]
            sub_r = norm_r[::step_samples]
            
            res_l = analyze_channel(sub_l, config=self.cfg, is_live_buffer=False)
            res_r = analyze_channel(sub_r, config=self.cfg, is_live_buffer=False)
            score_res = score_bilateral_window(
                sub_l, sub_r,
                res_l.get("suma_bins", {}), res_r.get("suma_bins", {}),
                res_l.get("active_apdf", {}).get(50, np.nan), res_r.get("active_apdf", {}).get(50, np.nan),
                self.accumulator_l, self.accumulator_r,
                config=self.cfg
            )
        else:
            spec_res = getattr(self, "last_spec_res", {})
            score_res = getattr(self, "last_score_res", {})
            res_l = getattr(self, "last_res_l", {})
            res_r = getattr(self, "last_res_r", {})
            sub_l = getattr(self, "last_sub_norm_l", None)
            sub_r = getattr(self, "last_sub_norm_r", None)

        if sub_l is not None and sub_r is not None:
            if not is_final or (epoch_idx > self.last_logged_window) or (getattr(self.accumulator_l, "completed_windows", 0) == 0):
                self.accumulator_l.submit_completed_window(sub_l)
                self.accumulator_r.submit_completed_window(sub_r)

        # Calculate true relaxation gap frequency per minute within this specific epoch duration
        epoch_dur_min = (len(sub_l) * self.cfg.RMS_STEP_MS / 1000.0) / 60.0 if (sub_l is not None and len(sub_l) > 0) else max(0.1, elapsed_min)

        score_left_dict = {
            "eindex_live": score_res.get("eindex_live_left", 0.0),
            "eindex_cumulative": getattr(self.accumulator_l, "eindex_session_cumulative", 0.0),
            "short_suma_penalty": score_res.get("short_suma_penalty_left", 0.0),
            "suma_count": sum(res_l.get("suma_bins", {}).values()) if isinstance(res_l.get("suma_bins"), dict) else 0,
            "gap_frequency_per_min": res_l.get("gaps_count", 0) / max(0.1, epoch_dur_min),
            "apdf_10": res_l.get("active_apdf", {}).get(10, ""),
            "apdf_50": res_l.get("active_apdf", {}).get(50, ""),
            "apdf_90": res_l.get("active_apdf", {}).get(90, ""),
        }
        score_right_dict = {
            "eindex_live": score_res.get("eindex_live_right", 0.0),
            "eindex_cumulative": getattr(self.accumulator_r, "eindex_session_cumulative", 0.0),
            "short_suma_penalty": score_res.get("short_suma_penalty_right", 0.0),
            "suma_count": sum(res_r.get("suma_bins", {}).values()) if isinstance(res_r.get("suma_bins"), dict) else 0,
            "gap_frequency_per_min": res_r.get("gaps_count", 0) / max(0.1, epoch_dur_min),
            "apdf_10": res_r.get("active_apdf", {}).get(10, ""),
            "apdf_50": res_r.get("active_apdf", {}).get(50, ""),
            "apdf_90": res_r.get("active_apdf", {}).get(90, ""),
        }

        self.local_logger.log_epoch(
            epoch_index=epoch_idx,
            elapsed_minutes=elapsed_min,
            composite_score=score_res.get("composite", {}).get("composite_score", 0.0),
            score_left_dict=score_left_dict,
            score_right_dict=score_right_dict,
            spectral_left_dict=spec_res.get("left", {}),
            spectral_right_dict=spec_res.get("right", {}),
            asymmetry_index=score_res.get("asymmetry_index", 0.0),
            asymmetry_penalty_applied=abs(score_res.get("asymmetry_index", 0.0)) > 0.5,
            subjective_strain_cr10=getattr(self, "current_strain_rating", None)
        )
        if hasattr(self, "epoch_filt_l"):
            self.epoch_filt_l.clear()
        if hasattr(self, "epoch_filt_r"):
            self.epoch_filt_r.clear()

    def _trigger_epoch_logging(self, epoch_idx=None, is_final=False):
        elapsed_min = (self.session_sample_count / max(1, self.sampling_rate)) / 60.0
        # Prompt interactive modal popup during live runs without blocking signal processing thread
        if self.isVisible() and self.mode == "live" and not os.getenv("TB_HEADLESS") and not is_final:
            dialog = BorgStrainPopupDialog(epoch_idx=epoch_idx or self.last_logged_window, elapsed_min=elapsed_min, parent=self)
            dialog.finished.connect(lambda _result, d=dialog, ei=epoch_idx: self._on_borg_popup_finished(d, ei))
            dialog.open()
        else:
            self._log_current_epoch(epoch_idx=epoch_idx, is_final=is_final)

    def _on_borg_popup_finished(self, dialog, epoch_idx):
        self.current_strain_rating = dialog.selected_rating
        self._log_current_epoch(epoch_idx=epoch_idx, is_final=False)
        # Reset rating to NULL after logging so future epochs require explicit evaluation
        self.current_strain_rating = None
        if hasattr(self, "strain_combo"):
            self.strain_combo.setCurrentIndex(0)

    def closeEvent(self, event):
        # Ensure clean disconnection, final Borg rating prompt, and Supabase sync if window is closed during streaming
        if self.stream_active:
            self.on_toggle_stream()
        elif self.local_logger and self.local_logger.is_active:
            self.local_logger.end_session(password=getattr(self, "_session_password", None))
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    window = TensionBudgetApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
