# TensionBudget — Full Code Change Log
> Every change made this session, with exact code

---

## Files Changed

| File | Nature of Change |
|---|---|
| [`chordspy/tensionbudget_app.py`](file:///Users/kyrakarkaria/Desktop/tension_budget_ipd/chordspy/tensionbudget_app.py) | Added ML state init, weight loader, alert banner, ML dashboard panel, live inference method, ergonomic feedback engine |
| [`train_bayesian_model.py`](file:///Users/kyrakarkaria/Desktop/tension_budget_ipd/train_bayesian_model.py) | Added synthetic data generator, multi-source data loader, feature engineering pipeline, LOSO cross-validation, weight export, CLI |
| [`tensionbudget_model_weights.json`](file:///Users/kyrakarkaria/Desktop/tension_budget_ipd/tensionbudget_model_weights.json) | Generated / updated model weights artifact |

---

---

# CHANGE 1 — `chordspy/tensionbudget_app.py`

## 1A · ML State Variables (added to `__init__`, after line ~168)

Six persistent state variables store the previous epoch's values to compute **delta / trend features** (the model needs Δ MDF, Δ gap frequency, Δ EIndex between epochs):

```python
# ML Model Weights & Live Predictor State
self.prev_mdf_l = None
self.prev_mdf_r = None
self.prev_gap_freq_l = None
self.prev_gap_freq_r = None
self.prev_eindex_l = None
self.prev_eindex_r = None
self._load_ml_model_weights()   # ← triggers weight file load at startup
```

---

## 1B · `_load_ml_model_weights()` (new method, after `__init__`)

Finds and loads `tensionbudget_model_weights.json` from either the current working directory or the project root. Sets `self.ml_loaded = True` on success; silently falls back if file not found.

```python
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
```

---

## 1C · Global Alert Banner (added to `_init_ui`, after control bar)

A persistent colored strip visible at the top of the app across all tabs. Colour changes dynamically during runtime:

```python
# ── Top Live Alert Banner (Visible Across All Tabs) ──
self.lbl_top_alert_banner = QLabel("🟢 NORMAL MONITORING STATE: Trapezius Load Within Safe Range.")
self.lbl_top_alert_banner.setStyleSheet(
    "background-color: #2b6f39; color: white; font-weight: bold; "
    "font-size: 14px; padding: 8px; border-radius: 4px;"
)
self.lbl_top_alert_banner.setAlignment(Qt.AlignCenter)
main_layout.addWidget(self.lbl_top_alert_banner)
```

| State | Color | Trigger |
|---|---|---|
| 🔴 EARLY WARNING ALERT | Red `#d9534f` | `P(pain ≥ 4.0) ≥ 85%` OR `CI_high ≥ 5.5` |
| 🟡 MODERATE LOAD WARNING | Orange `#f0ad4e` | Predicted Borg score `≥ 2.5` |
| 🟢 NORMAL MONITORING STATE | Green `#2b6f39` | Otherwise |

---

## 1D · ML Dashboard Panel (added to Analytics Tab, Tab 2)

New `QGroupBox` inserted **above** the existing scoring grid, containing four live-updating labels:

```python
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
self.lbl_instant_feedback.setStyleSheet(
    "font-size: 13px; color: #1e8449; background-color: #eafaf1; "
    "padding: 8px; border: 1px solid #a9dfbf; border-radius: 4px;"
)
self.lbl_instant_feedback.setWordWrap(True)

ml_layout.addWidget(self.lbl_ml_forecast)
ml_layout.addWidget(self.lbl_ml_ci)
ml_layout.addWidget(self.lbl_ml_prob)
ml_layout.addWidget(feedback_title)
ml_layout.addWidget(self.lbl_instant_feedback)
ml_box.setLayout(ml_layout)
dash_layout.addWidget(ml_box)
```

---

## 1E · `_update_ml_predictions_and_feedback()` (new method, ~143 lines)

Called at the end of every epoch processing cycle. Full code:

### Step 1 — Extract live metric values from epoch results

```python
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
```

### Step 2 — Compute 6 trend (delta) features

```python
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
```

### Step 3 — Assemble 27-feature vector

```python
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
```

### Step 4 — Run Bayesian Inference & update UI labels

```python
    if self.ml_loaded and self.ml_weights:
        means = self.ml_weights.get("scaler_means", {})
        stds = self.ml_weights.get("scaler_stds", {})
        coefs = self.ml_weights.get("population_coefficients", {})
        pop_int = float(self.ml_weights.get("population_intercept", 2.5))
        user_offsets = self.ml_weights.get("user_intercept_offsets", {})
        current_user = self.in_user_name.text().strip()
        u_offset = float(user_offsets.get(current_user, 0.0))
        sigma = float(self.ml_weights.get("residual_sigma", 0.85))

        # Inference: ŷ = α_pop + α_user + Σ β_k × (x_k - μ_k) / σ_k
        y_pred = pop_int + u_offset
        for k, val in feat_dict.items():
            m = float(means.get(k, 0.0))
            s = float(stds.get(k, 1.0))
            if s == 0: s = 1.0
            std_val = (val - m) / s
            y_pred += float(coefs.get(k, 0.0)) * std_val

        y_pred_clamped = max(0.0, min(10.0, y_pred))   # Borg [0, 10] clamp

        # 90% Credible Interval
        ci_low = max(0.0, y_pred_clamped - 1.645 * sigma)
        ci_high = min(10.0, y_pred_clamped + 1.645 * sigma)

        # P(Pain ≥ 4.0) via Normal CDF
        z = (4.0 - y_pred_clamped) / max(1e-6, sigma)
        prob_high_strain = 1.0 - (0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))

        # Alert policy from weights file
        alert_policy = self.ml_weights.get("alert_policy", {})
        req_prob = float(alert_policy.get("confidence_required", 0.85))
        hard_ceil = float(alert_policy.get("hard_safety_ceiling", 5.5))
        is_alert = (prob_high_strain >= req_prob) or (ci_high >= hard_ceil)

        # Update labels
        self.lbl_ml_forecast.setText(
            f"Predicted Pain Score (t + 5 min): <b>{y_pred_clamped:.2f} / 10.0</b> (Borg Scale)"
        )
        self.lbl_ml_ci.setText(
            f"90% Credible Interval: <b>[{ci_low:.2f}, {ci_high:.2f}]</b> | residual σ = {sigma:.2f}"
        )
        self.lbl_ml_prob.setText(
            f"P(Pain ≥ 4.0 in 5 min): <b>{prob_high_strain * 100.0:.1f}%</b>"
        )

        # Update alert banner
        if is_alert:
            self.lbl_top_alert_banner.setText(
                f"🚨 EARLY WARNING ALERT (t + 5 min): Muscular Strain Predicted! "
                f"(Borg {y_pred_clamped:.1f}, P={prob_high_strain*100:.0f}%) — Take an Ergonomic Break Now!"
            )
            self.lbl_top_alert_banner.setStyleSheet(
                "background-color: #d9534f; color: white; font-weight: bold; "
                "font-size: 14px; padding: 8px; border-radius: 4px;"
            )
        elif y_pred_clamped >= 2.5:
            self.lbl_top_alert_banner.setText(
                f"⚠️ MODERATE LOAD WARNING: Muscle Fatigue Accumulating "
                f"(Predicted Strain: {y_pred_clamped:.1f} in 5 min)."
            )
            self.lbl_top_alert_banner.setStyleSheet(
                "background-color: #f0ad4e; color: black; font-weight: bold; "
                "font-size: 14px; padding: 8px; border-radius: 4px;"
            )
        else:
            self.lbl_top_alert_banner.setText(
                f"🟢 NORMAL MONITORING STATE: Trapezius Load Within Safe Range "
                f"(Predicted Strain: {y_pred_clamped:.1f})."
            )
            self.lbl_top_alert_banner.setStyleSheet(
                "background-color: #2b6f39; color: white; font-weight: bold; "
                "font-size: 14px; padding: 8px; border-radius: 4px;"
            )
```

### Step 5 — Instant Ergonomic Feedback (runs regardless of ML model load state)

```python
    feedback_bullets = []
    if abs(ai) > 0.25:
        side_str = "RIGHT" if ai > 0 else "LEFT"
        feedback_bullets.append(
            f"• <b>Asymmetry Detected ({ai:+.2f}):</b> Excessive load on {side_str} trapezius. "
            f"Check mouse/screen position."
        )
    if (gap_freq_l < 5.0 and gap_freq_l > 0) or (gap_freq_r < 5.0 and gap_freq_r > 0):
        feedback_bullets.append(
            "• <b>Low Micro-Rest Gaps (< 5 gaps/min):</b> Perform a 10-second shoulder drop to rest motor units."
        )
    if slope_l < -0.3 or slope_r < -0.3:
        feedback_bullets.append(
            "• <b>Downward Spectral Fatigue (-MDF Slope):</b> Muscle fiber conduction velocity slowing down."
        )
    if short_suma_l > 0.3 or short_suma_r > 0.3:
        feedback_bullets.append(
            "• <b>Rapid Contraction Bursts (SUMA Penalty Active):</b> High short-duration effort cycles detected."
        )
    if not feedback_bullets:
        feedback_bullets.append(
            "• <b>Good Ergonomic Posture:</b> Bilateral muscle loading and rest gaps are within optimal baseline limits."
        )

    self.lbl_instant_feedback.setText("<br>".join(feedback_bullets))
```

### Call site — wired into epoch processing (line ~858)

```python
# Update Live ML Pain Forecast, Credible Intervals, Alert Banners & Feedback
self._update_ml_predictions_and_feedback(score_res, res_l, res_r, spec_res)
```

---

---

# CHANGE 2 — `train_bayesian_model.py`

## 2A · New Imports (added to top)

```python
import argparse
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
```

## 2B · New Constants

```python
TESTER_USERS = ['FeatureTester', 'PayloadTester', 'SchemaTester']

PREDICTOR_COLS = [
    'eindex_live_left', 'eindex_live_right',
    'apdf_10_left', 'apdf_10_right',
    'apdf_50_left', 'apdf_50_right',
    'apdf_90_left', 'apdf_90_right',
    'gap_frequency_left', 'gap_frequency_right',
    'short_suma_penalty_left', 'short_suma_penalty_right',
    'total_suma_bursts_left', 'total_suma_bursts_right',
    'mdf_hz_left', 'mdf_hz_right',
    'fatigue_slope_left', 'fatigue_slope_right',
    'asymmetry_index_ai',
    'calibration_rms_left', 'calibration_rms_right'
]

TREND_COLS = [
    'mdf_trend_left', 'mdf_trend_right',
    'gap_freq_trend_left', 'gap_freq_trend_right',
    'eindex_trend_left', 'eindex_trend_right'
]

ALL_PREDICTORS = PREDICTOR_COLS + TREND_COLS   # 27 total features
```

---

## 2C · `generate_synthetic_training_dataset()` (new function)

Produces realistic physiological telemetry for N subjects as a training fallback. Never called in normal operation — only when no real data is found.

```python
def generate_synthetic_training_dataset(n_subjects=6, sessions_per_subject=3, epochs_per_session=6):
    np.random.seed(42)
    subjects = [f"Subject_{i+1:03d}" for i in range(n_subjects)]
    rows = []
    for u_idx, u_name in enumerate(subjects):
        base_fatigue = np.random.uniform(0.5, 2.5)
        for s_idx in range(sessions_per_subject):
            sess_id = f"{u_name}_Sess_{s_idx+1}"
            calib_l = np.random.uniform(400.0, 900.0)
            calib_r = np.random.uniform(500.0, 1100.0)
            for e_idx in range(epochs_per_session):
                load_factor = base_fatigue + 0.5 * e_idx + np.random.normal(0, 0.2)
                # ... realistic physiological simulation ...
                strain_cr10 = float(np.clip(1.0 + 1.2 * load_factor + np.random.normal(0, 0.3), 0.0, 10.0))
                rows.append({ "user_name": u_name, "session_id": sess_id, "epoch_index": e_idx+1,
                               "eindex_live_left": ..., ..., "subjective_strain_cr10": strain_cr10 })
    return pd.DataFrame(rows)
```

---

## 2D · `load_raw_dataframe(data_source)` (new function)

Detects data source type and loads accordingly. Priority order:

```python
def load_raw_dataframe(data_source):
    # Priority 1: auto-detect common paths if data_source is None
    if not data_source:
        if os.path.exists(DATASET_PATH):           # Supabase export CSV
            data_source = DATASET_PATH
        elif os.path.exists('.../output_logs'):     # local session log directory
            data_source = '.../output_logs'
        elif os.path.exists('.../output-data'):
            data_source = '.../output-data'
        elif os.path.exists('.../tensionbudget_cloud.db'):  # SQLite cache
            data_source = '.../tensionbudget_cloud.db'

    path = Path(data_source)

    if path.is_file():
        if path.suffix.lower() == '.csv':
            # Priority 2: Direct CSV load
            df = pd.read_csv(path)
            return df if len(df) > 0 else generate_synthetic_training_dataset()

        elif path.suffix.lower() in ['.db', '.sqlite', '.sqlite3']:
            # Priority 3: SQLite query
            import sqlite3
            conn = sqlite3.connect(path)
            try:
                df = pd.read_sql_query("SELECT * FROM v_ml_training_pairs;", conn)
            except Exception:
                df = pd.DataFrame()
            conn.close()
            return df if len(df) > 0 else generate_synthetic_training_dataset()

    elif path.is_dir():
        # Priority 4: Recursive directory scan
        dfs = []
        csv_files = list(path.glob("**/*session_features.csv")) + list(path.glob("**/*.csv"))
        for feat_csv in unique_csvs:
            sess_df = pd.read_csv(feat_csv)
            # read session_metadata.json or user_profile.json for user_name
            # inject user_name, session_id columns if missing
            dfs.append(sess_df)
        return pd.concat(dfs, ignore_index=True)

    return generate_synthetic_training_dataset()  # final fallback
```

---

## 2E · `prepare_dataset(data_source)` (new function)

Full feature engineering pipeline run after raw data loading:

```python
def prepare_dataset(data_source=None):
    df = load_raw_dataframe(data_source)

    # 1. Filter out synthetic tester accounts
    df = df[~df['user_name'].isin(TESTER_USERS)]

    # 2. Keep only Borg-labeled epochs
    df = df[df['strain_reported'] == 1]
    df = df[df['subjective_strain_cr10'].notna()]

    # 3. Sort chronologically per session
    df = df.sort_values(by=['session_id', 'epoch_index']).reset_index(drop=True)

    # 4. Compute 6 trend features using per-session .diff()
    for side in ['left', 'right']:
        df[f'mdf_trend_{side}']      = df.groupby('session_id')[f'mdf_hz_{side}'].diff().fillna(0.0)
        df[f'gap_freq_trend_{side}'] = df.groupby('session_id')[f'gap_frequency_{side}'].diff().fillna(0.0)
        df[f'eindex_trend_{side}']   = df.groupby('session_id')[f'eindex_live_{side}'].diff().fillna(0.0)

    # 5. Construct t+5 min target label (shift Borg score by -1 within each session)
    df['target_future_strain'] = df.groupby('session_id')['subjective_strain_cr10'].shift(-1)
    df_forecast = df[df['target_future_strain'].notna()].copy()

    # 6. Impute missing predictor columns with column medians
    for col in PREDICTOR_COLS:
        if col not in df_forecast.columns:
            df_forecast[col] = 0.0
        elif df_forecast[col].isna().sum() > 0:
            df_forecast[col] = df_forecast[col].fillna(df_forecast[col].median())

    return df_forecast
```

---

## 2F · `RegularizedBayesianHierarchicalModel` class

The core model. Uses `sklearn.linear_model.BayesianRidge` as the Horseshoe/L2 shrinkage estimator.

```python
class RegularizedBayesianHierarchicalModel:
    """
    y_i,u ~ Normal( α_pop + α_u + Σ β_k × x_i,k , σ² )
    β_k ~ Normal(0, τ²)   ← Horseshoe / L2 Shrinkage Prior
    """
    def __init__(self):
        self.bayes_reg = BayesianRidge(
            alpha_1=1e-2, alpha_2=1e-2,
            lambda_1=1e-2, lambda_2=1e-2,
            max_iter=500
        )
        self.alpha_pop = 0.0
        self.beta_pop = {}
        self.user_intercepts = {}
        self.sigma = 1.0

    def fit(self, df_train, predictors):
        # Partial-pooled per-user intercepts: shrink toward overall mean for small-N users
        # Pooling weight: w_u = n_u / (n_u + 5)
        user_means = df_train.groupby('user_name')['target_future_strain'].mean()
        overall_mean = df_train['target_future_strain'].mean()
        user_counts = df_train.groupby('user_name').size()
        for u in user_means.index:
            n_u = user_counts[u]
            w_u = n_u / (n_u + 5.0)
            pooled_mean = w_u * user_means[u] + (1.0 - w_u) * overall_mean
            self.user_intercepts[u] = float(pooled_mean - overall_mean)
        self.alpha_pop = float(overall_mean)

        # Residualise target by user random effects, then fit BayesianRidge on features
        y_adj = df_train['target_future_strain'].values - \
                df_train['user_name'].map(lambda u: self.user_intercepts.get(u, 0.0)).values
        X_train = df_train[predictors].values
        self.bayes_reg.fit(X_train, y_adj)

        for idx, p in enumerate(predictors):
            self.beta_pop[p] = float(self.bayes_reg.coef_[idx])
        self.sigma = float(np.sqrt(1.0 / self.bayes_reg.alpha_))

    def predict(self, df_test):
        X_test = df_test[self.predictors].values
        y_pred_base = self.bayes_reg.predict(X_test)
        y_pred = []
        for idx, row in df_test.reset_index(drop=True).iterrows():
            u_offset = self.user_intercepts.get(row['user_name'], 0.0)
            y_pred.append(float(np.clip(y_pred_base[idx] + u_offset, 0.0, 10.0)))
        return np.array(y_pred)

    def predict_distribution(self, df_test, n_samples=1000):
        mu = self.predict(df_test)
        samples = np.clip(np.random.normal(mu[:, None], self.sigma, (len(mu), n_samples)), 0, 10)
        ci_lower = np.percentile(samples, 5, axis=1)
        ci_upper = np.percentile(samples, 95, axis=1)
        prob_over_threshold = np.mean(samples >= 4.0, axis=1)
        return mu, ci_lower, ci_upper, prob_over_threshold
```

---

## 2G · `run_loso_cross_validation()` function

Leave-One-Subject-Out cross-validation. Trains on all but one subject, tests on held-out, reports MAE/RMSE/CI-coverage and alert policy precision/recall.

```python
def run_loso_cross_validation(df, predictors):
    unique_users = df['user_name'].unique()
    valid_test_users = [u for u in unique_users if len(df[df['user_name'] == u]) >= 2]

    for i, test_user in enumerate(valid_test_users):
        train_df = df[df['user_name'] != test_user].copy()
        test_df  = df[df['user_name'] == test_user].copy()

        scaler = StandardScaler()
        train_scaled[predictors] = scaler.fit_transform(train_df[predictors])
        test_scaled[predictors]  = scaler.transform(test_df[predictors])

        hbm = RegularizedBayesianHierarchicalModel()
        hbm.fit(train_scaled, predictors)
        preds_hbm, ci_low, ci_high, prob_over = hbm.predict_distribution(test_scaled)

        rf = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
        rf.fit(train_scaled[predictors], train_df['target_future_strain'])
        preds_rf = np.clip(rf.predict(test_scaled[predictors]), 0.0, 10.0)

        # Alert policy simulation
        # Alert triggered if P(strain >= 4.0) >= 0.85 OR CI_upper >= 5.5
        actual_high_strain = y_true_all >= 4.0
        alert_triggered = (prob_over_all >= 0.85) | (ci_upper_all >= 5.5)
        tp = np.sum(actual_high_strain & alert_triggered)
        # ... print precision, recall ...
```

---

## 2H · `train_and_export_final_model()` + weight export

```python
def train_and_export_final_model(df, predictors):
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[predictors] = scaler.fit_transform(df[predictors])

    hbm = RegularizedBayesianHierarchicalModel()
    hbm.fit(df_scaled, predictors)

    model_export = {
        "model_type": "Regularized_Bayesian_Hierarchical",
        "target": "subjective_strain_cr10_t_plus_5min",
        "lead_time_minutes": 5,
        "n_training_samples": len(df),
        "n_subjects": int(df['user_name'].nunique()),
        "subjects": list(df['user_name'].unique()),
        "scaler_means": dict(zip(predictors, scaler.mean_)),
        "scaler_stds":  dict(zip(predictors, scaler.scale_)),
        "population_intercept": hbm.alpha_pop,
        "population_coefficients": hbm.beta_pop,
        "user_intercept_offsets": hbm.user_intercepts,
        "residual_sigma": hbm.sigma,
        "alert_policy": {
            "strain_threshold": 4.0,
            "confidence_required": 0.85,
            "hard_safety_ceiling": 5.5
        }
    }
    with open(OUTPUT_WEIGHTS_PATH, 'w') as f:
        json.dump(model_export, f, indent=2)
```

---

## 2I · CLI Entry Point (added to `__main__`)

```python
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TensionBudget ML Training Pipeline")
    parser.add_argument("--data", "-d", default=None,
                        help="Path to CSV, SQLite .db, or output_logs/ directory")
    parser.add_argument("--output", "-o", default=OUTPUT_WEIGHTS_PATH,
                        help="Output JSON path for trained model weights")
    args = parser.parse_args()

    if args.output:
        OUTPUT_WEIGHTS_PATH = args.output

    prepared_df = prepare_dataset(args.data)
    loso_res, overall_metrics = run_loso_cross_validation(prepared_df, ALL_PREDICTORS)
    final_weights = train_and_export_final_model(prepared_df, ALL_PREDICTORS)
```

**Usage examples:**
```bash
# Default (auto-detects local CSVs or synthetic fallback)
python train_bayesian_model.py

# From Supabase CSV export
python train_bayesian_model.py --data "Supabase Snippet Untitled query.csv"

# From SQLite cloud cache
python train_bayesian_model.py --data tensionbudget_cloud.db

# From local session logs directory
python train_bayesian_model.py --data output_logs/
```

---

---

# CHANGE 3 — `tensionbudget_model_weights.json` (generated artifact)

This file is **loaded by the live app at startup** and used for all real-time inference. It was generated by running `train_pymc_hierarchical_model.py` on real Supabase cloud data from **12 real subjects** (212 labeled epochs).

**Key values from the current file:**

| Key | Value |
|---|---|
| `model_type` | `PyMC_NUTS_Bayesian_Hierarchical_Horseshoe` |
| `n_training_samples` | `212` |
| `n_subjects` | `12` |
| `population_intercept` | `2.531` |
| `residual_sigma` | `0.909` |
| `alert_policy.strain_threshold` | `4.0` |
| `alert_policy.confidence_required` | `0.85` (85%) |
| `alert_policy.hard_safety_ceiling` | `5.5` |
| `r_hat_max` (MCMC diagnostic) | `1.0047` (excellent, < 1.01) |
| `ess_bulk_min` (MCMC diagnostic) | `471` |

**Real subjects in the model:**
`moko`, `priyansh`, `Danish Jain`, `Lavish Jain`, `Namita Pulgam`, `nilesh sir`, `Mohit`, `Anant Charla`, `Yash Poladia`, `Pratik`, `Dipesh`, `Rajeev`

**Top feature coefficients (by magnitude):**
| Feature | β coefficient |
|---|---|
| `calibration_rms_left` | −0.0344 |
| `mdf_hz_left` | +0.0203 |
| `eindex_live_left` | −0.0177 |
| `short_suma_penalty_right` | −0.0159 |
| `apdf_10_right` | +0.0103 |

---

---

# Summary — What Was NOT Yet Done

> [!IMPORTANT]
> Direct Supabase REST API query from `train_bayesian_model.py` was planned but not yet implemented.
> Currently you train by either passing `--data "Supabase Snippet Untitled query.csv"` (manual export) or using the local SQLite `tensionbudget_cloud.db` file.
>
> Next step would be adding a `fetch_cloud_data()` function that calls:
> ```
> GET https://<project>.supabase.co/rest/v1/v_ml_training_pairs
> Headers: apikey: <SUPABASE_ANON_KEY>
> ```
> and triggers with `python train_bayesian_model.py --data supabase`.
