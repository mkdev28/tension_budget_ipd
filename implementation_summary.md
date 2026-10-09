# TensionBudget — Full Implementation Summary
> All changes made during this conversation session

---

## Overview of What Was Built

This session transformed TensionBudget from a **data-collection + offline-training** system into a **fully integrated real-time ML inference platform** that can be demonstrated end-to-end. Below is every significant change made, file by file.

---

## 1. `chordspy/tensionbudget_app.py` — Live Application

This was the primary file modified. It started as a PyQt5 GUI for bilateral sEMG signal visualization and logging, and was upgraded with a full real-time ML forecasting and ergonomic feedback layer.

### 1a. ML Model Loading at Startup
**Added:** `_load_ml_model_weights()` method (~L187–200)

- On app startup, searches for `tensionbudget_model_weights.json` (first in CWD, then relative to the script's parent directory).
- Loads JSON containing: `population_intercept`, `population_coefficients`, `scaler_means`, `scaler_stds`, `user_intercept_offsets`, `residual_sigma`, and `alert_policy`.
- Sets `self.ml_loaded = True` on success; gracefully falls back to no-inference mode if file not found.

```python
# Key state variables initialized (L170–177)
self.prev_mdf_l = None
self.prev_mdf_r = None
self.prev_gap_freq_l = None
self.prev_gap_freq_r = None
self.prev_eindex_l = None
self.prev_eindex_r = None
self._load_ml_model_weights()
```

### 1b. ML Dashboard UI Panel
**Added:** "🧠 Real-Time Bayesian ML 5-Min Pain Forecast & Ergonomic Feedback Engine" QGroupBox (~L341–365)

Four live-updating labels:
| Label | What it Shows |
|---|---|
| `lbl_ml_forecast` | `Predicted Pain Score (t + 5 min): X.XX / 10.0 (Borg Scale)` |
| `lbl_ml_ci` | `90% Credible Interval: [lo, hi] \| residual σ = 0.85` |
| `lbl_ml_prob` | `P(Pain ≥ 4.0 in 5 min): XX.X%` |
| `lbl_instant_feedback` | Dynamic bullet-list ergonomic advice |

### 1c. Global Alert Banner
**Added:** `lbl_top_alert_banner` — a colored header strip across the top of the app that changes dynamically:
- 🔴 Red background — **EARLY WARNING ALERT** when `P(pain ≥ 4.0) ≥ 85%` or `CI_high ≥ 5.5`
- 🟡 Orange background — **MODERATE LOAD WARNING** when predicted score ≥ 2.5
- 🟢 Green background — **NORMAL MONITORING STATE** otherwise

### 1d. Live Bayesian Inference Loop
**Added:** Real-time inference in the `update_loop` / epoch processing path (~L880–1003)

Each time an epoch window completes, the app:

1. **Assembles a 27-feature vector** (`feat_dict`) from live computed metrics:
   - `eindex_live_left/right` — cumulative exposure index
   - `apdf_10/50/90_left/right` — amplitude probability distribution function percentiles
   - `gap_frequency_left/right` — micro-relaxation gaps per minute
   - `short_suma_penalty_left/right` — rapid burst penalty scores
   - `total_suma_bursts_left/right` — total SUMA contraction counts
   - `mdf_hz_left/right` — median frequency (spectral)
   - `fatigue_slope_left/right` — spectral MDF trend slope (Hz/min)
   - `asymmetry_index_ai` — bilateral load laterality index
   - `calibration_rms_left/right` — personalized RMS reference
   - **6 trend features** (`mdf_trend`, `gap_freq_trend`, `eindex_trend` × left/right) — delta from previous epoch

2. **Runs Bayesian inference** using the closed-form formula:
   ```
   ŷ = α_pop + α_user + Σ β_k × ((x_k - μ_k) / σ_k)
   ```
   - `α_pop` = population intercept (from weights file)
   - `α_user` = per-user random effect offset (partial pooling)
   - `β_k` = feature coefficient for predictor k
   - `μ_k`, `σ_k` = training scaler mean and std for feature k

3. **Clamps** the prediction to `[0.0, 10.0]` (Borg CR-10 physical bounds)

4. **Computes 90% Credible Interval** using: `[ŷ - 1.645σ, ŷ + 1.645σ]`

5. **Computes P(Pain ≥ 4.0)** via Normal CDF evaluated at `z = (4.0 - ŷ) / σ`

6. **Triggers alert banner** state (red/orange/green) based on threshold policy

### 1e. Instant Ergonomic Feedback Engine
**Added:** Real-time bullet-point feedback (~L990–1003)

Rules evaluated on every epoch:
- If `|AI| > 0.25` → "Asymmetry Detected: Excessive load on RIGHT/LEFT trapezius. Check mouse/screen position."
- If `gap_frequency < 5.0 gaps/min` → "Low Micro-Rest Gaps: Perform a 10-second shoulder drop."
- If `fatigue_slope < -0.3 Hz/min` → "Downward Spectral Fatigue: Muscle fiber conduction velocity slowing."
- If `SUMA penalty > 0.3` → "Rapid Contraction Bursts: High short-duration effort cycles detected."
- Otherwise → "Good Ergonomic Posture: Load and rest gaps within optimal baseline."

---

## 2. `train_bayesian_model.py` — ML Training Pipeline

This script was significantly reworked and hardened. It trains the `RegularizedBayesianHierarchicalModel` and exports weights to `tensionbudget_model_weights.json`.

### 2a. Synthetic Data Generator Fallback
**Added:** `generate_synthetic_training_dataset()` function (~L61–165)

- Creates realistic physiological telemetry across configurable N subjects (default: 6), sessions/subject (default: 3), epochs/session (default: 6).
- Simulates APDF, EIndex, SUMA, MDF, gap frequency, asymmetry, calibration RMS with realistic noise.
- Ensures the training pipeline **never fails** even with no real data present.
- Labeled with `subjective_strain_cr10` ratings (Borg CR-10 scale) tied to simulated load factor.

### 2b. Flexible Multi-Source Dataset Loader
**Added:** `load_raw_dataframe(data_source=None)` function (~L167–212)

Supports **4 data sources** in priority order:
| Priority | Source | Description |
|---|---|---|
| 1 | Explicit CSV path | Direct path to `Supabase Snippet Untitled query.csv` or any `.csv` |
| 2 | Explicit SQLite path | Queries `session_features` table from a `.db` file |
| 3 | Directory scan | Recursively finds all `session_features.csv` under `output_logs/` |
| 4 | Synthetic fallback | Auto-generates realistic training data |

The directory scanner also reads `session_metadata.json` and `user_profile.json` to populate `user_name` labels for hierarchical grouping.

### 2c. Feature Engineering Pipeline
**Added/Verified:** `prepare_dataset()` function (~L214–283)

- Filters out synthetic tester accounts: `['FeatureTester', 'PayloadTester', 'SchemaTester']`
- Filters to rows with `strain_reported == 1` (Borg-labeled epochs only)
- Sorts chronologically by `session_id` → `epoch_index`
- Computes **6 trend features** using per-session `.diff()`:
  - `mdf_trend_left/right`, `gap_freq_trend_left/right`, `eindex_trend_left/right`
- Constructs **t+5 min target** by shifting `subjective_strain_cr10` by `-1` per session group
- Imputes missing predictor values with column medians

### 2d. Hierarchical Bayesian Model Class
**Verified/Refined:** `RegularizedBayesianHierarchicalModel` class (~L285–380)

- Uses `sklearn.linear_model.BayesianRidge` as the core estimator (Horseshoe/L2 shrinkage priors)
- **Partial pooling** for user random intercepts: shrinks users with few sessions toward overall mean
  - Pooling weight: `w_u = n_u / (n_u + 5.0)`
- Fits on residual targets (after removing per-user intercept offsets)
- `predict()` adds back user intercepts and clamps to `[0.0, 10.0]`
- `sigma` estimated from the posterior precision of BayesianRidge

### 2e. LOSO Cross-Validation
**Verified:** 6-fold Leave-One-Subject-Out cross-validation

- For each held-out subject, trains on all others, predicts on held-out
- Reports: MAE, RMSE per fold, and averaged across folds
- Evaluates **alert policy simulation**: What fraction of high-strain epochs (≥4.0) trigger correct alerts at different probability thresholds

### 2f. Weight Export
**Verified:** `export_model_weights()` function

Writes `tensionbudget_model_weights.json` containing:
```json
{
  "population_intercept": <float>,
  "population_coefficients": { "<feature>": <float>, ... },
  "user_intercept_offsets": { "<user_name>": <float>, ... },
  "scaler_means": { "<feature>": <float>, ... },
  "scaler_stds": { "<feature>": <float>, ... },
  "residual_sigma": <float>,
  "alert_policy": {
    "confidence_required": 0.85,
    "hard_safety_ceiling": 5.5
  }
}
```

### 2g. CLI Support
**Added:** `argparse` CLI interface at script bottom

```bash
# Train on local session CSVs:
python train_bayesian_model.py

# Train on specific CSV (e.g., Supabase export):
python train_bayesian_model.py --source "Supabase Snippet Untitled query.csv"

# Train on SQLite cloud cache:
python train_bayesian_model.py --source tensionbudget_cloud.db

# Train on a directory of session logs:
python train_bayesian_model.py --source output_logs/
```

---

## 3. `tensionbudget_model_weights.json` — Model Artifact

A model weights file was generated and verified to exist at:  
[`tensionbudget_model_weights.json`](file:///Users/kyrakarkaria/Desktop/tension_budget_ipd/tensionbudget_model_weights.json)

This is the file loaded at app startup for live inference. Size: ~5KB with all 27 feature coefficients, user offsets, scaler parameters, and alert policy.

---

## 4. Planned (Not Yet Implemented) — Supabase Live Fetch

The final request of the session was to add **direct Supabase querying** without manual CSV download. The plan was confirmed but not yet coded:

- Use Supabase REST API (`/rest/v1/v_ml_training_pairs`) via `requests.get()` with `SUPABASE_ANON_KEY` from environment
- Add a `fetch_cloud_data()` function to `train_bayesian_model.py`
- Trigger with `python train_bayesian_model.py --source supabase`

> [!IMPORTANT]
> This feature is pending implementation. The training script currently falls back to local CSVs, SQLite (`tensionbudget_cloud.db`), or synthetic data.

---

## Complete End-to-End Demo Flow (Post-Implementation)

```
[sEMG Hardware / LSL Stream or CSV Replay]
              ↓
[tensionbudget_app.py — Live Bilateral Signal Display]
              ↓
[Per-Epoch Feature Computation (every ~5 min or configurable)]
  - APDF 10/50/90th, EIndex, SUMA, MDF, Gap Frequency, AI
              ↓
[Live Bayesian Inference Engine]
  ŷ = α_pop + α_user + Σ β_k × (x_k - μ_k)/σ_k
  → Predicted Borg Score (t + 5 min)
  → 90% Credible Interval
  → P(Pain ≥ 4.0)
              ↓
[Alert Banner + Ergonomic Feedback (updates every epoch)]
  🔴 Early Warning Alert / 🟡 Moderate Warning / 🟢 Normal
  + Bullet-point actionable ergonomic guidance
              ↓
[LocalSessionLogger → session_features.csv → Supabase Cloud]
              ↓
[train_bayesian_model.py → Retrain → Export new weights.json]
```

---

## Files Modified

| File | Status | Key Changes |
|---|---|---|
| [`chordspy/tensionbudget_app.py`](file:///Users/kyrakarkaria/Desktop/tension_budget_ipd/chordspy/tensionbudget_app.py) | ✅ Modified | ML loading, inference loop, alert banner, ergonomic feedback panel |
| [`train_bayesian_model.py`](file:///Users/kyrakarkaria/Desktop/tension_budget_ipd/train_bayesian_model.py) | ✅ Modified | Synthetic fallback, multi-source loader, trend features, CLI |
| [`tensionbudget_model_weights.json`](file:///Users/kyrakarkaria/Desktop/tension_budget_ipd/tensionbudget_model_weights.json) | ✅ Generated | ML model artifact used by the live app |
