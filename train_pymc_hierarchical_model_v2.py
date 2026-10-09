"""

train_pymc_hierarchical_model_v2.py

TensionBudget — ML Pipeline v2



Steps implemented:

  STEP 0  — Secret loading from .env

  STEP 1  — Data audit (user/session/epoch counts, strain distribution)

  STEP 2a — Target/trend construction bug fix

  STEP 2b — Remove calibration_rms from ALL_PREDICTORS

  STEP 2c — Calibration QC gate (pre-specified rules, applied identically)

  STEP 2d — Outlier policy declaration (Geeta stays)

  STEP 2e — Personalized-protocol eligibility table

  STEP 3a — Winsorize heavy-tailed features (1st/99th pct, training data only)

  STEP 3b — Near-constant check (binary indicator if std < NEAR_CONST_STD)

  STEP 3c — Per-subject causal expanding z-score (+ global-scaler variant)



Hard rules:

  - Original train_pymc_hierarchical_model.py is NOT modified.

  - No DSP/analytics/scoring code touched.

  - All thresholds in the CONFIG block below — no magic numbers elsewhere.

  - Feature sets are fixed in advance.

  - Chronological split per subject only.

"""



import os

import sys

import json

import math

import datetime

import requests

import numpy as np

import pandas as pd

from dotenv import load_dotenv



# ─────────────────────────────────────────────────────────────────────────────

# 0. ENVIRONMENT & SECURITY

# ─────────────────────────────────────────────────────────────────────────────

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")

SUPABASE_KEY = os.getenv("SUPABASE_KEY")



if not SUPABASE_URL or not SUPABASE_KEY:

    print("FATAL: SUPABASE_URL or SUPABASE_KEY missing from .env. Aborting.")

    sys.exit(1)



# ─────────────────────────────────────────────────────────────────────────────

# 1. CONFIG  (all thresholds live here — no magic numbers in logic below)

# ─────────────────────────────────────────────────────────────────────────────

CONFIG = {

    # Users to exclude as synthetic test accounts

    "TESTER_USERS": [

        "FeatureTester", 

        "PayloadTester", 

        "SchemaTester",

        "cloud_smoke_desktop_subject",

        "cloud_smoke_web_subject"

    ],



    # Step 1: minimum labeled epochs to avoid flagging

    "MIN_LABELED_EPOCHS": 8,



    # Step 2c: QC bounds for calibration sessions

    # Rule 1: null / zero is always flagged (no threshold needed)

    # Rule 2: population robust z-score of log(L/R) — flag if |z| > this

    "QC_RATIO_MAD_Z": 3.0,

    # Rule 3: within-subject deviation on log(L/R) — flag if |dev| > this

    #         Only applied to subjects with >= 3 sessions

    "QC_WITHIN_SUBJECT_MIN_SESSIONS": 3,

    "QC_WITHIN_SUBJECT_LOG_DEV": 1.5,

    # Rule 4: weak-shrug check — flag if rms < fraction * pop_median for that side

    "QC_MIN_FRACTION_OF_POP_MEDIAN": 0.15,



    # Step 2e: personalization eligibility

    "K_CAL_VALUES": [3, 5],

    "MIN_SCORED_AFTER_CAL": 3,



    # Step 3a: winsorization — features clipped at these percentiles (training data only)

    "WINSOR_LOW_PCT": 1.0,

    "WINSOR_HIGH_PCT": 99.0,

    "WINSOR_FEATURES": [

        "apdf_10_left", "apdf_10_right",

        "apdf_50_left", "apdf_50_right",

        "apdf_90_left", "apdf_90_right",

        "fatigue_slope_left", "fatigue_slope_right",

        "mdf_trend_left", "mdf_trend_right",

        "gap_freq_trend_left", "gap_freq_trend_right",

        "eindex_trend_left", "eindex_trend_right",

        "total_suma_bursts_left", "total_suma_bursts_right",

    ],



    # Step 3b: near-constant threshold — if training std < this, binarize (value > 0)

    "NEAR_CONST_STD": 0.05,



    # Step 3c: per-subject causal scaling

    # Minimum past epochs required before switching from global fallback to subject z-score

    "MIN_HISTORY": 3,

    # Flag to compare global StandardScaler vs per-subject causal scaler

    "USE_GLOBAL_SCALER": False,  # set True to force global scaler for comparison



    # Feature sets (calibration_rms excluded)

    "SET_A": [

        "eindex_live_left", "eindex_live_right",

        "gap_frequency_left", "gap_frequency_right",

        "short_suma_penalty_left", "short_suma_penalty_right",

        "mdf_trend_left", "mdf_trend_right",

    ],

    "SET_B_EXTRA": [

        "fatigue_slope_left", "fatigue_slope_right",

        "apdf_50_left", "apdf_50_right",

        "asymmetry_index_ai",

    ],

    "SET_C_EXTRA": [

        "apdf_10_left", "apdf_10_right",

        "apdf_90_left", "apdf_90_right",

        "total_suma_bursts_left", "total_suma_bursts_right",

        "mdf_hz_left", "mdf_hz_right",

        "gap_freq_trend_left", "gap_freq_trend_right",

        "eindex_trend_left", "eindex_trend_right",

    ],

}



# ─────────────────────────────────────────────────────────────────────────────

# 2. DATA FETCH

# ─────────────────────────────────────────────────────────────────────────────

def fetch_raw() -> pd.DataFrame:

    """Fetch the full v_ml_training_pairs view from Supabase."""

    headers = {

        "apikey": SUPABASE_KEY,

        "Authorization": f"Bearer {SUPABASE_KEY}",

    }

    resp = requests.get(

        f"{SUPABASE_URL}/rest/v1/v_ml_training_pairs?select=*",

        headers=headers,

    )

    resp.raise_for_status()

    df = pd.DataFrame(resp.json())

    if len(df) == 0:

        raise ValueError("Supabase returned 0 rows. Check API key and URL.")

    return df





# ─────────────────────────────────────────────────────────────────────────────

# 3. STEP 1 — DATA AUDIT

# ─────────────────────────────────────────────────────────────────────────────

def step1_audit(df_raw: pd.DataFrame) -> pd.DataFrame:

    print("\n" + "=" * 60)

    print("STEP 1 — DATA AUDIT")

    print("=" * 60)



    print(f"Total raw records fetched: {len(df_raw)}")



    # --- tester account rows ---

    tester_mask = df_raw["user_name"].isin(CONFIG["TESTER_USERS"])

    n_tester = tester_mask.sum()

    print(f"Tester account rows removed: {n_tester}")

    df = df_raw[~tester_mask].copy()



    # --- duplicate (session_id, epoch_index) ---

    dups = df.duplicated(subset=["session_id", "epoch_index"], keep=False)

    print(f"Duplicate (session_id, epoch_index) rows: {dups.sum()}")

    if dups.sum() > 0:

        print("  (These duplicates are KEPT; just reporting.)")



    # --- name unification ---

    df["user_name"] = df["user_name"].replace({"Lavish": "Lavish Jain"})

    print("Applied user-name unification: {'Lavish': 'Lavish Jain'}")



    # --- labeled subset ---

    df_labeled = df[(df["strain_reported"] == 1) & (df["subjective_strain_cr10"].notna())].copy()

    users = sorted(df_labeled["user_name"].unique())

    print(f"\nTotal distinct users with labeled data: {len(users)}")



    rows = []

    for u in users:

        ud = df_labeled[df_labeled["user_name"] == u]

        s = ud["subjective_strain_cr10"]

        rows.append({

            "User": u,

            "Sessions": ud["session_id"].nunique(),

            "Labeled Epochs": len(ud),

            "Mean Strain": round(s.mean(), 3),

            "Std Strain": round(s.std(), 3),

            "Min Strain": s.min(),

            "Max Strain": s.max(),

        })



    audit_df = pd.DataFrame(rows)

    print("\n--- Audit Table (strain_reported=1 rows only) ---")

    print(audit_df.to_string(index=False))

    print("-" * 50)



    flagged = audit_df[audit_df["Labeled Epochs"] < CONFIG["MIN_LABELED_EPOCHS"]]

    if len(flagged):

        print(f"\nFLAGGED (< {CONFIG['MIN_LABELED_EPOCHS']} labeled epochs):")

        print(flagged[["User", "Labeled Epochs"]].to_string(index=False))

    else:

        print(f"\nNo users flagged for < {CONFIG['MIN_LABELED_EPOCHS']} labeled epochs.")



    return df  # return full (not labeled-only) for step 2a





# ─────────────────────────────────────────────────────────────────────────────

# 4. STEP 2a — TARGET & TREND CONSTRUCTION BUG FIX

# ─────────────────────────────────────────────────────────────────────────────

def step2a_construct_targets(df_full: pd.DataFrame) -> pd.DataFrame:

    print("\n" + "=" * 60)

    print("STEP 2a — TARGET & TREND CONSTRUCTION (bug-fixed)")

    print("=" * 60)



    df_full = df_full.copy()

    df_full["session_id"] = df_full["session_id"].astype(str)

    df_full["epoch_index"] = df_full["epoch_index"].astype(int)



    # Sort by (session, epoch) — strict chronological order

    df_full = df_full.sort_values(["user_name", "session_id", "epoch_index"]).reset_index(drop=True)



    # ── OLD COUNT (how the v1 script counts pairs) ──

    # v1 filters to strain_reported==1 first, then shift(-1) per session

    old_labeled = df_full[(df_full["strain_reported"] == 1) & (df_full["subjective_strain_cr10"].notna())].copy()

    old_pairs_per_user = {}

    for sess, grp in old_labeled.groupby("session_id"):

        shifted = grp["subjective_strain_cr10"].shift(-1)

        old_pairs_per_user.setdefault(grp["user_name"].iloc[0], 0)

        old_pairs_per_user[grp["user_name"].iloc[0]] += shifted.notna().sum()



    # ── TREND FEATURES (computed on FULL table, then joined back) ──

    trend_rows = []

    gap_flag_total = 0



    for sess_id, grp in df_full.groupby("session_id"):

        grp = grp.sort_values("epoch_index").reset_index(drop=True)

        for side in ["left", "right"]:

            mdf_col = f"mdf_hz_{side}"

            gap_col = f"gap_frequency_{side}"

            ei_col  = f"eindex_live_{side}"



            for feat_col, trend_name in [

                (mdf_col, f"mdf_trend_{side}"),

                (gap_col, f"gap_freq_trend_{side}"),

                (ei_col,  f"eindex_trend_{side}"),

            ]:

                vals = grp[feat_col].values if feat_col in grp.columns else np.zeros(len(grp))

                epo  = grp["epoch_index"].values

                trends = np.zeros(len(grp), dtype=float)

                gap_flags = np.zeros(len(grp), dtype=int)

                for i in range(1, len(grp)):

                    if epo[i] == epo[i - 1] + 1:

                        trends[i] = vals[i] - vals[i - 1]

                    else:

                        trends[i] = 0.0

                        gap_flags[i] = 1

                        gap_flag_total += 1

                grp[trend_name] = trends

                grp["trend_gap_flag"] = gap_flags  # overwrite per iter (same for all sides per row)



        trend_rows.append(grp)



    df_trends = pd.concat(trend_rows, ignore_index=True)

    print(f"Rows with non-consecutive epoch_index (trend_gap_flag=1): {gap_flag_total}")



    # ── CORRECT TARGET: look forward on FULL table ──

    # For each row t, the target is the *next* row in the same session

    # where strain_reported=1. We only keep pairs where t+1 is the

    # immediately-next epoch (consecutive epoch_index) to ensure t+5 min semantics.

    # Build lookup: {(session_id, epoch_index) -> subjective_strain_cr10 if labeled}

    strain_lookup = {}

    for _, row in df_trends.iterrows():

        if row["strain_reported"] == 1 and not pd.isna(row["subjective_strain_cr10"]):

            strain_lookup[(row["session_id"], row["epoch_index"])] = row["subjective_strain_cr10"]



    targets = []

    for _, row in df_trends.iterrows():

        next_epoch = row["epoch_index"] + 1

        key = (row["session_id"], next_epoch)

        if key in strain_lookup:

            targets.append(strain_lookup[key])

        else:

            targets.append(np.nan)



    df_trends["target_future_strain"] = targets



    # Keep only rows where target exists (i.e., a valid forecast pair)

    df_pairs = df_trends[df_trends["target_future_strain"].notna()].copy().reset_index(drop=True)



    # ── COMPARE OLD vs NEW ──

    new_pairs_per_user = df_pairs.groupby("user_name").size().to_dict()



    print("\n--- Forecast Pair Counts: OLD (v1 buggy) vs NEW (v2 fixed) ---")

    all_users = sorted(set(list(old_pairs_per_user.keys()) + list(new_pairs_per_user.keys())))

    compare_rows = []

    for u in all_users:

        compare_rows.append({

            "User": u,

            "Old Pairs (v1)": old_pairs_per_user.get(u, 0),

            "New Pairs (v2)": new_pairs_per_user.get(u, 0),

            "Delta": new_pairs_per_user.get(u, 0) - old_pairs_per_user.get(u, 0),

        })

    print(pd.DataFrame(compare_rows).to_string(index=False))

    print(f"\nTotal old pairs: {sum(old_pairs_per_user.values())}  |  Total new pairs: {len(df_pairs)}")



    return df_pairs





# ─────────────────────────────────────────────────────────────────────────────

# 5. STEP 2b — REMOVE CALIBRATION RMS FROM PREDICTORS

# ─────────────────────────────────────────────────────────────────────────────

def step2b_define_feature_sets():

    print("\n" + "=" * 60)

    print("STEP 2b — FEATURE SETS (calibration_rms EXCLUDED)")

    print("=" * 60)



    SET_A = CONFIG["SET_A"]

    SET_B = SET_A + CONFIG["SET_B_EXTRA"]

    SET_C = SET_B + CONFIG["SET_C_EXTRA"]



    for name, fs in [("SET_A", SET_A), ("SET_B", SET_B), ("SET_C", SET_C)]:

        print(f"\n{name} ({len(fs)} features):")

        for f in fs:

            print(f"  {f}")



    assert "calibration_rms_left" not in SET_C, "BUG: calibration_rms_left still present!"

    assert "calibration_rms_right" not in SET_C, "BUG: calibration_rms_right still present!"

    print("\ncalibration_rms_left  [NOT IN any predictor set]")

    print("calibration_rms_right [NOT IN any predictor set]")



    return SET_A, SET_B, SET_C





# ─────────────────────────────────────────────────────────────────────────────

# 6. STEP 2c — CALIBRATION QC GATE

# ─────────────────────────────────────────────────────────────────────────────

def step2c_qc_gate(df_full: pd.DataFrame):

    print("\n" + "=" * 60)

    print("STEP 2c — CALIBRATION QC GATE")

    print("=" * 60)



    # One row per session: take the first row's calibration (session-constant)

    sess_df = (

        df_full

        .sort_values(["session_id", "epoch_index"])

        .groupby("session_id", as_index=False)

        .first()[["user_name", "session_id", "calibration_rms_left", "calibration_rms_right"]]

    )



    sess_df["lr_ratio"] = sess_df["calibration_rms_left"] / sess_df["calibration_rms_right"]

    sess_df["log_lr_ratio"] = np.log(sess_df["lr_ratio"].replace(0, np.nan))



    # ── Print raw distribution ──

    print("\n--- Raw calibration_rms per session ---")

    disp = sess_df[["user_name", "session_id",

                    "calibration_rms_left", "calibration_rms_right",

                    "lr_ratio"]].copy()

    disp["lr_ratio"] = disp["lr_ratio"].round(3)

    disp["calibration_rms_left"] = disp["calibration_rms_left"].round(1)

    disp["calibration_rms_right"] = disp["calibration_rms_right"].round(1)

    print(disp.to_string(index=False))



    pop_med_l = sess_df["calibration_rms_left"].median()

    pop_med_r = sess_df["calibration_rms_right"].median()

    log_ratio_vals = sess_df["log_lr_ratio"].dropna()

    pop_median_log = log_ratio_vals.median()

    pop_mad_log    = (log_ratio_vals - pop_median_log).abs().median()



    print(f"\nPopulation medians — L: {pop_med_l:.1f}  R: {pop_med_r:.1f}")

    print(f"Log(L/R) population median: {pop_median_log:.4f}  MAD: {pop_mad_log:.4f}")



    # ── Choose and log QC bounds (chosen from distribution ONLY, before any model) ──

    bounds_rationale = {

        "QC_RATIO_MAD_Z":                CONFIG["QC_RATIO_MAD_Z"],

        "QC_WITHIN_SUBJECT_MIN_SESSIONS": CONFIG["QC_WITHIN_SUBJECT_MIN_SESSIONS"],

        "QC_WITHIN_SUBJECT_LOG_DEV":      CONFIG["QC_WITHIN_SUBJECT_LOG_DEV"],

        "QC_MIN_FRACTION_OF_POP_MEDIAN":  CONFIG["QC_MIN_FRACTION_OF_POP_MEDIAN"],

        "QC_MIN_ABSOLUTE_RMS_L":          round(CONFIG["QC_MIN_FRACTION_OF_POP_MEDIAN"] * pop_med_l, 1),

        "QC_MIN_ABSOLUTE_RMS_R":          round(CONFIG["QC_MIN_FRACTION_OF_POP_MEDIAN"] * pop_med_r, 1),

        "Rationale": (

            "QC_RATIO_MAD_Z=3.0: conservative 3-sigma equivalent on a robust scale, "

            "chosen before seeing LOSO results to avoid cherry-picking. "

            "QC_WITHIN_SUBJECT_LOG_DEV=1.5: ~4.5x deviation of session ratio from subject median, "

            "flags obvious electrode swap / sensor error. "

            "QC_MIN_FRACTION_OF_POP_MEDIAN=0.15: below 15% of population median indicates "

            "a failed shrug calibration (electrode detachment or no contraction)."

        ),

    }



    print("\n--- QC Bounds declared (written BEFORE any model run) ---")

    for k, v in bounds_rationale.items():

        if k == "Rationale":

            print(f"  Rationale: {v}")

        else:

            print(f"  {k}: {v}")



    # ── Apply rules ──

    flagged = []



    for _, row in sess_df.iterrows():

        rms_l = row["calibration_rms_left"]

        rms_r = row["calibration_rms_right"]

        log_r = row["log_lr_ratio"]

        rule_fired = []



        # Rule 1

        if pd.isna(rms_l) or rms_l <= 0 or pd.isna(rms_r) or rms_r <= 0:

            rule_fired.append("Rule1: null/zero rms")



        # Rule 2

        if not pd.isna(log_r) and pop_mad_log > 0:

            robust_z = abs(log_r - pop_median_log) / (1.4826 * pop_mad_log)

            if robust_z > CONFIG["QC_RATIO_MAD_Z"]:

                rule_fired.append(f"Rule2: pop robust-z={robust_z:.2f}")



        # Rule 4 (weak shrug)

        if not pd.isna(rms_l) and rms_l < CONFIG["QC_MIN_FRACTION_OF_POP_MEDIAN"] * pop_med_l:

            rule_fired.append(f"Rule4: rms_L={rms_l:.0f} < {CONFIG['QC_MIN_FRACTION_OF_POP_MEDIAN']*pop_med_l:.0f}")

        if not pd.isna(rms_r) and rms_r < CONFIG["QC_MIN_FRACTION_OF_POP_MEDIAN"] * pop_med_r:

            rule_fired.append(f"Rule4: rms_R={rms_r:.0f} < {CONFIG['QC_MIN_FRACTION_OF_POP_MEDIAN']*pop_med_r:.0f}")



        if rule_fired:

            flagged.append({

                "user_name": row["user_name"],

                "session_id": row["session_id"],

                "rms_L": round(rms_l, 1) if not pd.isna(rms_l) else None,

                "rms_R": round(rms_r, 1) if not pd.isna(rms_r) else None,

                "lr_ratio": round(row["lr_ratio"], 3) if not pd.isna(row["lr_ratio"]) else None,

                "rules_fired": "; ".join(rule_fired),

            })



    # Rule 3 — within-subject, only for subjects with >= MIN_SESSIONS

    subject_session_counts = sess_df.groupby("user_name")["session_id"].nunique()

    multi_sess_users = subject_session_counts[

        subject_session_counts >= CONFIG["QC_WITHIN_SUBJECT_MIN_SESSIONS"]

    ].index



    for user in multi_sess_users:

        user_sess = sess_df[sess_df["user_name"] == user]

        subj_median_log = user_sess["log_lr_ratio"].median()

        for _, row in user_sess.iterrows():

            log_r = row["log_lr_ratio"]

            if not pd.isna(log_r):

                dev = abs(log_r - subj_median_log)

                if dev > CONFIG["QC_WITHIN_SUBJECT_LOG_DEV"]:

                    already = next(

                        (f for f in flagged if f["session_id"] == row["session_id"]),

                        None,

                    )

                    rule_str = f"Rule3: within-subj log-dev={dev:.3f}"

                    if already:

                        already["rules_fired"] += "; " + rule_str

                    else:

                        flagged.append({

                            "user_name": row["user_name"],

                            "session_id": row["session_id"],

                            "rms_L": round(row["calibration_rms_left"], 1),

                            "rms_R": round(row["calibration_rms_right"], 1),

                            "lr_ratio": round(row["lr_ratio"], 3),

                            "rules_fired": rule_str,

                        })



    print("\n--- QC Flagged Sessions (NOT silently dropped) ---")

    if flagged:

        flag_df = pd.DataFrame(flagged)

        print(flag_df.to_string(index=False))

        print(f"\nTotal flagged sessions: {len(flagged)} / {len(sess_df)}")

    else:

        print("  No sessions flagged.")



    flagged_session_ids = {f["session_id"] for f in flagged}

    return flagged_session_ids, bounds_rationale





# ─────────────────────────────────────────────────────────────────────────────

# 7. STEP 2d — OUTLIER POLICY (declaration only)

# ─────────────────────────────────────────────────────────────────────────────

def step2d_outlier_policy():

    print("\n" + "=" * 60)

    print("STEP 2d — OUTLIER POLICY (pre-declared)")

    print("=" * 60)

    policy = (

        "Geeta Srivastava and ALL subjects are retained in headline results. "

        "No subject is removed based on model error. "

        "A sensitivity table with Geeta excluded will be reported in Step 6 "

        "and clearly labeled as 'Sensitivity Analysis — Not Headline'. "

        "Geeta's strain range 7.0–8.0 is a subject-level baseline offset; "

        "this is the cold-start problem addressed by the personalized protocol."

    )

    print(policy)





# ─────────────────────────────────────────────────────────────────────────────

# 8. STEP 2e — PERSONALIZED PROTOCOL ELIGIBILITY

# ─────────────────────────────────────────────────────────────────────────────

def step2e_eligibility(df_pairs: pd.DataFrame):

    print("\n" + "=" * 60)

    print("STEP 2e — PERSONALIZED PROTOCOL ELIGIBILITY")

    print("=" * 60)

    print(

        "Calibration epochs are taken as the first K_CAL epochs "

        "ordered by (session_id ASC, epoch_index ASC) across all of a "

        "subject's sessions. This is the strict chronological rule."

    )



    MIN_AFTER = CONFIG["MIN_SCORED_AFTER_CAL"]

    rows = []

    for user in sorted(df_pairs["user_name"].unique()):

        ud = df_pairs[df_pairs["user_name"] == user].sort_values(

            ["session_id", "epoch_index"]

        ).reset_index(drop=True)

        n = len(ud)

        row = {"User": user, "N forecast pairs": n}

        for k in CONFIG["K_CAL_VALUES"]:

            eligible = n >= k + MIN_AFTER

            n_scored = max(0, n - k) if eligible else 0

            row[f"Eligible k={k}"] = "YES" if eligible else "NO"

            row[f"n_scored k={k}"] = n_scored if eligible else "—"

        rows.append(row)



    elig_df = pd.DataFrame(rows)

    print("\n--- Personalized Protocol Eligibility ---")

    print(elig_df.to_string(index=False))





# ─────────────────────────────────────────────────────────────────────────────

# 9. SESSION LOG APPEND

# ─────────────────────────────────────────────────────────────────────────────

def append_session_log(bounds_rationale: dict, step3_report: dict = None):

    ts = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S+05:30")

    step3_block = ""

    if step3_report:

        step3_block = f"""

3a: Winsorized {len(step3_report.get('winsorized', []))} features at {CONFIG['WINSOR_LOW_PCT']}th/{CONFIG['WINSOR_HIGH_PCT']}th percentiles (training data only).

3b: Near-constant features binarized: {step3_report.get('binarized', [])}. Std threshold: {CONFIG['NEAR_CONST_STD']}.

3c: Scaling mode: {'Global StandardScaler' if CONFIG['USE_GLOBAL_SCALER'] else 'Per-subject causal expanding z-score'}.

    Global fallback used for first {CONFIG['MIN_HISTORY']} epochs per subject.

"""

    entry = f"""

## Phase X — ML Pipeline Correction (Steps 2 & 3)

Timestamp: {ts}

Status: Completed



### Summary

Step 2 (2a-2e): Fixed target/trend construction, removed calibration_rms from predictors,

applied 4-rule calibration QC gate, declared outlier policy, computed eligibility table.



### QC Bounds Used (Step 2c)

{json.dumps({k: v for k, v in bounds_rationale.items() if k != 'Rationale'}, indent=2)}

Rationale: {bounds_rationale.get('Rationale', '')}



### Step 3 Feature Engineering{step3_block}

### Files modified

- `train_pymc_hierarchical_model_v2.py`

- `logs/session_log.md`

"""

    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "session_log.md")

    with open(log_path, "a", encoding="utf-8") as f:

        f.write(entry)

    print(f"\nAppended to {log_path}")





# -----------------------------------------------------------------------------

# STEP 3 — FEATURE ENGINEERING FIXES

# -----------------------------------------------------------------------------


def step3_feature_engineering(
    df_pairs: pd.DataFrame,
    feature_set: list,
    feature_set_name: str,
    train_mask: pd.Series = None,
) -> tuple:
    print("\n" + "=" * 60)
    print(f"STEP 3 — FEATURE ENGINEERING  [{feature_set_name}, n={len(df_pairs)}]")
    print("=" * 60)

    df = df_pairs.copy()

    features = [f for f in feature_set if f in df.columns]
    missing = [f for f in feature_set if f not in df.columns]
    if missing:
        print(f"  WARNING: features not found in dataframe, skipping: {missing}")

    if train_mask is None:
        train_mask = pd.Series(True, index=df.index)

    report = {"winsorized": [], "binarized": [], "near_const_stds": {}, "scaler_params": {}}

    for feat in features:
        df[feat] = df[feat].astype(float)

    # 3a: WINSORIZATION
    winsor_features = [f for f in CONFIG["WINSOR_FEATURES"] if f in features]
    winsor_bounds = {}
    for feat in winsor_features:
        train_vals = df.loc[train_mask, feat].dropna()
        lo = np.percentile(train_vals, CONFIG["WINSOR_LOW_PCT"])
        hi = np.percentile(train_vals, CONFIG["WINSOR_HIGH_PCT"])
        winsor_bounds[feat] = (lo, hi)
        df[feat] = df[feat].clip(lower=lo, upper=hi)
        report["winsorized"].append(feat)

    print(f"\n3a — Winsorized {len(winsor_bounds)} features at "
          f"{CONFIG['WINSOR_LOW_PCT']}th / {CONFIG['WINSOR_HIGH_PCT']}th percentile")

    # 3b: NEAR-CONSTANT CHECK
    binarized = []
    near_const_stds = {}
    for feat in features:
        train_std = df.loc[train_mask, feat].std()
        near_const_stds[feat] = train_std
        if train_std < CONFIG["NEAR_CONST_STD"]:
            df[feat] = (df[feat] > 0).astype(float)
            binarized.append(feat)

    report["near_const_stds"] = near_const_stds
    report["binarized"] = binarized

    print(f"\n3b — Binarized {len(binarized)} features (<{CONFIG['NEAR_CONST_STD']} std)")

    # 3c: SCALING
    if CONFIG["USE_GLOBAL_SCALER"]:
        scaler_params = {}
        for feat in features:
            if feat in binarized: continue
            train_vals = df.loc[train_mask, feat].dropna()
            mu, sd = train_vals.mean(), train_vals.std()
            if sd < 1e-9: sd = 1.0
            df[feat] = (df[feat] - mu) / sd
            scaler_params[feat] = {"mean": float(mu), "std": float(sd)}
        report["scaler_params"] = scaler_params
    else:
        global_stats = {}
        for feat in features:
            if feat in binarized: continue
            train_vals = df.loc[train_mask, feat].dropna()
            mu, sd = train_vals.mean(), train_vals.std()
            global_stats[feat] = (mu, sd if sd > 1e-9 else 1.0)

        df = df.sort_values(["user_name", "session_id", "epoch_index"]).reset_index(drop=True)

        for user in df["user_name"].unique():
            u_idx = df.index[df["user_name"] == user]
            for feat in features:
                if feat in binarized: continue
                g_mu, g_sd = global_stats[feat]
                vals = df.loc[u_idx, feat].values.copy().astype(float)
                scaled = np.empty_like(vals)
                for i, row_idx in enumerate(u_idx):
                    past = vals[:i]
                    past_valid = past[~np.isnan(past)]
                    if len(past_valid) >= CONFIG["MIN_HISTORY"]:
                        mu, sd = past_valid.mean(), past_valid.std()
                        if sd < 1e-9: sd = 1.0
                        scaled[i] = (vals[i] - mu) / sd
                    else:
                        scaled[i] = (vals[i] - g_mu) / g_sd
                df.loc[u_idx, feat] = scaled

        report["scaler_params"] = {"mode": "per_subject_causal_expanding"}

    return df, report

"""
train_pymc_hierarchical_model_v2_part2.py
This script contains Steps 4, 5, 6, 7 and replaces the end of train_pymc_hierarchical_model_v2.py.
"""
# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — MODEL STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────
import pytensor
pytensor.config.cxx = ""  # Disable C++ compiler on Windows to avoid setup issues (slows down sampling but is safe)
import pymc as pm
import arviz as az
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, confusion_matrix
from scipy.stats import pearsonr

CONFIG["BETA_SIGMA"] = 0.5

def build_model(X_train, y_train, user_idx_train, n_users, n_features, use_horseshoe, p0=5):
    """
    Step 5: Random Intercept ONLY (no random slopes).
    If use_horseshoe is True, uses Regularized Horseshoe for population slopes.
    Otherwise uses Normal(0, BETA_SIGMA).
    """
    with pm.Model() as model:
        u_idx = pm.Data("u_idx", user_idx_train)
        X_data = pm.Data("X_data", X_train)
        y_data = pm.Data("y_data", y_train)
        
        # 1. Population Intercept & User Random Intercepts
        alpha_pop = pm.Normal("alpha_pop", mu=0, sigma=5.0)
        tau_alpha = pm.HalfNormal("tau_alpha", sigma=2.0)
        alpha_offset = pm.Normal("alpha_offset", mu=0, sigma=1, shape=n_users)
        alpha_u = pm.Deterministic("alpha_u", alpha_pop + alpha_offset * tau_alpha)
        
        # 2. Population Slopes
        if use_horseshoe:
            N = len(y_train)
            sigma_0 = 1.0
            tau_0 = (p0 / (n_features - p0)) * (sigma_0 / np.sqrt(N))
            tau = pm.HalfCauchy("tau", beta=tau_0)
            lam = pm.HalfCauchy("lam", beta=1.0, shape=n_features)
            c2 = pm.InverseGamma("c2", alpha=1.0, beta=1.0)
            tilde_lam = pm.Deterministic("tilde_lam", pm.math.sqrt(c2 * lam**2 / (c2 + tau**2 * lam**2)))
            beta_pop_offset = pm.Normal("beta_pop_offset", mu=0, sigma=1, shape=n_features)
            beta_pop = pm.Deterministic("beta_pop", beta_pop_offset * tau * tilde_lam)
        else:
            beta_pop = pm.Normal("beta_pop", mu=0, sigma=CONFIG["BETA_SIGMA"], shape=n_features)
            
        # 3. Linear Predictor (NO per-user random slopes)
        mu = alpha_u[u_idx] + pm.math.sum(beta_pop * X_data, axis=1)
        
        # 4. Residual Noise Variance & Likelihood
        sigma_y = pm.HalfNormal("sigma_y", sigma=2.0)
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma_y, observed=y_data)
        
    return model

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — EVALUATION PROTOCOL
# ─────────────────────────────────────────────────────────────────────────────
def run_loso_evaluation(df_pairs, feature_set, feature_set_name, use_horseshoe):
    print(f"\n{'='*60}\nSTEP 6 — EVALUATION: {feature_set_name}\n{'='*60}")
    
    users = sorted(df_pairs["user_name"].unique())
    results = []
    
    # We will aggregate all test predictions across all folds
    all_y_true = []
    all_cold_pred = []
    all_k3_pred, all_y_true_k3 = [], []
    all_k5_pred, all_y_true_k5 = [], []
    
    all_global_mean_pred = []
    all_persistence_pred = []
    
    for test_user in users:
        print(f"\n--- Fold: Held-out {test_user} ---")
        train_mask = df_pairs["user_name"] != test_user
        test_mask = df_pairs["user_name"] == test_user
        
        # Step 3: Feature Engineering on this fold
        df_scaled, _ = step3_feature_engineering(df_pairs, feature_set, feature_set_name, train_mask)
        
        train_df = df_scaled[train_mask].reset_index(drop=True)
        test_df = df_scaled[test_mask].sort_values("epoch_index").reset_index(drop=True)
        
        if len(test_df) == 0:
            continue
            
        # Map training users to 0..N-1
        train_users = sorted(train_df["user_name"].unique())
        user2idx = {u: i for i, u in enumerate(train_users)}
        train_df["u_idx"] = train_df["user_name"].map(user2idx)
        
        X_tr = train_df[feature_set].values
        y_tr = train_df["target_future_strain"].values
        u_tr = train_df["u_idx"].values
        
        X_te = test_df[feature_set].values
        y_te = test_df["target_future_strain"].values
        
        # Global mean for baselines
        global_mean_y = y_tr.mean()
        
        # Train PyMC model
        model = build_model(X_tr, y_tr, u_tr, len(train_users), len(feature_set), use_horseshoe)
        
        with model:
            idata = pm.sample(
                draws=1000, 
                tune=1000, 
                chains=4, 
                target_accept=0.95, 
                progressbar=False,
                random_seed=42
            )
            
        # Extract posteriors (shape: n_samples = chains * draws)
        post = idata.posterior
        alpha_pop_samples = post["alpha_pop"].values.flatten()
        beta_pop_samples = post["beta_pop"].values.reshape(-1, len(feature_set))
        tau_alpha_samples = post["tau_alpha"].values.flatten()
        sigma_y_samples = post["sigma_y"].values.flatten()
        
        # 1. COLD-START PREDICTIONS
        # pred = alpha_pop + X * beta_pop
        cold_preds = []
        for i in range(len(test_df)):
            mu_samples = alpha_pop_samples + np.dot(beta_pop_samples, X_te[i])
            cold_preds.append(np.mean(mu_samples))
            
        all_y_true.extend(y_te)
        all_cold_pred.extend(cold_preds)
        
        # Baselines (full test set)
        all_global_mean_pred.extend([global_mean_y] * len(y_te))
        # Persistence: strain at t is the prediction for t+5
        persistence = test_df["subjective_strain_cr10"].values
        all_persistence_pred.extend(persistence)
        
        # 2. PERSONALIZED PREDICTIONS (Analytic Bayesian Update on Intercept)
        # alpha_post_mu = Var_post * (alpha_pop/tau_alpha^2 + sum(residual)/sigma_y^2)
        for K in CONFIG["K_CAL_VALUES"]:
            if len(test_df) >= K + CONFIG["MIN_SCORED_AFTER_CAL"]:
                # Use first K as calibration
                X_cal = X_te[:K]
                y_cal = y_te[:K]
                
                # Test on remaining
                X_score = X_te[K:]
                y_score = y_te[K:]
                
                k_preds = []
                for i in range(len(X_score)):
                    # For each posterior sample, what is the personalized intercept?
                    # sum(residual) = sum(y_cal - X_cal * beta_pop)
                    sum_resid = np.sum(y_cal[:, None] - np.dot(X_cal, beta_pop_samples.T), axis=0) # shape: (n_samples,)
                    
                    var_post = 1.0 / (1.0 / (tau_alpha_samples**2) + K / (sigma_y_samples**2))
                    alpha_u_post = var_post * (alpha_pop_samples / (tau_alpha_samples**2) + sum_resid / (sigma_y_samples**2))
                    
                    mu_score = alpha_u_post + np.dot(beta_pop_samples, X_score[i])
                    k_preds.append(np.mean(mu_score))
                    
                if K == 3:
                    all_k3_pred.extend(k_preds)
                    all_y_true_k3.extend(y_score)
                else:
                    all_k5_pred.extend(k_preds)
                    all_y_true_k5.extend(y_score)
                    
    # Return aggregated metrics
    def calc_metrics(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        mask = ~np.isnan(y_pred)
        if mask.sum() == 0: return None
        y_t, y_p = y_true[mask], y_pred[mask]
        mae = mean_absolute_error(y_t, y_p)
        corr = pearsonr(y_t, y_p)[0] if len(y_t) > 1 else 0
        return {"mae": mae, "corr": corr, "n": len(y_t)}
        
    metrics = {
        "Cold Start": calc_metrics(all_y_true, all_cold_pred),
        "Global Mean (Baseline)": calc_metrics(all_y_true, all_global_mean_pred),
        "Persistence (Baseline)": calc_metrics(all_y_true, all_persistence_pred),
        "Personalized K=3": calc_metrics(all_y_true_k3, all_k3_pred),
        "Personalized K=5": calc_metrics(all_y_true_k5, all_k5_pred),
    }
    
    print("\n--- RESULTS ---")
    for k, v in metrics.items():
        if v:
            print(f"{k:25} | MAE: {v['mae']:.3f} | Corr: {v['corr']:.3f} | n: {v['n']}")
            
    return metrics

def run_all_models(df_pairs):
    print("\nStarting full Evaluation Protocol (This will take a while)...")
    results = {}
    
    # SET A (No Horseshoe)
    results["SET_A"] = run_loso_evaluation(df_pairs, SET_A, "SET_A", use_horseshoe=False)
    
    # SET B (No Horseshoe)
    results["SET_B"] = run_loso_evaluation(df_pairs, SET_B, "SET_B", use_horseshoe=False)
    
    # SET C (Horseshoe)
    results["SET_C"] = run_loso_evaluation(df_pairs, SET_C, "SET_C", use_horseshoe=True)
    
    # Step 7 Outputs
    with open("loso_report_v2.md", "w") as f:
        f.write("# TensionBudget PyMC Pipeline v2 Results\n\n")
        f.write("## Overview\nRandom intercepts only, buggy targets fixed, strictly causal scaling.\n\n")
        for sname, sres in results.items():
            f.write(f"### {sname}\n")
            f.write("| Protocol | MAE | Corr | N |\n|---|---|---|---|\n")
            for k, v in sres.items():
                if v:
                    f.write(f"| {k} | {v['mae']:.3f} | {v['corr']:.3f} | {v['n']} |\n")
            f.write("\n")
            
    print("\nSaved loso_report_v2.md")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df_raw = fetch_raw()
    df_full = step1_audit(df_raw)
    df_pairs = step2a_construct_targets(df_full)
    SET_A, SET_B, SET_C = step2b_define_feature_sets()
    flagged_session_ids, bounds_rationale = step2c_qc_gate(df_full)
    step2d_outlier_policy()
    step2e_eligibility(df_pairs)
    
    print("\n--- Running Step 3 on each feature set (full dataset, no LOSO split yet) ---")
    for set_name, fset in [("SET_A", SET_A), ("SET_B", SET_B), ("SET_C", SET_C)]:
        df_scaled, step3_report = step3_feature_engineering(df_pairs, fset, set_name, train_mask=None)
    
    append_session_log(bounds_rationale, step3_report)
    
    run_all_models(df_pairs)
    
    print("\n" + "=" * 60)
    print("STEP 4-7 COMPLETE. Models evaluated.")
    print("=" * 60)
