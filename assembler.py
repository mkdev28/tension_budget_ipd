import os

with open('train_pymc_hierarchical_model_v2_top.py', 'r', encoding='utf-8') as f:
    top = f.read()

step3 = """
def step3_feature_engineering(
    df_pairs: pd.DataFrame,
    feature_set: list,
    feature_set_name: str,
    train_mask: pd.Series = None,
) -> tuple:
    print("\\n" + "=" * 60)
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

    print(f"\\n3a — Winsorized {len(winsor_bounds)} features at "
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

    print(f"\\n3b — Binarized {len(binarized)} features (<{CONFIG['NEAR_CONST_STD']} std)")

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

with open('train_pymc_hierarchical_model_v2_part2.py', 'r', encoding='utf-8') as f:
    part2 = f.read()

main = """
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
    
    print("\\n--- Running Step 3 on each feature set (full dataset, no LOSO split yet) ---")
    for set_name, fset in [("SET_A", SET_A), ("SET_B", SET_B), ("SET_C", SET_C)]:
        df_scaled, step3_report = step3_feature_engineering(df_pairs, fset, set_name, train_mask=None)
    
    append_session_log(bounds_rationale, step3_report)
    
    run_all_models(df_pairs)
    
    print("\\n" + "=" * 60)
    print("STEP 4-7 COMPLETE. Models evaluated.")
    print("=" * 60)
"""

with open('train_pymc_hierarchical_model_v2.py', 'w', encoding='utf-8') as f:
    f.write(top)
    f.write(step3)
    f.write("\n")
    f.write(part2)
    f.write("\n")
    f.write(main)
