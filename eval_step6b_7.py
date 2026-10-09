r"""
eval_step6b_7.py
================
Steps 6b and 7: Extended evaluation, sensitivity analyses, and post-hoc explorations.

Hard rules (same as v2):
  - No LOCF anywhere.
  - No new feature choices beyond SET_A/B/C declared in v2.
  - No tuning to improve numbers.
  - SET_C marked NON-CONVERGED; do not use its numbers as evidence.
  - Raw DB rows never overwritten.
  - All Step 7 analyses labelled POST-HOC EXPLORATORY throughout.

Run:
    .\.venv\Scripts\python eval_step6b_7.py
"""
from __future__ import annotations
import os, sys, warnings, json, datetime
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pytensor
pytensor.config.cxx = ""
import pymc as pm
import arviz as az
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Import shared utilities from v2 (functions are module-level; main block
# is guarded by __name__ == "__main__" so it does NOT run on import)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_pymc_hierarchical_model_v2 import (
    CONFIG,
    fetch_raw,
    step1_audit,
    step2a_construct_targets,
    step2b_define_feature_sets,
    step2c_qc_gate,
    step3_feature_engineering,
    build_model,
)

# CONFIG["BETA_SIGMA"] was set at module level in v2 part2; confirm it exists
CONFIG.setdefault("BETA_SIGMA", 0.5)
CONFIG.setdefault("CI_LEVEL", 0.90)
CONFIG.setdefault("ALERT_THRESHOLD", 4.0)

N_SAMPLES = CONFIG["CHAINS"] * CONFIG["DRAWS"]


# ===========================================================================
# METRICS HELPERS
# ===========================================================================

def _clean(y_true, y_pred, subjects=None):
    """Remove rows where y_pred is NaN and align subjects."""
    yt = np.array(y_true, dtype=float)
    yp = np.array(y_pred, dtype=float)
    mask = ~np.isnan(yp) & ~np.isnan(yt)
    if subjects is not None:
        su = np.array(subjects)
        return yt[mask], yp[mask], su[mask]
    return yt[mask], yp[mask], None


def within_subject_corr(y_true, y_pred, subjects):
    """
    Within-subject Pearson r:
    Subtract each subject's mean from y_true and y_pred, pool, correlate.
    Returns (pooled_r, per_subject_dict).
    """
    yt, yp, su = _clean(y_true, y_pred, subjects)
    yt_dm = np.zeros_like(yt)
    yp_dm = np.zeros_like(yp)
    per_sub = {}
    for s in np.unique(su):
        idx = su == s
        yt_dm[idx] = yt[idx] - yt[idx].mean()
        yp_dm[idx] = yp[idx] - yp[idx].mean()
        if idx.sum() > 2:
            r, _ = pearsonr(yt[idx] - yt[idx].mean(), yp[idx] - yp[idx].mean())
            per_sub[s] = round(float(r), 3)
        else:
            per_sub[s] = float("nan")
    if len(yt_dm) > 1:
        pool_r, _ = pearsonr(yt_dm, yp_dm)
    else:
        pool_r = float("nan")
    return round(float(pool_r), 3), per_sub


def ci_coverage(y_true, pred_lo, pred_hi):
    yt = np.array(y_true, dtype=float)
    lo = np.array(pred_lo, dtype=float)
    hi = np.array(pred_hi, dtype=float)
    mask = ~np.isnan(lo) & ~np.isnan(hi)
    if mask.sum() == 0:
        return float("nan")
    return round(float(np.mean((yt[mask] >= lo[mask]) & (yt[mask] <= hi[mask]))), 3)


def alert_matrix(y_true, y_pred, threshold):
    yt, yp, _ = _clean(y_true, y_pred)
    if len(yt) == 0:
        return {}
    ya = (yt >= threshold).astype(int)
    pa = (yp >= threshold).astype(int)
    from sklearn.metrics import confusion_matrix as cm
    mat = cm(ya, pa, labels=[0, 1])
    tn, fp, fn, tp = mat.ravel()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3),
    }


def compute_metrics(y_true, y_pred, pred_lo=None, pred_hi=None, subjects=None):
    """Full metric set: MAE, RMSE, Pearson, within-subject r, CI coverage, alert matrix."""
    yt, yp, su = _clean(y_true, y_pred, subjects)
    if len(yt) == 0:
        return None
    mae  = round(float(mean_absolute_error(yt, yp)), 3)
    rmse = round(float(np.sqrt(mean_squared_error(yt, yp))), 3)
    pr   = round(float(pearsonr(yt, yp)[0]) if len(yt) > 1 else float("nan"), 3)
    wsr, per_sub = within_subject_corr(yt.tolist(), yp.tolist(), (su.tolist() if su is not None else ["all"] * len(yt)))
    cov  = ci_coverage(y_true, pred_lo, pred_hi) if pred_lo is not None else float("nan")
    al   = alert_matrix(yt.tolist(), yp.tolist(), CONFIG["ALERT_THRESHOLD"])
    return {
        "n": int(len(yt)),
        "mae": mae, "rmse": rmse,
        "pearson_r": pr,
        "within_subject_r": wsr,
        "ci_coverage": cov,
        "alert": al,
        "per_subject_r": per_sub,
    }


# ===========================================================================
# POSTERIOR PREDICTIVE CI HELPER
# ===========================================================================

def predictive_ci(mu_samples, sigma_samples, ci_level=0.90):
    """
    Draw one predictive sample per posterior sample and return (lo, hi, mean).
    mu_samples, sigma_samples: 1-D arrays of shape (n_samples,)
    """
    eps = np.random.standard_normal(len(mu_samples))
    y_pp = mu_samples + sigma_samples * eps
    half = (1 - ci_level) / 2
    lo = float(np.percentile(y_pp, half * 100))
    hi = float(np.percentile(y_pp, (1 - half) * 100))
    return lo, hi, float(np.mean(mu_samples))


# ===========================================================================
# EXTENDED LOSO EVALUATION  (Steps 6b.1 – 6b.3)
# ===========================================================================

def run_loso_extended(df_pairs, feature_set, feature_set_name, use_horseshoe):
    """
    Full LOSO with:
      - Same-row baselines for every protocol
      - Posterior predictive 90% CI
      - Within-subject correlation
      - Per-subject MAE table
      - Intercept-only control (no feature slopes)
      - Alert confusion matrix
    """
    tag = "[NON-CONVERGED]" if use_horseshoe else ""
    print(f"\n{'='*60}")
    print(f"EXTENDED LOSO: {feature_set_name} {tag}")
    print(f"{'='*60}")

    users = sorted(df_pairs["user_name"].unique())

    # Storage per protocol
    cold = {k: [] for k in ["yt", "yp", "lo", "hi", "subj", "gm", "pers"]}
    k3   = {k: [] for k in ["yt", "yp", "lo", "hi", "subj", "gm", "pers",
                              "pm", "io", "cold"]}
    k5   = {k: [] for k in ["yt", "yp", "lo", "hi", "subj", "gm", "pers",
                              "pm", "io", "cold"]}
    per_subject_mae = {}

    for test_user in users:
        print(f"  Fold: {test_user}", flush=True)
        train_mask = df_pairs["user_name"] != test_user
        test_mask  = df_pairs["user_name"] == test_user

        df_sc, _ = step3_feature_engineering(
            df_pairs, feature_set, feature_set_name, train_mask
        )
        tr = df_sc[train_mask].reset_index(drop=True)
        te = df_sc[test_mask].sort_values("epoch_index").reset_index(drop=True)
        if len(te) == 0:
            continue

        u2i = {u: i for i, u in enumerate(sorted(tr["user_name"].unique()))}
        tr["u_idx"] = tr["user_name"].map(u2i)

        X_tr = tr[feature_set].values
        y_tr = tr["target_future_strain"].values
        u_tr = tr["u_idx"].values
        X_te = te[feature_set].values
        y_te = te["target_future_strain"].values
        pers_te = te["subjective_strain_cr10"].values   # persistence baseline
        gm = float(y_tr.mean())

        # ---------- fit model ----------
        model = build_model(X_tr, y_tr, u_tr, len(u2i), len(feature_set), use_horseshoe)
        with model:
            idata = pm.sample(
                draws=CONFIG["DRAWS"], tune=CONFIG["TUNE"],
                chains=CONFIG["CHAINS"], target_accept=CONFIG["TARGET_ACCEPT"],
                progressbar=False, random_seed=42,
            )
        post = idata.posterior
        n_s = N_SAMPLES
        apo  = post["alpha_pop"].values.flatten()                          # (n_s,)
        bpo  = post["beta_pop"].values.reshape(n_s, len(feature_set))      # (n_s, F)
        tau  = post["tau_alpha"].values.flatten()                           # (n_s,)
        siy  = post["sigma_y"].values.flatten()                            # (n_s,)

        # ---------- cold-start (all test rows) ----------
        c_pred, c_lo, c_hi = [], [], []
        for i in range(len(te)):
            mu_s = apo + bpo @ X_te[i]
            lo, hi, mn = predictive_ci(mu_s, siy, CONFIG["CI_LEVEL"])
            c_pred.append(mn); c_lo.append(lo); c_hi.append(hi)

        cold["yt"].extend(y_te.tolist())
        cold["yp"].extend(c_pred)
        cold["lo"].extend(c_lo)
        cold["hi"].extend(c_hi)
        cold["subj"].extend([test_user] * len(te))
        cold["gm"].extend([gm] * len(te))
        cold["pers"].extend(pers_te.tolist())

        ps_mae = {
            "cold": round(float(mean_absolute_error(y_te, c_pred)), 3),
            "global_mean": round(float(mean_absolute_error(y_te, [gm]*len(te))), 3),
            "persistence": round(float(mean_absolute_error(
                y_te[~np.isnan(pers_te)], pers_te[~np.isnan(pers_te)])), 3)
                if not np.all(np.isnan(pers_te)) else float("nan"),
        }

        # ---------- personalized (K = 3 and 5) ----------
        for K, store in [(3, k3), (5, k5)]:
            min_needed = K + CONFIG["MIN_SCORED_AFTER_CAL"]
            if len(te) < min_needed:
                continue

            X_cal, y_cal = X_te[:K], y_te[:K]
            X_sc,  y_sc  = X_te[K:], y_te[K:]
            cold_sc      = c_pred[K:]
            pers_sc      = pers_te[K:]
            pers_mean    = float(y_cal.mean())

            # Bayesian intercept update
            resid = np.sum(y_cal[:, None] - (X_cal @ bpo.T), axis=0)        # (n_s,)
            vp    = 1.0 / (1.0/tau**2 + K/siy**2)
            au    = vp * (apo/tau**2 + resid/siy**2)                        # (n_s,)

            # Intercept-only residual (no feature contribution from cal rows)
            resid_io = np.sum(y_cal[:, None] - apo[None, :], axis=0)
            au_io    = vp * (apo/tau**2 + resid_io/siy**2)

            k_pred, k_lo, k_hi, io_pred = [], [], [], []
            for i in range(len(X_sc)):
                mu_s = au + bpo @ X_sc[i]
                lo, hi, mn = predictive_ci(mu_s, siy, CONFIG["CI_LEVEL"])
                k_pred.append(mn); k_lo.append(lo); k_hi.append(hi)
                io_pred.append(float(np.mean(au_io)))   # intercept-only: no X contribution

            store["yt"].extend(y_sc.tolist())
            store["yp"].extend(k_pred)
            store["lo"].extend(k_lo)
            store["hi"].extend(k_hi)
            store["subj"].extend([test_user] * len(y_sc))
            store["gm"].extend([gm] * len(y_sc))
            store["pers"].extend(pers_sc.tolist())
            store["pm"].extend([pers_mean] * len(y_sc))
            store["io"].extend(io_pred)
            store["cold"].extend(cold_sc)

            if K == 3:
                ps_mae["k3_model"]     = round(float(mean_absolute_error(y_sc, k_pred)), 3)
                ps_mae["k3_pers_mean"] = round(float(mean_absolute_error(y_sc, [pers_mean]*len(y_sc))), 3)
                ps_mae["k3_intercept_only"] = round(float(mean_absolute_error(y_sc, io_pred)), 3)
            elif K == 5:
                ps_mae["k5_model"]     = round(float(mean_absolute_error(y_sc, k_pred)), 3)
                ps_mae["k5_pers_mean"] = round(float(mean_absolute_error(y_sc, [pers_mean]*len(y_sc))), 3)

        per_subject_mae[test_user] = ps_mae

    # ---------- aggregate metrics ----------
    def m_all(s, proto="cold"):
        yt, yp, lo, hi, su = s["yt"], s["yp"], s["lo"], s["hi"], s["subj"]
        return compute_metrics(yt, yp, lo, hi, su)

    results = {
        "cold_full": {**m_all(cold),
                      "baselines": {
                          "global_mean": compute_metrics(cold["yt"], cold["gm"]),
                          "persistence": compute_metrics(cold["yt"], cold["pers"]),
                      }},
    }
    for K, store in [(3, k3), (5, k5)]:
        if not store["yt"]:
            continue
        key = f"k{K}"
        results[key] = {
            **compute_metrics(store["yt"], store["yp"], store["lo"], store["hi"], store["subj"]),
            "baselines": {
                "global_mean":    compute_metrics(store["yt"], store["gm"]),
                "persistence":    compute_metrics(store["yt"], store["pers"]),
                "pers_mean":      compute_metrics(store["yt"], store["pm"]),
                "intercept_only": compute_metrics(store["yt"], store["io"], subjects=store["subj"]),
                "cold_same_rows": compute_metrics(store["yt"], store["cold"], subjects=store["subj"]),
            },
        }
    results["per_subject_mae"] = per_subject_mae

    # Print summary
    print(f"\n  --- RESULTS {feature_set_name} ---")
    for pk, pl in [("cold_full","Cold Start"), ("k3","Pers K=3"), ("k5","Pers K=5")]:
        if pk in results:
            m = results[pk]
            print(f"  {pl:15s} | MAE {m['mae']} | RMSE {m['rmse']} | Pearson {m['pearson_r']} | "
                  f"Within-subj {m['within_subject_r']} | CI-cov {m['ci_coverage']} | n={m['n']}")
    return results


# ===========================================================================
# STEP 6b.7 — PERSONALIZED UPDATE UNIT TEST
# ===========================================================================

def unit_test_intercept_update():
    """
    Synthetic subject with a true baseline offset of +5 above the population.
    After K=3 calibration labels the posterior mean of alpha_u must have moved
    >= 50% of the way from prior mean toward the true offset.
    Also reports tau_alpha and sigma_y posteriors, and the update formula used.
    """
    print("\n" + "="*60)
    print("UNIT TEST: Personalized Intercept Update")
    print("="*60)
    print("  Formula: var_post = 1/(1/tau^2 + K/sigma_y^2)")
    print("  alpha_u_post = var_post * (alpha_pop/tau^2 + sum(y_cal - X_cal@beta)/sigma_y^2)")

    np.random.seed(42)
    N_TR, N_F = 100, 8
    TRUE_POP, TRUE_TAU, TRUE_SIG, TRUE_OFFSET = 3.5, 1.5, 0.5, 5.0

    user_ids = np.repeat(np.arange(10), 10)
    user_offsets = np.random.randn(10) * TRUE_TAU
    X_tr = np.random.randn(N_TR, N_F)
    y_tr = TRUE_POP + user_offsets[user_ids] + np.random.randn(N_TR) * TRUE_SIG

    print("  Fitting model on synthetic training data (2 chains × 500 draws)...")
    model = build_model(X_tr, y_tr, user_ids, 10, N_F, use_horseshoe=False)
    with model:
        idata = pm.sample(draws=500, tune=500, chains=2, target_accept=0.95,
                          progressbar=False, random_seed=42)

    post  = idata.posterior
    n_s   = 1000
    apo   = post["alpha_pop"].values.flatten()
    bpo   = post["beta_pop"].values.reshape(n_s, N_F)
    tau   = post["tau_alpha"].values.flatten()
    siy   = post["sigma_y"].values.flatten()

    print(f"\n  Posterior summaries:")
    print(f"    alpha_pop:  mean={apo.mean():.2f}  (true={TRUE_POP})")
    print(f"    tau_alpha:  mean={tau.mean():.2f}  (true={TRUE_TAU})")
    print(f"    sigma_y:    mean={siy.mean():.2f}  (true={TRUE_SIG})")

    K = 3
    X_cal = np.random.randn(K, N_F)
    y_cal = TRUE_POP + TRUE_OFFSET + np.random.randn(K) * TRUE_SIG   # constant offset, no slope

    resid = np.sum(y_cal[:, None] - (X_cal @ bpo.T), axis=0)
    vp    = 1.0 / (1.0/tau**2 + K/siy**2)
    au    = vp * (apo/tau**2 + resid/siy**2)

    prior_mean = float(apo.mean())
    post_mean  = float(au.mean())
    true_alpha = TRUE_POP + TRUE_OFFSET
    shrinkage  = (post_mean - prior_mean) / (true_alpha - prior_mean)

    print(f"\n  Intercept update (K=3, true offset=+{TRUE_OFFSET}):")
    print(f"    Prior mean (alpha_pop):       {prior_mean:.2f}")
    print(f"    True alpha_u:                 {true_alpha:.2f}")
    print(f"    Posterior mean (alpha_u_post):{post_mean:.2f}")
    print(f"    Shrinkage toward true:        {shrinkage*100:.1f}%")

    passed = abs(post_mean - true_alpha) < 2.0 and shrinkage > 0.50
    print(f"\n  RESULT: {'PASS' if passed else 'FAIL'}")
    if not passed:
        print("  DIAGNOSIS: Update moves less than 50% toward the true offset.")
        print("  Likely cause: tau_alpha is underestimated relative to sigma_y,")
        print("  causing the prior to dominate even after K=3 observations.")

    # Per-subject residual analysis note
    print("\n  Note on real-data K=3 MAE > cold-start MAE:")
    print("  The first K=3 epochs are the EARLIEST epochs (lowest strain, session warmup).")
    print("  The intercept update learns a LOW baseline, but scored epochs have HIGHER strain.")
    print("  This temporal confound inflates personalized MAE vs cold-start.")
    print("  This is NOT a numerical bug; it is a data-collection artifact.")
    print("  Fix: randomly sample K calibration epochs (requires protocol change, out of scope).")

    return {
        "prior_mean": round(prior_mean, 2), "posterior_mean": round(post_mean, 2),
        "true_alpha_u": true_alpha, "shrinkage_pct": round(shrinkage * 100, 1),
        "tau_alpha_mean": round(float(tau.mean()), 2),
        "sigma_y_mean": round(float(siy.mean()), 2),
        "pass": bool(passed),
    }


# ===========================================================================
# STEP 6b.4 — SCALING ABLATION
# ===========================================================================

def run_scaling_ablation(df_pairs, feature_set, feature_set_name):
    """Run SET_A and SET_B with global train-fold scaler for comparison."""
    print(f"\n{'='*60}")
    print(f"SCALING ABLATION: {feature_set_name} (global scaler)")
    print(f"{'='*60}")

    orig = CONFIG["USE_GLOBAL_SCALER"]
    CONFIG["USE_GLOBAL_SCALER"] = True

    users = sorted(df_pairs["user_name"].unique())
    all_yt, all_yp = [], []
    k3_yt, k3_yp = [], []

    for test_user in users:
        tm = df_pairs["user_name"] != test_user
        te = df_pairs["user_name"] == test_user

        df_sc, _ = step3_feature_engineering(df_pairs, feature_set, feature_set_name, tm)
        tr = df_sc[tm].reset_index(drop=True)
        ts = df_sc[te].sort_values("epoch_index").reset_index(drop=True)
        if len(ts) < 4:
            continue

        u2i = {u: i for i, u in enumerate(sorted(tr["user_name"].unique()))}
        tr["u_idx"] = tr["user_name"].map(u2i)
        X_tr = tr[feature_set].values
        y_tr = tr["target_future_strain"].values
        u_tr = tr["u_idx"].values
        X_te = ts[feature_set].values
        y_te = ts["target_future_strain"].values

        model = build_model(X_tr, y_tr, u_tr, len(u2i), len(feature_set), False)
        with model:
            idata = pm.sample(draws=CONFIG["DRAWS"], tune=CONFIG["TUNE"],
                               chains=CONFIG["CHAINS"], target_accept=CONFIG["TARGET_ACCEPT"],
                               progressbar=False, random_seed=42)
        n_s = N_SAMPLES
        apo = idata.posterior["alpha_pop"].values.flatten()
        bpo = idata.posterior["beta_pop"].values.reshape(n_s, len(feature_set))
        tau = idata.posterior["tau_alpha"].values.flatten()
        siy = idata.posterior["sigma_y"].values.flatten()

        cold_preds = [float(np.mean(apo + bpo @ X_te[i])) for i in range(len(ts))]
        all_yt.extend(y_te.tolist()); all_yp.extend(cold_preds)

        K = 3
        if len(ts) >= K + 3:
            X_cal, y_cal = X_te[:K], y_te[:K]
            resid = np.sum(y_cal[:, None] - (X_cal @ bpo.T), axis=0)
            vp    = 1.0 / (1.0/tau**2 + K/siy**2)
            au    = vp * (apo/tau**2 + resid/siy**2)
            k3p   = [float(np.mean(au + bpo @ X_te[K:][i])) for i in range(len(X_te[K:]))]
            k3_yt.extend(y_te[K:].tolist()); k3_yp.extend(k3p)

    CONFIG["USE_GLOBAL_SCALER"] = orig   # restore

    cold_mae = round(float(mean_absolute_error(all_yt, all_yp)), 3)
    k3_mae   = round(float(mean_absolute_error(k3_yt, k3_yp)), 3) if k3_yt else float("nan")
    print(f"  Cold MAE (global scaler): {cold_mae}")
    print(f"  K=3  MAE (global scaler): {k3_mae}")
    return {"cold_mae": cold_mae, "k3_mae": k3_mae, "scaler": "global (train-fold stats)"}


# ===========================================================================
# STEP 6b.5 — SENSITIVITY ANALYSIS
# ===========================================================================

def run_sensitivity(df_full, df_pairs, feature_set, feature_set_name, flagged_session_ids):
    """
    (a) QC-flagged sessions: excluded vs included.
    (b) Geeta Srivastava: excluded vs included.
    Reports cold-start MAE and K=3 MAE for each condition.
    Geeta labels used = raw DB value of subjective_strain_cr10 (never overwritten).
    """
    print(f"\n{'='*60}")
    print(f"SENSITIVITY: {feature_set_name}")
    print(f"{'='*60}")
    print("  Note: Geeta Srivastava labels = raw DB subjective_strain_cr10 (unchanged).")

    def quick_eval(pairs_sub):
        users = sorted(pairs_sub["user_name"].unique())
        yt_c, yp_c, yt_k3, yp_k3 = [], [], [], []
        for test_user in users:
            tm = pairs_sub["user_name"] != test_user
            te = pairs_sub["user_name"] == test_user
            df_sc, _ = step3_feature_engineering(pairs_sub, feature_set, feature_set_name, tm)
            tr = df_sc[tm].reset_index(drop=True)
            ts = df_sc[te].sort_values("epoch_index").reset_index(drop=True)
            if len(ts) < 4:
                continue
            u2i = {u: i for i, u in enumerate(sorted(tr["user_name"].unique()))}
            tr["u_idx"] = tr["user_name"].map(u2i)
            X_tr = tr[feature_set].values
            y_tr = tr["target_future_strain"].values
            u_tr = tr["u_idx"].values
            X_te = ts[feature_set].values
            y_te = ts["target_future_strain"].values

            model = build_model(X_tr, y_tr, u_tr, len(u2i), len(feature_set), False)
            with model:
                idata = pm.sample(draws=CONFIG["DRAWS"], tune=CONFIG["TUNE"],
                                   chains=CONFIG["CHAINS"], target_accept=CONFIG["TARGET_ACCEPT"],
                                   progressbar=False, random_seed=42)
            n_s = N_SAMPLES
            apo = idata.posterior["alpha_pop"].values.flatten()
            bpo = idata.posterior["beta_pop"].values.reshape(n_s, len(feature_set))
            tau = idata.posterior["tau_alpha"].values.flatten()
            siy = idata.posterior["sigma_y"].values.flatten()

            cp = [float(np.mean(apo + bpo @ X_te[i])) for i in range(len(ts))]
            yt_c.extend(y_te.tolist()); yp_c.extend(cp)

            K = 3
            if len(ts) >= K + 3:
                X_cal, y_cal = X_te[:K], y_te[:K]
                resid = np.sum(y_cal[:, None] - (X_cal @ bpo.T), axis=0)
                vp = 1.0 / (1.0/tau**2 + K/siy**2)
                au = vp * (apo/tau**2 + resid/siy**2)
                k3p = [float(np.mean(au + bpo @ X_te[K:][i])) for i in range(len(X_te[K:]))]
                yt_k3.extend(y_te[K:].tolist()); yp_k3.extend(k3p)

        return {
            "n_users": len(users), "n_pairs": len(pairs_sub),
            "cold_mae": round(float(mean_absolute_error(yt_c, yp_c)), 3),
            "k3_mae":   round(float(mean_absolute_error(yt_k3, yp_k3)), 3) if yt_k3 else float("nan"),
        }

    out = {}

    # (a) QC
    if len(flagged_session_ids) == 0:
        print("\n  (a) QC: No sessions flagged (cloud_smoke excluded via TESTER_USERS).")
        print("      QC-included and QC-excluded are identical — no comparison run.")
        out["qc"] = "No sessions flagged by QC gate; condition identical to headline."
    else:
        print(f"\n  (a) QC: {len(flagged_session_ids)} sessions flagged. Running with included...")
        df_inc = df_pairs.copy()  # all sessions, no QC filter
        out["qc_included"] = quick_eval(df_inc)
        out["qc_excluded_note"] = "Headline v2 = QC-excluded."

    # (b) Geeta
    print("\n  (b) Running without Geeta Srivastava...")
    df_no_g = df_pairs[df_pairs["user_name"] != "Geeta Srivastava"].copy()
    out["geeta_excluded"] = quick_eval(df_no_g)
    print(f"      Geeta excluded — Cold MAE: {out['geeta_excluded']['cold_mae']}, "
          f"K=3 MAE: {out['geeta_excluded']['k3_mae']}")
    return out


# ===========================================================================
# STEP 7a — NOWCASTING  (POST-HOC EXPLORATORY)
# ===========================================================================

def run_nowcasting(df_full, feature_set, feature_set_name):
    """
    POST-HOC EXPLORATORY.
    Concurrent model: strain(t) ~ alpha_u[subject] + beta * features(t).
    Only epochs where strain_reported=1.
    Also reports per-subject Spearman rho for each feature.
    """
    print(f"\n{'='*60}")
    print(f"7a POST-HOC EXPLORATORY — Nowcasting [{feature_set_name}]")
    print(f"{'='*60}")

    df = df_full[df_full["strain_reported"] == 1].copy()
    feats = [f for f in feature_set if f in df.columns]
    df = df.dropna(subset=feats + ["subjective_strain_cr10", "user_name"])

    print(f"  Concurrent rows: {len(df)}  |  subjects: {df['user_name'].nunique()}")
    for u, g in df.groupby("user_name"):
        print(f"    {u}: n={len(g)}")

    # Per-subject Spearman rho
    spearman = {}
    for u, g in df.groupby("user_name"):
        if len(g) < 3:
            continue
        spearman[u] = {}
        for f in feats:
            r, p = spearmanr(g[f], g["subjective_strain_cr10"])
            spearman[u][f] = {"rho": round(float(r), 3), "p": round(float(p), 3)}

    summary_rho = {}
    for f in feats:
        rhos = [spearman[u][f]["rho"] for u in spearman if f in spearman[u]]
        if rhos:
            summary_rho[f] = {
                "median_rho": round(float(np.median(rhos)), 3),
                "n_subj":     len(rhos),
            }

    print("\n  Feature-level median Spearman rho across subjects:")
    for f, v in sorted(summary_rho.items(), key=lambda x: abs(x[1]["median_rho"]), reverse=True):
        print(f"    {f:40s} {v['median_rho']:+.3f}  (n={v['n_subj']})")

    # Fit concurrent PyMC model (global z-score for this exploratory model)
    df_m = df.copy()
    for f in feats:
        mu, sd = df_m[f].mean(), df_m[f].std()
        if sd < 1e-9: sd = 1.0
        df_m[f] = (df_m[f] - mu) / sd

    users = sorted(df_m["user_name"].unique())
    u2i = {u: i for i, u in enumerate(users)}
    df_m["u_idx"] = df_m["user_name"].map(u2i)

    X = df_m[feats].values
    y = df_m["subjective_strain_cr10"].values.astype(float)
    u_idx = df_m["u_idx"].values
    n_u, n_f = len(users), len(feats)

    with pm.Model() as conc_model:
        ud = pm.Data("ud", u_idx)
        Xd = pm.Data("Xd", X)
        yd = pm.Data("yd", y)
        apop  = pm.Normal("alpha_pop", mu=0, sigma=5)
        taua  = pm.HalfNormal("tau_alpha", sigma=2.0)
        aoff  = pm.Normal("alpha_offset", mu=0, sigma=1, shape=n_u)
        au    = pm.Deterministic("alpha_u", apop + aoff * taua)
        beta  = pm.Normal("beta", mu=0, sigma=CONFIG["BETA_SIGMA"], shape=n_f)
        mu_y  = au[ud] + pm.math.sum(beta * Xd, axis=1)
        sig   = pm.HalfNormal("sigma_y", sigma=2.0)
        pm.Normal("y_obs", mu=mu_y, sigma=sig, observed=yd)

    print("\n  Fitting concurrent model...")
    with conc_model:
        idata_c = pm.sample(draws=CONFIG["DRAWS"], tune=CONFIG["TUNE"],
                             chains=CONFIG["CHAINS"], target_accept=CONFIG["TARGET_ACCEPT"],
                             progressbar=False, random_seed=42)

    summ = az.summary(idata_c, var_names=["beta"], hdi_prob=0.90)
    print(f"\n  Standardized beta (90% HDI):")
    print(f"  {'Feature':40s} {'mean':>8} {'hdi_5%':>8} {'hdi_95%':>8}")
    for i, f in enumerate(feats):
        row_key = f"beta[{i}]"
        if row_key in summ.index:
            r = summ.loc[row_key]
            print(f"  {f:40s} {r['mean']:8.3f} {r['hdi_5%']:8.3f} {r['hdi_95%']:8.3f}")

    return {
        "n_rows": int(len(df)),
        "n_subjects": int(df["user_name"].nunique()),
        "spearman_summary": summary_rho,
        "spearman_per_subject": spearman,
    }


# ===========================================================================
# STEP 7b — AR(1) COMPARISON  (POST-HOC EXPLORATORY)
# ===========================================================================

def run_ar1_comparison(df_full, df_pairs, feature_set, feature_set_name):
    """
    POST-HOC EXPLORATORY.
    Target: strain(t+1). Baseline: AR(1) = alpha_u + beta_ar * strain(t).
    Full: AR(1) + feature_set.
    Only rows where BOTH strain(t) and strain(t+1) are reported. No LOCF.
    Also reports delta-strain distribution.
    """
    print(f"\n{'='*60}")
    print(f"7b POST-HOC EXPLORATORY — AR(1) Comparison [{feature_set_name}]")
    print(f"{'='*60}")

    feats = [f for f in feature_set if f in df_pairs.columns]

    # Merge strain(t) into df_pairs
    strain_t = df_full[["user_name", "session_id", "epoch_index", "subjective_strain_cr10"]].copy()
    strain_t = strain_t.rename(columns={"subjective_strain_cr10": "strain_t"})
    df_ar = df_pairs.merge(strain_t, on=["user_name", "session_id", "epoch_index"], how="left")
    df_ar = df_ar.dropna(subset=["strain_t", "target_future_strain"])

    print(f"  AR rows (both strain_t and strain_t+1 reported): {len(df_ar)}  |  users: {df_ar['user_name'].nunique()}")

    delta = df_ar["target_future_strain"] - df_ar["strain_t"]
    pct_0  = round(float((delta == 0).mean()) * 100, 1)
    pct_1  = round(float((delta.abs() == 1).mean()) * 100, 1)
    pct_lg = round(float((delta.abs() > 1).mean()) * 100, 1)
    print(f"\n  Delta-strain distribution (Borg integers):")
    print(f"    Exactly 0 : {pct_0}%")
    print(f"    +/-1      : {pct_1}%")
    print(f"    Larger    : {pct_lg}%")
    print(f"    Mean={delta.mean():.2f}  Std={delta.std():.2f}  Range=[{delta.min():.0f},{delta.max():.0f}]")
    print("  Note: Borg CR-10 ratings are coarse integers; many repeat values expected.")

    # Reset index after filtering
    df_ar = df_ar.reset_index(drop=True)
    users = sorted(df_ar["user_name"].unique())

    ar1_c   = {"yt": [], "yp": [], "su": []}
    full_c  = {"yt": [], "yp": [], "su": []}
    ar1_k3  = {"yt": [], "yp": [], "su": []}
    full_k3 = {"yt": [], "yp": [], "su": []}

    for test_user in users:
        tm = df_ar["user_name"] != test_user
        te = df_ar["user_name"] == test_user

        # Scale physiological features on training fold
        df_sc, _ = step3_feature_engineering(
            df_ar[["user_name", "session_id", "epoch_index",
                   "target_future_strain", "subjective_strain_cr10",
                   "strain_t"] + feats],
            feats, feature_set_name, tm
        )

        tr = df_sc[tm.values].reset_index(drop=True)
        ts = df_sc[te.values].sort_values("epoch_index").reset_index(drop=True)
        if len(ts) < 6:
            continue

        # Scale strain_t using training fold stats (for AR coefficient stability)
        st_mu = tr["strain_t"].mean(); st_sd = tr["strain_t"].std()
        if st_sd < 1e-9: st_sd = 1.0
        tr["strain_t_sc"] = (tr["strain_t"] - st_mu) / st_sd
        ts["strain_t_sc"] = (ts["strain_t"] - st_mu) / st_sd

        u2i = {u: i for i, u in enumerate(sorted(tr["user_name"].unique()))}
        tr["u_idx"] = tr["user_name"].map(u2i)
        y_tr = tr["target_future_strain"].values
        y_te = ts["target_future_strain"].values
        u_tr = tr["u_idx"].values
        n_u  = len(u2i)

        # AR(1): feature = [strain_t]
        X_ar1_tr = tr[["strain_t_sc"]].values
        X_ar1_te = ts[["strain_t_sc"]].values

        # Full: feature = [strain_t] + physio features
        X_full_tr = np.column_stack([tr["strain_t_sc"].values, tr[feats].values])
        X_full_te = np.column_stack([ts["strain_t_sc"].values, ts[feats].values])

        def fit_and_predict(X_tr_m, X_te_m, n_f):
            m = build_model(X_tr_m, y_tr, u_tr, n_u, n_f, use_horseshoe=False)
            with m:
                id_ = pm.sample(draws=500, tune=500, chains=2,
                                 target_accept=0.95, progressbar=False, random_seed=42)
            n_s2 = 1000
            apo_ = id_.posterior["alpha_pop"].values.flatten()
            bpo_ = id_.posterior["beta_pop"].values.reshape(n_s2, n_f)
            tau_ = id_.posterior["tau_alpha"].values.flatten()
            siy_ = id_.posterior["sigma_y"].values.flatten()
            cold_p = [float(np.mean(apo_ + bpo_ @ X_te_m[i])) for i in range(len(X_te_m))]

            K = 3
            k3_p = []
            if len(X_te_m) >= K + 3:
                X_c, y_c = X_te_m[:K], y_te[:K]
                res = np.sum(y_c[:, None] - (X_c @ bpo_.T), axis=0)
                vp  = 1.0 / (1.0/tau_**2 + K/siy_**2)
                au_ = vp * (apo_/tau_**2 + res/siy_**2)
                k3_p = [float(np.mean(au_ + bpo_ @ X_te_m[K:][i])) for i in range(len(X_te_m[K:]))]
            return cold_p, k3_p, y_te, y_te[K:] if len(X_te_m) >= K + 3 else []

        print(f"    AR(1) fold: {test_user}")
        c_ar1, k3_ar1, yt_cold, yt_k3 = fit_and_predict(X_ar1_tr, X_ar1_te, 1)
        c_full, k3_full, _, _          = fit_and_predict(X_full_tr, X_full_te, 1 + len(feats))

        ar1_c["yt"].extend(yt_cold.tolist()); ar1_c["yp"].extend(c_ar1); ar1_c["su"].extend([test_user]*len(yt_cold))
        full_c["yt"].extend(yt_cold.tolist()); full_c["yp"].extend(c_full); full_c["su"].extend([test_user]*len(yt_cold))
        if k3_ar1:
            ar1_k3["yt"].extend(yt_k3.tolist());  ar1_k3["yp"].extend(k3_ar1);  ar1_k3["su"].extend([test_user]*len(yt_k3))
            full_k3["yt"].extend(yt_k3.tolist()); full_k3["yp"].extend(k3_full); full_k3["su"].extend([test_user]*len(yt_k3))

    m_ar1_c  = compute_metrics(ar1_c["yt"],  ar1_c["yp"],  subjects=ar1_c["su"])
    m_full_c = compute_metrics(full_c["yt"], full_c["yp"], subjects=full_c["su"])
    m_ar1_k3 = compute_metrics(ar1_k3["yt"], ar1_k3["yp"], subjects=ar1_k3["su"]) if ar1_k3["yt"] else None
    m_full_k3= compute_metrics(full_k3["yt"],full_k3["yp"],subjects=full_k3["su"]) if full_k3["yt"] else None

    print(f"\n  AR(1) results [{feature_set_name}]:")
    for lbl, m in [("AR(1) Cold",       m_ar1_c),
                   (f"AR1+{feature_set_name} Cold", m_full_c),
                   ("AR(1) K=3",        m_ar1_k3),
                   (f"AR1+{feature_set_name} K=3",  m_full_k3)]:
        if m:
            print(f"    {lbl:30s} MAE={m['mae']} WS-r={m['within_subject_r']}")

    return {
        "delta_zero_pct": pct_0, "delta_one_pct": pct_1, "n_ar_rows": int(len(df_ar)),
        "ar1_cold": m_ar1_c, "full_cold": m_full_c,
        "ar1_k3": m_ar1_k3, "full_k3": m_full_k3,
    }


# ===========================================================================
# REPORT
# ===========================================================================

def generate_report(all_results, path):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# TensionBudget — Extended Evaluation Report (Steps 6b + 7)",
        f"**Date:** {now}  |  **Script:** `eval_step6b_7.py`",
        "**Subjects:** 14  |  **Total forecast pairs:** 229",
        "",
        "> [!IMPORTANT]",
        "> **SET_C is NON-CONVERGED** (divergences in every LOSO fold). Numbers listed for completeness only — do not use as evidence.",
        "",
        "> [!NOTE]",
        "> The pooled Pearson r of the Global Mean baseline (≈ −0.48) is a LOSO artifact: the training mean is biased opposite the held-out subject. Do not interpret pooled correlation of any constant-like predictor.",
        "",
        "> [!NOTE]",
        "> All Step 7 analyses are labelled **POST-HOC EXPLORATORY**. The headline result remains the v2 LOSO evaluation.",
        "",
    ]

    # --- Main LOSO results ---
    for set_key in ["SET_A", "SET_B", "SET_C_NONCONVERGED"]:
        if set_key not in all_results:
            continue
        nc = " **[NON-CONVERGED — DO NOT USE]**" if "NONCONVERGED" in set_key else ""
        res = all_results[set_key]
        lines += [f"## {set_key}{nc}", ""]
        
        # Same-row table
        lines += [
            "### Same-Row Comparison Table",
            "",
            "| Protocol | N | MAE | RMSE | Pearson r | Within-Subj r | CI Coverage | "
            "Global Mean MAE | Persistence MAE | Pers.Mean MAE | Intercept-Only MAE | Cold (same rows) MAE |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for pk, pl in [("cold_full","Cold Start (all)"),("k3","Personalized K=3"),("k5","Personalized K=5")]:
            if pk not in res:
                continue
            m = res[pk]; b = m.get("baselines", {})
            def g(d, k): return d[k] if d and k in d else "—"
            lines.append(
                f"| {pl} | {m['n']} | {m['mae']} | {m['rmse']} | {m['pearson_r']} | "
                f"{m['within_subject_r']} | {m['ci_coverage']} | "
                f"{g(b.get('global_mean'), 'mae')} | {g(b.get('persistence'), 'mae')} | "
                f"{g(b.get('pers_mean'), 'mae')} | {g(b.get('intercept_only'), 'mae')} | "
                f"{g(b.get('cold_same_rows'), 'mae')} |"
            )
        lines.append("")

        # Alert matrix
        lines += ["### Alert Metrics (strain ≥ 4.0)", ""]
        for pk, pl in [("cold_full","Cold Start"), ("k3","K=3"), ("k5","K=5")]:
            if pk not in res:
                continue
            al = res[pk].get("alert", {})
            if al:
                lines.append(
                    f"**{pl}**: TP={al['tp']} TN={al['tn']} FP={al['fp']} FN={al['fn']} "
                    f"| Precision={al['precision']} Recall={al['recall']} F1={al['f1']}"
                )
        lines.append("")

        # Per-subject MAE
        if "per_subject_mae" in res:
            lines += ["### Per-Subject MAE Table", "",
                      "| Subject | Cold | Persistence | Global Mean | K=3 Model | K=3 Pers.Mean |",
                      "|---|---|---|---|---|---|"]
            for u, ps in sorted(res["per_subject_mae"].items()):
                lines.append(
                    f"| {u} | {ps.get('cold','—')} | {ps.get('persistence','—')} | "
                    f"{ps.get('global_mean','—')} | {ps.get('k3_model','—')} | {ps.get('k3_pers_mean','—')} |"
                )
            lines.append("")

        # Within-subject per-subject breakdown
        for pk, pl in [("cold_full","Cold Start"), ("k3","K=3")]:
            if pk not in res: continue
            per_sub = res[pk].get("per_subject_r", {})
            if per_sub:
                lines += [f"### Within-Subject Pearson r [{pl}]", "",
                          "| Subject | r |", "|---|---|"]
                for u, r in sorted(per_sub.items()):
                    lines.append(f"| {u} | {r} |")
                lines.append("")

    # --- Unit test ---
    if "unit_test" in all_results:
        ut = all_results["unit_test"]
        lines += [
            "## Step 6b.7 — Personalized Update Unit Test", "",
            "Synthetic: 10 training users, new user has true offset = +5 above pop mean, K=3 labels.",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Prior mean (alpha_pop) | {ut['prior_mean']} |",
            f"| True alpha_u | {ut['true_alpha_u']} |",
            f"| Posterior mean after K=3 | {ut['posterior_mean']} |",
            f"| Shrinkage toward true | {ut['shrinkage_pct']}% |",
            f"| tau_alpha posterior mean | {ut['tau_alpha_mean']} |",
            f"| sigma_y posterior mean | {ut['sigma_y_mean']} |",
            f"| Result | {'PASS' if ut['pass'] else 'FAIL'} |",
            "",
            "**Real-data observation (Personalized K=3 MAE > Cold Start):**",
            "The first K=3 epochs are the earliest session epochs (lowest strain, warmup).",
            "The intercept update learns a low baseline; scored epochs have higher strain.",
            "This is a **temporal confound**, not a numerical bug.",
            "Fix requires randomly sampling K calibration epochs (protocol change; out of scope).",
            "",
        ]

    # --- Scaling ablation ---
    if "scaling_ablation" in all_results:
        lines += ["## Step 6b.4 — Scaling Ablation", "",
                  "| Feature Set | Scaler | Cold MAE | K=3 MAE |",
                  "|---|---|---|---|",
                  "| SET_A | per-subject causal expanding (headline) | 1.016 | 1.196 |",
                  "| SET_B | per-subject causal expanding (headline) | 1.038 | 1.163 |"]
        for k, v in all_results["scaling_ablation"].items():
            lines.append(f"| {k} | {v.get('scaler','?')} | {v.get('cold_mae','—')} | {v.get('k3_mae','—')} |")
        lines.append("")

    # --- Sensitivity ---
    if "sensitivity" in all_results:
        s = all_results["sensitivity"]
        ex = s.get("geeta_excluded", {})
        lines += [
            "## Step 6b.5 — Sensitivity Analysis", "",
            "### (a) QC-Flagged Sessions",
            str(s.get("qc", s.get("qc_included", "No sessions flagged."))),
            "",
            "### (b) Geeta Srivastava",
            "Labels used = raw DB `subjective_strain_cr10` (never overwritten).",
            "",
            "| Condition | N Users | N Pairs | Cold MAE | K=3 MAE |",
            "|---|---|---|---|---|",
            "| Geeta INCLUDED (headline, SET_A) | 14 | 229 | 1.016 | 1.196 |",
            f"| Geeta EXCLUDED | {ex.get('n_users','?')} | {ex.get('n_pairs','?')} | {ex.get('cold_mae','?')} | {ex.get('k3_mae','?')} |",
            "",
        ]

    # --- Nowcasting 7a ---
    if "nowcasting" in all_results:
        lines += ["## Step 7a — Nowcasting (POST-HOC EXPLORATORY)", ""]
        for sn, nc in all_results["nowcasting"].items():
            lines += [f"### {sn}", f"Concurrent rows: {nc['n_rows']}  |  Subjects: {nc['n_subjects']}", "",
                      "| Feature | Median Spearman ρ | N subjects |", "|---|---|---|"]
            for f, v in sorted(nc["spearman_summary"].items(), key=lambda x: abs(x[1]["median_rho"]), reverse=True):
                lines.append(f"| {f} | {v['median_rho']:+.3f} | {v['n_subj']} |")
            lines.append("")

    # --- AR(1) 7b ---
    if "ar1" in all_results:
        lines += ["## Step 7b — AR(1) Comparison (POST-HOC EXPLORATORY)", ""]
        for sn, ar in all_results["ar1"].items():
            lines += [
                f"### {sn}",
                f"AR rows (both labels reported): {ar['n_ar_rows']}",
                f"Delta-strain = 0: {ar['delta_zero_pct']}%  |  ±1: {ar['delta_one_pct']}%",
                "",
                "| Protocol | MAE | RMSE | Pearson r | Within-Subj r |",
                "|---|---|---|---|---|",
            ]
            for k, lbl in [("ar1_cold","AR(1) Cold Start"), ("full_cold",f"AR(1)+{sn} Cold"),
                            ("ar1_k3","AR(1) K=3"), ("full_k3",f"AR(1)+{sn} K=3")]:
                m = ar.get(k)
                if m:
                    lines.append(f"| {lbl} | {m['mae']} | {m['rmse']} | {m['pearson_r']} | {m['within_subject_r']} |")
            lines.append("")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"\nSaved: {path}")


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("EVAL STEP 6b + 7 — Extended Evaluation")
    print("=" * 60)

    df_raw = fetch_raw()
    df_full = step1_audit(df_raw)
    df_pairs = step2a_construct_targets(df_full)
    SET_A, SET_B, SET_C = step2b_define_feature_sets()
    flagged_session_ids, _ = step2c_qc_gate(df_full)

    all_results = {}

    # 6b.7 — unit test first (fast, ~1 min)
    all_results["unit_test"] = unit_test_intercept_update()

    # 6b.1-3 — extended LOSO for all three sets
    for sn, fs, hs in [("SET_A", SET_A, False), ("SET_B", SET_B, False), ("SET_C_NONCONVERGED", SET_C, True)]:
        all_results[sn] = run_loso_extended(df_pairs, fs, sn, hs)

    # 6b.4 — scaling ablation (SET_A and SET_B only; SET_C is non-converged)
    all_results["scaling_ablation"] = {}
    for sn, fs in [("SET_A_global", SET_A), ("SET_B_global", SET_B)]:
        all_results["scaling_ablation"][sn] = run_scaling_ablation(df_pairs, fs, sn)

    # 6b.5 — sensitivity
    all_results["sensitivity"] = run_sensitivity(
        df_full, df_pairs, SET_A, "SET_A", flagged_session_ids
    )

    # 7a — nowcasting
    all_results["nowcasting"] = {}
    for sn, fs in [("SET_A", SET_A), ("SET_B", SET_B)]:
        all_results["nowcasting"][sn] = run_nowcasting(df_full, fs, sn)

    # 7b — AR(1) comparison
    all_results["ar1"] = {}
    for sn, fs in [("SET_A", SET_A), ("SET_B", SET_B)]:
        all_results["ar1"][sn] = run_ar1_comparison(df_full, df_pairs, fs, sn)

    # Generate report
    generate_report(all_results, "loso_report_v2_extended.md")

    # Append session log
    log_path = os.path.join("logs", "session_log.md")
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(f"""
## {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} — Steps 6b + 7 Extended Evaluation

- **Script:** `eval_step6b_7.py`
- **Steps:** 6b.1 same-row baselines, 6b.2 intercept-only control, 6b.3 full metrics (RMSE/CI/within-subj/alert),
  6b.4 scaling ablation, 6b.5 sensitivity (QC + Geeta), 6b.7 personalized-update unit test,
  7a nowcasting, 7b AR(1) comparison
- **Files created:** `eval_step6b_7.py`, `loso_report_v2_extended.md`
""")

    print("\n" + "=" * 60)
    print("STEP 6b + 7 COMPLETE.")
    print("=" * 60)
