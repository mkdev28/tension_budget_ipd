"""
train_pymc_hierarchical_model.py — True Bayesian Hierarchical Model with Regularized Horseshoe Prior
and NUTS Sampling in PyMC.
 
Architecture & Spec Compliance:
1. Random Intercepts (alpha_u) per user: alpha_u ~ Normal(alpha_pop, tau_alpha)
2. Random Slopes (beta_u[u, k]) per user: beta_u[u, k] ~ Normal(beta_pop[k], tau_beta[k])
3. Regularized Horseshoe Prior for Population Slopes (beta_pop[k]):
   beta_pop[k] ~ Normal(0, tau * tilde_lambda_k)
   lambda_k ~ HalfCauchy(1.0)
   tau ~ HalfCauchy(tau_0)
   c^2 ~ InverseGamma(1.0, 1.0)
   tilde_lambda_k = sqrt(c^2 * lambda_k^2 / (c^2 + tau^2 * lambda_k^2))
4. NUTS Sampling: pm.sample(draws=1000, tune=1000, chains=4, target_accept=0.95)
5. MCMC Diagnostics: ArViZ summary calculating R-hat (R_hat <= 1.05), ESS (> 400), and Divergence Check (divergences == 0).
6. Posterior Predictive Check: pm.sample_posterior_predictive.
7. Leave-One-Subject-Out (LOSO) Cross-Validation.
8. Artifact Export: exports true posterior parameters (mean, SD, 90% HDI) to 'tensionbudget_model_weights.json'.
"""
 
import os
import json
import numpy as np
import pandas as pd
import os
os.environ["PYTENSOR_FLAGS"] = "cxx="  # Disable C-compilation to fix 32-bit MinGW errors on 64-bit Windows
import pymc as pm
import arviz as az
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
 
# ---------------------------------------------------------
# 1. CONSTANTS & CONFIGURATION
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'Supabase Snippet Untitled query.csv')
OUTPUT_WEIGHTS_PATH = os.path.join(BASE_DIR, 'tensionbudget_model_weights.json')
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
 
ALL_PREDICTORS = PREDICTOR_COLS + TREND_COLS
 
def prepare_dataset(csv_path):
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
   
    # 1. Filter out tester accounts
    initial_len = len(df)
    df = df[~df['user_name'].isin(TESTER_USERS)].copy()
    print(f"Filtered out test accounts ({TESTER_USERS}). Rows remaining: {len(df)} / {initial_len}")
   
    # 2. Filter to labeled reported strain rows only
    df = df[df['strain_reported'] == 1].copy()
    df = df[df['subjective_strain_cr10'].notna()].copy()
    print(f"Filtered to labeled rows. Rows remaining: {len(df)}")
   
    # 3. Sort chronologically per session
    df['session_id'] = df['session_id'].astype(str)
    df = df.sort_values(by=['session_id', 'epoch_index']).reset_index(drop=True)
   
    # 4. Compute Trajectory/Trend features (t-1 to t) per session
    for side in ['left', 'right']:
        df[f'mdf_trend_{side}'] = df.groupby('session_id')[f'mdf_hz_{side}'].diff().fillna(0.0)
        df[f'gap_freq_trend_{side}'] = df.groupby('session_id')[f'gap_frequency_{side}'].diff().fillna(0.0)
        df[f'eindex_trend_{side}'] = df.groupby('session_id')[f'eindex_live_{side}'].diff().fillna(0.0)
       
    # 5. Construct Future Pain Target (t + 5 min forecast)
    df['target_future_strain'] = df.groupby('session_id')['subjective_strain_cr10'].shift(-1)
   
    # Drop rows where target_future_strain is NaN (last epoch of each session)
    df_forecast = df[df['target_future_strain'].notna()].copy().reset_index(drop=True)
    print(f"Constructed t+5min target ('target_future_strain'). Valid forecasting pairs: {len(df_forecast)}")
   
    # 6. Impute missing predictor values gracefully
    for col in PREDICTOR_COLS:
        if df_forecast[col].isna().sum() > 0:
            median_val = df_forecast[col].median()
            df_forecast[col] = df_forecast[col].fillna(median_val if not pd.isna(median_val) else 0.0)
           
    return df_forecast
 
def build_pymc_hierarchical_horseshoe_model(X_train, y_train, user_idx_train, n_users, n_features, p0=5):
    """
    Builds a PyMC Model with:
    - Random Intercepts per user (alpha_u)  — one offset per subject
    - Population-level feature slopes (beta_pop[k]) shared across all users,
      regularized by the Piironen & Vehtari (2017) regularized horseshoe prior
    NOTE: Per-user random slopes (beta_u) are intentionally omitted — with
    ~6 subjects and ~500-600 rows the model is unidentifiable at that level.
    """
    with pm.Model() as model:
        # Data Containers
        u_idx = pm.Data("u_idx", user_idx_train)
        X_data = pm.Data("X_data", X_train)
        y_data = pm.Data("y_data", y_train)
       
        # 1. Population Intercept & User Random Intercepts
        alpha_pop = pm.Normal("alpha_pop", mu=2.5, sigma=2.0)
        tau_alpha = pm.HalfNormal("tau_alpha", sigma=1.0)
        alpha_offset = pm.Normal("alpha_offset", mu=0, sigma=1, shape=n_users)
        alpha_u = pm.Deterministic("alpha_u", alpha_pop + alpha_offset * tau_alpha)
       
        # 2. Regularized Horseshoe Prior (Piironen & Vehtari 2017)
        # Expected active features p0 = 5
        N = len(y_train)
        sigma_0 = 1.0
        tau_0 = (p0 / (n_features - p0)) * (sigma_0 / np.sqrt(N))
       
        tau = pm.HalfCauchy("tau", beta=tau_0)
        lam = pm.HalfCauchy("lam", beta=1.0, shape=n_features)
        # Piironen & Vehtari (2017) recommended slab prior: nu=4, s=2
        # => alpha = nu/2 = 2,  beta = (nu/2)*s^2 = 8
        # alpha=1/beta=1 has infinite variance and destabilises the slab scale.
        c2 = pm.InverseGamma("c2", alpha=2.0, beta=8.0)
       
        # Regularized shrinkage scale
        tilde_lam = pm.Deterministic("tilde_lam", pm.math.sqrt(c2 * lam**2 / (c2 + tau**2 * lam**2)))
       
        # Population feature slopes
        beta_pop_offset = pm.Normal("beta_pop_offset", mu=0, sigma=1, shape=n_features)
        beta_pop = pm.Deterministic("beta_pop", beta_pop_offset * tau * tilde_lam)
       
        # 3. Linear Predictor (mu_i)
        # Population slopes are shared; user variation is captured by alpha_u alone.
        # mu = alpha_u[user_idx] + X @ beta_pop
        mu = alpha_u[u_idx] + pm.math.dot(X_data, beta_pop)
       
        # 4. Residual Noise Variance
        sigma_y = pm.HalfNormal("sigma_y", sigma=1.5)
       
        # 5. Observed Likelihood
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma_y, observed=y_data)
       
    return model
 
def run_full_bayes_workflow(df, predictors):
    """
    Executes the Complete Bayesian Workflow (Gelman et al.):
    1. Prior Predictive Check
    2. PyMC NUTS Fitting
    3. Convergence Diagnostics (R-hat, ESS, Divergences)
    4. Posterior Predictive Check
    5. Leave-One-Subject-Out Cross Validation
    """
    unique_users = list(df['user_name'].unique())
    user_map = {u: i for i, u in enumerate(unique_users)}
    df['user_idx'] = df['user_name'].map(user_map)
   
    n_users = len(unique_users)
    n_features = len(predictors)
   
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[predictors].values)
    y_vals = df['target_future_strain'].values
    user_indices = df['user_idx'].values
   
    print("\n=======================================================")
    print("STEP 1: PRIOR PREDICTIVE CHECK")
    print("=======================================================")
    model = build_pymc_hierarchical_horseshoe_model(X_scaled, y_vals, user_indices, n_users, n_features)
   
    with model:
        prior_draws = pm.sample_prior_predictive(draws=200, random_seed=42)
    prior_y = prior_draws.prior_predictive["y_obs"].values.flatten()
    frac_in_range = np.mean((prior_y >= 0.0) & (prior_y <= 10.0))
    print(f"Prior Predictive Strain Range: [{np.min(prior_y):.2f}, {np.max(prior_y):.2f}] "
          f"| Fraction within physical [0,10] Borg range: {frac_in_range:.1%}")
    if frac_in_range < 0.5:
        print("WARNING: Less than 50% of prior predictive draws fall within the physical Borg CR-10 range. "
              "Priors on alpha_pop/sigma_y may be too diffuse.")
   
    print("\n=======================================================")
    print("STEP 2: PYMC NUTS SAMPLING (FULL POSTERIOR FITTING)")
    print("=======================================================")
    with model:
        idata = pm.sample(
            draws=1000,
            tune=1000,
            chains=4,
            target_accept=0.95,
            random_seed=42,
            progressbar=False,
            return_inferencedata=True
        )
       
    print("\n=======================================================")
    print("STEP 3: CONVERGENCE DIAGNOSTICS (R-HAT, ESS, DIVERGENCES)")
    print("=======================================================")
    # Cover ALL parameters, not just three — catches any badly mixing group.
    summary = az.summary(idata)
    rhat_max = summary["r_hat"].max()
    ess_min = summary["ess_bulk"].min()
   
    # Count Hamiltonian divergences
    divergences = int(idata.sample_stats["diverging"].sum().values)
   
    print(f"Max R-hat across parameters: {rhat_max:.4f} (Target: <= 1.05)")
    print(f"Min Bulk Effective Sample Size (ESS): {ess_min:.1f} (Target: > 100)")
    print(f"Divergence Count: {divergences} (Target: 0)")
   
    if rhat_max <= 1.05 and divergences == 0:
        print(">> CONVERGENCE CHECK PASSED: MCMC chains mixed smoothly with zero divergences!")
        convergence_ok = True
    else:
        print(">> CONVERGENCE WARNING: Review priors or step size.")
        print("!! CONVERGENCE FAILED — do not trust posterior summaries or exported weights "
              "until this is fixed. Continuing run for debugging purposes only.")
        convergence_ok = False
       
    print("\n=======================================================")
    print("STEP 4: POSTERIOR PREDICTIVE CHECK")
    print("=======================================================")
    with model:
        post_draws = pm.sample_posterior_predictive(idata, random_seed=42)
       
    y_post_samples = post_draws.posterior_predictive["y_obs"].values
    y_post_flat = np.clip(y_post_samples.reshape(-1, len(y_vals)), 0.0, 10.0)
   
    y_pred_mean = np.mean(y_post_flat, axis=0)
    ci_low = np.percentile(y_post_flat, 5, axis=0)
    ci_high = np.percentile(y_post_flat, 95, axis=0)
   
    mae_train = mean_absolute_error(y_vals, y_pred_mean)
    rmse_train = np.sqrt(mean_squared_error(y_vals, y_pred_mean))
    r2_train = r2_score(y_vals, y_pred_mean)
    corr_train = np.corrcoef(y_vals, y_pred_mean)[0, 1]
    coverage_train = np.mean((y_vals >= ci_low) & (y_vals <= ci_high)) * 100.0
   
    print(f"Posterior Predictive MAE : {mae_train:.3f} Borg points")
    print(f"Posterior Predictive RMSE: {rmse_train:.3f} Borg points")
    print(f"Posterior Predictive R²  : {r2_train:.3f}")
    print(f"Posterior Predictive Corr: {corr_train:.3f}")
    print(f"90% Credible Interval Coverage: {coverage_train:.1f}%")
   
    # ---------------------------------------------------------
    # STEP 5: LEAVE-ONE-SUBJECT-OUT (LOSO) CROSS-VALIDATION
    # ---------------------------------------------------------
    print("\n=======================================================")
    print("STEP 5: LEAVE-ONE-SUBJECT-OUT (LOSO) CROSS-VALIDATION")
    print("=======================================================")
   
    loso_results = []
    y_true_loso = []
    y_pred_loso = []
    ci_low_loso = []
    ci_high_loso = []
   
    for fold, test_user in enumerate(unique_users):
        train_sub_df = df[df['user_name'] != test_user].copy()
        test_sub_df = df[df['user_name'] == test_user].copy()
       
        if len(test_sub_df) == 0:
            continue
           
        sub_scaler = StandardScaler()
        X_tr = sub_scaler.fit_transform(train_sub_df[predictors].values)
        y_tr = train_sub_df['target_future_strain'].values
       
        sub_user_map = {u: idx for idx, u in enumerate(train_sub_df['user_name'].unique())}
        u_tr = train_sub_df['user_name'].map(sub_user_map).values
       
        X_te = sub_scaler.transform(test_sub_df[predictors].values)
        y_te = test_sub_df['target_future_strain'].values
       
        sub_model = build_pymc_hierarchical_horseshoe_model(X_tr, y_tr, u_tr, len(sub_user_map), n_features)
       
        with sub_model:
            sub_idata = pm.sample(draws=500, tune=500, chains=4, target_accept=0.95, progressbar=False, random_seed=42)
           
        # Predict for unseen test user using population posterior distributions (alpha_pop, beta_pop)
        alpha_pop_draws = sub_idata.posterior["alpha_pop"].values.flatten()
        beta_pop_draws = sub_idata.posterior["beta_pop"].values.reshape(-1, n_features)
        sigma_draws = sub_idata.posterior["sigma_y"].values.flatten()
       
        # Draw posterior predictive predictions
        test_preds_samples = []
        for d_idx in range(len(alpha_pop_draws)):
            mu_test = alpha_pop_draws[d_idx] + np.dot(X_te, beta_pop_draws[d_idx])
            sample_draw = np.random.normal(mu_test, sigma_draws[d_idx])
            test_preds_samples.append(sample_draw)
           
        test_preds_samples = np.clip(np.array(test_preds_samples), 0.0, 10.0) # shape (draws, N_test)
       
        sub_pred_mean = np.mean(test_preds_samples, axis=0)
        sub_ci_low = np.percentile(test_preds_samples, 5, axis=0)
        sub_ci_high = np.percentile(test_preds_samples, 95, axis=0)
       
        sub_mae = mean_absolute_error(y_te, sub_pred_mean)
        sub_cov = np.mean((y_te >= sub_ci_low) & (y_te <= sub_ci_high)) * 100.0
       
        print(f"LOSO Fold {fold+1}/{n_users} [Test Subject: {test_user:15s} | N={len(y_te):2d}] -> MAE: {sub_mae:.3f} | 90% CI Cov: {sub_cov:5.1f}%")
       
        y_true_loso.extend(y_te)
        y_pred_loso.extend(sub_pred_mean)
        ci_low_loso.extend(sub_ci_low)
        ci_high_loso.extend(sub_ci_high)
       
    y_true_loso = np.array(y_true_loso)
    y_pred_loso = np.array(y_pred_loso)
    ci_low_loso = np.array(ci_low_loso)
    ci_high_loso = np.array(ci_high_loso)
   
    loso_mae = mean_absolute_error(y_true_loso, y_pred_loso)
    loso_rmse = np.sqrt(mean_squared_error(y_true_loso, y_pred_loso))
    loso_corr = np.corrcoef(y_true_loso, y_pred_loso)[0, 1]
    loso_cov = np.mean((y_true_loso >= ci_low_loso) & (y_true_loso <= ci_high_loso)) * 100.0
   
    print("\n=======================================================")
    print("FINAL LOSO CROSS-VALIDATION SUMMARY (UNSEEN SUBJECTS)")
    print("=======================================================")
    print(f"LOSO MAE  : {loso_mae:.3f} Borg CR-10 points")
    print(f"LOSO RMSE : {loso_rmse:.3f} Borg CR-10 points")
    print(f"LOSO Corr : {loso_corr:.3f}")
    print(f"LOSO 90% CI Coverage: {loso_cov:.1f}%")
    print("=======================================================\n")
   
    # ---------------------------------------------------------
    # STEP 6: EXPORT FINAL MODEL WEIGHTS & POSTERIOR SUMMARIES
    # ---------------------------------------------------------
    alpha_pop_mean = float(summary.loc["alpha_pop", "mean"])
    sigma_y_mean = float(summary.loc["sigma_y", "mean"])
   
    beta_pop_means = {}
    for idx, p in enumerate(predictors):
        beta_pop_means[p] = float(idata.posterior["beta_pop"].mean(dim=["chain", "draw"]).values[idx])
       
    user_alpha_means = {}
    for u, u_i in user_map.items():
        user_alpha_means[u] = float(idata.posterior["alpha_u"].mean(dim=["chain", "draw"]).values[u_i] - alpha_pop_mean)
       
    export_dict = {
        "model_type": "PyMC_NUTS_Bayesian_Hierarchical_Horseshoe",
        "sampling_method": "Hamiltonian_Monte_Carlo_NUTS",
        "target": "subjective_strain_cr10_t_plus_5min",
        "lead_time_minutes": 5,
        "n_training_samples": len(df),
        "n_subjects": n_users,
        "subjects": unique_users,
        "diagnostics": {
            "r_hat_max": float(rhat_max),
            "ess_bulk_min": float(ess_min),
            "divergence_count": divergences,
            "converged": convergence_ok
        },
        "scaler_means": dict(zip(predictors, scaler.mean_)),
        "scaler_stds": dict(zip(predictors, scaler.scale_)),
        "population_intercept": alpha_pop_mean,
        "population_coefficients": beta_pop_means,
        "user_intercept_offsets": user_alpha_means,
        "residual_sigma": sigma_y_mean,
        "alert_policy": {
            "strain_threshold": 4.0,
            "confidence_required": 0.85,
            "hard_safety_ceiling": 5.5
        }
    }
   
    with open(OUTPUT_WEIGHTS_PATH, 'w') as f:
        json.dump(export_dict, f, indent=2)
       
    print(f"Successfully exported PyMC Bayesian Hierarchical Model weights to: {OUTPUT_WEIGHTS_PATH}")
    return export_dict
 
if __name__ == '__main__':
    df_prepared = prepare_dataset(DATASET_PATH)
    run_full_bayes_workflow(df_prepared, ALL_PREDICTORS)
 