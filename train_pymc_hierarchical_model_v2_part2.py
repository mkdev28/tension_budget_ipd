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
        if len(y_true) == 0: return None
        mae = mean_absolute_error(y_true, y_pred)
        corr = pearsonr(y_true, y_pred)[0] if len(y_true) > 1 else 0
        return {"mae": mae, "corr": corr, "n": len(y_true)}
        
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
