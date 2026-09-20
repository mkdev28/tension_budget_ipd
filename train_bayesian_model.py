
"""
train_bayesian_model.py — TensionBudget Machine Learning Training Pipeline

Trains a Regularized Hierarchical Bayesian Model on real human trapezius sEMG telemetry
to forecast subjective muscle strain/discomfort 5 minutes in advance (t + 5 min).

Features & Pipeline Design:
1. Ingests 'Supabase Snippet Untitled query.csv' (Supabase v_ml_training_pairs export).
2. Filters out synthetic test accounts ('FeatureTester', 'PayloadTester', 'SchemaTester').
3. Constructs t + 5 min target labels ('target_future_strain') and trailing trajectory features (t-1 to t trends).
4. Standardizes feature vectors and applies Regularized Bayesian Shrinkage (Horseshoe/L2 prior) to prevent rank-deficiency explosion on 27 features.
5. Evaluates model performance using 6-Fold Leave-One-Subject-Out (LOSO) Cross-Validation across real subjects with physical Borg scale clamping [0.0, 10.0].
6. Evaluates 90% Credible Interval Coverage & Alert Policy simulation.
7. Exports learned population baselines (beta_pop), user-specific random effects, and scaler parameters to 'tensionbudget_model_weights.json'.
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
    # Shift subjective_strain_cr10 by -1 within each session
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

class RegularizedBayesianHierarchicalModel:
    """
    Hierarchical Regularized Bayesian Model using Bayesian Ridge Shrinkage Priors:
    y_i,u ~ Normal( alpha_pop + alpha_u + sum(beta_k * x_i,k), sigma^2 )
    beta_k ~ Normal(0, tau^2) (Horseshoe / L2 Shrinkage Prior)
    """
    def __init__(self):
        self.bayes_reg = BayesianRidge(alpha_1=1e-2, alpha_2=1e-2, lambda_1=1e-2, lambda_2=1e-2, max_iter=500)
        self.alpha_pop = 0.0
        self.beta_pop = {}
        self.user_intercepts = {}
        self.sigma = 1.0
        self.predictors = []
        
    def fit(self, df_train, predictors):
        self.predictors = predictors
        
        # Calculate per-user baseline intercepts
        user_means = df_train.groupby('user_name')['target_future_strain'].mean()
        overall_mean = df_train['target_future_strain'].mean()
        
        # Partial pooling for random intercepts: shrink small sample users toward overall mean
        user_counts = df_train.groupby('user_name').size()
        for u in user_means.index:
            n_u = user_counts[u]
            w_u = n_u / (n_u + 5.0)
            pooled_mean = w_u * user_means[u] + (1.0 - w_u) * overall_mean
            self.user_intercepts[u] = float(pooled_mean - overall_mean)
            
        self.alpha_pop = float(overall_mean)
        
        # Adjust target by user random intercept for feature slope training
        y_adj = df_train['target_future_strain'].values - df_train['user_name'].map(lambda u: self.user_intercepts.get(u, 0.0)).values
        
        # Fit Bayesian Ridge with shrinkage prior on feature vectors
        X_train = df_train[predictors].values
        self.bayes_reg.fit(X_train, y_adj)
        
        for idx, p in enumerate(predictors):
            self.beta_pop[p] = float(self.bayes_reg.coef_[idx])
            
        self.sigma = float(np.sqrt(1.0 / self.bayes_reg.alpha_)) if hasattr(self.bayes_reg, 'alpha_') else 1.0
        
    def predict(self, df_test):
        X_test = df_test[self.predictors].values
        y_pred_base = self.bayes_reg.predict(X_test)
        
        # Add user random intercept offset if known, else 0.0 for new unseen user
        y_pred = []
        for idx, row in df_test.reset_index(drop=True).iterrows():
            u = row['user_name']
            u_offset = self.user_intercepts.get(u, 0.0)
            pred_val = y_pred_base[idx] + u_offset
            # Clamp predictions to Borg CR-10 scale boundaries [0.0, 10.0]
            clamped_val = float(np.clip(pred_val, 0.0, 10.0))
            y_pred.append(clamped_val)
            
        return np.array(y_pred)
        
    def predict_distribution(self, df_test, n_samples=1000):
        """
        Generates Posterior Predictive Samples (Mean, SD, 90% Credible Interval)
        """
        mu = self.predict(df_test)
        samples = np.random.normal(loc=mu[:, None], scale=self.sigma, size=(len(mu), n_samples))
        samples = np.clip(samples, 0.0, 10.0)
        ci_lower = np.percentile(samples, 5, axis=1)
        ci_upper = np.percentile(samples, 95, axis=1)
        prob_over_threshold = np.mean(samples >= 4.0, axis=1)
        return mu, ci_lower, ci_upper, prob_over_threshold

def run_loso_cross_validation(df, predictors):
    """
    Performs Leave-One-Subject-Out (LOSO) Cross Validation across all unique subjects.
    """
    unique_users = df['user_name'].unique()
    valid_test_users = [u for u in unique_users if len(df[df['user_name'] == u]) >= 2]
    
    print(f"\n=======================================================")
    print(f"RUNNING LEAVE-ONE-SUBJECT-OUT (LOSO) CROSS-VALIDATION")
    print(f"Subjects ({len(valid_test_users)}): {list(valid_test_users)}")
    print(f"=======================================================")
    
    loso_results = []
    y_true_all = []
    y_pred_hbm_all = []
    y_pred_rf_all = []
    ci_lower_all = []
    ci_upper_all = []
    prob_over_all = []
    
    for i, test_user in enumerate(valid_test_users):
        train_df = df[df['user_name'] != test_user].copy()
        test_df = df[df['user_name'] == test_user].copy()
        
        if len(test_df) == 0 or len(train_df) == 0:
            continue
            
        # Standardize features using training scaler
        scaler = StandardScaler()
        train_scaled = train_df.copy()
        test_scaled = test_df.copy()
        
        train_scaled[predictors] = scaler.fit_transform(train_df[predictors])
        test_scaled[predictors] = scaler.transform(test_df[predictors])
        
        # 1. Regularized Hierarchical Bayesian Model
        hbm = RegularizedBayesianHierarchicalModel()
        hbm.fit(train_scaled, predictors)
        preds_hbm, ci_low, ci_high, prob_over = hbm.predict_distribution(test_scaled)
        
        # 2. Random Forest Baseline
        rf = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
        rf.fit(train_scaled[predictors], train_df['target_future_strain'])
        preds_rf = np.clip(rf.predict(test_scaled[predictors]), 0.0, 10.0)
        
        y_true = test_df['target_future_strain'].values
        
        mae_hbm = mean_absolute_error(y_true, preds_hbm)
        mae_rf = mean_absolute_error(y_true, preds_rf)
        
        covered = np.mean((y_true >= ci_low) & (y_true <= ci_high)) * 100.0
        
        print(f"Fold {i+1}/{len(valid_test_users)} [Subject: {test_user:15s} | Test N={len(test_df):2d}] -> HBM MAE: {mae_hbm:.3f} (90% CI Cov: {covered:5.1f}%) | RF MAE: {mae_rf:.3f}")
        
        y_true_all.extend(y_true)
        y_pred_hbm_all.extend(preds_hbm)
        y_pred_rf_all.extend(preds_rf)
        ci_lower_all.extend(ci_low)
        ci_upper_all.extend(ci_high)
        prob_over_all.extend(prob_over)
        
        loso_results.append({
            'subject': test_user,
            'n_samples': len(test_df),
            'mae_hbm': mae_hbm,
            'mae_rf': mae_rf,
            'ci_coverage_pct': covered
        })
        
    y_true_all = np.array(y_true_all)
    y_pred_hbm_all = np.array(y_pred_hbm_all)
    y_pred_rf_all = np.array(y_pred_rf_all)
    ci_lower_all = np.array(ci_lower_all)
    ci_upper_all = np.array(ci_upper_all)
    prob_over_all = np.array(prob_over_all)
    
    total_mae_hbm = mean_absolute_error(y_true_all, y_pred_hbm_all)
    total_rmse_hbm = np.sqrt(mean_squared_error(y_true_all, y_pred_hbm_all))
    corr_hbm = np.corrcoef(y_true_all, y_pred_hbm_all)[0, 1]
    overall_coverage = np.mean((y_true_all >= ci_lower_all) & (y_true_all <= ci_upper_all)) * 100.0
    
    total_mae_rf = mean_absolute_error(y_true_all, y_pred_rf_all)
    total_rmse_rf = np.sqrt(mean_squared_error(y_true_all, y_pred_rf_all))
    corr_rf = np.corrcoef(y_true_all, y_pred_rf_all)[0, 1]
    
    # Alert policy evaluation: Alert fired if P(strain >= 4.0) >= 0.85 OR Upper 95% >= 5.5
    actual_high_strain = y_true_all >= 4.0
    alert_triggered = (prob_over_all >= 0.85) | (ci_upper_all >= 5.5)
    tp = np.sum(actual_high_strain & alert_triggered)
    fp = np.sum((~actual_high_strain) & alert_triggered)
    fn = np.sum(actual_high_strain & (~alert_triggered))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    print("\n=======================================================")
    print("FINAL LOSO CROSS-VALIDATION PERFORMANCE OVERVIEW")
    print("=======================================================")
    print(f"Hierarchical Bayesian Model -> MAE: {total_mae_hbm:.3f} | RMSE: {total_rmse_hbm:.3f} | Corr: {corr_hbm:.3f}")
    print(f"                            -> 90% Credible Interval Coverage: {overall_coverage:.1f}%")
    print(f"                            -> Alert Policy (Threshold >= 4.0): Precision: {precision*100:.1f}% | Recall: {recall*100:.1f}%")
    print(f"Random Forest Model         -> MAE: {total_mae_rf:.3f} | RMSE: {total_rmse_rf:.3f} | Corr: {corr_rf:.3f}")
    print("=======================================================\n")
    
    return loso_results, (total_mae_hbm, total_rmse_hbm, corr_hbm, overall_coverage)

def train_and_export_final_model(df, predictors):
    """
    Trains the final Hierarchical Bayesian Model on 100% of the real human dataset
    and exports learned weights and parameters to JSON.
    """
    print("Training final Hierarchical Bayesian Model on 100% of real human dataset...")
    
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[predictors] = scaler.fit_transform(df[predictors])
    
    hbm = RegularizedBayesianHierarchicalModel()
    hbm.fit(df_scaled, predictors)
    
    print("\nTop Learned Population Feature Weights (Beta_pop):")
    sorted_params = sorted(hbm.beta_pop.items(), key=lambda x: abs(x[1]), reverse=True)
    for k, v in sorted_params[:10]:
        print(f"  - {k:30s}: Beta = {v:+7.4f}")
        
    # Build export dictionary
    model_export = {
        "model_type": "Regularized_Bayesian_Hierarchical",
        "target": "subjective_strain_cr10_t_plus_5min",
        "lead_time_minutes": 5,
        "n_training_samples": len(df),
        "n_subjects": int(df['user_name'].nunique()),
        "subjects": list(df['user_name'].unique()),
        "scaler_means": dict(zip(predictors, scaler.mean_)),
        "scaler_stds": dict(zip(predictors, scaler.scale_)),
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
        
    print(f"Successfully exported final model weights to: {OUTPUT_WEIGHTS_PATH}")
    return model_export

if __name__ == '__main__':
    prepared_df = prepare_dataset(DATASET_PATH)
    loso_res, overall_metrics = run_loso_cross_validation(prepared_df, ALL_PREDICTORS)
    final_weights = train_and_export_final_model(prepared_df, ALL_PREDICTORS)
