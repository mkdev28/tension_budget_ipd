"""
generate_statistical_report.py — Generates complete statistical matrices,
confusion matrices, per-subject breakdown, and model evaluation metrics for TensionBudget.
"""

import sys
import json
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, mean_absolute_error, mean_squared_error, r2_score

from train_bayesian_model import prepare_dataset, ALL_PREDICTORS, RegularizedBayesianHierarchicalModel

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def generate_report():
    df = prepare_dataset('Supabase Snippet Untitled query.csv')
    
    with open('tensionbudget_model_weights.json') as f:
        weights = json.load(f)
        
    print("==========================================================================")
    print("1. FINAL TRAINED MODEL METADATA & PARAMETERS")
    print("==========================================================================")
    print(f"Model Architecture       : {weights['model_type']}")
    print(f"Target Variable          : {weights['target']} (Subjective Strain on Borg CR-10)")
    print(f"Lead Time                : {weights['lead_time_minutes']} minutes ahead")
    print(f"Total Valid Training Pairs: {weights['n_training_samples']} (5-min epochs)")
    print(f"Number of Subjects       : {weights['n_subjects']}")
    print(f"Population Intercept (alpha_pop): {weights['population_intercept']:.4f}")
    print(f"Residual Std Error (sigma)     : {weights['residual_sigma']:.4f}")
    
    print("\n--------------------------------------------------------------------------")
    print("User-Specific Random Intercept Offsets (Personalization Baseline alpha_u):")
    print("--------------------------------------------------------------------------")
    for u, offset in weights['user_intercept_offsets'].items():
        user_baseline = weights['population_intercept'] + offset
        print(f"  - Subject {u:15s}: Offset = {offset:+6.4f}  => Personal Baseline = {user_baseline:5.2f}")

    print("\n==========================================================================")
    print("2. TOP FEATURE COEFFICIENTS MATRIX (Beta_pop)")
    print("==========================================================================")
    coeff_df = pd.DataFrame(list(weights['population_coefficients'].items()), columns=['Feature', 'Beta_Weight'])
    coeff_df['Abs_Weight'] = coeff_df['Beta_Weight'].abs()
    coeff_df = coeff_df.sort_values(by='Abs_Weight', ascending=False).reset_index(drop=True)
    print(coeff_df[['Feature', 'Beta_Weight']].head(15).to_string(index=False))

    # Evaluate full model predictions
    scaler_means = np.array([weights['scaler_means'][p] for p in ALL_PREDICTORS])
    scaler_stds = np.array([weights['scaler_stds'][p] for p in ALL_PREDICTORS])
    
    X_raw = df[ALL_PREDICTORS].values
    X_scaled = (X_raw - scaler_means) / scaler_stds
    
    df_scaled = df.copy()
    df_scaled[ALL_PREDICTORS] = X_scaled
    
    hbm = RegularizedBayesianHierarchicalModel()
    hbm.fit(df_scaled, ALL_PREDICTORS)
    y_true = df['target_future_strain'].values
    y_pred, ci_low, ci_high, prob_over = hbm.predict_distribution(df_scaled)
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    coverage = np.mean((y_true >= ci_low) & (y_true <= ci_high)) * 100.0

    print("\n==========================================================================")
    print("3. OVERALL STATISTICAL ACCURACY METRICS")
    print("==========================================================================")
    print(f"Mean Absolute Error (MAE)   : {mae:.4f} Borg CR-10 points")
    print(f"Root Mean Sq Error (RMSE)   : {rmse:.4f} Borg CR-10 points")
    print(f"Coefficient of Determination: R² = {r2:.4f}")
    print(f"Pearson Correlation (R)     : {corr:.4f}")
    print(f"90% Credible Interval Cov   : {coverage:.1f}%")

    print("\n==========================================================================")
    print("4. PER-SUBJECT ACCURACY MATRIX")
    print("==========================================================================")
    subject_rows = []
    for u in df['user_name'].unique():
        sub_mask = df['user_name'] == u
        sub_y = y_true[sub_mask]
        sub_pred = y_pred[sub_mask]
        sub_low = ci_low[sub_mask]
        sub_high = ci_high[sub_mask]
        
        sub_mae = mean_absolute_error(sub_y, sub_pred)
        sub_cov = np.mean((sub_y >= sub_low) & (sub_y <= sub_high)) * 100.0
        subject_rows.append({
            'Subject': u,
            'N Epochs': int(np.sum(sub_mask)),
            'Mean Actual Strain': round(float(np.mean(sub_y)), 2),
            'Mean Pred Strain': round(float(np.mean(sub_pred)), 2),
            'MAE': round(sub_mae, 3),
            '90% CI Coverage': f"{sub_cov:.1f}%"
        })
    print(pd.DataFrame(subject_rows).to_string(index=False))

    print("\n==========================================================================")
    print("5. ALERT POLICY CONFUSION MATRIX (Strain Alert Threshold >= 4.0)")
    print("==========================================================================")
    actual_alert = (y_true >= 4.0).astype(int)
    predicted_alert = ((prob_over >= 0.85) | (ci_high >= 5.5)).astype(int)
    
    cm = confusion_matrix(actual_alert, predicted_alert)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    print(f"Confusion Matrix (Actual vs Predicted Alert):")
    print(f"                Predicted NO ALERT (0)   Predicted ALERT (1)")
    print(f"Actual NO (0)           TN = {tn:2d}                 FP = {fp:2d}")
    print(f"Actual YES (1)          FN = {fn:2d}                 TP = {tp:2d}")
    print(f"\nAlert Accuracy Metrics:")
    print(f"  - Overall Alert Accuracy : {accuracy*100:.1f}%")
    print(f"  - Alert Precision        : {precision*100:.1f}% (positive predictive value)")
    print(f"  - Alert Recall (Sensitivity): {recall*100:.1f}% (true positive rate)")
    print(f"  - Alert F1-Score         : {f1:.4f}")
    print("==========================================================================")

if __name__ == '__main__':
    generate_report()
