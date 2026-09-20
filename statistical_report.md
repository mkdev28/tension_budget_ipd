# BioEcho — Final PyMC Bayesian Hierarchical Model Report (top 5)
==========================================================================
1. FINAL TRAINED PYMC MODEL METADATA & DIAGNOSTICS
==========================================================================
Model Architecture       : Regularized_Bayesian_Hierarchical
Sampling Method          : Hamiltonian Monte Carlo (NUTS Sampler)
Target Variable          : subjective_strain_cr10_t_plus_5min (Subjective Strain on Borg CR-10)
Lead Time                : 5 minutes ahead
Total Valid Training Pairs: 568 (5-min epochs)
Number of Subjects       : 6
Population Intercept (alpha_pop): 2.5821
Residual Std Error (sigma)     : 1.3547

--------------------------------------------------------------------------
User-Specific Random Intercept Offsets (Personalization Baseline alpha_u):
--------------------------------------------------------------------------
  - Subject Danish Jain    : Offset = -0.3093  => Personal Baseline =  2.27
  - Subject Lavish Jain    : Offset = +0.1634  => Personal Baseline =  2.75
  - Subject Namita Pulgam  : Offset = +0.5303  => Personal Baseline =  3.11
  - Subject moko           : Offset = +1.6418  => Personal Baseline =  4.22
  - Subject nilesh sir     : Offset = -0.6910  => Personal Baseline =  1.89

==========================================================================
2. TOP FEATURE COEFFICIENTS MATRIX (Beta_pop)
==========================================================================
                 Feature  Beta_Weight
     fatigue_slope_right     0.165465
        eindex_live_left    -0.079780
       eindex_trend_left    -0.062736
short_suma_penalty_right    -0.058859
      fatigue_slope_left     0.057940
           apdf_90_right     0.050958
    calibration_rms_left    -0.047000
            mdf_hz_right     0.033363
            apdf_50_left     0.031618
            apdf_10_left     0.028214
       eindex_live_right    -0.027750
         mdf_trend_right    -0.026060
          mdf_trend_left     0.021651
     gap_frequency_right    -0.019324
   calibration_rms_right    -0.019013

==========================================================================
3. OVERALL STATISTICAL ACCURACY METRICS
==========================================================================
Mean Absolute Error (MAE)   : 0.9671 Borg CR-10 points
Root Mean Sq Error (RMSE)   : 1.2953 Borg CR-10 points
Coefficient of Determination: R² = 0.5076
Pearson Correlation (R)     : 0.7428
90% Credible Interval Cov   : 91.0%

==========================================================================
4. PER-SUBJECT ACCURACY MATRIX
==========================================================================
      Subject  N Epochs  Mean Actual Strain  Mean Pred Strain   MAE 90% CI Coverage
         moko         8                5.25              4.52 1.531           75.0%
     priyansh        14                1.07              1.38 0.744          100.0%
  Danish Jain        13                2.15              2.21 0.868           92.3%
  Lavish Jain        15                2.80              2.59 1.489           80.0%
Namita Pulgam        12                3.33              3.30 0.423          100.0%
   nilesh sir         5                1.20              1.83 0.691          100.0%

==========================================================================
5. ALERT POLICY CONFUSION MATRIX (Strain Alert Threshold >= 4.0)
==========================================================================
Confusion Matrix (Actual vs Predicted Alert):
                Predicted NO ALERT (0)   Predicted ALERT (1)
Actual NO (0)           TN = 415                 FP =  26
Actual YES (1)          FN =  68                 TP =  59

Alert Accuracy Metrics:
  - Overall Alert Accuracy : 83.6%
  - Alert Precision        : 70.0% (positive predictive value)
  - Alert Recall (Sensitivity): 46.7% (true positive rate)
  - Alert F1-Score         : 0.5600
==========================================================================