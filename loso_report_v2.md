# TensionBudget PyMC Pipeline v2 Results

## Overview
Random intercepts only, buggy targets fixed, strictly causal scaling.

### SET_A
| Protocol | MAE | Corr | N |
|---|---|---|---|
| Cold Start | 1.016 | -0.080 | 229 |
| Global Mean (Baseline) | 0.989 | -0.482 | 229 |
| Persistence (Baseline) | 0.515 | 0.751 | 227 |
| Personalized K=3 | 1.196 | 0.193 | 185 |
| Personalized K=5 | 1.238 | 0.114 | 157 |

### SET_B
| Protocol | MAE | Corr | N |
|---|---|---|---|
| Cold Start | 1.038 | -0.088 | 229 |
| Global Mean (Baseline) | 0.989 | -0.482 | 229 |
| Persistence (Baseline) | 0.515 | 0.751 | 227 |
| Personalized K=3 | 1.163 | 0.208 | 185 |
| Personalized K=5 | 1.235 | 0.154 | 157 |

### SET_C
| Protocol | MAE | Corr | N |
|---|---|---|---|
| Cold Start | 0.991 | -0.119 | 229 |
| Global Mean (Baseline) | 0.989 | -0.482 | 229 |
| Persistence (Baseline) | 0.515 | 0.751 | 227 |
| Personalized K=3 | 1.099 | 0.260 | 185 |
| Personalized K=5 | 1.179 | 0.172 | 157 |

