# Predictive Modeling Report

## Part A: Classification (breast cancer)

| Model               | CV AUC (train)   |   Accuracy |   Precision |   Recall |    F1 |   Test AUC |
|:--------------------|:-----------------|-----------:|------------:|---------:|------:|-----------:|
| Logistic Regression | 0.996 +/- 0.005  |      0.982 |       0.986 |    0.986 | 0.986 |      0.995 |
| Decision Tree       | 0.927 +/- 0.018  |      0.939 |       0.958 |    0.944 | 0.951 |      0.934 |
| Random Forest       | 0.990 +/- 0.007  |      0.947 |       0.958 |    0.958 | 0.958 |      0.994 |

Tuned Random Forest params: `{'max_depth': 5, 'min_samples_leaf': 1, 'n_estimators': 300}`

## Part B: Regression (diabetes progression)

| Model             |   CV R2 (train) |   Test R2 |   Test MAE |   Test RMSE |
|:------------------|----------------:|----------:|-----------:|------------:|
| Linear Regression |           0.449 |     0.453 |     42.794 |      53.853 |
| Ridge             |           0.38  |     0.419 |     46.139 |      55.474 |
| Decision Tree     |           0.274 |     0.329 |     48.097 |      59.605 |
| Random Forest     |           0.412 |     0.459 |     43.357 |      53.516 |

Baseline MAE (predict the mean): 64.0
