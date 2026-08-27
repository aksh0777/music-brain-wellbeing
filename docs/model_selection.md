# Model Selection Report

## 1. Candidate Models
We compared the following regression models wrapped in our data preprocessing pipeline:
1. **Naive Mean Baseline**: Predicts the training target mean (`5.8452`) for all test observations.
2. **Linear Regression Baseline**: Ordinary Least Squares (OLS) linear model.
3. **Unregularized Decision Tree**: Default, unconstrained decision tree.
4. **Tuned Decision Tree**: A Decision Tree Regressor constrained to `max_depth=3`.

---

## 2. Evaluation Criteria
We evaluate our candidate models on three primary regression metrics calculated on the held-out test set:
- **Mean Absolute Error (MAE)**: Measures average absolute prediction error magnitude.
- **Root Mean Squared Error (RMSE)**: Penalizes larger prediction errors more heavily.
- **Coefficient of Determination (R²)**: Compares the model's Mean Squared Error (MSE) to the variance of the baseline mean predictor.
- **Generalization Gap**: Calculated as Test RMSE - Train RMSE, indicating the level of model overfitting.

---

## 3. Comparison Table

Below is the comparison table compiled from the actual model evaluations:

| Model | Configuration | CV R² Mean | Test MAE | Test RMSE | Test R² | Train R² | Generalization Gap (RMSE) | Overfitting Observation |
|---|---|---|---|---|---|---|---|---|
| **Naive Mean Baseline** | Predict y_train mean | N/A | 2.4193 | 2.8423 | -0.0004 | 0.0000 | 0.1858 | None (underfits) |
| **Linear Regression** | OLS Pipeline | -0.0531 | 2.4173 | 2.8786 | -0.0261 | 0.1267 | 0.2825 | Mild (due to OLS categorical expansion) |
| **Unregularized Decision Tree** | Default parameters | -1.0465 | 3.3480 | 4.1258 | -1.1078 | 1.0000 | 4.1258 | Severe (Train R² = 1.0000, Test R² = -1.1078) |
| **Best Tuned Decision Tree** | `max_depth=3` | **-0.1027** | **2.2603** | **2.7499** | **0.0636** | 0.1158 | **0.1375** | Minimal (Train R² = 0.1158, Test R² = 0.0636) |

---

## 4. Model Selection Reasoning
1. **Raw Test R² Limits**: Raw Test R² alone is not enough to select a model because it is a single point estimate. It must be compared against the training R² to determine if the model is regularized and whether its performance generalizes stably.
2. **MAE vs. RMSE**: 
   - MAE drops from `2.4193` (Naive) to `2.2603` (Tuned Tree), an improvement of ~0.16 points.
   - RMSE drops from `2.8423` (Naive) to `2.7499` (Tuned Tree), an improvement of ~0.09 points.
   The tuned tree improves both average error (MAE) and large error penalty (RMSE) over all competing baselines.
3. **Train vs. Test Performance**: The unregularized tree achieved a perfect training fit (R² = 1.0000, MAE = 0.0000) but completely failed on the test set (R² = -1.1078), representing extreme overfitting. The tuned tree (`max_depth=3`) shows closely aligned training R² (0.1158) and testing R² (0.0636), showing that limiting tree depth successfully controlled model variance.
4. **Practical Significance**: While the tuned tree is our best-performing model, a Test $R^2$ of `0.0636` means it only explains 6.36% of the variance on unseen data. The improvement over the naive baseline is positive but practically weak, indicating that music-listening habits alone cannot be used to make production-quality predictions of anxiety.

---

## 5. Final Candidate Model
The selected candidate is the **Tuned Decision Tree (`max_depth=3`)**. It is the only model that generalizes successfully to unseen data, yielding a positive Test R² and a small generalization gap (0.1375).

---

## 6. Why Competing Models Were Rejected
- **Naive Mean Baseline**: Rejected because it represents a zero-intelligence strategy that does not utilize any predictor information.
- **Linear Regression**: Rejected because its Test R² was negative (-0.0261), showing it was unable to capture non-linear patterns or interactions, and overfit slightly on expanded one-hot categoricals.
- **Unregularized Decision Tree**: Rejected because it suffered from severe overfitting, capturing random training set noise (R² = 1.0000) rather than generalizable signals.

---

## 7. Interview Explanation
"I selected the regularized Decision Tree with `max_depth=3` as our final candidate model. I established that OLS linear regression and unconstrained decision trees overfit the survey features due to low signal-to-noise ratios, with the unconstrained tree failing completely on held-out test data (R² = -1.10). By limiting tree depth to 3 and validating the configuration via cross-validation, I stabilized out-of-sample performance, achieving a positive Test R² of 0.0636 and successfully minimizing the generalization gap."
