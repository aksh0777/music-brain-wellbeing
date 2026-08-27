# Model Baselines

## 1. Objective
The objective of this modeling stage is to verify our validation pipeline and establish benchmark prediction performances. We investigate whether daily music-listening habits, preferences, and basic demographics contain useful predictive signal for self-reported anxiety scores before trying complex ensemble algorithms.

---

## 2. Data Used
- **Dataset**: [`data/processed/mxmh_cleaned.csv`](file:///c:/Users/aksha/OneDrive/Desktop/music-brain-wellbeing/data/processed/mxmh_cleaned.csv) (736 rows, 31 columns).
- **Target Variable**: `Anxiety` (continuous, numerical scale 0-10).
- **Predictors**: 27 input features. Excluded administrative metadata (`Timestamp`, `Permissions`) and target-leakage variables (`Depression`, `Insomnia`, `OCD`).

---

## 3. Train/Test Strategy
We implement a randomized 80/20 train/test split with a fixed seed (`random_state=42`) to guarantee reproducibility.
- **Training Set**: 588 observations.
- **Testing Set**: 148 observations.
To prevent data leakage, all preprocessing statistics (scaling means, standard deviations, categorical modes) are fitted strictly on the training set, and applied (transformed) to the test set.

---

## 4. Naive Baseline
Our naive reference benchmark predicts the training mean of the target variable (`5.8452`) for all test set observations.
- **Test MAE**: 2.4193
- **Test RMSE**: 2.8423
- **Test R²**: -0.0004
This represents a benchmark of zero predictive intelligence. A model is only useful if it outperforms these metrics.

---

## 5. Linear Regression
We train an Ordinary Least Squares (OLS) Linear Regression model wrapped in our preprocessing pipeline.
- **Training Metrics**: MAE = 2.1321 | RMSE = 2.5961 | R² = 0.1267
- **Testing Metrics**: MAE = 2.4173 | RMSE = 2.8786 | R² = -0.0261
- **Observation**: The model explains 12.67% of the variance on training data, but its Test R² is negative (-0.0261), meaning it performs slightly worse than always predicting the training mean. This indicates a low signal-to-noise ratio and slight overfitting due to unpenalized one-hot feature expansion.

---

## 6. Decision Tree
We train an untuned, default Decision Tree Regressor wrapped in our preprocessing pipeline.
- **Training Metrics**: MAE = 0.0000 | RMSE = 0.0000 | R² = 1.0000
- **Testing Metrics**: MAE = 3.3480 | RMSE = 4.1258 | R² = -1.1078
- **Observation**: The untuned tree memorizes the training data perfectly (R² = 1.0000), but fails completely on the test set (R² = -1.1078), performing significantly worse than both Linear Regression and the naive baseline.

---

## 7. Model Comparison
Below is the summary comparison table of all initial model baselines:

| Model | Train MAE | Test MAE | Train RMSE | Test RMSE | Train R² | Test R² |
|---|---|---|---|---|---|---|
| **Naive Mean Baseline** | 2.1963 | 2.4193 | 2.6565 | 2.8423 | 0.0000 | -0.0004 |
| **Linear Regression** | 2.1321 | 2.4173 | 2.5961 | 2.8786 | 0.1267 | -0.0261 |
| **Decision Tree** | 0.0000 | 3.3480 | 0.0000 | 4.1258 | 1.0000 | -1.1078 |

---

## 8. Generalization Gap
The **generalization gap** represents the difference between a model's training error and testing error (typically measured as Test RMSE - Train RMSE):
- **Linear Regression Generalization Gap**: $2.8786 - 2.5961 = \mathbf{0.2825}$
- **Decision Tree Generalization Gap**: $4.1258 - 0.0000 = \mathbf{4.1258}$

A large generalization gap is a quantitative symptom of **overfitting**. The Decision Tree's massive gap demonstrates that its unconstrained split logic captured training sample noise rather than generalizable signals, rendering it useless on unseen test observations.

---

## 9. Interpretation
- **Scientific Framing**: The baseline experiments provide evidence about predictive signal in the available survey variables, but they do not establish causality or clinical effectiveness.
- **Prediction**: Neither model can predict Anxiety scores with meaningful accuracy on unseen data (Test R² values are negative). This suggests the current dataset has a very weak predictive signal under simple OLS linear or unconstrained non-linear assumptions.
- **Association**: Continuous features such as `Age`, `BPM`, and `Hours per day` show predictive association inside the Decision Tree's split criteria, but unregularized models overfit these patterns rather than generalize them.
- **Causation**: The model does **NOT** establish that music-listening habits cause changes in anxiety.


---

## 10. Limitations
1. **Self-Reported Data**: Subjective scales (0-10) vary in interpretation by individual.
2. **Cross-Sectional design**: Data is a single snapshot; we cannot establish temporal ordering.
3. **Sample Characteristics**: High skew towards younger respondents recruited from Reddit, limiting generalizability.
4. **Confounding**: Other lifestyle factors (work hours, stress levels) are unmeasured.

---

## 11. Next Step
To resolve the high variance and overfitting of our models, the next step is to implement a **Random Forest Regressor** to average predictions over many randomized bootstrapped trees, and perform systematic hyperparameter tuning to restrict tree depth.
