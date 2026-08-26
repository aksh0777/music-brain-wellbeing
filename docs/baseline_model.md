# Baseline Model Report

This document records the configuration, training, evaluation, and interpretation of our baseline model for the **Music, Brain & Wellbeing** project.

---

## 1. Problem Statement
The objective of this project is to investigate whether self-reported music listening habits and demographics contain predictive information about a participant's self-reported psychological wellbeing. We frame this as an observational regression task.

## 2. Target
- **Primary Target**: **`Anxiety`** (numerical, self-reported severity score ranging from `0` to `10`).

## 3. Features
We utilize a feature space $X$ of **27 input columns** (which expands to **54 columns** after one-hot encoding categorical variables). 
- **Excluded Features**: `Depression`, `Insomnia`, and `OCD` are excluded from features to prevent target leakage. `Timestamp` and `Permissions` are also excluded.

## 4. Baseline Approach
We establish a naive benchmark: predicting the training target mean for all test set observations. This represents a zero-intelligence, zero-variance baseline.
- **Training target mean**: **`5.8452`** (on the 0-10 Anxiety scale).

## 5. Model Used
We wrap our preprocessor `ColumnTransformer` and a standard **`LinearRegression`** model into a scikit-learn `Pipeline`.
- **Training**: Fitted strictly on the training set `(X_train, y_train)` of size `588`.
- **Testing**: Evaluated strictly on the unseen test set `(X_test, y_test)` of size `148`.

## 6. Evaluation Metrics
We use three standard regression metrics:
1. **MAE (Mean Absolute Error)**: Measures average absolute prediction error.
2. **RMSE (Root Mean Squared Error)**: Standard deviation of residuals; penalizes larger errors more heavily.
3. **$R^2$ (Coefficient of Determination)**: Proportion of target variance explained by features.

---

## 7. Results & Baseline Comparison

| Model | MAE | RMSE | $R^2$ |
|---|---|---|---|
| **Naive Mean Baseline** | 2.4193 | 2.8423 | -0.0004 |
| **Linear Regression Pipeline** | 2.4173 | 2.8786 | -0.0261 |

### Interpretation
- **How well did the baseline perform?**: The Linear Regression pipeline performs poorly. It achieves an $R^2$ of **`-0.0261`** on the test set, indicating that it performs slightly worse than a simple naive mean predictor.
- **Is the model better than the naive baseline?**: No. While the MAE is marginally lower (2.4173 vs 2.4193), the RMSE is higher (2.8786 vs 2.8423) and the $R^2$ is negative. This indicates mild overfitting on the training set.

---

## 8. Important Observations & Interpretability
We inspect the model's coefficients to understand which features were given the largest weights:

| Feature | Coefficient | Absolute Magnitude |
|---|---|---|
| `Fav genre_Latin` | -1.6693 | 1.6693 |
| `Primary streaming service_Pandora` | +0.8950 | 0.8950 |
| `Fav genre_Hip hop` | +0.8894 | 0.8894 |
| `Primary streaming service_Apple Music` | +0.8613 | 0.8613 |
| `Primary streaming service_YouTube Music` | -0.7834 | 0.7834 |

### Rationale for Weak Performance
1. **Low Signal-to-Noise Ratio**: Self-reported music metadata alone carries very weak predictive signal for self-reported anxiety.
2. **High-Dimensional Categorical Expansion**: One-hot encoding creates many binary features for rare categories. For example, `Fav genre_Latin` has only 3 occurrences in the entire dataset of 736 rows. A standard unregularized Linear Regression model can assign extremely large coefficients to these rare categories to minimize training loss, leading to poor generalization (overfitting) on the test set.
3. **Non-Linear Relationships**: The true relationship between music preferences, listening hours, and mental health symptoms is highly non-linear and context-dependent.

---

## 9. Limitations
- **Observational Constraint**: The coefficients show statistical associations used by the model, **not** causal relationships. We cannot state that listening to Latin music *causes* a decrease in anxiety.
- **Sample Bias**: The survey dataset is self-selected (Reddit users) and skews young, so results do not generalize to the wider population.
- **Linear Assumption**: The model is restricted to additive linear boundaries.

---

## 10. Next Recommended Step
Since standard unregularized Linear Regression overfits due to categorical expansion and has no mechanism to penalize large weights, the next step is to introduce regularized regression (Ridge/Lasso) or tree-based models (Decision Trees and Random Forests). Trees do not require scaling and can capture non-linear, step-wise thresholds.
