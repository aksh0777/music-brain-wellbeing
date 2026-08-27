# Model Comparison Report

This document compares our regression models progressively: from the Naive Mean Baseline to Linear Regression and the Decision Tree Regressor.

---

## 1. Regression Target & Validation Setup
- **Target**: **`Anxiety`** (numerical, self-reported severity score from `0` to `10`).
- **Validation**: 80/20 train/test split.
  - Training set size: `588` observations.
  - Testing set size: `148` observations.
- **Preprocessing**: Applied a ColumnTransformer (median imputation + scaling for numerical columns; mode imputation + one-hot encoding for categorical columns) wrapped inside our model pipelines to ensure zero-leakage.

---

## 2. Models Compared
1. **Naive Mean Baseline**: Always predicts the training mean of the target.
2. **Linear Regression Baseline**: Ordinary Least Squares (OLS) linear model wrapped in the preprocessing pipeline.
3. **Decision Tree Regressor (Untuned)**: A default, depth-unbounded decision tree model wrapped in the preprocessing pipeline.
4. **Best Tuned Decision Tree**: A Decision Tree Regressor constrained to `max_depth=3` to prevent overfitting.

---

## 3. Comparison Metrics Table
Below is the comparison table compiled from the actual training and testing evaluation metrics:

| Model | Train MAE | Test MAE | Train RMSE | Test RMSE | Train R² | Test R² |
|---|---|---|---|---|---|---|
| **Naive Mean Baseline** | 2.1963 | 2.4193 | 2.6565 | 2.8423 | 0.0000 | -0.0004 |
| **Linear Regression** | 2.1321 | 2.4173 | 2.5961 | 2.8786 | 0.1267 | -0.0261 |
| **Unregularized Decision Tree** | 0.0000 | 3.3480 | 0.0000 | 4.1258 | 1.0000 | -1.1078 |
| **Best Tuned Decision Tree (depth=3)** | 2.1633 | 2.2603 | 2.6124 | 2.7499 | 0.1158 | **0.0636** |

---

## 4. Overfitting Observation
- The **Unregularized Decision Tree** exhibits extreme overfitting: Train $R^2 = 1.0000$, Test $R^2 = -1.1078$. Without constraints, it splits nodes recursively until every training point is isolated, memorizing noise rather than signal.
- The **Tuned Decision Tree (`max_depth=3`)** restricts decision levels to 3 (8 leaf nodes). This reduces model variance, successfully mitigating overfitting (Train $R^2 = 0.1158$, Test $R^2 = 0.0636$). The generalization gap drops from `4.1258` to `0.1375`.

---

## 5. Model Feature Importance
The relative predictive importance assigned to features by the unregularized Decision Tree:

| Feature | Importance Score | Type |
|---|---|---|
| `Age` | 0.1265 | Numerical Demographic |
| `BPM` | 0.0933 | Numerical Preference |
| `Hours per day` | 0.0670 | Numerical Behaviour |
| `Frequency [Rap]` | 0.0491 | Ordinal Context |
| `Frequency [Pop]` | 0.0479 | Ordinal Context |

---

## 6. Interpretation
1. **Did Decision Tree outperform Linear Regression?**: The unregularized tree did not, but the **tuned tree (`max_depth=3`)** did, improving the Test R² from `-0.0261` (Linear Regression) to **`0.0636`**.
2. **Did it improve test RMSE?**: Yes, the tuned tree reduced test RMSE to `2.7499` (compared to `2.8786` for Linear Regression and `2.8423` for the Naive Mean).
3. **Is there evidence of overfitting?**: The unregularized tree shows severe overfitting. The tuned tree shows minimal overfitting, as its train and test metrics are closely aligned.
4. **What does this tell us about the structure of our dataset?**: The dataset has a very low signal-to-noise ratio. Complex models overfit rapidly. Shallow decision trees are more robust because they act as coarse categorizers, capturing broad structural patterns (such as age or listening duration boundaries) without memorizing individual records.

---

## 7. Limitations
- **No Causal Inference**: Observational data only. We cannot state that age or genre frequency *causes* anxiety.
- **Coarse Splits**: A shallow tree with depth 3 is restricted to only 8 leaf nodes, which simplifies the prediction function significantly.
- **Sample Representation**: Limited generalizability (young-skewed Reddit convenience sample).

---

## 8. Next Recommended Step
To capture non-linear patterns more stably, we should implement a **`RandomForestRegressor`**. An ensemble of trees trains multiple randomized decision trees on bootstrapped samples, averaging their predictions to reduce variance and smooth out decision boundaries without overfitting.

