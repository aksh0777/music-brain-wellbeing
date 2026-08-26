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

---

## 3. Comparison Metrics Table
Below is the comparison table compiled from the actual training and testing evaluation metrics:

| Model | Train MAE | Test MAE | Train RMSE | Test RMSE | Train R² | Test R² |
|---|---|---|---|---|---|---|
| **Naive Mean Baseline** | 2.1963 | 2.4193 | 2.6565 | 2.8423 | 0.0000 | -0.0004 |
| **Linear Regression** | 2.1321 | 2.4173 | 2.5961 | 2.8786 | 0.1267 | -0.0261 |
| **Decision Tree** | 0.0000 | 3.3480 | 0.0000 | 4.1258 | 1.0000 | -1.1078 |

---

## 4. Overfitting Observation
The Decision Tree Regressor exhibits **extreme overfitting**:
- **Train $R^2$**: **`1.0000`** (Perfect prediction on training data, MAE = 0.0000)
- **Test $R^2$**: **`-1.1078`** (Severe failure on unseen test data, MAE = 3.3480, RMSE = 4.1258)

### Explanation from First Principles:
An untuned Decision Tree has no leaf constraints or depth limits. It continues splitting nodes until every node contains a single training sample or is completely pure. This allows it to perfectly memorize the exact values of the training set. 
However, this memorized structure is highly specific to the training sample noise and fails to generalize to the test set. In fact, a Test $R^2$ of `-1.1078` indicates that the model predictions are significantly worse than simply predicting the training target mean (which would yield an $R^2$ close to `0.0`).

---

## 5. Model Feature Importance
The relative predictive importance assigned to features by the Decision Tree:

| Feature | Importance Score | Type |
|---|---|---|
| `Age` | 0.1265 | Numerical Demographic |
| `BPM` | 0.0933 | Numerical Preference |
| `Hours per day` | 0.0670 | Numerical Behaviour |
| `Frequency [Rap]` | 0.0491 | Ordinal Context |
| `Frequency [Pop]` | 0.0479 | Ordinal Context |

### Important Scientific Note:
Feature importance measures how much the Decision Tree's split decisions reduced Mean Squared Error during training. It represents **predictive importance** within the model's structure. It does **not** establish a physical cause-and-effect relationship between these variables and a participant's anxiety levels, and holds no clinical significance.

---

## 6. Interpretation
1. **Did Decision Tree outperform Linear Regression?**: No, it performed significantly worse on the test set.
2. **Did it improve test RMSE?**: No, the test RMSE increased from `2.8786` (Linear Regression) to `4.1258` (Decision Tree).
3. **Did it improve test $R^2$ ?**: No, the test $R^2$ dropped from `-0.0261` (Linear Regression) to `-1.1078` (Decision Tree).
4. **Is there evidence of overfitting?**: Yes, the gap between Train $R^2$ (1.0000) and Test $R^2$ (-1.1078) is a definitive indicator of overfitting.
5. **Which features were most important to the tree?**: The continuous numerical features `Age` (12.7%), `BPM` (9.3%), and `Hours per day` (6.7%) were partitioned into highly specific splits to separate individual training instances.
6. **What does this tell us about the structure of our dataset?**: The dataset has a low signal-to-noise ratio. Using a complex, unconstrained model like a default Decision Tree allows the algorithm to fit noise rather than signal.

---

## 7. Limitations
- **No Causal Inference**: Observational data only. We cannot state that age or genre frequency *causes* anxiety.
- **Unconstrained Depth**: Default trees are guaranteed to overfit on small-to-medium datasets.
- **Sample Representation**: Limited generalizability (skewed young-adult Reddit convenience sample).

---

## 8. Next Recommended Step
To combat overfitting, we must constrain the tree's capacity to memorize the training data. The next step is to introduce ensemble methods (**`RandomForestRegressor`**) which average predictions across multiple randomized decision trees to reduce variance, followed by systematic hyperparameter tuning (constraining `max_depth`, `min_samples_split`, and `min_samples_leaf`).
