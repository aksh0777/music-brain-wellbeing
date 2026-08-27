# Final Analysis

This document provides a comprehensive analysis of the **Music, Brain & Wellbeing** machine learning project. It synthesizes our research framework, preprocessing, modeling results, limitations, and key interview takeaways.

---

## 1. Project Objective
The objective of this project is to investigate whether daily music-listening habits (listening hours, context, genre preferences, and streaming platforms) contain predictive or associative signals for a person's self-reported psychological wellbeing.
* **Scope**: This study is predictive and associative, **not** causal or diagnostic. The goal is to determine if features carry information that correlates with symptom levels on unseen data.

---

## 2. Dataset
We analyzed the **Music & Mental Health Survey (MXMH)** dataset.
- **Size**: 736 records, 31 cleaned columns.
- **Target Variable**: `Anxiety` (continuous rating from `0` to `10`).
- **Demographics**: Participant age ranges from 10 to 89, with a high concentration of young adults.
- **Music Habits**: Hours of daily listening range from 0 to 24 hours. BPM ranges from 40 to 220 (clipped physiologically).

---

## 3. Feature Pipeline
We constructed a robust, zero-leakage preprocessing pipeline using scikit-learn's `ColumnTransformer`:
1. **Numerical Columns** (`Age`, `Hours per day`, `BPM`, 16 `Frequency [genre]` ordinal mappings): Imputed with the training median and standardized.
2. **Categorical Columns** (`Primary streaming service`, `Fav genre`, `While working`, `Instrumentalist`, `Composer`, `Exploratory`, `Foreign languages`, `Music effects`): Imputed with the training mode and one-hot encoded.
3. **Target Split**: Preprocessing statistics were fitted strictly on `X_train` to prevent test-set leakage.

---

## 4. Models Tested
1. **Naive Mean Baseline**: Predicts the training target mean (`5.8452`) for all test samples.
2. **Linear Regression Baseline**: Ordinary Least Squares (OLS) regression fitted on the preprocessed pipeline.
3. **Unregularized Decision Tree**: A default depth-unbounded tree regressor.
4. **Tuned Decision Tree**: A Decision Tree Regressor constrained to `max_depth=3` to manage model variance.

---

## 5. Model Comparison
Below is the final comparison of model performance on the held-out test set:

| Model | Configuration | CV R² Mean | Test MAE | Test RMSE | Test R² | Train R² | Overfitting Observation |
|---|---|---|---|---|---|---|---|
| **Naive Mean Baseline** | Predict y_train mean | N/A | 2.4193 | 2.8423 | -0.0004 | 0.0000 | None (underfits; zero variance) |
| **Linear Regression** | OLS Pipeline | -0.0531 | 2.4173 | 2.8786 | -0.0261 | 0.1267 | Mild (unpenalized OLS coefficients) |
| **Unregularized Decision Tree** | Default parameters | -1.0465 | 3.3480 | 4.1258 | -1.1078 | 1.0000 | Severe (Train R² = 1.0000, Test R² = -1.1078) |
| **Tuned Decision Tree** | `max_depth=3` | **-0.1027** | **2.2603** | **2.7499** | **0.0636** | 0.1158 | Minimal (Train R² = 0.1158, Test R² = 0.0636) |

---

## 6. Best Model
The **Tuned Decision Tree (`max_depth=3`)** is our best performing model on unseen test data.
- It was selected strictly using 5-fold cross-validation on `X_train` (where it achieved the highest mean R² of `-0.1027` among the candidate tree models).
- The held-out test set was isolated and only evaluated once at the very end.
- The tuned tree outperformed Linear Regression (Test $R^2 = -0.0261$) and the Naive Mean (Test $R^2 = -0.0004$) by achieving a positive Test $R^2$ of **`0.0636`** and reducing Test RMSE to **`2.7499`**.

---

## 7. Generalization Analysis
- The **unregularized tree** has a generalization gap (Test RMSE - Train RMSE) of **`4.1258`**, indicating extreme overfitting. It memorized individual training samples.
- The **tuned tree (`max_depth=3`)** has a generalization gap of **`0.1375`** (Test RMSE = 2.7499 vs. Train RMSE = 2.6124).
- By restricting tree depth, we successfully controlled model variance. The train and test metrics are closely aligned, demonstrating that the regularized tree generalizes to new observations.

---

## 8. Error Analysis
We analyzed the errors of the tuned tree model:
- **Prediction Compression**: Model predictions are clustered in a narrow band between **`3.03` and `8.20`** ($\sigma = 0.98$), while actual scores span the entire `0` to `10` range ($\sigma = 2.85$).
- **Under/Overprediction Bias**: The model overpredicts low anxiety scores (minimum prediction is 3.03) and underpredicts high anxiety scores (maximum prediction is 8.20).
- **Extreme Target values**: Since depth is limited to 3, the model can only output 8 unique leaf averages. This prevents the tree from splitting deeply enough to isolate extreme scores (0 or 10) without overfitting, rendering predictions near the boundaries inaccurate.

---

## 9. Scientific Interpretation
- **Association**: Features like `Age`, `BPM`, and `Hours per day` carry associative weight during tree splitting, suggesting correlations exist in the survey sample.
- **Prediction**: While out-of-sample prediction is statistically positive ($R^2 = 0.0636$), it is practically weak. The features leave 93.64% of the target's variance unexplained.
- **Causation**: This project does **NOT** establish that music listening habits cause or alleviate anxiety. Confounding variables are uncontrolled, and the data is cross-sectional.

---

## 10. Limitations

### A. Observational Survey Design
* **WHAT**: The dataset was collected via a cross-sectional public survey.
* **WHY**: There is no experimental control group or longitudinal tracking. We can only model correlation, not causation.

### B. Subjective self-reports
* **WHAT**: Anxiety and other mental health metrics are rated on an arbitrary 0-10 scale.
* **WHY**: Individuals interpret rating scales differently, introducing substantial subjective measurement noise.

### C. Missing Explanatory Variables
* **WHAT**: The dataset lacks crucial covariates (e.g. daily work hours, job stress, relationship status, family clinical history).
* **WHY**: These missing variables likely explain the majority of the target's variance, leaving the model with a weak predictive signal from music habits alone.

---

## 11. Challenges & How They Were Solved

### A. Python Import Path Resolution
* **Challenge**: Encountering `ModuleNotFoundError: No module named 'src'` when executing notebooks in subdirectories.
* **Why**: Python searches the local execution directory for imports first.
* **Diagnosis**: The traceback showed import failures for custom modules located in the project root.
* **Solution**: Programmatically inserted the project root directory into `sys.path` at the start of our notebooks.
* **What I Learned**: Subdirectory imports must manage search paths programmatically to guarantee notebook portability.

### B. Decision Tree Overfitting
* **Challenge**: Default trees achieved a training MAE of 0.0000 but a test $R^2$ of -1.1078.
* **Why**: Unconstrained splits continue until all leaves are pure, memorizing training noise.
* **Diagnosis**: Compared train and test metrics and observed a massive generalization gap.
* **Solution**: Constrained depth to `max_depth=3` and used 5-fold cross-validation to select parameters.
* **What I Learned**: Training performance is deceptive. Regularization parameters must restrict model capacity to prevent overfitting.

---

## 12. Final Conclusion
The tuned Decision Tree (`max_depth=3`) successfully prevented overfitting, generalizing to unseen test data with a positive Test $R^2$ of **`0.0636`** (outperforming OLS linear regression and naive benchmarks). However, the overall predictive signal remains weak. Music listening habits alone do not carry sufficient information to predict self-reported anxiety scores with high precision.

---

## 13. Future Work

### Within the Current Dataset
1. **Regularized Linear Models**: Train Lasso or Ridge regression to penalize the 54 categorical indicator weights.
2. **Ensemble Modeling**: Train a Random Forest Regressor to average predictions across multiple bootstrap samples, which stabilizes predictions and reduces variance.

### Requiring New Data
1. **Acquire Longitudinal Data**: Track participants' listening logs and self-reported anxiety over weeks or months.
2. **Collect Clinical Status and Confounders**: Include clinical diagnoses, daily work hours, and stress levels to control for background confounding.

---

## 14. Interview Summary
* **The Project**: Analyzed the MXMH survey dataset to predict continuous self-reported anxiety levels from music listening habits.
* **The Baseline**: Established a naive mean baseline (Test RMSE = 2.84) and Linear Regression baseline (Test RMSE = 2.87, Test R² = -0.026).
* **The Overfitting**: An untuned Decision Tree memorized training noise (Train R² = 1.0, Test R² = -1.10).
* **The Solution**: Regularized the tree to `max_depth=3` using 5-fold cross-validation on the training set, achieving a positive Test $R^2$ of `0.0636` and reducing Test RMSE to `2.7499`.
* **The Takeaway**: Proved that restricting model complexity prevents overfitting, while honestly documenting that daily music habits carry a very weak predictive signal for subjective anxiety scores.
