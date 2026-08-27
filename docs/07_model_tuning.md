# Model Tuning, Validation, and Interpretation

This document details our hyperparameter tuning process, validation strategy, model selection checks, error analysis, and final modeling conclusions.

---

## 1. Problem and Concept
An untuned Decision Tree has unbounded depth, allowing it to split recursively until it perfectly separates every training observation. In datasets with a low signal-to-noise ratio, this unconstrained model complexity results in the model memorizing random training noise, leading to extreme overfitting and a failure to generalize to new unseen test data.

To prevent overfitting, we regularize the tree using hyperparameters:
- **`max_depth`**: Limits the number of sequential decision levels. Lower depth restricts the tree to simple, coarse decisions.
- **`min_samples_leaf`**: Restricts splits by requiring each leaf node to contain at least $N$ observations, preventing the model from writing specific rules for single sample noise.

---

## 2. Model Selection Leakage Check
We performed a strict audit of our model selection methodology to ensure the integrity of our validation pipeline:
1. **Were hyperparameters selected using only training data / cross-validation?**  
   *Yes*. The choice of `max_depth=3` was based on 5-fold cross-validation scores computed strictly on the training set `X_train` (where `max_depth=3` achieved the highest mean validation $R^2$ of `-0.1027`, compared to `-0.2765` for `max_depth=5` and `-1.0465` for default parameters).
2. **Was the test set used to select `max_depth`?**  
   *No*. The test set was not used to select the best hyperparameter configuration.
3. **Was the test set only used for final evaluation?**  
   *Yes*. The test set `X_test` remained completely untouched during model selection and was only evaluated once at the very end to estimate out-of-sample generalization.

> [!NOTE]
> This strict test set isolation is a major methodological strength. It guarantees that our final test performance metrics are entirely unbiased and reflect true generalizability.

---

## 3. Final Model Evaluation
We evaluate our candidate model configuration (`DecisionTreeRegressor(max_depth=3)`) against the baselines on the holdout test set:

| Model | Configuration | CV R² Mean | Test MAE | Test RMSE | Test R² | Overfitting Observation |
|---|---|---|---|---|---|---|
| **Naive Mean Baseline** | Predict $y_{train}$ mean | N/A | 2.4193 | 2.8423 | -0.0004 | None (underfits; zero variance) |
| **Linear Regression** | OLS Pipeline | -0.0531 | 2.4173 | 2.8786 | -0.0261 | Mild (unpenalized OLS coefficients) |
| **Unregularized Decision Tree** | Default parameters | -1.0465 | 3.3480 | 4.1258 | -1.1078 | Severe (Train R² = 1.0000, Test R² = -1.1078) |
| **Tuned Decision Tree** | `max_depth=3` | **-0.1027** | **2.2603** | **2.7499** | **0.0636** | Minimal (Train R² = 0.1158, Test R² = 0.0636) |

### Statistical Meaning of R² = 0.0636:
An $R^2$ of `0.0636` on the test set means that our tuned model explains **6.36% of the variance** in self-reported anxiety scores on unseen test data compared to a naive baseline that always predicts the training mean. 
* **What it does NOT mean**: It does **not** mean the model explains "6.36% of mental health."
* **Reality**: The model's predictive performance is very weak. While it successfully captures some non-linear patterns that OLS missed (raising test $R^2$ from `-0.0261` to `+0.0636`), 93.64% of the variance in self-reported anxiety is unexplained by our predictors.

---

## 4. Error Analysis
We analyzed the errors (residuals) of our tuned model (`max_depth=3`) on the test set:
1. **Prediction Compression**: The model predictions are concentrated in a narrow band around the mean (**`3.03` to `8.20`**), with a standard deviation of only `0.98` (compared to the actual test target standard deviation of `2.85`).
2. **Reason**: A tree with depth 3 has a maximum of $2^3 = 8$ leaf nodes. The prediction for any test sample must be one of these 8 unique leaf averages. Since leaf nodes average training targets, their values are naturally pulled toward the overall sample mean (`5.85`).
3. **Under/Overprediction Bias**:
   * **Low actual scores (0–2)** are systematically **overpredicted** (the minimum predicted score is `3.03`).
   * **High actual scores (8–10)** are systematically **underpredicted** (the maximum predicted score is `8.20`).
4. **Extreme Target Values**: Extreme values (0 or 10) are poorly predicted because the tree does not split deeply enough to isolate extreme individuals, which prevents it from memorizing noise but also prevents it from fitting actual extreme cases.

---

## 5. First-Principles Interpretation
Why can a correctly implemented model yield Train $R^2 = 0.1158$ and Test $R^2 = 0.0636$?
- **Model Correctness**: Means that the code is free of implementation bugs, the training loop is mathematical, and the train/test split has zero data leakage.
- **Predictive Signal**: Represents the actual relationship between our features and target. If daily music listening habits simply do not strongly relate to a person's anxiety levels, the features contain little predictive signal.
- **Data Limitations**: Observational survey data has inherent limits. A model can be 100% correct in its math but fail to achieve high performance because the features do not contain enough information to predict the target.

---

## 6. Model Limitations
1. **Noisy Self-Reports**: Survey responses (rating scale 0-10) are highly subjective. One person's "5" is another's "8", adding measurement noise.
2. **Missing Confounders**: The dataset lacks crucial background variables (such as work hours, relationship status, family history, and clinical status) that co-vary with anxiety.
3. **Cross-Sectional**: The survey is a snapshot in time, preventing us from capturing longitudinal trends or establishing temporal sequence.
4. **Sample Skew**: Most respondents are young adults recruited from Reddit, limiting generalizability.

---

## 7. Modeling Conclusion
1. **Best Model**: The regularized Decision Tree (`max_depth=3`) performed best on unseen test data.
2. **Performance vs. Baselines**: It outperformed the Naive Baseline ($R^2 = -0.0004$) and Linear Regression ($R^2 = -0.0261$) by achieving a positive Test $R^2$ of **`0.0636`**.
3. **Generalization**: It generalizes successfully, as shown by its closely aligned training $R^2$ (`0.1158`) and testing $R^2$ (`0.0636`) and a small generalization gap (`0.1375`).
4. **Signal Strength**: The overall predictive signal remains **weak**. The features carry some predictive association, but most of the variance in anxiety remains unexplained.
5. **Next Steps**: Introduce ensemble models (**`RandomForestRegressor`**) to average predictions across multiple bootstrap samples, which stabilizes predictions and reduces variance.

---

## 8. Interview Preparation

### 1. Why did the original Decision Tree overfit?
Without constraints, the tree splits nodes recursively until every leaf is pure (contains a single training sample). This allows the model to perfectly memorize the specific noise of training observations (Train $R^2 = 1.0$), which fails to generalize to test data.

### 2. What does `max_depth` do?
`max_depth` restricts the maximum number of split levels from the root to the leaves. Limiting depth forces the model to make coarse splits on major structural patterns, regularizing the model by reducing variance.

### 3. Why did you use cross-validation?
A single train/test split can produce noisy validation estimates because the validation partition might happen to contain unusually easy or hard observations. K-fold cross-validation averages metrics across multiple partitions, providing a stable estimate for hyperparameter selection.

### 4. Why shouldn't the test set be used for hyperparameter tuning?
If we tune hyperparameters by optimizing test set scores, we leak information from the test set into our model selection process. This makes the test set a pseudo-validation set, inflating our performance estimate.

### 5. Why can R² be negative?
$R^2$ compares the model's MSE to the MSE of a baseline that always predicts the training target mean. If a model makes large prediction errors (such as our overfit tree), its MSE is larger than the mean baseline's variance, yielding a negative $R^2$.

### 6. Why can a correctly implemented model still have weak performance?
Model correctness refers to the integrity of the code and validation pipeline (no leakage, correct math). Model performance depends on the predictive signal in the data. If the features do not contain enough information to explain the target, even a mathematically correct model will have low predictive power.

### 7. Why didn't you choose the model with the highest training R²?
Training $R^2$ only measures how well the model memorizes the training data. A model with Train $R^2 = 1.0$ (the unregularized tree) failed completely on unseen data. We must always select models based on out-of-sample generalization (Test $R^2$ or CV score).

### 8. Does this project prove that music affects mental health?
No. The dataset is observational and cross-sectional, which restricts us to modeling prediction and association. We cannot control for confounding variables, and therefore cannot establish causal relationships.

### 9. What would you try next if more data were available?
I would evaluate ensemble techniques (Random Forests, Gradient Boosting) to reduce variance, and seek to acquire longitudinal data or control for key confounding variables like daily work hours or baseline clinical history.
