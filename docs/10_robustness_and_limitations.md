# Robustness and Limitations

This document presents our robustness checks to evaluate the stability of our findings and outlines the structured data, model, and scientific limitations of the project.

---

## 1. Robustness Checks

### A. Check 1: Exclude Extreme Listening Hours (>12 hours/day)
* **WHAT**: We checked whether dropping the 10 participants who reported listening to music for more than 12 hours per day affects the correlation between listening hours and anxiety.
* **WHY**: Extreme values can act as leverage points, skewing correlation coefficients.
* **RESULT**:
  - Overall correlation ($N=736$): `0.0493`
  - Subgroup correlation $\le 12$ hours ($N=726$): `0.0730`
* **INTERPRETATION**: The relationship remains extremely weak. This confirms that listening duration has virtually no linear predictive signal for anxiety, and our baseline is not dominated by extreme listening outliers.

### B. Check 2: Exclude Elderly Participants (Age > 70)
* **WHAT**: We checked whether dropping the 6 participants older than 70 years alters the negative correlation between age and anxiety.
* **WHY**: Older individuals can act as leverage points in linear relationships on skewed demographic samples.
* **RESULT**:
  - Overall correlation ($N=736$): `-0.1770`
  - Subgroup correlation $\le 70$ years ($N=730$): `-0.1522`
* **INTERPRETATION**: The correlation remains negative and stable in magnitude. This confirms that the negative relationship between age and anxiety is consistent across the core population, rather than being driven by a small number of elderly observations.

### C. Check 3: Imputation Audit (BPM Imputation vs. Row Deletion)
* **WHAT**: We compared the correlation between BPM and Anxiety in our cleaned, median-imputed dataset against the raw dataset where rows with missing or out-of-bounds BPM were simply deleted.
* **WHY**: Imputing 107 missing BPM values with the median could artificially compress covariance and distort relationships.
* **RESULT**:
  - Cleaned (Median-Imputed) correlation ($N=736$): `0.0512`
  - Raw (Dropped NaNs) correlation ($N=622$): `0.0557`
* **INTERPRETATION**: The correlation coefficient changed by only `0.0045`. This demonstrates that our median imputation decision did not distort the feature's covariance structure, verifying that the imputation method is robust.

### D. Check 4: Cross-Validation Hyperparameter Stability
* **WHAT**: We checked if `max_depth=3` remains the best performing hyperparameter configuration across all 5 cross-validation folds.
* **WHY**: If different folds prefer different hyperparameters, our selection is unstable and depends on partition noise.
* **RESULT**: Across all 5 training folds, `max_depth=3` consistently minimized validation mean squared error compared to deeper configurations (`max_depth=5` or unconstrained).
* **INTERPRETATION**: Hyperparameter selection is highly stable and represents a robust generalizable complexity boundary.

---

## 2. Limitations

### A. Data Limitations
1. **Self-Reported Survey Data**: Symptoms (`Anxiety`, `Depression`) and habits (`Hours per day`, `BPM`) are self-reported. This introduces recall bias (e.g. estimating listening hours inaccurately) and rating scale subjectivity (anxiety "5" to one person might feel like an "8" to another).
2. **Missing Confounders**: The dataset lacks crucial covariates (e.g. daily work stress, clinical history, career status, socio-economic variables) that are strongly associated with mental health.
3. **Convenience Sampling Bias**: Most participants were recruited from online forums (Reddit), skewing the sample heavily towards younger, tech-literate populations, which limits generalizability.

### B. Model Limitations
1. **Shallow Split Constraints**: Restricting the Decision Tree to `max_depth=3` limits the model to only 8 unique predictions. This prevents the model from predicting extreme values (0 or 10) and compresses predictions toward the mean (predictions range strictly from `3.03` to `8.20`).
2. **Decision Tree Instability**: Single decision trees are high-variance estimators; small changes in the training set can lead to completely different root splits.
3. **Sparse Representation**: One-hot encoding categorical variables (like `Fav genre`) generates 54 sparse dimensions, which are difficult for simple trees to split on without overfitting.

### C. Scientific Interpretation Limitations
1. **Observational Cross-Sectional Design**: Data was collected at a single point in time. We cannot establish temporal ordering (e.g., whether music listening changes anxiety, or anxiety changes music listening).
2. **No Causal Inference**: The study cannot prove causal relationships. All findings are strictly limited to out-of-sample prediction and association.
3. **No Clinical Validity**: Self-reported scores do not correspond to clinical diagnoses.
