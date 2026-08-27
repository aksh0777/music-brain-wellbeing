# Model Interpretability Report

## 1. Problem
A machine learning model can successfully predict out-of-sample observations, but we must understand *how* it makes those decisions. Identifying which features carry the most weight in the model's split decisions helps us validate whether its rules align with domain knowledge and highlights the primary variables associated with the target.

---

## 2. Concept
In a Decision Tree, we evaluate feature contributions using **feature importances** (calculated via scikit-learn's `feature_importances_` attribute).
- **Gini Importance / Mean Decrease in Impurity (MDI)**: Measures the total reduction in Mean Squared Error (MSE) brought by splits on a given feature, weighted by the number of training samples passing through those splits.
- Features that are selected early (near the root node) or split repeatedly receive higher importance scores.

---

## 3. First-Principles Intuition
A Decision Tree splits data by partitioning feature spaces to reduce the variance of the target variable. Feature importance represents a feature's cumulative contribution to reducing that variance during training. If a feature is never used to partition the data, its importance is zero. If a feature is used at the root split to make a major partition, it receives a very high importance score.

---

## 4. Implementation
We fit our candidate model (`max_depth=3`) and extract the feature importances from the fitted pipeline. We match them with the feature names generated after one-hot encoding.

Below is the relative predictive importance assigned to features by the unconstrained and shallow Decision Tree models:

| Feature Name | Gini Importance (Unconstrained) | Gini Importance (max_depth=3) | Feature Type / Domain |
|---|---|---|---|
| `Age` | 12.65% | 29.83% | Continuous Demographic |
| `BPM` | 9.33% | 0.00% | Continuous Preference |
| `Hours per day` | 6.70% | 40.54% | Continuous Behavioral |
| `Frequency [Rap]` | 4.91% | 0.00% | Ordinal Context |
| `Frequency [Pop]` | 4.79% | 0.00% | Ordinal Context |
| `Music effects_Improve` | 0.00% | 29.63% | One-Hot Context / Perception |

*Note: In the constrained tree (`max_depth=3`), the model only split on three features: `Hours per day` (40.54%), `Age` (29.83%), and `Music effects_Improve` (29.63%). All other features (including BPM and individual genre frequencies) receive an importance of 0.00% because they were never split on within the top 3 levels.*

---

## 5. Domain Sense and Interpretation
- **`Hours per day` (40.54%)**: Makes domain sense. People experiencing higher anxiety levels may listen to music for longer durations as a coping mechanism or background distraction.
- **`Age` (29.83%)**: Matches demographic expectations. Younger populations typically report higher levels of anxiety in surveys compared to older cohorts.
- **`Music effects_Improve` (29.63%)**: Reflects subjective self-perception. Respondents who actively state that music improves their mental health state show different average anxiety distributions compared to those reporting "No effect" or "Worsen".

---

## 6. Association vs. Causation
We must establish a strict scientific boundary when interpreting these feature importances:
- **Association (Correlation)**: The model identifies that older/younger age or longer listening hours are *associated* with different average anxiety scores.
- **No Causality**: This does **not** prove that listening to music longer *causes* anxiety to increase, nor that music *causes* anxiety to decrease. It is highly likely that confounding variables (e.g. daily stress levels, job type, or underlying clinical conditions) drive both the listening hours and the anxiety level.

---

## 7. Next Modeling Step
Since a single regularized Decision Tree only splits on three broad variables, its decision function is very simple. To capture a richer set of associations across the remaining features without overfitting, we should implement a **`RandomForestRegressor`**. An ensemble of trees trains multiple randomized decision trees on bootstrapped samples, averaging their predictions to reduce variance and smooth out decision boundaries without overfitting.
