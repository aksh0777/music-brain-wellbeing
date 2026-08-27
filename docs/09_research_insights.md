# Research Insights

This document outlines the primary statistical and exploratory relationships investigated between music behavior, preferences, and self-reported anxiety levels, and discusses the distinction between association and causality.

---

## 1. Relationships Investigated

### A. Age vs. Anxiety
* **Variables compared**: `Age` (continuous numerical) vs. `Anxiety` (continuous target score 0-10).
* **Interesting Rationale**: Age acts as a key demographic control. We want to verify if younger cohorts report higher anxiety.
* **Findings**: We observed a **weak negative correlation** ($\rho = -0.1770$). As age increases, self-reported anxiety levels tend to decrease slightly. 
* **Model Consistency**: Consistent with our models; `Age` was selected for early splits in the Decision Tree, indicating it carries the most prominent linear and partition information.
* **Confounding Risk**: High. The relationship could be explained by lifestyle shifts, career stability, or differences in survey participation (most elderly participants did not use social media convenience samples, introducing sampling selection bias).

### B. Hours per Day vs. Anxiety
* **Variables compared**: `Hours per day` (continuous numerical) vs. `Anxiety` (continuous target score).
* **Interesting Rationale**: Our tuned Decision Tree (`max_depth=3`) relied heavily on `Hours per day` (40.54% Gini importance) to split samples.
* **Findings**: The overall linear correlation is **virtually zero** ($\rho = 0.0493$). 
* **Model Consistency**: Although the linear correlation is near zero, the Decision Tree selected this feature because it can partition the feature space non-linearly (e.g. isolating groups with very low or very high listening hours that exhibit different average anxiety scores).
* **Confounding Risk**: High. People with high anxiety might listen to music longer as a coping mechanism, while people with low anxiety might listen to music for background focus, confounding any simple relationship.

### C. Perceived Music Effects vs. Anxiety
* **Variables compared**: `Music effects` (categorical perception: Improve, No effect, Worsen) vs. `Anxiety` (continuous target score).
* **Interesting Rationale**: We want to analyze whether baseline anxiety varies across respondents' perceived subjective responses to music.
* **Findings**: 
  - **Improve**: Mean Anxiety = **`6.0282`**
  - **No effect**: Mean Anxiety = **`5.1243`**
  - **Worsen**: Mean Anxiety = **`6.7647`**
  Interestingly, respondents who report that music "improves" their wellbeing actually have a *higher* baseline anxiety average (`6.03`) than those reporting "no effect" (`5.12`). The small cohort reporting music "worsens" symptoms has the highest mean (`6.76`).
* **Model Consistency**: Consistent; the tuned tree used `Music effects_Improve` as a split indicator (29.63% Gini importance).
* **Confounding Risk**: High. People experiencing high baseline anxiety are more likely to seek out music to cope, leading to a high self-report of "improvement" despite their elevated symptoms. This is a classic case of reverse causation.

---

## 2. Association vs. Causation

### Scientific Boundary:
The baseline experiments and statistical summaries provide evidence of **predictive association** in the available variables, but they do **NOT** establish causality or clinical effectiveness.

* **Association (Correlation)**: Means that two variables co-vary together (e.g., younger participants report higher anxiety). It is represented by a non-zero correlation coefficient or partition split.
* **Causation**: Means that intervening to change variable $X$ directly causes a change in variable $Y$ ($P(Y \mid \text{do}(X))$).

### Why Observational Data Cannot Establish Causality:
1. **No Interventions**: We did not randomly assign participants to different listening groups (e.g. forcing one group to listen to 5 hours of Pop per day and another to listen to 0 hours) as in a Randomized Controlled Trial (RCT).
2. **Confounding Variables**: Confounders (e.g. daily work stress, clinical history, socio-economic status) are unmeasured and co-vary with both music habits and anxiety.
3. **Reverse Causality**: It is unclear if listening to music longer leads to higher anxiety, or if having higher anxiety causes people to listen to music longer to cope.

> [!IMPORTANT]
> To maintain scientific integrity, we do not claim that music listening habits *cause* changes in mental symptoms. We limit our language to predictive association: "associated with," "predictive of," "correlated with," and "the model estimates."
