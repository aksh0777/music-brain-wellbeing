# Project Findings

This document summarizes the core research and machine learning findings of the **Music, Brain & Wellbeing** project.

---

## 1. Research Question
To what extent can daily music listening behaviors (listening hours, preferences, BPM, and context) predict or explain a person's self-reported anxiety symptom severity on a 0-10 scale?

---

## 2. Dataset
We analyzed the **Music & Mental Health Survey (MXMH)** dataset, consisting of **`736`** processed records and **`31`** cleaned columns (representing demographics, behavioral listening habits, preferred genres, context frequencies, and self-reported mental wellbeing scores).

---

## 3. Main Findings

### A. Weak Association Between Music Listening Duration and Anxiety
* **Finding**: Daily music listening duration carries virtually zero linear predictive correlation with self-reported anxiety levels.
* **Evidence**: The Pearson correlation coefficient is extremely weak ($\rho = 0.0493$). Dropping outliers listening to music >12 hours per day only slightly shifts the correlation to $\rho = 0.0730$.
* **Interpretation**: Simply listening to music longer is not associated with higher or lower anxiety.
* **Caveat**: The relationship is cross-sectional and does not capture contextual variation (e.g. listening to relaxing classical music vs. listening to high-energy metal during work).

### B. Self-Reported Improvement Co-occurs with Elevated Baseline Anxiety
* **Finding**: Respondents who report that music "improves" their wellbeing actually exhibit higher baseline anxiety averages compared to those reporting "no effect".
* **Evidence**: Average anxiety for the "Improve" group is **`6.03`** compared to **`5.12`** for the "No effect" group.
* **Interpretation**: Individuals experiencing higher baseline anxiety levels are more likely to seek out music as a coping tool, resulting in high self-reports of perceived improvement.
* **Caveat**: This represents **reverse causality**; the subjective perception of improvement does not imply that music successfully lowered their baseline anxiety score below the average population.

### C. Age is Associated with Lower Reported Anxiety
* **Finding**: Older age cohorts tend to report slightly lower anxiety scores in this survey population.
* **Evidence**: We observed a weak negative correlation between age and anxiety ($\rho = -0.1770$). This remains stable ($\rho = -0.1522$) when excluding elderly leverage points (>70 years).
* **Interpretation**: Demographics are the strongest indicators of baseline self-reported anxiety.
* **Caveat**: The survey convenience sample is heavily skewed toward young adults recruited online (Reddit), which may introduce sampling selection bias.

---

## 4. Machine Learning Findings
- **Baseline Performance**: Naive Mean predictions yield a Test $R^2$ of `-0.0004` (RMSE = 2.8423). OLS Linear Regression overfits categorical features, yielding a negative Test $R^2$ of `-0.0261` (RMSE = 2.8786).
- **Tuned Model Performance**: Restricting a Decision Tree Regressor to `max_depth=3` successfully prevents overfitting. It achieves a positive Test $R^2$ of **`0.0636`** and reduces Test RMSE to **`2.7499`**.
- **Generalization**: The tuned tree's training $R^2$ (0.1158) and test $R^2$ (0.0636) are closely aligned, with a small generalization gap (0.1375).
- **Most Important Features**: Gini variance reduction identified three active splitting features: `Hours per day` (40.54%), `Age` (29.83%), and `Music effects_Improve` (29.63%).

---

## 5. Statistical / Exploratory Findings
- Pearson correlation values show that continuous listening attributes like `BPM` ($\rho = 0.0512$) and `Hours per day` ($\rho = 0.0493$) have almost zero linear association with anxiety scores, explaining why Linear Regression was unable to find linear boundaries.
- Group summaries show that the small cohort reporting music "worsens" symptoms has the highest mean anxiety score (**`6.76`**).

---

## 6. What We Can Conclude
1. **Model Complexity Control**: Limiting model complexity (`max_depth=3`) is essential for out-of-sample generalizability when features have a high noise-to-signal ratio.
2. **Weak Predictive Signal**: Demographics (`Age`), daily duration (`Hours per day`), and perceived effect (`Music effects`) carry a small but positive predictive signal (Test $R^2 = 0.0636$, RMSE = 2.7499), outperforming naive and OLS benchmarks.
3. **Scientific Value of Weak $R^2$**: Explaining 6.36% of the variance is an essential research finding: it empirically demonstrates that daily music listening behaviors alone cannot reliably predict or diagnose subjective mental health states.

---

## 7. What We Cannot Conclude
1. **No Causation**: We **cannot** conclude that music-listening habits *cause* or *alleviate* anxiety.
2. **No Clinical Diagnosis**: We **cannot** state that high self-reported survey ratings represent clinical psychiatric diagnoses.
3. **No Clinical Overclaims for Spotify Users**: We **cannot** assume that individual Spotify listening streams can be used to diagnose mental health.

---

## 8. How This Finding Influenced Project Evolution
The weak predictive signal directly guided the architectural evolution of the **Music Brain Wellbeing Intelligence System**:
- **From Population Prediction to Individual Intelligence**: Recognizing that static survey features cannot diagnose anxiety, we evolved the system toward analyzing individual listening streams (30-minute sessionization, cyclical temporal patterns, K-Means acoustic profiles).
- **From Outcome Modeling to Content-Based Recommendation**: We repurposed audio features into a personalized recommendation engine that matches user acoustic habit vectors without making unsupported therapeutic claims.
- **Data Boundary**: The MXMH dataset established the scientific boundaries of the research; the Spotify Web API serves as the data provider for individual listening intelligence. The two datasets represent different populations with **no row-level join**.

---

## 9. Limitations
- **Data Limits**: Subjective self-reporting (0-10 scales), missing key confounders (daily stress, career, clinical history), and convenience sampling bias.
- **Model Limits**: Shallow tree splits compress predictions near the overall mean, preventing the model from predicting extreme scores.

---

## 10. Future Work
- **Research Grounding (RAG)**: Index peer-reviewed music psychology literature to contextualize recommendations with published scientific evidence.
- **Explainability**: Integrate natural language AI explanations to interpret acoustic habit matches responsibly without generating medical advice.

