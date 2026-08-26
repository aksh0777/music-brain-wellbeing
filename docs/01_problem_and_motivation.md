# Chapter 01 — Problem and Motivation

## 1. Why are we learning this?
Before writing a single line of code or importing Pandas, a data scientist must precisely define the problem, understand business/scientific motivation, state formal research questions, and establish boundaries between prediction, association, and causation. 

In a Citi Data Science interview, candidates who immediately start talking about algorithms without explaining *why* the problem matters or *how* data was framed fail the problem-framing stage.

## 2. First-Principles Intuition
My earlier music project focused on **Audio Signal Classification**: analyzing raw audio waveforms or acoustic features (tempo, pitch, energy, spectral centroid) to classify a song's mood (e.g. Happy vs Sad).

That project proved that *music audio contains intrinsic emotional characteristics*.

However, audio alone tells only half the story. Music does not exist in a vacuum; it is consumed by **humans**. Two people listening to the exact same energetic song may experience completely different internal states based on their mental health, personality, listening environment, and physiological state.

This leads to our fundamental pivot:
* **Old Question**: What mood does this song express? (Audio-centric)
* **New Question**: How does a person's music listening behavior and physiological response relate to their psychological wellbeing? (Human-centric)

## 3. Core Concept
* **1. What are we building?**
  A data science pipeline that ingests tabular listening behavior metrics, self-reported psychological wellbeing indices, and physiological biosignals (EEG, ECG, GSR) to quantify associations, build predictive models of wellbeing, and scale signal features efficiently.
* **2. Research / Business Question**:
  *Primary Question*: To what extent can daily music listening behaviors (e.g., listening duration, playlist diversity, time of day) predict self-reported human wellbeing scores?
  *Secondary Question*: Do physiological biosignals (EEG spectral power, ECG heart rate variability, GSR skin response) during music listening provide statistically significant incremental predictive power beyond self-reported behavioral logs?
* **3. Why does this problem matter?**
  * *Human Impact*: Digital wellbeing, personalized mental health tools, and adaptive music recommendation systems.
  * *Data Science Relevance*: It presents classic enterprise data science challenges: heterogeneous data types (tabular + time series), missing values, noisy self-reported labels, high-dimensional feature spaces, and strict demands for scientific integrity (avoiding false causal claims).

## 4. How It Works Internally

### The Core Research Story
```text
[ Audio Mood Classifier ] ──(Taught us)──> Music audio carries emotional signals
                                                 │
                                                 ▼
[ Behavioral Study ]     ──(Asks)─────> Can listener behavior predict wellbeing?
                                                 │
                                                 ▼
[ Biosignal Study ]      ──(Extends)───> Do EEG/ECG/GSR physiological signals confirm response?
```

### Scientific Rigor Framework: Defining Valid vs Invalid Conclusions

| Concept | Definition | What Counts as a Meaningful Answer? | What Does NOT Count (Scientific Violation)? |
| :--- | :--- | :--- | :--- |
| **Association** | $X$ and $Y$ co-vary statistically ($\text{Corr}(X, Y) \neq 0$). | Finding that listening to acoustic music correlates with lower anxiety scores ($r = -0.34, p < 0.01$). | Claiming that acoustic music *reduces* anxiety. (Correlation $\neq$ Causation). |
| **Prediction** | Model predicts $Y$ given $X$ on unseen data ($\hat{Y} = f(X)$). | Out-of-sample $R^2 = 0.42$ or ROC-AUC $= 0.78$ using strict cross-validation. | Claiming high prediction performance using training set accuracy or leaked features. |
| **Causation** | Intervention on $X$ directly causes a change in $Y$ ($P(Y \mid \text{do}(X))$). | Demonstrating causal effect through a randomized controlled trial (RCT) or valid causal inference design. | Claiming music *causes* wellbeing changes from observational/survey data. |

## 5. Tiny Example
Imagine a dataset with 3 columns:
* $X_1$: Daily listening time (hours)
* $X_2$: Skip rate (%)
* $Y$: Psychological Wellbeing Score (1–100)

If we observe that higher daily listening time ($X_1$) co-occurs with higher wellbeing ($Y$), we can state:
* **Association**: "Daily listening time is positively correlated with wellbeing ($p < 0.05$)."
* **Prediction**: "A linear model trained on $X_1$ and $X_2$ predicts wellbeing with an RMSE of 8.2 points on test data."
* **Causation (INVALID)**: "Listening to music for 2 hours per day increases your wellbeing score by 5 points." (Invalid because happier people might simply have more free time to listen to music—a confounding variable).

## 6. Python Implementation
In our project architecture, expected inputs and outputs are defined explicitly:

* **Expected Inputs**:
  1. `behavioral_data.csv`: Tabular metrics (User ID, listening duration, genre preference, time of day, self-reported stress/wellbeing scores).
  2. `biosignals/`: Time-series signal recordings (EEG channel voltages in $\mu V$, ECG heart rate waveforms in $mV$, GSR conductance in $\mu S$).
* **Expected Outputs**:
  1. Statistical correlation matrix and hypothesis test results ($p$-values, effect sizes).
  2. Trained predictive machine learning models (`.pkl` / `.joblib`) evaluating out-of-sample performance.
  3. Feature importance rankings (interpreting which behavioral/biosignal metrics drive predictions).
  4. Scalable PySpark feature processing script for batch processing.

## 7. Connection to Our Project
This problem formulation governs every module we build in `src/`:
* `src/features/`: Generates behavioral aggregations and biosignal domain features.
* `src/models/`: Implements baseline and advanced predictors.
* `src/evaluation/`: Evaluates metrics strictly on out-of-sample test splits.

## 8. Why Did We Choose This Approach?
We explicitly choose an observational and predictive modeling approach paired with biosignal feature extraction because:
1. Observational behavioral logs reflect real-world human habits (ecological validity).
2. Biosignal data (EEG/ECG) provides objective physiological metrics to cross-validate subjective self-reported surveys.

## 9. Alternatives
* **Pure Audio Analysis**: Analyzing song MP3s only. (*Rejected*: Ignores individual human physiological and psychological differences).
* **Pure Survey Study**: Using only self-reported questionnaires. (*Rejected*: Subject to recall bias, subjective reporting bias, and social desirability bias).

## 10. Tradeoffs
* **Advantage**: Combining behavioral surveys with objective physiological biosignals yields a rich, multidimensional dataset that demonstrates both tabular and signal processing capability.
* **Disadvantage**: Biosignal data is noisy, requires specialized preprocessing (bandpass filtering, artifact removal), and cannot be naively combined across unrelated datasets without maintaining subject identity integrity.

## 11. Common Mistakes
* **Mistake**: Merging participants from Dataset A (e.g. behavioral survey) with Dataset B (e.g. separate EEG study) and pretending they are the exact same individuals.
* **Why it happens**: Desire to create a "complete" dataset without having overlapping subject keys.
* **Correction**: Maintain strict subject identity tracking (`participant_id`). Never fabricate linkages between disparate datasets.

## 12. Debugging Notes
* Always verify primary key integrity (`participant_id` / `user_id`) across tabular and signal files before attempting any merge operation.

## 13. Interview Questions

### Basic
* **Q**: What is the difference between correlation, prediction, and causation?
* **A**: Correlation measures linear association between two variables. Prediction estimates unknown target values using input features. Causation proves that manipulating one variable directly changes another, requiring experimental control or causal inference methods.

### Citi-Style Practical
* **Q**: How does this project map directly to the Citi Data Science job description?
* **A**:
  1. **Python / Pandas / NumPy**: End-to-end data cleaning, tabular aggregation, and vectorization.
  2. **Scikit-Learn**: Building and evaluating supervised ML baseline models.
  3. **Statistics**: Hypothesis testing, correlation analysis, and p-value interpretation.
  4. **PySpark**: Scaling data transformations on large-scale behavioral event logs.
  5. **Model Risk & Explainability**: Feature importance analysis, baseline model comparisons, and avoiding data leakage.

## 14. One-Minute Explanation
"My previous project focused on music mood classification from audio signals. I realized audio only tells half the story—human response depends heavily on the listener. In this project, I study how music listening behavior relates to psychological wellbeing, and extend it into neuroscience using EEG, ECG, and GSR biosignals. We maintain strict scientific integrity by distinguishing association, prediction, and causation while building an enterprise-ready pipeline covering statistical testing, ML modeling, and PySpark scalability tailored for financial data science roles."

## 15. Key Takeaways
1. Audio features describe the song; behavioral and biosignal data describe the listener.
2. Observational data allows prediction and association, but NOT causal claims without experimental design.
3. Heterogeneous datasets must preserve participant identity (`participant_id`).
4. Pipeline inputs: Tabular logs + Time-series biosignals. Outputs: Statistical tests + ML predictors + PySpark pipelines.
5. Every technical choice directly maps to key competencies in the Citi Data Science interview.

## 16. Status
COMPLETED
