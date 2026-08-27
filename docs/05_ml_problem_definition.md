# ML Problem Definition

## 1. Research Question
To what extent can daily music listening behaviors (such as listening duration, preferred genres, and streaming context) predict a person's self-reported anxiety symptom severity?

---

## 2. What We Are Predicting
We are predicting the continuous numerical variable **`Anxiety`** (self-reported score ranging from `0` (no anxiety symptoms) to `10` (severe anxiety symptoms)). 

---

## 3. What We Are NOT Claiming
- **No Causal Claims**: We are **not** claiming that specific music habits *cause* or *cure* anxiety. This is an observational, cross-sectional survey dataset. Confounding variables (e.g. lifestyle, occupation, environment) are uncontrolled.
- **No Clinical Diagnoses**: A high score (e.g. 9/10) represents high self-reported symptom severity. It does **not** equal a clinical diagnosis of Generalized Anxiety Disorder (GAD) or any other medical condition.

---

## 4. Candidate Targets
The dataset contains five potential wellbeing/mental health variables:

### A. Anxiety
- **Type**: Continuous (`float64`, scale 0-10)
- **Meaning**: Self-reported anxiety symptom severity.
- **Number of valid observations**: 736
- **Range/categories**: 0.0 to 10.0
- **Potential leakage**: Strong correlation with concurrent symptom scores (`Depression`, `Insomnia`, `OCD`).
- **Suitability for ML**: High (good variance, fully populated).
- **Scientific limitations**: Subjective self-assessment; scale interpretation varies by individual.

### B. Depression
- **Type**: Continuous (`float64`, scale 0-10)
- **Meaning**: Self-reported depression symptom severity.
- **Number of valid observations**: 736
- **Range/categories**: 0.0 to 10.0
- **Potential leakage**: Strongly correlates with `Anxiety`, `Insomnia`, and `OCD`.
- **Suitability for ML**: High (good variance, fully populated).
- **Scientific limitations**: Subjective self-assessment.

### C. Insomnia
- **Type**: Continuous (`float64`, scale 0-10)
- **Meaning**: Self-reported sleep difficulties.
- **Number of valid observations**: 736
- **Range/categories**: 0.0 to 10.0
- **Potential leakage**: Correlates with anxiety and depression.
- **Suitability for ML**: High.
- **Scientific limitations**: Subjective self-assessment.

### D. OCD
- **Type**: Continuous (`float64`, scale 0-10)
- **Meaning**: Self-reported OCD symptom severity.
- **Number of valid observations**: 736
- **Range/categories**: 0.0 to 10.0
- **Potential leakage**: Correlates with anxiety and depression.
- **Suitability for ML**: Moderate (highly right-skewed; most values are close to 0, which makes modeling variance difficult).
- **Scientific limitations**: Subjective self-assessment.

### E. Music effects
- **Type**: Categorical (`str`, 3 categories: `Improve`, `No effect`, `Worsen`)
- **Meaning**: Perceived effect of music on mental health.
- **Number of valid observations**: 736 (8 missing values were mode-imputed to `Improve`).
- **Range/categories**: `Improve` (550), `No effect` (169), `Worsen` (17)
- **Potential leakage**: High (represents a post-outcome subjective belief that directly reflects the respondent's current mood).
- **Suitability for ML**: Moderate (highly imbalanced, with very few observations in the "Worsen" category).
- **Scientific limitations**: Represents a participant's self-belief rather than an objective measurement of change.

---

## 5. Selected Target
We select **`Anxiety`** as our primary target because:
1. It is a continuous numerical score that naturally fits regression modeling.
2. It exhibits a wide, balanced distribution across the entire 0-10 range (offering more variance to model than OCD).
3. It has no missing values, providing a clean ground truth vector of size 736.

---

## 6. Feature Groups

We organize our variables into distinct roles:

| Feature Group | Variable Names | Role / Type |
|---|---|---|
| **Demographics** | `Age` | Demographic control variable |
| **Music Behaviour** | `Hours per day`, `While working`, `Instrumentalist`, `Composer`, `Exploratory`, `Foreign languages`, `Primary streaming service` | Input features |
| **Music Preferences** | `Fav genre`, `BPM` | Input features |
| **Listening Context** | 16 ordinal `Frequency [genre]` variables | Input features |
| **Other Variables** | `Music effects` | Input feature (perception check) |
| **Primary Target** | `Anxiety` | Target outcome (y) |
| **Excluded (Leakage)** | `Depression`, `Insomnia`, `OCD` | Excluded to prevent target leakage |
| **Excluded (Metadata)**| `Timestamp`, `Permissions` | Excluded (irrelevant or zero-variance) |

---

## 7. Leakage Analysis

### Direct Leakage
Including `Anxiety` in the feature matrix $X$ would result in a model that trivially learns the identity function $f(x) = x$, which has zero predictive value for new subjects.

### Indirect Leakage
Including the other mental health variables (`Depression`, `Insomnia`, `OCD`) represents indirect target leakage. In a real-world scenario, we want to predict a person's anxiety based on their *music listening habits*. If our model relies on their depression score to predict their anxiety, we are simply mapping one concurrent symptom profile to another, rather than learning anything about music-symptom relationships.

### Post-Outcome Information
`Music effects` is a perceived subjective response to music. While it can be included as a behavioral check feature, it represents information reported *after* or *during* the experience of mental symptoms, making its temporal relationship to anxiety ambiguous.

---

## 8. Prediction Formulation
We formulate this task as a **Regression** problem. The target `Anxiety` is natively continuous on a 11-point bounded interval scale (0 to 10). Forcing it into a classification task (e.g. grouping into "High" vs "Low" anxiety) would throw away valuable threshold information and make the prediction boundary arbitrary.

---

## 9. Limitations
1. **Cross-Sectional**: Data was collected at a single point in time, meaning we cannot capture longitudinal patterns or prove temporal sequence.
2. **Subjective Scales**: Rating scale interpretations are highly individual (e.g., a "5" to one person might feel like an "8" to another).
3. **Imbalanced Representation**: Some favorite genres have very few observations (e.g. Latin has 3, Gospel has 6), which makes estimating genre-specific coefficients or splits noisy.

---

## 10. Next Modeling Step
Build a **`RandomForestRegressor`** using the pipeline. This allows us to capture complex non-linear feature splits and interaction effects, while using tree averaging to reduce the high variance and extreme overfitting observed in the single, untuned Decision Tree.
