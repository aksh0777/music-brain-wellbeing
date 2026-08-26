# Chapter 03 — Data Understanding

## 1. Why Data Understanding Comes Before Modeling

A recurring mistake in applied data science is treating data as a transparent input to an algorithm. The assumption is: "I have data, therefore I can model." This is wrong.

Every dataset carries hidden assumptions about how it was collected, who was measured, what the numbers mean, and where gaps exist. Jumping directly to modelling without understanding these assumptions leads to:

- **Overfitting on noise** — If a column contains data entry errors or meaningless entries, a model will memorise them.
- **Leaking the target** — Including variables that implicitly encode the answer to the prediction task produces models that cannot generalise.
- **Wrong framing** — If the target variable does not mean what you think it means, the entire modelling exercise answers the wrong question.
- **Hidden confounders** — Variables correlated with both the features and the outcome can generate false associations.

The CRISP-DM framework (Cross-Industry Standard Process for Data Mining) dedicates two of its six phases to data understanding and data preparation before any modelling begins. This notebook implements those phases.

---

## 2. First-Principles Mental Model

Raw data arrives as a structured file (CSV, database table, JSON). We can think of it as a rectangular grid:

```
        Column_0  Column_1  Column_2  ...  Column_k
Row_0   val        val        val            val
Row_1   val        val        NaN            val
...
Row_n   val        val        val            val
```

Every **row** represents one unit of observation (here: one survey participant).  
Every **column** represents one measured attribute (here: a survey question response).  
Every **cell** contains either a value or a missing indicator (`NaN`).

Before modelling, we answer:
- What does each row represent?
- What does each column measure?
- Are the values plausible?
- Are any values missing, extreme, duplicated, or inconsistent?
- What types of variables are present (numerical, categorical, ordinal, binary)?
- Which variables are candidates for prediction targets?
- Which variables carry information that could help predict the target?

---

## 3. Dataset Sources

### Primary Dataset: `mxmh_survey_results.csv`
- **Origin**: Public dataset collected via Reddit (r/SurveyExchange) and similar communities.
- **Collection method**: Self-administered online survey.
- **Sample size**: 736 respondents.
- **Variables**: 33 columns covering demographics, music listening behaviours, genre listening frequencies, and self-reported mental health symptom scores.
- **Known limitations**: Self-selected convenience sample; online survey respondents skew young; no clinical validation of mental health scores.

### Secondary Dataset: `music_brain_wellbeing.csv`
- **Origin**: Synthetic practice dataset created for this project. It simulates what physiological + survey data might look like.
- **Sample size**: 26 rows (25 unique participants; 1 accidental duplicate row).
- **Variables**: 10 columns — demographics, genre preference, listening hours, skip rate, EEG alpha power, heart rate, anxiety score, wellbeing score.
- **Important note**: This dataset was created as a teaching scaffold. It does not represent real participants. It CANNOT be merged with `mxmh_survey_results.csv`.

---

## 4. Dataset Structure

### mxmh_survey_results.csv

| Metric | Value |
|---|---|
| Rows | 736 |
| Columns | 33 |
| Numeric columns | 7 (Age, Hours per day, BPM, Anxiety, Depression, Insomnia, OCD) |
| Categorical/text columns | 26 |
| Total missing values | 129 |
| Duplicate rows | 0 |

### music_brain_wellbeing.csv

| Metric | Value |
|---|---|
| Rows | 26 (25 unique) |
| Columns | 10 |
| Numeric columns | 8 |
| Categorical/text columns | 2 |
| Total missing values | 2 |
| Duplicate rows | 1 (P001 appears twice) |

---

## 5. Understanding Columns

### mxmh_survey_results.csv — Complete Column Reference

| # | Column | Meaning | Category | Type | Missing % | Notes |
|---|---|---|---|---|---|---|
| 0 | `Timestamp` | Survey submission date/time | E — Metadata | str | 0% | Not a feature. Nearly unique (735 of 736). |
| 1 | `Age` | Age in years | B — Demographic | float64 | 0.14% | Range 10–89. Median 21. |
| 2 | `Primary streaming service` | Main music platform used | A — Music behaviour | str | 0.14% | 6 categories. Spotify dominates (62%). |
| 3 | `Hours per day` | Hours of daily music listening | A — Music behaviour | float64 | 0% | Range 0–24. Median 3. |
| 4 | `While working` | Listens while working (Yes/No) | A — Music behaviour | str | 0.41% | Binary. 79% Yes. |
| 5 | `Instrumentalist` | Plays an instrument (Yes/No) | A — Music behaviour | str | 0.54% | Binary. 32% Yes. |
| 6 | `Composer` | Composes music (Yes/No) | A — Music behaviour | str | 0.14% | Binary. 17% Yes. |
| 7 | `Fav genre` | Self-reported favourite genre | A — Music preference | str | 0% | 16 genres. Rock dominates (26%). |
| 8 | `Exploratory` | Actively explores new music (Yes/No) | A — Music behaviour | str | 0% | Binary. 71% Yes. |
| 9 | `Foreign languages` | Listens in foreign languages (Yes/No) | A — Music behaviour | str | 0.54% | Binary. 55% Yes. |
| 10 | `BPM` | Self-reported preferred tempo | A — Music preference | float64 | **14.54%** | ⚠️ Severe outliers: BPM=999,999,999 and BPM=624 found. 107 missing. |
| 11–26 | `Frequency [genre]` × 16 | How often each genre is listened to | A — Music behaviour | str | 0% | Ordinal: Never / Rarely / Sometimes / Very frequently |
| 27 | `Anxiety` | Self-reported anxiety symptom level | C — Self-reported mental health | float64 | 0% | Scale 0–10. NOT a clinical diagnosis. |
| 28 | `Depression` | Self-reported depression symptom level | C — Self-reported mental health | float64 | 0% | Scale 0–10. NOT a clinical diagnosis. |
| 29 | `Insomnia` | Self-reported insomnia level | C — Self-reported mental health | float64 | 0% | Scale 0–10. NOT a clinical diagnosis. |
| 30 | `OCD` | Self-reported OCD symptom level | C — Self-reported mental health | float64 | 0% | Scale 0–10. NOT a clinical diagnosis. |
| 31 | `Music effects` | Perceived effect of music on mental health | A/C crossover | str | 1.09% | 3 values: Improve / No effect / Worsen. |
| 32 | `Permissions` | Consent acknowledgement | E — Metadata | str | 0% | 100% identical. Exclude completely. |

---

## 6. Data Types

In Pandas, columns are assigned one of several storage types:

- **`float64`**: 64-bit floating-point number. Used for decimal values (Age, Hours per day, BPM, mental health scores).
- **`int64`**: 64-bit integer. Used for whole numbers with no missing values.
- **`str` / `object`**: Python string. Used for text, categorical labels, binary Yes/No fields.

**Important note on ordinal columns**: The 16 `Frequency [genre]` columns are stored as `str` even though they represent an ordered scale (`Never < Rarely < Sometimes < Very frequently`). Before modelling, these need to be encoded as numeric ordinal values (e.g. 0, 1, 2, 3).

---

## 7. Missing Data

### Definition
A missing value (`NaN` — Not a Number) occurs when a cell contains no observation. Pandas represents missing values as `NaN` for float columns and `None` for object columns.

### Missing Value Profile — mxmh_survey_results.csv

| Column | Missing Count | Missing % | Severity |
|---|---|---|---|
| `BPM` | 107 | 14.54% | ⚠️ High — requires explicit strategy |
| `Music effects` | 8 | 1.09% | Low |
| `Instrumentalist` | 4 | 0.54% | Very low |
| `Foreign languages` | 4 | 0.54% | Very low |
| `While working` | 3 | 0.41% | Very low |
| `Age` | 1 | 0.14% | Very low |
| `Primary streaming service` | 1 | 0.14% | Very low |
| `Composer` | 1 | 0.14% | Very low |

### Why Missing Data is Problematic

**Blindly deleting rows** reduces sample size and can introduce selection bias — if the rows with missing BPM are systematically different from rows with valid BPM, deletion distorts our sample.

**Blindly filling** missing values with the mean or median introduces artificial precision and underestimates variance. For BPM specifically, the column also contains extreme outliers, meaning even the mean is corrupted.

**Decision (deferred to cleaning stage)**: For `BPM`, we will first remove extreme outliers (BPM ≤ 0, BPM > 250), then evaluate whether to impute or exclude the column entirely. This decision belongs in the data cleaning notebook.

---

## 8. Duplicate Data

### mxmh_survey_results.csv
**0 duplicate rows found.** Every row is distinct.

### music_brain_wellbeing.csv
**1 duplicate row found.** Participant `P001` appears twice with identical values on all 10 columns. This is a data entry error — the same individual's record was recorded twice. The duplicate row should be dropped in the cleaning stage.

---

## 9. Invalid and Suspicious Values

### BPM — Most Critical Issue

Two rows in `mxmh_survey_results.csv` contain implausible BPM values:

| Row Index | Age | BPM | Assessment |
|---|---|---|---|
| 568 | 16 | 999,999,999 | Clearly a data entry error — not a real BPM |
| 644 | 16 | 624 | Physiologically implausible human listening preference |

Typical music BPM ranges from ~40 (very slow Adagio) to ~200 (very fast Presto). Values outside this range should be treated as invalid.

Additionally, **6 rows have BPM < 40** — also suspicious. These should be reviewed during cleaning.

### Age
- Minimum age is 10. This is plausible for an online survey but should be noted as a boundary case.
- Maximum age is 89. Plausible.
- No negative ages, no ages above 100.

### Hours per Day
- Range 0–24. Maximum value of 24 is technically possible but a boundary case (could be a maximum-response click error).
- Several respondents reported 20–24 hours — worth noting but not conclusively invalid.

### Mental Health Scores
- All four scores (`Anxiety`, `Depression`, `Insomnia`, `OCD`) fall within the stated 0–10 range.
- No out-of-range values detected.

### Categorical Consistency
- No leading/trailing whitespace found in any categorical column.
- No case inconsistencies (e.g. "spotify" vs "Spotify") found.
- All 16 `Frequency [genre]` columns contain exactly the same 4 ordinal values: `Never`, `Rarely`, `Sometimes`, `Very frequently`.

---

## 10. Music Variables

### Conceptual Organisation

The music-related variables in this dataset can be understood in a hierarchical structure from behaviour to preference to perceived response:

```
Music Behaviour
  Hours per day | While working | Instrumentalist | Composer | Exploratory | Foreign languages
         |
         v
Music Preference
  Fav genre | BPM
         |
         v
Listening Platform / Context
  Primary streaming service
         |
         v
Genre-Specific Listening Frequency
  Frequency [Classical] | Frequency [Rock] | ... (×16 genres)
         |
         v
Self-Reported Response to Music
  Music effects: Improve / No effect / Worsen
```

This is a conceptual organisation only — it does not imply a causal pathway. It is useful for grouping variables when building feature matrices.

---

## 11. Wellbeing Variables

The dataset contains **four self-reported mental health symptom variables**:

| Variable | Scale | Description | Missing |
|---|---|---|---|
| `Anxiety` | 0–10 | Respondent rates how anxious they feel | 0% |
| `Depression` | 0–10 | Respondent rates how depressed they feel | 0% |
| `Insomnia` | 0–10 | Respondent rates their sleep difficulties | 0% |
| `OCD` | 0–10 | Respondent rates their OCD symptom severity | 0% |

**Critical scientific distinction**: These are **self-reported symptom severity scores**, not clinical diagnoses. A person scoring 8/10 on Anxiety has self-reported high anxiety — they have not been diagnosed with an Anxiety Disorder by a clinician. This distinction must be maintained in all analysis language and any reports or presentations.

The `Music effects` column (`Improve` / `No effect` / `Worsen`) is a subjective perception variable — it represents the respondent's own belief about whether music affects their mental health. It is different from the symptom scores and could serve as either a feature or a target.

---

## 12. Candidate Targets

We do not fix the final prediction target in the data understanding stage. The following are candidates:

| Variable | Type | Role |
|---|---|---|
| `Anxiety` | Continuous 0–10 | Primary regression target |
| `Depression` | Continuous 0–10 | Primary regression target |
| `Insomnia` | Continuous 0–10 | Secondary regression target |
| `OCD` | Continuous 0–10 | Secondary regression target |
| `Music effects` | Categorical (3-class) | Classification target candidate |

If `Anxiety` is chosen as the primary target, then `Depression`, `Insomnia`, and `OCD` become candidate features — but including them risks **target leakage** if they all measure the same underlying wellbeing construct. This will be addressed during feature selection.

---

## 13. Exploratory Analysis — Key Observations

### Age Distribution
- Median age: 21 years.
- Distribution is right-skewed — predominantly young adults.
- The platform (Reddit) explains this skew. Results may not generalise to older populations.

### Listening Hours
- Median: 3 hours/day. Mean: ~3.6 hours/day.
- Distribution is right-skewed.
- A small number of respondents report 15–24 hours/day — boundary cases.

### Favourite Genre
- Rock (26%) and Pop (15%) are most common.
- Latin (0.4%) and Gospel (0.8%) are least represented.
- Genre imbalance matters for any genre-stratified analysis or genre-as-feature encoding.

### Mental Health Scores
- `Anxiety` and `Depression` are broadly distributed across 0–10.
- `OCD` is right-skewed — most respondents report low OCD scores.
- `Anxiety` and `Depression` are moderately correlated (~0.5 Pearson r), which is expected.

### Music Variables vs Mental Health
- Simple linear correlations between music frequency variables and mental health scores are small (|r| < 0.20).
- This does not mean relationships are absent — it means linear correlation alone is a weak detector.
- Non-linear relationships, interaction effects, or subgroup patterns may exist.

---

## 14. Important Observations

1. **BPM is the most problematic variable** — extreme outliers and 14.5% missingness make it unreliable without significant cleaning.
2. **The `Permissions` column contains no information** — 100% identical consent text. Exclude completely.
3. **All Frequency columns are ordinal strings** — they must be encoded before any numerical analysis.
4. **The two datasets cannot be merged** — different populations, different variable sets, different scales.
5. **The sample skews young** — median age 21 from a Reddit-based survey. Generalisability is limited.
6. **Music effects column straddles domains** — it relates both to music behaviour (listening practice) and to mental health (perceived outcome). Its role in modelling requires careful thought.

---

## 15. What We Cannot Conclude

- We cannot claim that music **causes** changes in mental health based on this dataset.
- We cannot diagnose any survey respondent with any mental health condition.
- We cannot generalise results beyond the sample (Reddit-recruited, English-speaking, predominantly young, self-selected).
- We cannot assume the BPM column reflects actual listening behavior (it is self-reported preference, not measured tempo).
- We cannot merge this dataset with `music_brain_wellbeing.csv` — they measure different populations with different instruments.

---

## 16. Dataset Limitations

| Limitation | Implication |
|---|---|
| Self-selected online convenience sample | Results may not generalise to general population |
| Young-skewed age distribution (median 21) | Limited insights into older adult listening behaviour |
| All mental health variables are self-reported | Cannot be treated as clinical measurements |
| `BPM` is self-reported preferred tempo, not measured | Unreliable; large missingness and outliers |
| Survey context is unknown | We do not know when (season, time of day) or why respondents took the survey |
| No longitudinal component | We cannot observe change over time — this is a cross-sectional snapshot |
| No control condition | Observational data only — causal claims are not possible |

---

## 17. Decisions for the Next Stage

The following decisions are deferred to the **data cleaning** stage (next notebook):

1. Handle `BPM` outliers (999,999,999; 624; values < 40). Strategy options: winsorisation, removal, or full column exclusion.
2. Drop the `Permissions` column — contains no analytical value.
3. Encode all 16 `Frequency [genre]` columns as ordinal integers (Never=0, Rarely=1, Sometimes=2, Very frequently=3).
4. Handle remaining missing values (Age, Primary streaming service, While working, etc.).
5. Drop the duplicate row in `music_brain_wellbeing.csv`.
6. Choose the primary target variable for modelling.

---

## 18. Interview Questions

**Q: What is the difference between self-reported mental health scores and clinical diagnoses?**  
A: Self-reported scores are based on a participant rating their own perceived symptoms on a numerical scale. Clinical diagnoses require evaluation by a qualified mental health professional using validated diagnostic criteria (e.g. DSM-5). Survey datasets like this one contain self-reported scores — not diagnoses.

**Q: Why should raw data remain unchanged?**  
A: Raw data is the reproducibility anchor of an analysis. If the raw file is modified, we lose the ability to trace any result back to its original source. All transformations should be applied in code so the entire pipeline from raw data to output is auditable and reproducible.

**Q: You find that `BPM` has 14.5% missing values and contains 999,999,999 as a value. What do you do?**  
A: First, document the issue clearly. Second, remove extreme outliers (values outside a plausible range of 40–220 BPM) before computing any statistics. Third, evaluate whether the remaining non-null BPM values are reliable enough to include as a feature. If the column cannot be made reliable, exclude it from modelling and document why.

**Q: Two datasets contain a column both named "Anxiety" — can you merge them?**  
A: Not automatically. First, verify that "Anxiety" measures the same construct using the same scale in both datasets. If one uses 0–10 and the other uses 0–100, the columns are not compatible. If the populations are different (e.g. one is clinical patients, one is Reddit users), merging may introduce systematic bias. Dataset compatibility must be verified before any join or concatenation.

**Q: What is the difference between `pd.concat()` and `pd.merge()`?**  
A: `pd.concat()` stacks DataFrames physically — it appends rows or columns without requiring matching keys. `pd.merge()` performs a relational join — it matches rows across two DataFrames based on shared key column values. Use `merge()` when rows in two tables correspond to the same entities. Use `concat()` when rows in two tables are independent observations that can be stacked together.
