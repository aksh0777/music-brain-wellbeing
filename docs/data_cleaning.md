# Data Cleaning and Preprocessing Report

This document records the exact cleaning and preprocessing decisions made for the **Music, Brain & Wellbeing** project to ensure transparency, reproducibility, and scientific integrity.

---

## 1. Original Dataset Size
- **Source File**: `data/raw/mxmh_survey_results.csv`
- **Dimensions**: 736 rows, 33 columns

---

## 2. Missing-Value Findings
A systematic audit of missing values revealed the following:

- **`BPM`**: 107 missing values (14.54%). This is the largest and most critical gap.
- **`Music effects`**: 8 missing values (1.09%).
- **`Instrumentalist`**: 4 missing values (0.54%).
- **`Foreign languages`**: 4 missing values (0.54%).
- **`While working`**: 3 missing values (0.41%).
- **`Age`**: 1 missing value (0.14%).
- **`Primary streaming service`**: 1 missing value (0.14%).
- **`Composer`**: 1 missing value (0.14%).
- **Other columns**: 0 missing values.

---

## 3. Duplicate Findings
- **Duplicates in `mxmh_survey_results.csv`**: **0** duplicate rows. Every participant response is unique.
- **Duplicates in `music_brain_wellbeing.csv`**: **1** duplicate row (participant `P001` recorded twice). Note: We did not merge or modify this dataset since it represents a separate synthetic practice dataset.

---

## 4. Invalid-Value Findings
We audited numerical ranges and detected:
- **`BPM`**: 
  - Row 568 contained a BPM of `999,999,999` (clearly a typo/data-entry error).
  - Row 644 contained a BPM of `624` (physiologically implausible as a listening preference).
  - There were also several very low values (< 40) that were highly suspicious.
- **`Age`**: Min age is 10, max age is 89. All are within realistic human bounds.
- **`Hours per day`**: Max value is 24. This is a boundary case but physically possible (e.g. continuous background music or maximum-click entry). We did not alter it.
- **Mental health scores** (`Anxiety`, `Depression`, `Insomnia`, `OCD`): All values are within the correct bounded scale of `0` to `10`.

---

## 5. Cleaning Decisions
Our first-principles decisions were:
- **Preserve raw data**: We loaded the raw CSV and created a working copy in Pandas. The raw CSV was never modified.
- **Exclusion vs Imputation**: Since dropping rows with missing BPM would discard over 14.5% of the dataset (107 records), we decided to clean outliers and impute missing values using median imputation to preserve sample size and prevent selection bias.
- **Drop useless metadata columns**: Drop `Timestamp` (survey entry metadata) and `Permissions` (zero variance consent column).
- **Ordinal encoding**: Convert categorical frequency columns into numeric values.

---

## 6. Columns Excluded and Why
- **`Timestamp`**: Excluded because it represents submission metadata and does not carry predictive information for psychological state.
- **`Permissions`**: Excluded because it has zero variance (100% "I understand."). A feature with no variance provides no predictive value to machine learning models.

---

## 7. Columns Retained and Why
- **All 16 `Frequency [genre]` columns**: Retained because they capture the contextual diversity and intensity of listening habits across genres.
- **`BPM`**: Retained despite high missingness, as musical tempo is a key physical acoustic attribute that could relate to autonomic arousal (heart rate/insomnia). We handled missingness using robust median imputation.
- **`Age`, `Primary streaming service`, `While working`, `Instrumentalist`, `Composer`, `Fav genre`, `Exploratory`, `Foreign languages`**: Retained as essential demographic, contextual, and behavioral predictors.
- **`Anxiety` (Primary Target)**, `Depression`, `Insomnia`, `OCD`: Retained as self-reported mental health scores.

---

## 8. Imputation Strategy
We implemented simple, robust imputation:
- **Numerical variables**:
  - `Age`: 1 missing value filled with the column **median** (`21.0`).
  - `BPM`: Outliers were replaced with `NaN` first. The remaining valid values were used to compute the column **median** (`120.0`), which was then used to fill all missing values. This prevents the outlier `999,999,999` from corrupting the imputed value.
- **Categorical variables**:
  - Imputed missing values with the **mode** (most common category):
    - `Primary streaming service` -> `Spotify`
    - `While working` -> `Yes`
    - `Instrumentalist` -> `No`
    - `Composer` -> `No`
    - `Foreign languages` -> `Yes`
    - `Music effects` -> `Improve`

---

## 9. Categorical Cleaning and Ordinal Encoding
- **Categorical Consistency**: Checked for trailing whitespace and capitalization issues. None were found.
- **Ordinal Encoding**: The 16 `Frequency [genre]` columns were mapped to integers:
  - `Never` -> 0
  - `Rarely` -> 1
  - `Sometimes` -> 2
  - `Very frequently` -> 3
  This converts the ordinal text categories to numbers while preserving the natural rank order.

---

## 10. Outlier Decisions
- **Outlier vs Invalid value**:
  - A **valid extreme value** (e.g. listening to music 24 hours per day or being age 89) represents a real-world edge case. We do not modify or delete it, as it contains real signal.
  - An **invalid data value** (e.g. BPM = 999,999,999 or BPM = 624) represents a measurement or entry error that is physically impossible. We replace these with `NaN` and impute them.
- We set all BPM values > 250 and < 40 to `NaN` and imputed them with the median of valid BPM values.

---

## 11. Final Dataset Dimensions
- **File**: `data/processed/mxmh_cleaned.csv`
- **Dimensions**: **736 rows, 31 columns** (2 columns dropped: `Timestamp` and `Permissions`).
- **Remaining missing values**: **0** (fully clean and ready for modeling).
