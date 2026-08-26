# Data Dictionary — mxmh_survey_results.csv

**Dataset**: Music × Mental Health (MXMH) Survey Results  
**Source**: Public survey dataset, collected via Reddit (r/SurveyExchange) and similar communities.  
**Rows**: 736 | **Columns**: 33

---

## Possible Role Definitions

| Role | Meaning |
|---|---|
| `feature` | Candidate input variable for a predictive model |
| `target` | Candidate outcome variable to predict |
| `demographic/control` | Demographic or background variable; may be used as a control |
| `metadata` | Administrative or consent variable — not an analytical variable |
| `exclude` | Should not be used in analysis |

---

## Column Reference

| Column | Meaning | Type | Category | Missing % | Possible Role |
|---|---|---|---|---|---|
| `Timestamp` | Date and time when the survey was submitted | str | Metadata | 0% | `exclude` — administrative only |
| `Age` | Participant age in years | float64 | Demographic | 0.14% | `demographic/control` |
| `Primary streaming service` | Main music streaming platform used (Spotify, YouTube Music, Apple Music, Pandora, Other, None) | str | Music behaviour | 0.14% | `feature` |
| `Hours per day` | Number of hours per day the participant listens to music | float64 | Music behaviour | 0% | `feature` |
| `While working` | Whether the participant listens to music while working (Yes/No) | str | Music behaviour | 0.41% | `feature` |
| `Instrumentalist` | Whether the participant plays a musical instrument (Yes/No) | str | Music behaviour | 0.54% | `feature` |
| `Composer` | Whether the participant composes music (Yes/No) | str | Music behaviour | 0.14% | `feature` |
| `Fav genre` | Self-reported favourite music genre (16 genres) | str | Music preference | 0% | `feature` |
| `Exploratory` | Whether the participant actively explores new music genres (Yes/No) | str | Music behaviour | 0% | `feature` |
| `Foreign languages` | Whether the participant listens to music in foreign languages (Yes/No) | str | Music behaviour | 0.54% | `feature` |
| `BPM` | Self-reported preferred music tempo in beats per minute | float64 | Music preference | **14.54%** | `feature` ⚠️ — requires outlier removal and imputation or exclusion |
| `Frequency [Classical]` | How often participant listens to Classical music (Never / Rarely / Sometimes / Very frequently) | str (ordinal) | Music behaviour | 0% | `feature` — encode as 0/1/2/3 |
| `Frequency [Country]` | How often participant listens to Country music | str (ordinal) | Music behaviour | 0% | `feature` — encode as 0/1/2/3 |
| `Frequency [EDM]` | How often participant listens to EDM | str (ordinal) | Music behaviour | 0% | `feature` — encode as 0/1/2/3 |
| `Frequency [Folk]` | How often participant listens to Folk music | str (ordinal) | Music behaviour | 0% | `feature` — encode as 0/1/2/3 |
| `Frequency [Gospel]` | How often participant listens to Gospel music | str (ordinal) | Music behaviour | 0% | `feature` — encode as 0/1/2/3 |
| `Frequency [Hip hop]` | How often participant listens to Hip hop | str (ordinal) | Music behaviour | 0% | `feature` — encode as 0/1/2/3 |
| `Frequency [Jazz]` | How often participant listens to Jazz | str (ordinal) | Music behaviour | 0% | `feature` — encode as 0/1/2/3 |
| `Frequency [K pop]` | How often participant listens to K-pop | str (ordinal) | Music behaviour | 0% | `feature` — encode as 0/1/2/3 |
| `Frequency [Latin]` | How often participant listens to Latin music | str (ordinal) | Music behaviour | 0% | `feature` — encode as 0/1/2/3 |
| `Frequency [Lofi]` | How often participant listens to Lo-fi music | str (ordinal) | Music behaviour | 0% | `feature` — encode as 0/1/2/3 |
| `Frequency [Metal]` | How often participant listens to Metal | str (ordinal) | Music behaviour | 0% | `feature` — encode as 0/1/2/3 |
| `Frequency [Pop]` | How often participant listens to Pop | str (ordinal) | Music behaviour | 0% | `feature` — encode as 0/1/2/3 |
| `Frequency [R&B]` | How often participant listens to R&B | str (ordinal) | Music behaviour | 0% | `feature` — encode as 0/1/2/3 |
| `Frequency [Rap]` | How often participant listens to Rap | str (ordinal) | Music behaviour | 0% | `feature` — encode as 0/1/2/3 |
| `Frequency [Rock]` | How often participant listens to Rock | str (ordinal) | Music behaviour | 0% | `feature` — encode as 0/1/2/3 |
| `Frequency [Video game music]` | How often participant listens to video game music | str (ordinal) | Music behaviour | 0% | `feature` — encode as 0/1/2/3 |
| `Anxiety` | Self-reported anxiety symptom severity (0 = none, 10 = extreme) | float64 | Self-reported mental health | 0% | `target` or `feature/control` depending on chosen target |
| `Depression` | Self-reported depression symptom severity (0 = none, 10 = extreme) | float64 | Self-reported mental health | 0% | `target` or `feature/control` depending on chosen target |
| `Insomnia` | Self-reported insomnia severity (0 = none, 10 = extreme) | float64 | Self-reported mental health | 0% | `target` or `feature/control` depending on chosen target |
| `OCD` | Self-reported OCD symptom severity (0 = none, 10 = extreme) | float64 | Self-reported mental health | 0% | `target` or `feature/control` depending on chosen target |
| `Music effects` | Participant's perceived effect of music on mental health (Improve / No effect / Worsen) | str | Music perception / mental health crossover | 1.09% | `target` (classification) or `feature` |
| `Permissions` | Consent acknowledgement — always "I understand." | str | Metadata | 0% | `exclude` — contains no variance |

---

## Ordinal Encoding Reference

All 16 `Frequency [genre]` columns share the same ordinal encoding:

| String value | Numeric encoding |
|---|---|
| `Never` | 0 |
| `Rarely` | 1 |
| `Sometimes` | 2 |
| `Very frequently` | 3 |

---

## Known Data Quality Issues

| Column | Issue | Status |
|---|---|---|
| `BPM` | 107 missing values (14.54%); extreme outliers (999,999,999; 624; values < 40) | ⚠️ Requires cleaning before use |
| `Permissions` | 100% identical value — zero variance | ✅ Exclude |
| `Timestamp` | Near-unique metadata string | ✅ Exclude |
| `Age` | 1 missing value; minimum age 10 | 🔎 Monitor |
| `Music effects` | 8 missing values | 🔎 Impute or drop depending on use |

---

## Data Dictionary — music_brain_wellbeing.csv

**Dataset**: Synthetic practice dataset (not real participants)  
**Rows**: 26 (25 unique) | **Columns**: 10

> ⚠️ This dataset is a synthetic teaching scaffold. It cannot be merged with `mxmh_survey_results.csv`.

| Column | Meaning | Type | Category | Missing % | Possible Role |
|---|---|---|---|---|---|
| `participant_id` | Unique participant identifier (P001–P025) | str | Metadata/identifier | 0% | `exclude` (identifier) |
| `age` | Age in years | int64 | Demographic | 0% | `demographic/control` |
| `gender` | Gender identity (Female / Male / Non-binary) | str | Demographic | 0% | `demographic/control` |
| `primary_genre` | Favourite music genre (5 genres) | str | Music preference | 0% | `feature` |
| `daily_listening_hours` | Hours of music listening per day | float64 | Music behaviour | 0% | `feature` |
| `skip_rate` | Proportion of songs skipped (0.0–1.0) | float64 | Music behaviour | 0% | `feature` |
| `eeg_alpha_power` | Simulated EEG alpha wave power (μV²) — indicates relaxed state | float64 | Physiological | 3.85% | `feature` |
| `heart_rate_bpm` | Simulated average heart rate during listening (bpm) | float64 | Physiological | 3.85% | `feature` |
| `anxiety_score` | Self-reported anxiety score (0–10) | float64 | Self-reported mental health | 0% | `target` |
| `wellbeing_score` | Self-reported wellbeing score (0–100) | float64 | Self-reported mental health | 0% | `target` — **Note: different scale from MXMH** |
