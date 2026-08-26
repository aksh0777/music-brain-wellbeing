# Personal Learning Log (Interview Q&A Format)

This log records real learning interactions, mental model corrections, and first-principles explanations derived from constructing the **Music, Brain & Wellbeing** project. It is structured specifically for **Citi Data Science interview preparation**.

---

### Entry 001: Python Execution & Virtual Environments
* **Date**: 2026-08-24
* **Question / Problem**: How does Python run scripts and resolve package dependencies across projects?
* **My Initial Assumption**: Running `python` in terminal always executes the same global Python installation, and `pip install` makes packages globally available to all scripts.
* **First-Principles Truth**: `python` executes a specific binary (`python.exe`) that searches for packages strictly within its active `sys.path` and environment directory. An environment (`.venv`) is simply an isolated directory containing its own binary and `site-packages/` folder.
* **Code Example**:
  ```python
  import sys
  print(f"Active Interpreter: {sys.executable}")
  print(f"Search Paths: {sys.path[:2]}")
  ```
* **Citi Interview Takeaway**: When interviewers ask about environment troubleshooting or pipeline reproducibility, explain that `sys.executable` and `sys.path` dictate where Python imports modules from. Isolating dependencies via `.venv` and pinning `requirements.txt` prevents environment drift in production pipelines.

---

### Entry 002: Scientific Scope (Association vs Prediction vs Causation)
* **Date**: 2026-08-24
* **Question / Problem**: Can we claim that music habit changes improve human psychological wellbeing?
* **My Initial Assumption**: High statistical correlation between music listening duration and wellbeing score implies that music listening directly improves wellbeing.
* **First-Principles Truth**: Observational data allows us to measure statistical co-variation (**Association**) and train out-of-sample models (**Prediction**), but **NOT** infer **Causation**. Confounding variables (e.g. people with more free time may listen to more music and have higher wellbeing) cannot be ruled out without randomized controlled experiments.
* **Code Example**:
  ```python
  # Quantifies association (linear correlation), NOT causation
  correlation = df["listening_hours"].corr(df["wellbeing_score"])
  ```
* **Citi Interview Takeaway**: Senior interviewers look for scientific rigor. Always distinguish between predictive models ($P(Y \mid X)$) and causal interventions ($P(Y \mid \text{do}(X))$). Never claim business or clinical causation from observational data.

---

### Entry 003: Dictionary Key Access (`KeyError` vs `.get()`) & Integer Immutability
* **Date**: 2026-08-25
* **Question / Problem**: What data type is `485` in `track = {"duration_seconds": 485}`, how does integer addition work in memory, and what happens when accessing missing keys?
* **My Initial Assumption**: Duration values default to strings; missing dictionary keys return `null`/`None` like in SQL or JavaScript.
* **First-Principles Truth**:
  1. Unquoted numbers like `485` are **Integers (`int`)**. Integers in Python are **immutable** `PyObject` heap structures. Adding `+ 15` allocates a **new integer object `500`** in memory and rebinds the dictionary pointer.
  2. Direct bracket access `track["missing_key"]` does **NOT** return `None`—it crashes with a **`KeyError`**. To access keys safely without crashing, use `.get(key, default)`.
* **Code Example**:
  ```python
  track = {"title": "Weightless", "duration_seconds": 485}

  # 1. Safe access on missing key (returns None instead of crashing)
  tempo = track.get("tempo_bpm", 0.0)

  # 2. Integer immutability (creates a NEW integer object in memory)
  print(id(track["duration_seconds"])) # Address A
  track["duration_seconds"] += 15
  print(id(track["duration_seconds"])) # Address B (different ID)
  ```
* **Citi Interview Takeaway**: In production data ingestion pipelines processing millions of rows, using direct bracket lookup `row["metric"]` will crash the job on incomplete rows. Always use `dict.get()` or schema-validated data structures to ensure pipeline resilience.

---

### Entry 004: List Slicing Syntax (`start:stop:step`) & Direct Iteration
* **Date**: 2026-08-25
* **Question / Problem**: What is the exact syntax for slicing sequences and how should we iterate over lists in Python?
* **My Initial Assumption**: Slicing syntax rules were unclear; looping required C-style index pointers (`for i in range(len(arr)): total += arr[i]`).
* **First-Principles Truth**:
  1. Slicing follows `sequence[start:stop:step]`. `start` is inclusive, `stop` is **exclusive**, and negative indices count backwards from the end (`-1` is the last item). Slicing `[-2:]` retrieves the last 2 elements.
  2. Python lists are direct iterators. `for item in list:` iterates directly over objects without needing index lookups (`arr[i]`).
* **Code Example**:
  ```python
  hours = [1.5, 2.0, 0.5, 3.0, 4.0, 2.5, 1.0]

  # 1. Negative slicing for last 2 elements
  weekend = hours[-2:]  # [2.5, 1.0]

  # 2. Direct Pythonic iteration
  total = 0
  for h in hours:
      total += h
  ```
* **Citi Interview Takeaway**: Direct iteration (`for item in list`) is cleaner, faster, and avoids `IndexError` risks compared to C-style index loops. Slicing with negative indices (`[-N:]`) is essential for windowing time-series features (e.g. taking the last $N$ seconds of a biosignal stream).

---

### Entry 005: Python Foundations Schema Standardization & Notebook Execution Verification
* **Date**: 2026-08-26
* **Question / Problem**: How do we systematically structure Python teaching materials, mental models, and exercise verification without runtime failure or scope creep?
* **What Actually Happened**:
  1. Standardized `notebooks/01_python_foundations.ipynb` across 16 core concepts using a unified 9-part template (`Concept`, `Why it exists`, `First-principles intuition`, `Tiny example`, `Code`, `Expected output`, `Music/Data Science connection`, `Common mistake`, `Interview question`).
  2. Integrated 5 unsolved practice exercises (`Lists`, `Dictionaries`, `Loops`, `Conditions`, `Functions`) at the end of the notebook.
  3. Expanded `docs/02_python_foundations.md` to articulate the underlying mental models (heap memory pointers, IEEE 754 float precision, Hash Table bucket mapping, Python iterator protocol, call stack unwinding, and `sys.path` namespace resolution).
  4. Executed and verified all 21 notebook code cells in the project `.venv` Python environment, confirming zero runtime or syntax errors.
* **Code Example**:
  ```python
  # Safe dictionary lookup pattern preventing KeyError crashes
  participant = {"id": "P005", "age": 24}
  score = participant.get("wellbeing_score", 50)  # Defaults safely to 50
  ```
* **Citi Interview Takeaway**: Senior data engineering interviews assess code structure and execution reliability. Writing modular, defensively typed Python functions guarded by safe accessor methods (`dict.get()`) and specific exception handling (`try / except`) ensures production data pipelines run continuously without crashing on incomplete rows.

---

### Entry 006: NumPy Array Foundations (31 Sequential Concepts), Zero-Copy Reshaping & Axis Reductions
* **Date**: 2026-08-26
* **Question / Problem**: How do we systematically structure NumPy foundational concepts from first principles for technical interview mastery?
* **What Actually Happened**:
  1. Created `notebooks/02_numpy_foundations.ipynb` implementing all 31 concepts in explicit sequential order (Why NumPy exists, Python lists vs NumPy arrays, ndarray, shape, ndim, size, dtype, creating arrays, np.array, np.zeros, np.ones, np.arange, np.linspace, indexing, slicing, 2D arrays, rows and columns, reshape, reshape(-1, 1), reshape(1, -1), element-wise operations, comparisons, sum, mean, min, max, std, axis=0, axis=1, vectorization, basic broadcasting) along with 8 unsolved practice exercises.
  2. Structured every concept with the exact 10 sub-element template: `Concept`, `Why it exists`, `First-principles intuition`, `Tiny example`, `Code`, `Expected output`, `What Python/NumPy is doing`, `Data Science connection`, `Common mistake`, `Interview question`.
  3. Updated `docs/04_numpy_foundations.md` explaining first-principles mental models (contiguous C RAM byte buffers, stride offset arithmetic, zero-copy reshaping, vertical vs horizontal axis reductions, and SIMD vectorization speedups).
  4. Verified all 39 code cells in `notebooks/02_numpy_foundations.ipynb` in the `.venv` Python environment, confirming zero syntax or execution errors.
* **Code Example**:
  ```python
  import numpy as np

  # Vectorized 2D matrix sum along axis=0 (collapses rows down vertically)
  matrix = np.array([[1, 2, 3],
                     [4, 5, 6]])
  col_sums = np.sum(matrix, axis=0)  # Output: array([5, 7, 9])
  ```
* **Citi Interview Takeaway**: In quantitative technical interviews, explain that NumPy achieves SIMD vectorization speedups by replacing Python's pointer-indirection loops with compiled C loops iterating over contiguous memory byte buffers. Zero-copy `reshape()` operations re-interpret metadata (`shape`, `strides`) in $O(1)$ time without copying physical RAM data.

---

### Entry 007: Data Understanding and Data Auditing — mxmh_survey_results.csv and music_brain_wellbeing.csv
* **Date**: 2026-08-26
* **Question / Problem**: How do we systematically understand the structure, quality, and semantics of a raw dataset before performing analysis or modelling?
* **What Actually Happened**:
  1. Audited both CSV files in `data/raw/` — confirmed `mxmh_survey_results.csv` (736 rows × 33 columns) and `music_brain_wellbeing.csv` (26 rows × 10 columns, including 1 duplicate row).
  2. Created `notebooks/02_data_understanding.ipynb` (54 cells) implementing all 12 phases: data location and verification, loading and inspection, column-by-column classification, data quality audit, target variable identification, music variable organisation, basic EDA, initial relationship analysis, dataset comparison, and conceptual grouping.
  3. Discovered the most critical quality issue: `BPM` column contains a value of `999,999,999` (clearly a data entry error) and `624` (physiologically implausible). Additionally, 107 values are missing (14.54% of rows). This column cannot be used without careful cleaning.
  4. Established a key scientific distinction: the four mental health variables (`Anxiety`, `Depression`, `Insomnia`, `OCD`) are **self-reported symptom severity scores on a 0–10 scale** — NOT clinical diagnoses. This distinction must appear in all analysis language.
  5. Confirmed that the two datasets CANNOT be merged — different populations, different scales (`wellbeing_score` is 0–100 in MBW vs 0–10 in MXMH), different variables (MXMH has no EEG/heart rate; MBW has no genre frequency columns), and `music_brain_wellbeing.csv` is synthetic practice data not representing real participants.
  6. Generated 10 EDA figures saved to `docs/figures/` — age distribution, hours per day distribution, mental health score distributions (4 variables), favourite genre bar chart, streaming service bar chart, mental health correlation heatmap, hours vs anxiety scatter, anxiety by genre bar chart, and music variables vs mental health correlation heatmap.
  7. Created `docs/03_data_understanding.md` (18 sections) explaining reasoning behind every data understanding decision.
  8. Created `docs/data_dictionary.md` — complete column reference for both datasets including meaning, type, category, missing percentage, and possible role (feature / target / demographic/control / metadata / exclude).
* **Discoveries Made During This Session**:
  - `BPM` column: mean inflated to 1,589,948 by the extreme outlier 999,999,999. Any statistics computed on raw BPM are meaningless until outliers are removed.
  - `Permissions` column: 100% identical "I understand." response — zero variance. Must be excluded.
  - All 16 `Frequency [genre]` columns share exactly 4 ordinal values (`Never`, `Rarely`, `Sometimes`, `Very frequently`) with no whitespace or case inconsistencies — this simplifies later encoding.
  - `Anxiety` and `Depression` are moderately positively correlated (~0.50 Pearson r). If we predict one as the target, including the other as a feature risks target leakage.
  - Simple linear correlations between music frequency variables and mental health scores are weak (|r| < 0.20). This is expected — linear correlation is a weak detector of complex relationships.
* **Code Example**:
  ```python
  # BPM outlier identification
  import pandas as pd
  df = pd.read_csv('data/raw/mxmh_survey_results.csv')

  # Step 1: Identify extreme outliers
  extreme_bpm = df[df['BPM'] > 250][['Age', 'Fav genre', 'BPM']]
  print(extreme_bpm)
  # Row 568: Age=16, BPM=999,999,999 — data entry error
  # Row 644: Age=16, BPM=624 — physiologically implausible

  # Step 2: Missing value percentage
  missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
  print(missing_pct[missing_pct > 0])
  # BPM: 14.54%  <-- most critical
  ```
* **Citi Interview Takeaway**: Data understanding is not just exploratory — it is the foundation for defensible modelling decisions. When asked "what would you do first with a new dataset?", the correct answer covers: shape/dtypes check, missing value audit, range/plausibility check on numerics, categorical consistency check, and explicit distinction between what a variable measures vs what we might want to infer from it. Never assume a column means what its name suggests without verification.

---

### Entry 008: Data Cleaning, Preprocessing and Exploratory Data Analysis (EDA)
* **Date**: 2026-08-26
* **Question / Problem**: How do we clean a noisy real-world dataset (missingness, outliers, metadata) and perform goal-oriented EDA without introducing target leakage or invalid causal claims?
* **What Actually Happened**:
  1. Created `notebooks/04_cleaning_and_eda.ipynb` to clean the raw `mxmh_survey_results.csv` dataset and perform EDA.
  2. Implemented copy-on-write integrity by starting with `df = raw_df.copy()` to isolate raw files from modification.
  3. Audited and imputed missing values:
     - Imputed `Age` with the median (`21.0`).
     - Replaced extreme, invalid BPM values (`999,999,999` and `624`) with `NaN`, and then imputed all `BPM` missing values using the valid median (`120.0`).
     - Imputed missing categorical values (`Primary streaming service`, `While working`, etc.) using the mode.
  4. Dropped uninformative columns: `Timestamp` (metadata) and `Permissions` (zero variance consent column).
  5. Performed ordinal encoding mapping: `Never`->0, `Rarely`->1, `Sometimes`->2, `Very frequently`->3 across all 16 `Frequency [genre]` columns.
  6. Documented `Anxiety` as the primary numeric target.
  7. Performed target-focused EDA saving 5 key figures to `docs/figures/` (Anxiety distribution, listening hours distribution, boxplots of anxiety vs perceived music effects, scatterplot of age vs anxiety with trendlines, and a numerical correlation heatmap).
  8. Wrote a comprehensive preprocessing report at `docs/data_cleaning.md`.
  9. Executed the entire notebook through jupyter kernel verification, ensuring zero runtime or cell errors.
* **Discoveries Made During This Session**:
  - **Copy vs Original**: Mutating a slice of a DataFrame without `.copy()` yields a `SettingWithCopyWarning`. Creating an explicit copy guarantees the underlying block-manager creates a new array object.
  - **Imputation Trade-offs**: Blindly imputing `BPM` with raw mean would lead to an average BPM of ~1.5 million due to the `999,999,999` outlier. Setting outliers to `NaN` before computing median prevents outlier propagation.
  - **Valid Extreme vs Invalid Data**: An age of 89 or a listening duration of 24 hours are valid extreme observations. A BPM of 999,999,999 is an invalid data value. We keep valid extremes but clean invalid data.
  - **Association vs Causation**: A slight negative slope in Age vs Anxiety scatter indicates younger participants in our survey report higher anxiety. This is an association (correlation), not a causal relationship. Age could confound the relationship between music and anxiety.
* **Code Example**:
  ```python
  # Copy-on-write and robust outlier cleaning + imputation
  import pandas as pd
  import numpy as np

  df = raw_df.copy()

  # Set physically invalid BPM values to NaN
  df.loc[(df["BPM"] < 40) | (df["BPM"] > 250), "BPM"] = np.nan

  # Impute using median of valid values
  median_bpm = df["BPM"].median()
  df["BPM"] = df["BPM"].fillna(median_bpm)
  ```
* **Citi Interview Takeaway**: In quantitative finance interviews, emphasize that data cleaning is not just about syntax. Walk through: 1) why you create deep copies to isolate inputs, 2) how you handle outliers defensively before computing imputation values to avoid statistic corruption, and 3) the difference between valid extreme values (which contain structural tail risk info) and invalid data entry errors (which must be filtered).

---

### Entry 009: Feature Engineering, Preprocessing and Leakage Prevention
* **Date**: 2026-08-26
* **Question / Problem**: How do we construct a robust, leakage-free preprocessing pipeline and prepare features for model training from first principles?
* **What Actually Happened**:
  1. Created `notebooks/05_feature_engineering.ipynb` to construct our preprocessing and validation pipeline.
  2. Defined `y` (target: `Anxiety`) and `X` (features: 27 input columns). Excluded `Depression`, `Insomnia`, and `OCD` to prevent target leakage.
  3. Separated features based on Pandas dtypes into 8 Categorical features and 19 Numerical features (including the 16 ordinal encoded columns).
  4. Created a train/test split (80% train, 20% test: 588 train rows, 148 test rows) with `random_state=42`.
  5. Built a `ColumnTransformer` with parallel pipelines:
     - Numerical: `SimpleImputer(strategy='median')` -> `StandardScaler()`
     - Categorical: `SimpleImputer(strategy='most_frequent')` -> `OneHotEncoder(handle_unknown='ignore')`
  6. Verified that fitting the preprocessor strictly on the training set (`X_train`) and transforming both sets completely prevents data leakage.
  7. Confirmed feature space expansion from 27 inputs to 54 outputs after one-hot encoding.
  8. Created a reusable preprocessor module under `src/features/preprocessing.py`.
  9. Authored the technical guide at `docs/feature_engineering.md`.
  10. Executed and verified the notebook using jupyter kernel preprocessing with zero errors.
* **Discoveries Made During This Session**:
  - **Categorical Feature Expansion**: The 8 categorical features expanded to 35 columns because of one-hot encoding (Spotify, genre categories, yes/no binary features). Combined with 19 numeric features, this results in 54 final features.
  - **Data Leakage Risk**: If scaling is applied before splitting, the test set's mean and standard deviation leak into training. Fitting the ColumnTransformer strictly on `X_train` ensures scaling is completely independent.
  - **Scaling Requirements**: Linear models with regularization require scaling because coefficients are penalized equally. Tree-based models are scale-invariant since they split on individual columns independently.
* **Code Example**:
  ```python
  # Reusable preprocessor construction
  from sklearn.compose import ColumnTransformer
  from sklearn.pipeline import Pipeline
  from sklearn.impute import SimpleImputer
  from sklearn.preprocessing import StandardScaler, OneHotEncoder

  num_pipeline = Pipeline(steps=[
      ("imputer", SimpleImputer(strategy="median")),
      ("scaler", StandardScaler())
  ])

  cat_pipeline = Pipeline(steps=[
      ("imputer", SimpleImputer(strategy="most_frequent")),
      ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
  ])

  preprocessor = ColumnTransformer(transformers=[
      ("num", num_pipeline, numerical_cols),
      ("cat", cat_pipeline, categorical_cols)
  ])
  ```
* **Citi Interview Takeaway**: When asked how you structure a model-ready pipeline, explain the role of `ColumnTransformer` and `Pipeline` in scikit-learn. Emphasize that splitting train/test must occur *before* any preprocessing to prevent data leakage, and explain why you strictly fit on the training set and transform the test set using those pre-fit parameters.

---

### Entry 010: Baseline Model Construction, Training, and Evaluation
* **Date**: 2026-08-26
* **Question / Problem**: How do we construct a baseline model, evaluate it against a naive predictor, and interpret the model coefficients without claiming causation?
* **What Actually Happened**:
  1. Created `notebooks/06_baseline_model.ipynb` to establish our reference benchmark and build a baseline Linear Regression pipeline.
  2. Defined the naive benchmark: predicting the training target mean (`5.8452` Anxiety score) for all observations.
  3. Built and trained a `LinearRegression` model using our scikit-learn `Pipeline` (reusing our custom preprocessing steps).
  4. Fitted the model strictly on `X_train` and `y_train` (588 samples), keeping `X_test` and `y_test` hidden.
  5. Generated test predictions and evaluated both models using MAE, RMSE, and $R^2$:
     - Naive Mean Baseline: MAE = 2.4193, RMSE = 2.8423, $R^2$ = -0.0004
     - Linear Regression: MAE = 2.4173, RMSE = 2.8786, $R^2$ = -0.0261
  6. Analyzed the regression coefficients, identifying `Fav genre_Latin` (-1.6693) and `Primary streaming service_Pandora` (+0.8950) as having the largest magnitudes.
  7. Written a detailed baseline report at `docs/baseline_model.md`.
  8. Executed the baseline notebook through validation verification with zero errors.
* **Discoveries Made During This Session**:
  - **Negative R² on Test Set**: The baseline Linear Regression model got a negative $R^2$ (-0.0261) on the test set. This means it performs slightly worse than always predicting the training mean. This is a common finding for unregularized models trained on high-dimensional categorical spaces (e.g. rare categories from one-hot encoding like Latin genre, which has only 3 occurrences total).
  - **Outlier/Overfitting Sensitivity**: Without regularization (L1/L2 penalties), standard ordinary least squares (OLS) can assign large coefficients to rare features to minimize training error, resulting in poor generalization.
  - **Coefficient Interpretation**: A large coefficient indicates statistical association used by the model for optimization, but **not** a causal link. Confounding variables are highly active in this observational dataset.
* **Code Example**:
  ```python
  # Evaluating regression pipeline against naive baseline
  from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
  import numpy as np

  # 1. Naive Baseline (mean)
  y_pred_naive = np.full(shape=y_test.shape, fill_value=y_train.mean())
  r2_naive = r2_score(y_test, y_pred_naive)

  # 2. Linear Regression Pipeline
  y_pred_lr = lr_pipeline.predict(X_test)
  r2_lr = r2_score(y_test, y_pred_lr)
  ```
* **Citi Interview Takeaway**: If asked about baseline evaluation, explain that you always compare models to a naive predictor (like target mean for regression). Be honest about negative $R^2$ test performance — explain that it indicates overfitting or low signal-to-noise ratio. Clarify that coefficients in observational studies show association, never causation, because of uncontrolled confounders.

---

### Entry 011: Decision Tree Regression and Overfitting Analysis
* **Date**: 2026-08-26
* **Question / Problem**: How does an untuned Decision Tree Regressor perform compared to Linear Regression, and how do we diagnose and explain overfitting from first principles?
* **What Actually Happened**:
  1. Created `notebooks/07_decision_tree_regression.ipynb` to train and evaluate an untuned Decision Tree Regressor on our preprocessing pipeline.
  2. Trained the Decision Tree strictly on `(X_train, y_train)` (588 samples) and evaluated on both train and test sets.
  3. Computed evaluation metrics (MAE, RMSE, $R^2$) and compared them to Linear Regression:
     - Linear Regression Test: MAE = 2.4173, RMSE = 2.8786, $R^2$ = -0.0261
     - Decision Tree Test: MAE = 3.3480, RMSE = 4.1258, $R^2$ = -1.1078
     - Decision Tree Train: MAE = 0.0000, RMSE = 0.0000, $R^2$ = 1.0000
  4. Extracted predictive feature importances, identifying `Age` (12.65%), `BPM` (9.33%), and `Hours per day` (6.70%) as the most important columns.
  5. Created two simple visualizations in `docs/figures/` (RMSE comparison and Top 10 feature importances).
  6. Created a comparison report at `docs/model_comparison.md`.
  7. Appended Entry 011 in `docs/learning_log.md`.
  8. Executed the decision tree notebook through validation verification with zero errors.
* **Discoveries Made During This Session**:
  - **Extreme Overfitting**: The untuned Decision Tree achieved a perfect $R^2$ of 1.0000 on the training set, but its test $R^2$ dropped to -1.1078. This is a severe example of overfitting caused by unconstrained tree depth, which allows the model to memorize individual training samples rather than learn general patterns.
  - **Tree Split Mechanics**: The tree relies heavily on continuous features (`Age`, `BPM`, `Hours per day`) for splitting, because they provide many unique values to split on. Without depth constraints, it creates deeply nested splits that fit noise.
  - **Predictive Importance**: Feature importance does not establish physical causation. It only indicates how much a feature's split thresholds reduced Mean Squared Error during training.
* **Code Example**:
  ```python
  # Initializing and training Decision Tree Regressor in pipeline
  from sklearn.tree import DecisionTreeRegressor
  from sklearn.pipeline import Pipeline

  dt_pipeline = Pipeline(steps=[
      ("preprocessor", preprocessor),
      ("regressor", DecisionTreeRegressor(random_state=42))
  ])
  dt_pipeline.fit(X_train, y_train)
  ```
* **Citi Interview Takeaway**: When asked about Decision Trees, explain that they can capture non-linear relationships and interactions without scaling features. However, default untuned trees are highly prone to overfitting because they grow until leaf nodes are pure. Highlight that a Train $R^2$ of 1.0 combined with a negative Test $R^2$ is the classic signature of overfitting, and explain that feature importance measures predictive association within the model, not physical causation.

---
