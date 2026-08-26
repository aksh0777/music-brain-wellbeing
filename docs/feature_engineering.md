# Feature Engineering and Preprocessing Foundation

This document describes the design, configuration, and execution of the preprocessing pipeline and feature engineering system constructed for the **Music, Brain & Wellbeing** project.

---

## 1. What is a Feature?
A **feature** is an individual measurable property or characteristic of an observation used as an input to a machine learning model. In our project, features include participant age, daily music listening hours, preferred genre, streaming service used, and frequency of listening to specific genres.

## 2. What is a Target?
A **target** (or label) is the outcome variable we want the model to learn to predict. For our primary regression model, the target is **`Anxiety`** (a self-reported score from 0 to 10).

## 3. What are X and y?
- **`X`**: The feature matrix (a 2D grid of size `[n_samples, n_features]`) containing all input variables for all observations.
- **`y`**: The target vector (a 1D array of size `[n_samples]`) containing the outcome values for all observations.
We define `y` as `df["Anxiety"]` and `X` as the remaining 27 columns (excluding `Anxiety`, `Depression`, `Insomnia`, and `OCD` to prevent target leakage).

## 4. Why Split the Data?
We split the data to simulate real-world prediction scenarios. If we train and evaluate our model on the same data, the model can overfit by memorizing the training set, leading to high training metrics but poor performance on new data.
- **Training set** (80% / 588 rows): Used to learn the weights/rules of the model and fit preprocessing parameters.
- **Test set** (20% / 148 rows): Kept completely hidden during training, used strictly for final generalization evaluation.

## 5. What is Numerical Preprocessing?
Numerical preprocessing transforms numerical variables to make them suitable for model fitting.
- **Imputation**: Missing values are filled using a statistic (like the median) to prevent models from failing.
- **Standardization/Scaling**: Translates values so they share a common scale.

## 6. What is Categorical Preprocessing?
Categorical preprocessing converts text-based discrete labels into numeric formats.
- **Imputation**: Fills missing categories with the mode (most frequent class).
- **Encoding**: Maps textual classes (e.g. "Spotify", "Apple Music") into numeric matrices.

## 7. What is One-Hot Encoding?
One-hot encoding creates a new binary column for each unique category within a nominal feature. For example, the `Primary streaming service` column is expanded into:
- `Primary streaming service_Spotify` (1 or 0)
- `Primary streaming service_Apple Music` (1 or 0)
- `Primary streaming service_YouTube Music` (1 or 0)
- (and so on)
This represents categorical labels numerically without implying an arbitrary mathematical order (like mapping them to 1, 2, 3, which would imply Spotify < Apple Music < Pandora).

## 8. What is Scaling?
Scaling standardizes numerical features so they share a comparable range. We use `StandardScaler`, which transforms each value as:
$$z = \frac{x - \mu}{\sigma}$$
Where $\mu$ is the feature mean and $\sigma$ is the standard deviation, setting the mean to 0 and variance to 1. This prevents features with large scales (like BPM or Age) from dominating features with small scales (like listening hours) in distance-based and regularized models.
*Note: Tree-based models are scale-invariant and do not require scaling.*

## 9. What is ColumnTransformer?
A `ColumnTransformer` allows different preprocessing pipelines to be applied to different subsets of columns in parallel. We configure:
- A numerical pipeline (`SimpleImputer` -> `StandardScaler`) applied to numerical columns.
- A categorical pipeline (`SimpleImputer` -> `OneHotEncoder`) applied to categorical columns.

## 10. What is a Pipeline?
An sklearn `Pipeline` chains multiple transformer steps and an estimator into a single object. It packages the `ColumnTransformer` (preprocessing) and the model placeholder together so they are executed in sequence.

## 11. What is Data Leakage?
Data leakage occurs when information from outside the training dataset (such as the target variable or future test data statistics) is accidentally introduced during model training.
For example, calculating the mean and standard deviation for scaling across the *entire* dataset before splitting is a form of leakage because the test set's distribution parameters influence training scaling. This makes test evaluations look artificially good.

## 12. Why Our Pipeline Avoids Leakage
Our pipeline avoids leakage because:
1. We split the data into `X_train` and `X_test` *before* any preprocessing.
2. We strictly call `.fit()` or `.fit_transform()` on `X_train`. This learns scaling parameters ($\mu$, $\sigma$) and modes *only* from the training set.
3. We call `.transform()` on `X_test` using the pre-fit parameters from the training set. The test set's statistics are never used.

## 13. Final Feature Groups
- **Demographics**: `Age` (Numerical)
- **Music Behaviour**: `Hours per day`, `While working`, `Instrumentalist`, `Composer`, `Exploratory`, `Foreign languages` (Categorical/Binary), `Primary streaming service` (Categorical)
- **Music Preferences**: `Fav genre` (Categorical), `BPM` (Numerical)
- **Listening Context**: 16 ordinal `Frequency [genre]` variables (Numerical)
- **Other relevant variables**: `Music effects` (Categorical)
- **Targets**: `Anxiety` (Primary Regression Target). `Depression`, `Insomnia`, and `OCD` are excluded from features to prevent target leakage.
