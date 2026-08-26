# Interview Questions — 03: Pandas Foundations

### 1. Basic Questions
* **Q: What is the difference between `.loc` and `.iloc` in Pandas?**
  * **A**: `.loc` is label-based indexing (selecting by index/column name), whereas `.iloc` is integer-position based indexing (selecting by row/column zero-indexed position).

### 2. Citi-Style Practical Questions
* **Q: Why should you avoid iterating over Pandas DataFrame rows using a `for` loop or `iterrows()`?**
  * **A**: `iterrows()` creates a Series for each row, destroying vectorization and introducing massive overhead. Vectorized Pandas/NumPy operations or `.apply()` (when vectorization is impossible) are order-of-magnitude faster.

### 3. Project Questions
* **Q: How do we prevent data leakage when filling missing values in Pandas?**
  * **A**: Imputation statistics (e.g. median/mean) must be computed ONLY on the training split, then applied to both train and test splits. Never compute global column medians before splitting.
