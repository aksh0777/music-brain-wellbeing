# Interview Questions — 04: SQL & Database Foundations

### 1. Basic Questions
* **Q: What is the difference between `WHERE` and `HAVING` in SQL?**
  * **A**: `WHERE` filters rows before any aggregation (`GROUP BY`), whereas `HAVING` filters grouped summary rows after aggregation.

### 2. Citi-Style Practical Questions
* **Q: Explain window functions (`ROW_NUMBER()`, `RANK()`, `DENSE_RANK()`) and when to use them.**
  * **A**: Window functions perform calculations across a set of table rows related to the current row without collapsing rows into a single output row. `ROW_NUMBER()` assigns unique sequential integers; `RANK()` leaves gaps on ties; `DENSE_RANK()` does not leave gaps.
