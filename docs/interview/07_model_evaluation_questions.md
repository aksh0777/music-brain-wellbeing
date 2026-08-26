# Interview Questions — 07: Model Evaluation

### 1. Basic Questions
* **Q: Why is accuracy a misleading metric for imbalanced classification problems?**
  * **A**: In a dataset where 99% of samples belong to Class 0, a trivial model that always predicts Class 0 achieves 99% accuracy while failing completely to detect Class 1.

### 2. Citi-Style Practical Questions
* **Q: What is the difference between Precision and Recall, and how do you choose between them?**
  * **A**: Precision is $\frac{TP}{TP + FP}$ (quality of positive predictions). Recall is $\frac{TP}{TP + FN}$ (coverage of actual positives). Optimize Precision when false positives are costly (e.g. flagging legit trades as fraud); optimize Recall when false negatives are dangerous (e.g. missing disease or credit default).
