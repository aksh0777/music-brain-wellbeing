# Citi Data Science JD & Skill Alignment Matrix

This document maps core requirements of a Citi Data Science role directly to technical evidence built into the **Music, Brain & Wellbeing** project.

| Citi JD Requirement | Project Evidence & Implementation | Learning Chapter | Interview Talking Point |
| :--- | :--- | :--- | :--- |
| **Python & Scientific Stack** | Modular package structure in `src/`, script execution, environment management. | Chapter 00, 02, 03 | "Built a modular Python project structure following software engineering best practices rather than relying solely on monolithic notebooks." |
| **NumPy & Vectorization** | Signal matrix processing, fast array operations for biosignal data. | Chapter 04, 13 | "Used vectorized array operations in NumPy for efficient computation of power spectral density across multi-channel EEG recordings." |
| **Pandas & Data Manipulation** | Data cleaning, indexing, handling missing values, aggregations on behavioral logs. | Chapter 05, 06, 07 | "Implemented leak-free data cleaning, imputation, and feature aggregation pipelines on tabular user survey logs." |
| **Statistics & Hypothesis Testing** | Pearson/Spearman correlation, null hypothesis testing, p-values, distribution checks. | Chapter 08 | "Performed rigorous statistical testing to distinguish true co-variation from random noise before feeding features into ML models." |
| **Machine Learning & Scikit-Learn** | Baseline Logistic Regression/Decision Trees, cross-validation, feature scaling. | Chapter 10, 11, 12 | "Trained baseline classification models using strict out-of-sample k-fold cross-validation to prevent data leakage." |
| **Model Evaluation & Risk Management** | Confusion matrix, ROC-AUC, precision-recall trade-offs, feature importance. | Chapter 12 | "Evaluated models using ROC-AUC and F1-score rather than raw accuracy to account for potential class imbalance in wellbeing labels." |
| **PySpark & Scalable Computing** | Distributed data processing pipeline for high-volume event logs. | Chapter 15 | "Implemented PySpark DataFrames for distributed feature aggregation to demonstrate scalability to enterprise log volumes." |
| **Model Serving & Production** | REST API deployment using FastAPI for real-time inference serving. | Chapter 16 | "Wrapped model artifacts in a FastAPI endpoint to serve real-time predictions with structured JSON request/response validation." |
| **Git & Software Engineering** | Version control, branch discipline, clear commit logging, unit testing with `pytest`. | Chapter 17, 18 | "Maintained strict Git commit hygiene and reproducible environments (`requirements.txt`) to ensure auditability." |
| **Communication & Storytelling** | End-to-end documentation from problem formulation to model trade-offs. | All Chapters | "Can explain every architectural decision, data transformation, and model limitation from first principles without black-box abstractions." |

---

### Non-Project Requirements (Interview Preparation Strategy)
* **NLP / LLMs / RAG**: While not artificially forced into this music & biosignal project, concepts (tokenization, embeddings, vector search) will be prepared separately under `docs/interview/`.
