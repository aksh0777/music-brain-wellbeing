# Interview Questions — 01: Python Foundations

### 1. Basic Questions
* **Q: How does memory management work in Python?**
  * **A**: Python uses reference counting as its primary memory management mechanism, supplemented by a generational garbage collector to detect and clean up circular references.
* **Q: What is the difference between `is` and `==`?**
  * **A**: `==` checks value equality (do the objects contain the same data?), whereas `is` checks reference identity (do both variables point to the exact same memory address?).

### 2. Citi-Style Practical Questions
* **Q: How does `sys.path` affect package imports in a production project?**
  * **A**: `sys.path` is a list of directory strings that Python searches sequentially when an `import` statement is executed. If a module is not in `sys.path` (or `site-packages`), Python raises `ModuleNotFoundError`.

### 3. Project Questions
* **Q: Why did we structure code into `.py` packages in `src/` instead of keeping everything in Jupyter Notebooks?**
  * **A**: Production pipelines require modularity, version control diff readability, automated unit testing (`pytest`), and execution reproducibility without hidden state bugs.
