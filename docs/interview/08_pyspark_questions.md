# Interview Questions — 08: PySpark & Distributed Computing

### 1. Basic Questions
* **Q: What is lazy evaluation in PySpark?**
  * **A**: PySpark does not execute transformations (like `.select()`, `.filter()`) immediately. Instead, it builds a Directed Acyclic Graph (DAG) of logical operations, which is only evaluated when an action (like `.count()`, `.collect()`, `.write()`) is invoked.

### 2. Citi-Style Practical Questions
* **Q: What is a shuffle operation in PySpark and why is it expensive?**
  * **A**: A shuffle redistributes data across cluster partitions (e.g. during `groupBy` or `join`). It is expensive because it requires serialization, disk I/O, and network transfer across worker nodes.
