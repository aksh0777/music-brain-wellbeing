# Interview Questions — 02: NumPy Foundations

### 1. Basic Questions
* **Q: Why is NumPy significantly faster than Python lists?**
  * **A**: NumPy arrays are stored in contiguous memory blocks with homogeneous data types (`dtype`). Operations are executed via pre-compiled C code without dynamic type checking or pointer dereferencing overhead.

### 2. Citi-Style Practical Questions
* **Q: What is broadcasting in NumPy and what are its rules?**
  * **A**: Broadcasting allows NumPy to perform arithmetic operations on arrays of different shapes. Trailing dimensions are compared from right to left; two dimensions are compatible if they are equal or one of them is 1.

### 3. Project Questions
* **Q: How will NumPy be used in the biosignal phase of our project?**
  * **A**: Multichannel EEG/ECG biosignals are represented as 2D arrays (`[channels, timepoints]`). NumPy allows vectorized matrix manipulations for baseline correction and signal filtering.
