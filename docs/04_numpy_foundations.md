# Chapter 04 — NumPy Foundations for Data Science

## 1. Executive Summary & Context
In Data Science and Machine Learning pipelines, raw numerical operations (matrix products, signal filtering, batch transformations) must execute over millions of observations per second. Standard Python collections (`list`, `dict`) are designed for dynamic flexibility, relying on heap-allocated object pointers and runtime type checking. This introduces massive memory overhead and slows execution by orders of magnitude.

NumPy addresses this by providing a C-implemented object structure: the **`ndarray`** (N-Dimensional Array). NumPy stores homogeneous numbers in contiguous RAM blocks and executes numerical operations using compiled C loops and SIMD (Single Instruction, Multiple Data) CPU hardware instructions.

Understanding NumPy from first principles—memory layout, strides, zero-copy reshaping, axis reductions, vectorization, and broadcasting—is essential before building data pipelines in Pandas or training models in Scikit-Learn.

---

## 2. First-Principles Mental Models

### A. Memory Layout: Python Lists vs NumPy Arrays
* **Python List**: An array of pointers pointing to scattered `PyObject` wrappers across heap memory. Each integer `PyObject` carries a ~28-byte header (`ob_refcnt`, `ob_type`). Accessing elements requires pointer dereferencing and runtime type checking, triggering frequent CPU cache misses.
* **NumPy `ndarray`**: A single contiguous block of raw numerical bytes (e.g. 4 bytes for `float32`, 8 bytes for `float64`) paired with a lightweight metadata header. All elements share the exact same primitive type (`dtype`).

```text
Python List Memory Layout (Scattered Pointers):
   +---+---+---+---+
   | * | * | * | * |  (Pointers in RAM)
   +---+---+---+---+
     |   |   |   |
     v   v   v   v
   [PyObject: 10] [PyObject: 20] [PyObject: 30]  (Scattered in Heap)

NumPy ndarray Memory Layout (Contiguous Bytes):
   +---------------+
   | Metadata Header| (shape: (3,), dtype: int64, strides: (8,))
   +---------------+
   | 10 | 20 | 30  | (Contiguous 24 raw bytes in C RAM)
   +---------------+
```

### B. Strides & Indexing Arithmetic
A NumPy array's `shape` describes its multidimensional dimensions, but RAM is fundamentally a 1D sequence of byte addresses. 

To bridge this, NumPy uses **`strides`**: a tuple indicating the number of bytes to step in memory to advance by one element along each axis.
For a 2D `float64` array (8 bytes per element) of shape `(3, 4)`:
* `strides = (32, 8)` $\to$ To move down 1 row (`axis=0`), skip 32 bytes ($4 \times 8$). To move right 1 column (`axis=1`), skip 8 bytes.
* Memory offset formula: $\text{Byte Offset} = \text{row} \times 32 + \text{col} \times 8$.

### C. Zero-Copy Reshaping
Because multidimensional indexing relies strictly on metadata (`shape` and `strides`), invoking `arr.reshape(rows, cols)` does **NOT** duplicate or move byte data in RAM. It creates a new lightweight `ndarray` view header pointing to the exact same underlying byte buffer (**$O(1)$ zero-copy operation**).

---

## 3. Core Array Fundamentals

| Attribute | Mental Model | First-Principles Meaning | Data Science Relevance |
| :--- | :--- | :--- | :--- |
| **`ndarray`** | Contiguous Memory Buffer | C-level array of uniform raw byte elements | Core array container for all numerical data. |
| **`shape`** | Spatial Grid Dimensions | Tuple `(d0, d1, ...)` defining element count per axis | Describing feature matrix rows/cols `(n_samples, n_features)`. |
| **`ndim`** | Dimensionality Rank | Length of the shape tuple (`len(shape)`) | Distinguishing 1D signal vectors from 2D matrices. |
| **`size`** | Total Item Volume | Total element count ($\prod \text{shape}_i$) | Quantifying total sample points across recording windows. |
| **`dtype`** | Binary Element Material | Fixed C-type specification (`float32`, `int64`) | Optimizing RAM footprint for large sensor streams. |
| **`axis`** | Directional Dimension | Coordinate direction index for operations | Specifying vertical (`axis=0`) vs horizontal (`axis=1`) reductions. |

---

## 4. Axis Reduction Mechanics

Reducing a 2D array along an axis collapses that specific dimension by performing summary operations (e.g. `sum()`, `mean()`, `std()`):

Given matrix $M$ of shape `(2, 3)`:
$$M = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}$$

* **`axis=0` (Collapse Rows Vertically)**: Operation runs down vertical columns, collapsing 2 rows into 1 row of column statistics.
  $$\text{np.sum}(M, \text{axis}=0) = [1+4, \; 2+5, \; 3+6] = [5, \; 7, \; 9] \quad (\text{shape: } (3,))$$
* **`axis=1` (Collapse Columns Horizontally)**: Operation runs across horizontal rows, collapsing 3 columns into 1 column of row statistics.
  $$\text{np.sum}(M, \text{axis}=1) = [1+2+3, \; 4+5+6] = [6, \; 15] \quad (\text{shape: } (2,))$$

---

## 5. Vectorization & Broadcasting

### A. Vectorization
Vectorization eliminates explicit Python bytecode loops (`for x in list:`) in favor of C-level SIMD operations over contiguous memory blocks. 
* **Python Loop**: Interprets bytecode, checks types, and dereferences pointers $N$ times $\to$ High overhead.
* **Vectorized NumPy**: Passes contiguous byte pointer directly to compiled C loop $\to$ Executes in CPU registers at near-hardware limits ($10\times - 100\times$ faster).

### B. Broadcasting
Broadcasting allows element-wise operations between arrays of different shapes without copying data:
1. **Scalar Expansion**: A scalar ($0D$) automatically broadcasts across every element of a 1D/2D array.
2. **Dimension Alignment**: When operating on a 2D array of shape `(M, N)` and a 1D array of shape `(N,)`, NumPy aligns trailing dimensions and stretches the 1D array across all $M$ rows virtually without replicating memory.

---

## 6. Domain Application (Music, Brain & Wellbeing)

* **Biosignal Feature Matrices (`ndarray`)**: Storing continuous EEG microvolt channels across time `(n_channels, n_timestamps)`.
* **Channel Normalization (`broadcasting`)**: Subtracting baseline mean microvolts from multichannel signals ($X - \mu_{\text{channel}}$).
* **Feature Aggregation (`axis=0` / `axis=1`)**: Computing mean power spectral density per electrode channel across recording windows.
* **Epoch Reshaping (`reshape`)**: Structuring long time-series vectors into fixed 2-second signal epochs `(n_epochs, n_samples_per_epoch)`.

---

## 7. Citi Data Science Interview Takeaways

1. **Memory Overhead Comparison**:
   * A Python list of 1,000,000 integers consumes ~8 MB for pointer arrays plus ~28 MB for individual `PyObject` wrappers (~36 MB total).
   * A NumPy `int64` array of 1,000,000 elements occupies exactly $1,000,000 \times 8 \text{ bytes} = 8 \text{ MB}$ contiguous RAM.
2. **Zero-Copy Reshaping**:
   * `reshape()` returns a **view** of the same memory buffer (instant $O(1)$ time). Modifying values in the reshaped array modifies the original array.
3. **Axis Direction Rule**:
   * In a 2D matrix of shape `(rows, cols)`, `axis=0` operates *across rows* (column-wise result), while `axis=1` operates *across columns* (row-wise result).

---

## 8. Status
COMPLETED
