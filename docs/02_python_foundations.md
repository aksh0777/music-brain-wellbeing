# Chapter 02 — Python Foundations for Data Science

## 1. Executive Summary & Context
Data science pipelines process raw signals, observational survey responses, and tabular metrics. Before leveraging domain-specific packages like Pandas or Scikit-Learn, data engineers and data scientists must thoroughly understand how the Python runtime executes code, manages memory, evaluates control flow, and guarantees exception safety.

In quantitative technical interviews (e.g. Citi Data Science), live coding tests your understanding of **first-principles execution mechanics**: pointer reference semantics, time/space complexity trade-offs ($O(1)$ vs $O(N)$), hash table mechanics, iterator protocols, and modular pipeline design.

---

## 2. First-Principles Mental Models

### A. The Memory Pointer Model (`PyObject` & Heap Allocation)
In Python, variables are **name tags (pointers)** referencing immutable or mutable `PyObject` structures residing in heap memory—they are **never** physical value containers.

* **Variables & Reference Binding**: Executing `x = 100` creates a `PyLongObject` in heap memory containing a reference count (`ob_refcnt`), object type pointer (`ob_type`), and value (`100`), then binds the string pointer `x` to its address.
* **Immutability**: Primitive integers, floats, strings, booleans, and tuples cannot be altered in place. Mutating operations re-evaluate and allocate a **new object** at a different memory address.
* **Mutability**: Lists, dictionaries, and sets are mutable. Modifying elements alters the underlying heap structure without changing the object reference address.

```text
Variable Pointer (Stack/Scope)                 Heap Memory Object
   +---+                                   +-------------------------+
   | x | --------------------------------> | PyLongObject            |
   +---+                                   |   ob_refcnt: 1          |
                                           |   ob_type:   &PyLong_Type|
                                           |   ob_ival:   100        |
                                           +-------------------------+
```

---

## 3. The 16 Core Foundation Mental Models

### 1. Variables and Objects
* **Mental Model**: Sticky name tags pointing to objects in heap space. Assignment (`b = a`) creates a second pointer to the exact same memory address rather than making a copy.
* **Pipeline Impact**: Prevents unintentional side-effects when passing raw dataset lists between processing functions.

### 2. Integers and Floats
* **Mental Model**: `int` objects handle arbitrary-precision discrete counting. `float` objects implement IEEE 754 double-precision (64-bit) representation.
* **Key Mechanics**: Floating-point binary representation cannot exactly represent certain decimal fractions (e.g. `0.1 + 0.2 == 0.30000000000000004`).
* **Data Science Impact**: Always use explicit tolerance checks (`abs(a - b) < 1e-9`) instead of exact `==` equality when checking floating-point signal amplitudes.

### 3. Strings
* **Mental Model**: Immutable sequences of Unicode code points locked in fixed memory blocks.
* **Complexity Insight**: Concatenating strings inside a loop using `+` re-allocates memory repeatedly ($O(N^2)$ time complexity). Joining a list of string tokens via `''.join(str_list)` pre-allocates memory once ($O(N)$ time complexity).

### 4. Boolean Values
* **Mental Model**: Primitive binary evaluation (`True` / `False`). Inherits from integer (`bool` is a subclass of `int` where `True == 1` and `False == 0`).
* **Truthy/Falsy Rules**: Empty containers (`[]`, `{}`, `set()`), numeric zeroes (`0`, `0.0`), `None`, and empty strings `""` evaluate to `False` in conditional contexts. Non-empty objects evaluate to `True`.

### 5. Lists
* **Mental Model**: A dynamic, contiguous array of memory pointers pointing to heterogeneous objects.
* **Operation Complexity**:
  * `list.append(item)`: $O(1)$ amortized time (allocates over-capacity headroom).
  * `list.insert(0, item)` / `list.pop(0)`: $O(N)$ time (requires shifting all $N$ pointers in memory).

### 6. Dictionaries
* **Mental Model**: Open-addressing Hash Table mapping hashable keys to object values.
* **Lookup Mechanics**: Executing `d[key]` computes `hash(key)` to map directly to an array bucket index, yielding $O(1)$ average lookup time.
* **Safety Pattern**: Direct access `d[key]` raises `KeyError` if key is missing; `d.get(key, default)` gracefully falls back without crashing.

### 7. Sets
* **Mental Model**: Unordered hash tables storing keys without values.
* **Set Operations**: Deduplicates duplicates automatically. Testing membership (`x in set`) takes $O(1)$ time versus $O(N)$ linear scanning in a `list`. Only hashable (immutable) objects can be set elements.

### 8. Tuples
* **Mental Model**: Immutable, fixed-length arrays of memory pointers.
* **Performance Benefit**: Tuples occupy a single contiguous block of memory without dynamic expansion capacity overhead, making them faster to allocate and evaluate than lists.

### 9. Indexing
* **Mental Model**: Positional offset arithmetic from the base memory address of a sequence (`address + index * pointer_size`).
* **Zero-Indexing**: Index `0` represents an offset of zero steps from the sequence head. Negative index `-k` offsets $k$ steps backward from the sequence length (`len - k`).

### 10. Slicing
* **Mental Model**: Extracting sub-sequence ranges via `sequence[start:stop:step]`.
* **Copy Behavior**: Slicing a list produces a **shallow copy** of the sliced pointer references from index `start` up to (excluding) `stop`.

### 11. if/elif/else
* **Mental Model**: Control flow decision branching based on boolean expression evaluation.
* **Short-Circuit Evaluation**: In `A and B`, if `A` is `False`, `B` is never evaluated. In `A or B`, if `A` is `True`, `B` is skipped.

### 12. for Loops
* **Mental Model**: The Python Iterator Protocol under the hood.
* **Under The Hood Mechanics**: Running `for item in container:` calls `iterator = iter(container)` and repeatedly evaluates `next(iterator)`. When elements run out, Python catches the raised `StopIteration` exception to exit cleanly.

### 13. while Loops
* **Mental Model**: Indefinite execution loops that continue as long as a state condition evaluates to `True`.
* **Use Case**: Real-time signal streaming, buffer drain loops, or numerical convergence algorithms where the iteration count is not known in advance.

### 14. Functions
* **Mental Model**: Modular execution frames created on the call stack with local variable scope.
* **Argument Passing**: Pass-by-object-reference. Mutating mutable arguments (like modifying a passed `list`) inside a function mutates the original object in the caller's scope.

### 15. Basic Exceptions
* **Mental Model**: Structured call-stack unwinding to handle runtime errors gracefully (`try / except`).
* **Best Practice**: Catch specific exception types (e.g. `ZeroDivisionError`, `KeyError`, `FileNotFoundError`) rather than using bare `except:`, preventing silent suppression of critical system bugs.

### 16. Imports and Modules
* **Mental Model**: Modular namespace isolation and package execution.
* **Execution Flow**: Executing `import module_name` searches paths listed in `sys.path`, parses the target Python source file into bytecode, executes it in a new module namespace, and caches the resulting module object in `sys.modules`.

---

## 4. Collection Complexity Summary

| Collection | Data Structure | Lookup Time | Insertion Time | Mutable? | Ordered? | Primary Use Case |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **List** | Dynamic Pointer Array | $O(N)$ search | $O(1)$ amortized end | Yes | Yes | Dynamic feature sequences, continuous signal values. |
| **Tuple** | Fixed Pointer Array | $O(N)$ search | N/A (Immutable) | No | Yes | Immutable sensor coordinates `(X, Y, Z)`, matrix shapes. |
| **Set** | Hash Table (Keys) | $O(1)$ avg | $O(1)$ avg | Yes | No | Deduplicating participant lists, $O(1)$ membership filtering. |
| **Dictionary** | Hash Table (Key-Value)| $O(1)$ avg | $O(1)$ avg | Yes | Yes (3.7+) | Feature metadata records, survey dictionaries, JSON outputs. |

---

## 5. Domain Connection (Music, Brain & Wellbeing)

* **Continuous Signal Buffers (`list`)**: Holding sliding time-series windows of EEG microvolt readings.
* **Spatial Channel Coordinates (`tuple`)**: Storing fixed 3D electrode positions `(x, y, z)` for spatial brain mapping.
* **Metadata Maps (`dict`)**: Storing participant responses (e.g., `{"participant_id": "P001", "listening_hours": 2.5}`).
* **Deduplication (`set`)**: Extracting unique genre tags across raw audio feature files.
* **Pipeline Resilience (`try / except`)**: Guarding metric calculation functions against divide-by-zero errors on empty audio windows.

---

## 6. Status
COMPLETED
