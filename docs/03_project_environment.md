# Chapter 03 — Project Environment

## 1. Why are we learning this?
In enterprise financial data science (such as at Citi), model auditability and environment reproducibility are non-negotiable requirements. If a data scientist builds a predictive pipeline that runs locally on Python 3.12 but breaks on a server or co-worker's machine due to unpinned dependencies or global package collisions, the code cannot be deployed to production.

Understanding Python environments, `pip`, and isolation guarantees that your code runs identically everywhere.

## 2. First-Principles Intuition
Think of a Python virtual environment like a **customized toolbox** built for a single specific project:

* **Global Python**: The master hardware store downtown. It contains thousands of tools and versions.
* **Virtual Environment (`.venv`)**: Your private workbench in your garage. You fetch only the specific tools required for your current dish or project into this workspace.
* **`pip`**: The delivery courier who goes to the central store (PyPI) and places packages directly into your private workbench (`.venv/Lib/site-packages`).
* **`requirements.txt`**: The exact manifest or receipt detailing every tool and its exact version number.

## 3. Core Concept
* **Python Interpreter**: An executable binary (`python.exe` on Windows) that compiles high-level Python code (`.py`) into intermediate bytecode (`.pyc`) and executes it line-by-line via the Python Virtual Machine (PVM).
* **Virtual Environment (`.venv`)**: A self-contained directory tree containing a dedicated copy/symlink of `python.exe`, its own `Scripts/` directory, and its own isolated `Lib/site-packages/` folder.
* **`pip`**: Python's standard package manager that fetches wheel (`.whl`) or source distributions from PyPI and installs them inside the active environment's `site-packages`.

## 4. How It Works Internally

When you run `python -m venv .venv`:
1. Python creates a new directory `.venv`.
2. It writes a file `.venv/pyvenv.cfg` storing configuration parameters (such as `home = C:\Users\...\Python312` and `include-system-site-packages = false`).
3. It copies/links `python.exe` into `.venv/Scripts/`.
4. It sets up an empty `.venv/Lib/site-packages/` directory.

When activated (`.\.venv\Scripts\Activate.ps1`):
* The operating system's `PATH` environment variable is prepended with the absolute path to `.venv\Scripts`.
* When you type `python` or `pip`, Windows resolves the executable inside `.venv\Scripts` FIRST before checking any global Python installation.

```text
c:\Users\...\music-brain-wellbeing\
├── .venv/
│   ├── pyvenv.cfg                  <-- Points to base interpreter & settings
│   ├── Scripts/
│   │   ├── python.exe              <-- Isolated executable
│   │   ├── pip.exe                 <-- Isolated package installer
│   │   └── Activate.ps1            <-- PowerShell activation script
│   └── Lib/
│       └── site-packages/          <-- Where numpy, pandas, etc. live
```

## 5. Tiny Example
Inspecting how Python resolves its executable path:

```python
import sys

print(f"Active Interpreter: {sys.executable}")
print(f"Is inside virtualenv: {sys.prefix != sys.base_prefix}")
```

If inside `.venv`, `sys.prefix` points to `.venv` directory, while `sys.base_prefix` points to the system-level Python installation.

## 6. Python Implementation
Commands used to initialize, activate, and populate our environment:

```powershell
# 1. Create virtual environment
python -m venv .venv

# 2. Verify creation
Test-Path .venv\Scripts\python.exe

# 3. Install core foundational packages
.\.venv\Scripts\python.exe -m pip install numpy pandas matplotlib seaborn scikit-learn jupyter

# 4. Export exact pinned dependencies
.\.venv\Scripts\python.exe -m pip freeze > requirements.txt
```

## 7. Connection to Our Project
In **Music, Brain & Wellbeing**, we use `.venv` to isolate tabular analytics (`pandas`), machine learning (`scikit-learn`), and signal visualization (`matplotlib`/`seaborn`). Isolating these dependencies ensures that when we later introduce PySpark or neuroscience packages (`mne`, `scipy`), our base environment remains stable and reproducible.

## 8. Why Did We Choose This Approach?
* **Standard `venv` over heavy frameworks**: Standard `venv` is lightweight, built directly into Python 3, and natively supported across all platforms and CI/CD runners without external package manager overhead.

## 9. Alternatives
* **Conda (`conda create`)**: Useful for C/C++ native binary dependencies, but adds significant disk footprint and environment resolution complexity.
* **Poetry / Pipenv**: Advanced dependency resolvers and lockfile managers. Excellent for complex software packages, but standard `venv` + `requirements.txt` is simpler for first-principles transparency.

## 10. Tradeoffs
* **Advantage**: Zero external dependencies required to set up `.venv`; explicit, transparent control over `site-packages/`.
* **Disadvantage**: Requires manual discipline to run `pip freeze` and maintain `requirements.txt`.

## 11. Common Mistakes
* **Mistake**: Running `pip install <package>` without checking if `.venv` is activated.
* **Why it happens**: Terminal defaults to global Python if PATH is not updated.
* **Correction**: Always verify interpreter path using `sys.executable` or `where python` in PowerShell before installing packages.

## 12. Debugging Notes
* If PowerShell throws `Execution_Policies` error when activating `.venv`, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` or invoke `.venv\Scripts\python.exe` directly.

## 13. Interview Questions

### Basic
* **Q**: What is the purpose of `pyvenv.cfg` in a virtual environment?
* **A**: It stores metadata defining the root Python binary (`home`), environment version, and whether global system site-packages should be accessible.

### Citi-Style Practical
* **Q**: How do you prevent dependency drift across development, staging, and production environments?
* **A**: By using pinned version requirements files (`pip freeze > requirements.txt` or `pip-compile` lockfiles) and building automated containerized environments (Docker) referencing exact base Python versions.

## 14. One-Minute Explanation
"A Python virtual environment is an isolated directory containing its own Python executable and `site-packages` directory. It prevents dependency conflicts between projects by prepending its own binary paths to the system `PATH`. We use `.venv` combined with a pinned `requirements.txt` to guarantee complete auditability, reproducibility, and deployment safety across development and production environments."

## 15. Key Takeaways
1. Virtual environments isolate project dependencies from global system Python.
2. `sys.executable` reveals the exact binary running your code.
3. `pip` fetches wheel archives from PyPI and installs them directly into `.venv/Lib/site-packages`.
4. `requirements.txt` acts as the exact manifest for environment reproduction.
5. Always verify interpreter paths before installing third-party libraries.

## 16. Status
COMPLETED
