# Data Directory

This directory stores datasets used throughout the project lifecycle.

## Directory Structure
- `raw/`: Original, immutable data dumps. Files in this directory should never be edited directly.
- `processed/`: Cleaned, transformed, and feature-engineered datasets ready for modeling and analysis.

## Usage Guidelines
- Raw data files are kept separate from transformed data to preserve reproducible pipelines.
- Large datasets are excluded from Git via `.gitignore` to prevent tracking heavy binary files in version control.
