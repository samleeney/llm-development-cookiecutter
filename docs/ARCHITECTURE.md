# SYSTEM ARCHITECTURE

<!-- PERMANENT INSTRUCTIONS - DO NOT REMOVE THIS SECTION -->
## How to Use This Document

This document describes the system architecture of your project. Update it as you develop to maintain a clear overview of:
- Project structure and organization
- Core modules and their responsibilities
- Data flow and processing pipelines
- Key dependencies and interfaces

Keep this document in sync with your actual implementation. It serves as the technical blueprint for understanding how components interact.

---

<!-- EXAMPLE CONTENT - REPLACE EVERYTHING BELOW WITH YOUR PROJECT SPECIFICS -->

## Project Structure
```
project_root/
├── src/           # Source code
├── tests/         # Test files
├── data/          # Data files
├── docs/          # Documentation
├── logs/          # Log files (all logs saved here)
├── notebooks/     # Jupyter notebooks
├── scripts/       # Utility scripts
└── results/       # Output files (format: YYYY-MM-DD_HH-MM-SS_description.ext)
```

## Core Modules

### Module: data_loader
- **Purpose**: Loads and preprocesses input data
- **Location**: `src/data_loader.py`
- **Dependencies**: pandas, numpy

### Module: processor
- **Purpose**: Main processing logic
- **Location**: `src/processor.py`
- **Dependencies**: data_loader, utils

## Data Flow
```
Raw Data → [Data Loader] → [Preprocessor] → [Main Processor] → [Output Writer] → Results
```

## Key Interfaces

### DataLoader Interface
```python
class DataLoader:
    def load(path: str) -> DataFrame
    def validate(data: DataFrame) -> bool
```

## Dependencies
- Python 3.8+
- NumPy for numerical operations
- Pandas for data manipulation
