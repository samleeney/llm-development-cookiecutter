# PROJECT GOAL

<!-- PERMANENT INSTRUCTIONS - DO NOT REMOVE THIS SECTION -->
## How to Use This Document

This is the north star document for your project. It defines:
- What you're building and why
- Who will use it and how
- What success looks like
- Technical constraints and choices

Update this document when project scope changes, but keep it focused and concise. This should be the first document anyone reads to understand your project.

---

## Overview
**Project**: JAX Radiometer Calibration Pipeline (jaxcal)
**Purpose**: GPU-accelerated radiometer calibration pipeline for 21cm cosmology using JAX

## Problem
Current radiometer calibration pipelines for 21cm cosmology experiments are CPU-bound and process data sequentially, limiting throughput for large datasets. The existing pipeline uses multiple file formats (text files, .s1p files) and lacks unified data handling, making it difficult to leverage modern GPU acceleration.

## Solution
A JAX-based calibration pipeline that provides:
- Unified HDF5 data format for all measurements (PSD, VNA, temperature)
- Fully parallelised operations across frequency channels using JAX
- JIT-compiled calibration algorithms for optimal performance
- Clean, modular architecture built from scratch
- GPU-compatible data structures throughout

## Users
- Primary: Radio astronomers working on global 21cm cosmology experiments
- Secondary: Instrument scientists calibrating radiometer systems

## Key Features
1. **Data Loading** - HDF5-based data loader with standardised structure, JAX array conversion, and GPU compatibility
2. **Least Squares Calibration** - Vectorised least squares method for noise wave parameter extraction
3. **Analysis Tools** - Comprehensive plotting functions for input data, noise wave parameters, and predicted temperatures
4. **Performance** - Full parallelisation across frequency channels with JIT compilation

## Technical Requirements

### Inputs
- PSD measurements (source, load, noise spectra)
- VNA measurements (S11 parameters)
- Temperature measurements
- Frequency channel information
- All data provided via unified HDF5 format

### Outputs
- Noise wave parameters (5 parameters: u, c, s, NS, L)
- Calibrated antenna temperatures
- Diagnostic plots and visualisations

## Success Criteria
- [ ] Process full frequency range (50-200 MHz) in under 10 seconds
- [ ] Achieve numerical agreement with existing pipeline (< 0.1% difference)
- [ ] Support batch processing of multiple calibration datasets
- [ ] Run on both CPU and GPU with automatic device selection
- [ ] Produce publication-quality diagnostic plots

## Technical Stack
- Language: Python 3.10+
- Core Framework: JAX
- Data Format: HDF5 (h5py)
- Visualisation: Matplotlib
- Testing: Pytest
- Type Hints: Throughout codebase

## Project Structure
- `src/data.py` - HDF5 data loading and JAX array management
- `src/models/` - Calibration models with common interface
  - `base.py` - Abstract base model class defining interface
  - `least_squares/lsq.py` - Least squares implementation
  - Future: `conjugate_priors/`, `marginalised_poly/`
- `src/analysis.py` - Plotting and visualisation functions
- `src/calibration.py` - Main pipeline orchestration
- `examples/` - Example scripts demonstrating usage

## Model Interface Standard
All models must implement:
- `__init__(config)` - Initialize with configuration
- `fit(calibration_data)` - Fit model to calibration data
- `predict(frequency, calibrator)` - Predict temperatures
- `get_parameters()` - Return noise wave parameters
- Common input: `CalibrationData` object
- Common output: `CalibrationResult` object
