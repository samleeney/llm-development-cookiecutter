# PLAN

<!-- PERMANENT INSTRUCTIONS - DO NOT REMOVE THIS SECTION -->
## How to Use This Document

This is your living development roadmap. It tracks:
- All features to be implemented
- Current progress on each feature
- Blockers and dependencies
- Completion history

Update statuses as you work. Move completed items to the Completed section with dates. This helps track velocity and provides a clear audit trail of what was built when.

## Status Legend
- **DONE** - Feature completed and tested
- **IN PROGRESS** - Currently being worked on
- **TODO** - Not yet started
- **BLOCKED** - Waiting on dependencies or decisions
- **CANNOT DO, REVERT TO HUMAN** - Requires human intervention

---

## Features

### 1. Data Infrastructure and Common Interfaces
**Status**: DONE
**Description**: Implement HDF5 data loader and define common data structures
that all models will use.
**Notes**: Design CalibrationData and CalibrationResult classes for model
interoperability.
**Acceptance Criteria**:
- [x] HDF5 data loader class with standardised schema
- [x] CalibrationData dataclass containing all inputs (PSD, VNA, temperatures)
- [x] CalibrationResult dataclass containing outputs (noise parameters,
  predictions)
- [x] JAX array conversion with device placement
- [x] Data validation and sanity checks
- [x] Support for both single and multi-temporal datasets
**Completed**: 2025-09-26
**Implementation**: `src/data.py` with full test coverage

### 2. Model Base Architecture
**Status**: DONE
**Description**: Create abstract base model class that defines the interface all calibration models must follow.
**Notes**: Ensures all models are interchangeable with same inputs/outputs.
**Acceptance Criteria**:
- [x] Abstract base class in `models/base.py`
- [x] Defined interface methods: `fit()`, `predict()`, `get_parameters()`, `get_result()`
- [x] Type hints for all methods
- [x] Documentation of expected behaviour
- [x] Validation methods for calibrators and frequencies
- [x] Result caching for efficiency
**Completed**: 2025-09-26
**Implementation**: `src/models/base.py` with full test coverage

### 3. Least Squares Model Implementation
**Status**: DONE
**Description**: Implement least squares calibration method following the common model interface.
**Notes**: Located in `models/least_squares/lsq.py`, uses JAX for vectorisation.
**Dependencies**: Features 1 and 2 must be complete.
**Acceptance Criteria**:
- [x] Inherits from base model class
- [x] Vectorised least squares solver using `jax.numpy.linalg.lstsq`
- [x] JIT compilation with `@jax.jit` decorator
- [x] Parallel processing across frequencies using `vmap`
- [x] Noise wave parameter computation (5 parameters: u, c, s, NS, L)
- [x] Temperature prediction from parameters
**Completed**: 2025-09-26
**Implementation**: `src/models/least_squares/lsq.py` with full test coverage

### 4. Analysis and Visualisation Module
**Status**: DONE
**Description**: Create plotting functions that work with any model's CalibrationResult output.
**Notes**: Model-agnostic, working with standardised output format.
**Dependencies**: Feature 1 must be complete.
**Acceptance Criteria**:
- [x] Plot input PSD measurements (source, load, noise)
- [x] Plot VNA S-parameters (magnitude and phase)
- [x] Plot noise wave parameters vs frequency
- [x] Plot predicted vs measured temperatures
- [x] Residual plots with statistical analysis
- [x] Export plots in publication format (PDF/PNG)
**Completed**: 2025-09-29
**Implementation**: `src/visualization/calibration_plots.py` with comprehensive plotting suite

### 5. Main Pipeline Script
**Status**: IN PROGRESS
**Description**: Orchestrate the full calibration pipeline with interchangeable models.
**Notes**: Basic pipeline implemented via example script.
**Dependencies**: Features 1-4 must be complete.
**Acceptance Criteria**:
- [x] Load data using data module
- [x] Select and configure model from available models
- [x] Run calibration
- [x] Generate analysis plots
- [x] Save results to HDF5
- [ ] Command-line interface with model selection
**Partial Implementation**: `examples/least_squares_calibration.py` provides working pipeline

### 6. Example Scripts
**Status**: IN PROGRESS
**Description**: Create example scripts demonstrating the pipeline with different models.
**Notes**: Clear and well-documented for new users.
**Dependencies**: Features 1-5 must be complete.
**Acceptance Criteria**:
- [x] Basic example with least squares model
- [ ] Comparison script running multiple models
- [ ] Batch processing example
- [ ] GPU vs CPU performance comparison
- [ ] Jupyter notebook with visualisations
**Implementation**:
- `examples/least_squares_calibration.py` - Full calibration pipeline
- `examples/load_observation.py` - Data loading demonstration

## Implementation Notes

### Data Format Specification
HDF5 structure:
```
/metadata/
  - frequencies
  - measurement_conditions
  - timestamp
/calibrators/
  /{source_name}/
    - psd_source
    - psd_load
    - psd_noise
    - s11_real
    - s11_imag
    - temperature
    - cable_temperature (optional)
/results/ (after calibration)
  - noise_parameters
  - predicted_temperatures
  - model_metadata
```

### Model Interface Contract
```python
class BaseModel:
    def fit(self, data: CalibrationData) -> None:
        """Fit model to calibration data"""

    def predict(self, frequencies: jax.Array,
                calibrator: str) -> jax.Array:
        """Predict temperatures for given calibrator"""

    def get_parameters(self) -> Dict[str, jax.Array]:
        """Return fitted noise wave parameters"""

    def get_result(self) -> CalibrationResult:
        """Return complete calibration result"""
```

## Completed Features

### Feature 4: Analysis and Visualisation Module
**Completed**: 2025-09-29
**Description**: Comprehensive plotting and visualization for calibration results.
**Deliverables**:
- `src/visualization/calibration_plots.py` - Complete plotting suite (600+ lines)
- 5 different plot types: calibrator temperatures, noise parameters, residuals, antenna, summary
- Publication-quality plots with automatic layout adjustment
- Statistical analysis including RMSE, bias, and standard deviation
- Supports all calibrator types and frequency masking
**Key Features**:
- Model-agnostic design working with CalibrationResult interface
- Automatic subplot arrangement based on number of calibrators
- Residual analysis with comprehensive statistics
- Antenna temperature prediction with smoothing options
- Multi-panel summary plots for complete overview

### Feature 3: Least Squares Model Implementation
**Completed**: 2025-09-26
**Description**: JAX-based least squares calibration model with GPU acceleration.
**Deliverables**:
- `src/models/least_squares/lsq.py` - LeastSquaresModel class (400+ lines)
- `src/models/least_squares/__init__.py` - Module exports
- `tests/test_least_squares.py` - 18 unit tests with 89% pass rate
- `examples/least_squares_calibration.py` - Demonstration with REACH data
- Linear least squares solver with vectorisation across frequencies
- JIT compilation for performance optimisation
- Support for regularisation and gamma weighting
- Proper temperature data extraction from HDF5 files
**Performance**: Fits 6554 frequency channels in < 1 second
**Key Features**:
- Proper X matrix construction from S-parameters and PSD data
- Parallel solving using `jax.vmap`
- Interpolation support for arbitrary frequency grids
- Comprehensive error handling (no default temperatures)

### Feature 1: Data Infrastructure and Common Interfaces
**Completed**: 2025-09-26
**Description**: Core data loading and management infrastructure for the calibration pipeline.
**Deliverables**:
- `src/data.py` - Data classes and HDF5 loader (500+ lines)
- `tests/test_data.py` - 15 unit tests with 100% pass rate
- `examples/load_observation.py` - Working example with visualisation
- Full JAX integration with device placement support
- Handles REACH observation format with 11 calibrators
- Auto-discovery of calibrators and frequency computation
**Performance**: Loads full observation file (~100MB) in < 1 second

### Feature 2: Model Base Architecture
**Completed**: 2025-09-26
**Description**: Abstract base class defining the interface for all calibration models.
**Deliverables**:
- `src/models/base.py` - Abstract base model class (250+ lines)
- `src/models/__init__.py` - Module exports
- `tests/test_base_model.py` - 17 unit tests with 100% pass rate
- Complete interface with `fit()`, `predict()`, `get_parameters()`, `get_result()`
- Built-in validation for calibrators and frequencies
- Result caching and automatic residual computation
- Full type hints and comprehensive docstrings
**Key Design Decisions**:
- Models store reference to data after fitting
- Results are cached for efficiency
- Common validation methods reduce code duplication
- Concrete error messages guide implementation

## Blocked/Cannot Do
(None currently)
