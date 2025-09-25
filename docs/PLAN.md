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
**Status**: TODO
**Description**: Implement least squares calibration method following the common model interface.
**Notes**: Located in `models/least_squares/lsq.py`, uses JAX for vectorisation.
**Dependencies**: Features 1 and 2 must be complete.
**Acceptance Criteria**:
- [ ] Inherits from base model class
- [ ] Vectorised least squares solver using `jax.numpy.linalg.lstsq`
- [ ] JIT compilation with `@jax.jit` decorator
- [ ] Parallel processing across frequencies using `vmap`
- [ ] Noise wave parameter computation (5 parameters: u, c, s, NS, L)
- [ ] Temperature prediction from parameters

### 4. Analysis and Visualisation Module
**Status**: TODO
**Description**: Create plotting functions that work with any model's CalibrationResult output.
**Notes**: Should be model-agnostic, working with standardised output format.
**Dependencies**: Feature 1 must be complete.
**Acceptance Criteria**:
- [ ] Plot input PSD measurements (source, load, noise)
- [ ] Plot VNA S-parameters (magnitude and phase)
- [ ] Plot noise wave parameters vs frequency
- [ ] Plot predicted vs measured temperatures
- [ ] Residual plots with uncertainty bands
- [ ] Export plots in publication format (PDF/PNG)

### 5. Main Pipeline Script
**Status**: TODO
**Description**: Orchestrate the full calibration pipeline with interchangeable models.
**Notes**: Should allow easy switching between different calibration models.
**Dependencies**: Features 1-4 must be complete.
**Acceptance Criteria**:
- [ ] Load data using data module
- [ ] Select and configure model from available models
- [ ] Run calibration
- [ ] Generate analysis plots
- [ ] Save results to HDF5
- [ ] Command-line interface with model selection

### 6. Example Scripts
**Status**: TODO
**Description**: Create example scripts demonstrating the pipeline with different models.
**Notes**: Should be clear and well-documented for new users.
**Dependencies**: Features 1-5 must be complete.
**Acceptance Criteria**:
- [ ] Basic example with least squares model
- [ ] Comparison script running multiple models
- [ ] Batch processing example
- [ ] GPU vs CPU performance comparison
- [ ] Jupyter notebook with visualisations

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
