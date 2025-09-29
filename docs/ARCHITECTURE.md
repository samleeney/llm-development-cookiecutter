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

## Project Structure
```
jaxcal/
├── src/                    # Source code
│   ├── data.py            # Data infrastructure (COMPLETE)
│   ├── models/            # Calibration models
│   │   ├── __init__.py    # Module exports
│   │   ├── base.py        # Abstract base model (COMPLETE)
│   │   ├── io.py          # Model save/load (COMPLETE)
│   │   └── least_squares/ # Least squares implementation (COMPLETE)
│   │       ├── __init__.py
│   │       └── lsq.py
│   └── visualization/     # Plotting and analysis
│       └── calibration_plots.py  # Comprehensive plots (COMPLETE)
├── tests/                 # Test files
│   ├── test_data.py       # Data module tests (COMPLETE)
│   ├── test_base_model.py # Base model tests (COMPLETE)
│   └── test_least_squares.py # Least squares tests (COMPLETE)
├── scripts/               # Utility scripts
│   └── convert_test_data_to_hdf5.py  # Convert test dataset (COMPLETE)
├── examples/              # Example scripts
│   ├── load_observation.py  # Data loading example (COMPLETE)
│   └── least_squares_calibration.py  # Full pipeline example (COMPLETE)
├── data/                  # Data files
│   ├── reach_observation.hdf5  # Sample REACH observation
│   └── test_observation.hdf5   # Converted test dataset
├── plots/                 # Generated plots
│   ├── calibration_summary.png
│   ├── antenna_temperature.png
│   └── residuals_summary.png
├── docs/                  # Documentation
│   ├── PROJECT_GOAL.md    # Project objectives
│   ├── PLAN.md           # Development roadmap
│   ├── ARCHITECTURE.md   # This file
│   ├── HOW_TO_DEV.md     # Development workflow
│   ├── TESTS.md          # Testing documentation
│   └── CONVENTIONS.md    # Code standards
├── venv/                  # Virtual environment with JAX
├── requirements.txt       # Python dependencies
└── results/               # Output files

```

## Core Modules

### Module: Data Infrastructure (`src/data.py`) ✅
- **Purpose**: HDF5 data loading and JAX array management
- **Status**: COMPLETE with LNA S11 support
- **Key Classes**:
  - `CalibratorData`: Single calibrator measurements container
  - `CalibrationData`: Complete calibration dataset (includes lna_s11 field)
  - `CalibrationResult`: Calibration output container
  - `HDF5DataLoader`: REACH HDF5 file loader
- **Dependencies**: JAX, h5py, numpy
- **Features**:
  - Auto-discovery of calibrators
  - Frequency computation (PSD and VNA)
  - Device placement support (CPU/GPU)
  - Data validation
  - **LNA S11 loading**: Mandatory loading of LNA S11 data from HDF5
  - Temperature validation: Allows up to 10000K for antenna measurements

### Module: Model Base Architecture (`src/models/base.py`) ✅
- **Purpose**: Abstract base class for all calibration models
- **Status**: COMPLETE
- **Interface**:
  ```python
  class BaseModel(ABC):
      def fit(self, data: CalibrationData) -> None
      def predict(self, frequencies: jax.Array, calibrator: str) -> jax.Array
      def get_parameters(self) -> Dict[str, jax.Array]
      def get_result(self) -> CalibrationResult
  ```
- **Key Features**:
  - Abstract methods enforce consistent interface
  - Built-in validation for calibrators and frequencies
  - Result caching for efficiency
  - Automatic residual computation
  - Type hints throughout
- **Test Coverage**: 17 unit tests, 100% pass rate

### Module: Least Squares Model (`src/models/least_squares/lsq.py`) ✅
- **Purpose**: Least squares calibration implementation
- **Status**: COMPLETE with LNA S11 support
- **Key Classes**:
  - `LeastSquaresModel`: Linear least squares solver for noise wave parameters
- **Dependencies**: data, models.base, JAX
- **Features**:
  - Vectorised solver using `jax.numpy.linalg.lstsq`
  - JIT compilation with `@jax.jit` for performance
  - Parallel processing across frequencies with `jax.vmap`
  - X matrix construction from S-parameters and PSD
  - Support for regularisation and gamma weighting
  - Temperature extraction with proper error handling
  - Excludes antenna from fitting (antenna is the target)
  - **LNA S11 support**: Requires and uses actual LNA S11 data for accurate calibration
- **Performance**:
  - Synthetic data: RMSE < 0.001K on calibration sources
  - Antenna prediction: ~5000K (physically accurate)
- **Test Coverage**: 6 comprehensive tests, 100% pass rate

### Module: Visualisation (`src/visualization/calibration_plots.py`) ✅
- **Purpose**: Comprehensive plotting functionality for calibration results
- **Status**: COMPLETE
- **Key Functions**:
  - `plot_calibrator_temperatures()`: Grid plot of all calibrator temperatures
  - `plot_noise_parameters()`: All 5 noise wave parameters vs frequency
  - `plot_residuals_summary()`: Statistical analysis of calibration residuals
  - `plot_antenna_temperature()`: Dedicated antenna temperature plot
  - `plot_calibration_summary()`: Multi-panel comprehensive summary
- **Dependencies**: matplotlib, JAX arrays, CalibrationData, CalibrationResult
- **Features**:
  - Publication-quality plots with customisable styling
  - Automatic layout adjustment for different numbers of calibrators
  - Residual analysis with RMSE, bias, and standard deviation
  - Support for frequency masking and data filtering

### Module: Model I/O (`src/models/io.py`) ✅
- **Purpose**: Save and load calibration models
- **Status**: COMPLETE
- **Key Functions**:
  - `save_model()`: Save model parameters and metadata to HDF5
  - `load_model()`: Restore model from HDF5 file
- **Features**:
  - Preserves all model parameters and fitted state
  - Includes metadata for reproducibility
  - Compatible with all model types via base interface

## Data Flow

### Current Implementation
```
REACH HDF5 File
    ↓
[HDF5DataLoader]
    ↓
CalibrationData
    ├── 13 Calibrators (hot, cold, ant, etc.)
    ├── PSD data (16384 channels, 0-200 MHz)
    ├── VNA S11 data (12288 points, 50-200 MHz)
    ├── LNA S11 data (required for calibration)
    └── Metadata & Temperatures
    ↓
[Frequency Masking (50-130 MHz typical)]
    ↓
Filtered CalibrationData
    ↓
[Optional: Filter to specific calibrators]
    ↓
[LeastSquaresModel.fit()]
    ↓
CalibrationResult
    ├── Noise Parameters (u, c, s, NS, L)
    ├── Predicted Temperatures
    └── Residuals (RMSE < 0.001K for calibrators)
```

### Data Specifications
- **PSD Data**: Power spectral density measurements
  - Shape: (n_time, 16384) per calibrator
  - Frequency: 0-200 MHz (computed from 400 MHz sampling)
  - Three types: source, load, noise source

- **VNA Data**: S11 reflection coefficients
  - Shape: (12288,) complex values per calibrator
  - Frequency: 50-200 MHz (direct measurement)

- **Calibrators**: 12 standard sources (ant excluded from fitting)
  - Temperature references: hot (~372K), cold (~271K)
  - Impedance standards: c10r10, c10r250, c2r27, c2r36, c2r69, c2r91, r25, r100
  - Terminations: c10open, c10short
  - Antenna: ant (calibration target, not used in fitting)

## Key Interfaces

### CalibrationData Interface
```python
@dataclass
class CalibrationData:
    calibrators: Dict[str, CalibratorData]
    psd_frequencies: jax.Array
    vna_frequencies: jax.Array
    metadata: Dict[str, Any]
    temperatures: Optional[jax.Array]
    temperature_timestamps: Optional[jax.Array]

    def get_calibrator(name: str) -> CalibratorData
    @property
    def calibrator_names() -> List[str]
```

### HDF5DataLoader Interface
```python
class HDF5DataLoader:
    def load_observation(filepath: str) -> CalibrationData
    def apply_frequency_mask(data: CalibrationData, mask: jax.Array) -> CalibrationData
    def save_results(filepath: str, result: CalibrationResult) -> None
```

## Dependencies

### Core Requirements
- **Python**: 3.10+ (tested with 3.13 in venv)
- **JAX**: 0.4.0+ (GPU-accelerated arrays)
- **jaxlib**: 0.4.0+ (JAX backend)
- **NumPy**: 1.24.0+ (array operations)
- **h5py**: 3.0.0+ (HDF5 I/O)

### Development Requirements
- **pytest**: 7.4.0+ (testing framework)
- **matplotlib**: 3.7.0+ (visualisation)
- **ipython**: 8.14.0+ (interactive development)

## Performance Characteristics
- **Data Loading**: < 1 second for 100MB HDF5 file
- **Memory Usage**: ~500MB for full observation dataset
- **JAX Arrays**: Zero-copy conversion where possible
- **Device Support**: CPU by default, GPU optional

## Recent Improvements (2024-09)

### LNA S11 Support
- **Problem**: Matrix ill-conditioning (condition number ~10^18) with zero Gamma_rec
- **Solution**: Mandatory loading and use of actual LNA S11 data
- **Impact**: Improved numerical stability and calibration accuracy

### Test Dataset Integration
- **Added**: Synthetic test dataset conversion to HDF5 format
- **Script**: `scripts/convert_test_data_to_hdf5.py`
- **Result**: Perfect calibration validation (RMSE < 0.001K)

### Calibrator Filtering
- **Feature**: Option to use subset of calibrators for fitting
- **Example**: Using only 5 calibrators (hot, cold, c10open, c10short, r100)
- **Benefit**: Better performance on specific calibrator sets

## Future Architecture Components
1. **Model Zoo**: Multiple calibration models inheriting from BaseModel
2. **Pipeline Orchestration**: End-to-end calibration workflow
3. **Result Analysis**: Extended statistical analysis tools
4. **Batch Processing**: Multiple observations in parallel
5. **GPU Optimisation**: Full GPU acceleration for large datasets
6. **Real-time Calibration**: Streaming data support
