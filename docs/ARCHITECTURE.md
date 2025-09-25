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
│   └── models/            # Calibration models
│       ├── __init__.py    # Module exports
│       ├── base.py        # Abstract base model (COMPLETE)
│       └── least_squares/ # Least squares implementation (TODO)
│           └── lsq.py
├── tests/                 # Test files
│   ├── test_data.py       # Data module tests (COMPLETE)
│   └── test_base_model.py # Base model tests (COMPLETE)
├── data/                  # Data files
│   └── reach_observation.hdf5  # Sample REACH observation
├── docs/                  # Documentation
│   ├── FEATURE_PLANS/     # Feature implementation plans
│   ├── PROJECT_GOAL.md    # Project objectives
│   ├── PLAN.md           # Development roadmap
│   └── ARCHITECTURE.md   # This file
├── examples/              # Example scripts
│   └── load_observation.py  # Data loading example (COMPLETE)
├── venv/                  # Virtual environment with JAX
├── requirements.txt       # Python dependencies
└── results/               # Output files

```

## Core Modules

### Module: Data Infrastructure (`src/data.py`) ✅
- **Purpose**: HDF5 data loading and JAX array management
- **Status**: COMPLETE
- **Key Classes**:
  - `CalibratorData`: Single calibrator measurements container
  - `CalibrationData`: Complete calibration dataset
  - `CalibrationResult`: Calibration output container
  - `HDF5DataLoader`: REACH HDF5 file loader
- **Dependencies**: JAX, h5py, numpy
- **Features**:
  - Auto-discovery of calibrators
  - Frequency computation (PSD and VNA)
  - Device placement support (CPU/GPU)
  - Data validation

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

### Module: Least Squares Model (`src/models/least_squares/lsq.py`) 🚧
- **Purpose**: Least squares calibration implementation
- **Status**: TODO
- **Dependencies**: data, models.base
- **Features**: Vectorised solver, JIT compilation, parallel processing

## Data Flow

### Current Implementation
```
REACH HDF5 File
    ↓
[HDF5DataLoader]
    ↓
CalibrationData
    ├── 11 Calibrators (hot, cold, ant, etc.)
    ├── PSD data (16384 channels, 0-200 MHz)
    ├── VNA S11 data (12288 points, 50-200 MHz)
    └── Metadata & Temperatures
    ↓
[Frequency Masking]
    ↓
Filtered CalibrationData
    ↓
[Model Processing] ← TODO
    ↓
CalibrationResult
    ├── Noise Parameters (u, c, s, NS, L)
    ├── Predicted Temperatures
    └── Residuals
```

### Data Specifications
- **PSD Data**: Power spectral density measurements
  - Shape: (n_time, 16384) per calibrator
  - Frequency: 0-200 MHz (computed from 400 MHz sampling)
  - Three types: source, load, noise source

- **VNA Data**: S11 reflection coefficients
  - Shape: (12288,) complex values per calibrator
  - Frequency: 50-200 MHz (direct measurement)

- **Calibrators**: 11 standard sources
  - Temperature references: hot, cold
  - Impedance standards: r25, r100, c2r27, c2r36, c2r69, c2r91
  - Terminations: c10open, c10short
  - Antenna: ant

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

## Future Architecture Components
1. **Model Zoo**: Multiple calibration models inheriting from BaseModel
2. **Pipeline Orchestration**: End-to-end calibration workflow
3. **Result Analysis**: Comprehensive plotting and diagnostics
4. **Batch Processing**: Multiple observations in parallel
5. **GPU Optimisation**: Full GPU acceleration for large datasets
