# TESTS

<!-- PERMANENT INSTRUCTIONS - DO NOT REMOVE THIS SECTION -->
## How to Use This Document

This document defines your testing strategy and standards. It should specify:
- How tests are organized
- How to run tests
- Coverage requirements
- Testing best practices for this project

Keep this updated with actual test commands and any special testing requirements. This ensures consistent testing practices across all development.

---

## Test Structure
```
tests/
├── test_data.py           # Data infrastructure tests (COMPLETE)
├── test_models/           # Model tests (TODO)
│   ├── test_base.py      # Base model interface tests
│   └── test_least_squares.py  # Least squares implementation tests
└── fixtures/             # Test data and mocks
```

## Running Tests

### Using Virtual Environment (Recommended)
```bash
# Activate virtual environment
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate     # On Windows

# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_data.py -v

# Run specific test class
python -m pytest tests/test_data.py::TestHDF5DataLoader -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html --cov-report=term

# Run with short traceback
python -m pytest tests/ -v --tb=short
```

### Without Virtual Environment
```bash
# Use the venv Python directly
./venv/bin/python -m pytest tests/ -v
```

## Current Test Files

### test_data.py ✅
**Status**: COMPLETE (15 tests, 100% passing)
**Coverage**: Full coverage of data module
**Test Classes**:
- `TestCalibratorData`: Tests for single calibrator data container
  - Valid creation with JAX arrays
  - PSD shape validation
  - S11 shape validation
  - Timestamp dimension checking
- `TestCalibrationData`: Tests for complete dataset
  - Multiple calibrator management
  - Frequency array validation
  - Calibrator access methods
- `TestCalibrationResult`: Tests for results container
  - Noise parameter validation
  - Missing parameter detection
- `TestHDF5DataLoader`: Tests for HDF5 loader
  - Loading actual REACH observation files
  - Frequency mask application
  - Results saving to HDF5
  - Error handling for missing files

## Test Data
- **Sample HDF5**: `data/reach_observation.hdf5`
  - Real REACH observation with 11 calibrators
  - 16384 PSD channels, 12288 VNA points
  - Multiple time samples per calibrator

## Coverage Requirements
- **Data Module**: ✅ 100% (achieved)
- **Model Base**: 🚧 90% minimum (pending)
- **Least Squares**: 🚧 85% minimum (pending)
- **Integration Tests**: 🚧 80% minimum (pending)

## Testing Best Practices

### 1. Test Naming Convention
```python
def test_{method}_{scenario}_{expected_outcome}():
    """Test that {method} {expected behavior} when {scenario}."""
```
Example: `test_load_observation_raises_error_when_file_missing`

### 2. JAX-Specific Testing
- Use `jnp.array` for test data creation
- Test device placement with `jax.device_put`
- Verify array types with `isinstance(arr, jax.Array)`
- Check complex types with `np.iscomplexobj()`

### 3. Data Validation Tests
- Always test shape mismatches
- Verify error messages are informative
- Test edge cases (empty data, single samples)
- Validate frequency range calculations

### 4. Fixture Usage
```python
@pytest.fixture
def mock_calibrator_data():
    """Create a mock CalibratorData for testing."""
    return CalibratorData(
        name="test",
        psd_source=jnp.ones((10, 100)),
        # ... etc
    )
```

### 5. Performance Tests
- Test loading time < 1 second for 100MB files
- Verify memory usage stays reasonable
- Check JAX compilation doesn't break

## Continuous Integration
Future CI should run:
```yaml
- python -m pytest tests/ -v --cov=src --cov-report=xml
- python -m pytest tests/ --durations=10  # Show slowest tests
```

## Test Utilities
Helper functions in tests:
- `create_mock_calibrator()`: Generate test calibrator data
- `assert_shapes_equal()`: Compare JAX array shapes
- `compare_frequencies()`: Validate frequency arrays

## Known Issues
- Python 3.13 requires virtual environment for JAX
- Complex dtype checking needs `np.iscomplexobj()` not direct comparison
- Large HDF5 files should use temporary directories in tests