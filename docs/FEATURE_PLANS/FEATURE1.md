# Feature 1: Data Infrastructure and Common Interfaces

## Overview
Create a robust data loading infrastructure and standardised interfaces that will serve as the foundation for all calibration models in the JAX-based radiometer calibration pipeline.

## Actual HDF5 Structure Analysis

The existing HDF5 file (`data/reach_observation.hdf5`) has the following structure:

### File Organisation
- **observation_data/** - Contains all calibrator measurements
  - Each calibrator has: `{name}_spectra`, `{name}_load_spectra`, `{name}_ns_spectra`, `{name}_s11`
  - Spectra shape: (n_time, 16384) where n_time varies (15 or 75)
  - S11 shape: (1, 3, 12288) where row 0=frequencies, row 1=real, row 2=imag
- **observation_metadata/** - Temperature and timing data
- **observation_info/** - Configuration metadata (stored as attributes)

### Available Calibrators
- ant, c10open, c10short, c2r27, c2r36, c2r69, c2r91, cold, hot, r100, r25

## Data Classes Design

### 1. CalibratorData - Single calibrator measurements
```python
@dataclass
class CalibratorData:
    name: str
    psd_source: jax.Array      # Shape: (n_time, n_freq)
    psd_load: jax.Array         # Shape: (n_time, n_freq)
    psd_ns: jax.Array           # Shape: (n_time, n_freq)
    s11_freq: jax.Array         # Shape: (n_freq_vna,)
    s11_complex: jax.Array      # Shape: (n_freq_vna,) complex
    timestamps: jax.Array       # Shape: (n_time, 2)
    temperature: Optional[jax.Array] = None  # Physical temperature
```

### 2. CalibrationData - Complete calibration dataset
```python
@dataclass
class CalibrationData:
    calibrators: Dict[str, CalibratorData]
    psd_frequencies: jax.Array  # Derived from sampling_rate/n_channels
    vna_frequencies: jax.Array  # From S11 data
    metadata: Dict[str, Any]    # Config info, timestamps
    temperatures: Optional[jax.Array] = None  # Shape: (n_time, n_sensors)
    temperature_timestamps: Optional[jax.Array] = None
```

### 3. CalibrationResult - Output container
```python
@dataclass
class CalibrationResult:
    noise_parameters: Dict[str, jax.Array]  # u, c, s, NS, L
    predicted_temperatures: Dict[str, jax.Array]  # Per calibrator
    residuals: Dict[str, jax.Array]
    model_name: str
    metadata: Dict[str, Any]  # Model config, timing, etc.
```

## HDF5DataLoader Implementation

### Core Methods
```python
class HDF5DataLoader:
    def load_observation(filepath: str) -> CalibrationData:
        """Load REACH observation HDF5 file"""

    def extract_calibrator(h5file, name: str) -> CalibratorData:
        """Extract single calibrator from HDF5"""

    def compute_psd_frequencies(sampling_rate: float, n_channels: int) -> jax.Array:
        """Calculate PSD frequency array"""

    def apply_frequency_mask(data: CalibrationData, mask: jax.Array) -> CalibrationData:
        """Apply frequency selection mask"""

    def save_results(filepath: str, result: CalibrationResult) -> None:
        """Save calibration results to HDF5"""
```

## Implementation Steps

### Phase 1: Core Data Structures (Day 1)
1. Create `src/data.py` file
2. Implement dataclasses:
   - `CalibratorData`
   - `CalibrationData`
   - `CalibrationResult`
3. Add type hints and comprehensive docstrings
4. Implement `__post_init__` validation

### Phase 2: HDF5 Loader (Days 2-3)
1. Implement `HDF5DataLoader` class
2. Handle the specific REACH HDF5 format:
   - Parse observation_data group
   - Extract S11 data (frequencies + complex values)
   - Load spectra data (source, load, ns)
   - Match timestamps with temperature data
3. Compute PSD frequencies from metadata (400 MHz sampling, 16384 channels)

### Phase 3: Data Processing (Day 4)
1. Implement frequency masking/selection
2. Handle multi-temporal averaging
3. Temperature interpolation to match measurement times
4. Convert to JAX arrays with proper device placement

### Phase 4: Calibration Results (Day 5)
1. Implement result saving to HDF5
2. Add metadata tracking (model config, timing)
3. Create validation utilities

### Phase 5: Testing & Documentation (Day 6)
1. Write unit tests for all components
2. Create integration tests with actual HDF5 file
3. Write API documentation
4. Create usage examples

## Technical Considerations

### Frequency Handling
- **PSD frequencies**: 16384 channels from 0-200 MHz (computed from 400 MHz sampling rate)
- **VNA frequencies**: 12288 points from 50-200 MHz (stored directly in S11 data)
- Need to handle frequency alignment/interpolation between PSD and VNA data

### Calibrator Discovery
```python
def discover_calibrators(h5file):
    """Auto-discover calibrators in HDF5 file"""
    calibrators = set()
    for key in h5file['observation_data'].keys():
        if '_spectra' in key and not any(x in key for x in ['_load', '_ns']):
            calibrators.add(key.replace('_spectra', ''))
    return calibrators
```

### Temperature Mapping
- 9 temperature sensors in the system
- Need to map sensor indices to physical locations
- Interpolate temperatures to match measurement timestamps

### JAX Array Management
- Use `jax.device_put` for explicit device placement
- Default to CPU, allow GPU override via config
- Ensure arrays are contiguous for performance
- Handle both single and batched operations

## Example Usage

```python
# Load observation data
loader = HDF5DataLoader()
data = loader.load_observation('data/reach_observation.hdf5')

# Access specific calibrator
hot_cal = data.calibrators['hot']
print(f"Hot load PSD shape: {hot_cal.psd_source.shape}")
print(f"Frequencies: {data.psd_frequencies[0]:.1f} - {data.psd_frequencies[-1]:.1f} MHz")

# Apply frequency mask (50-130 MHz)
mask = (data.vna_frequencies > 50e6) & (data.vna_frequencies < 130e6)
masked_data = loader.apply_frequency_mask(data, mask)

# Process with a model (future implementation)
from src.models.least_squares import LeastSquaresModel

model = LeastSquaresModel()
model.fit(masked_data)
result = model.get_result()

# Save results
loader.save_results('results/calibration_output.hdf5', result)
```

## Success Metrics

- [ ] Load full REACH observation file in < 1 second
- [ ] Handle variable time dimensions gracefully
- [ ] Zero-copy JAX array conversion where possible
- [ ] 100% test coverage for core functionality
- [ ] Compatible with both CPU and GPU execution
- [ ] Clean integration with downstream models
- [ ] Maintain numerical precision (< 1e-10 relative error)

## Dependencies

- `jax>=0.4.0` - Array operations and GPU support
- `jaxlib` - JAX backend
- `h5py>=3.0` - HDF5 file I/O
- `numpy>=1.20` - Array operations
- `dataclasses` - Data structures (Python 3.7+)
- `typing` - Type hints

## Deliverables

1. **Core Module**: `src/data.py`
   - All dataclasses
   - HDF5DataLoader implementation
   - Utility functions

2. **Tests**: `tests/test_data.py`
   - Unit tests for each class
   - Integration tests with sample HDF5
   - Performance benchmarks

3. **Documentation**:
   - Inline docstrings with examples
   - API reference documentation
   - Data format specification

4. **Examples**:
   - `examples/load_observation.ipynb` - Basic usage
   - `examples/batch_processing.ipynb` - Multiple files
   - `examples/gpu_usage.ipynb` - GPU acceleration

## Notes for Implementation

1. **Error Handling**: Provide clear error messages for:
   - Missing calibrators
   - Corrupted HDF5 files
   - Frequency mismatch
   - Invalid temperature data

2. **Logging**: Add debug logging for:
   - Data loading progress
   - Memory usage
   - Device placement

3. **Performance**: Optimise for:
   - Large datasets (>1GB files)
   - Batch processing
   - Minimal memory footprint

4. **Compatibility**: Ensure works with:
   - Existing REACH data format
   - Future extended formats
   - Different calibrator configurations