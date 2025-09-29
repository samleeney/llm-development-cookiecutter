# JAX Calibration Pipeline

A high-performance radio telescope calibration pipeline using JAX for GPU-accelerated computation, implementing least squares calibration for noise wave parameter extraction.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/samleeney/jaxcal.git
cd jaxcal

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run calibration example
python examples/least_squares_calibration.py data/test_observation.hdf5
```

## Documentation

- `docs/PROJECT_GOAL.md` - Project vision and objectives
- `docs/PLAN.md` - Development roadmap and status
- `docs/ARCHITECTURE.md` - System design and structure
- `docs/HOW_TO_DEV.md` - Development workflow
- `CONVENTIONS.md` - Code standards and best practices
- `docs/TESTS.md` - Testing strategy and coverage
- `results/README.md` - Output files documentation

## For AI Assistants

**Read this first!** Essential context for working on this project:
1. Start by reading `docs/PROJECT_GOAL.md` to understand objectives
2. Check `docs/PLAN.md` for current tasks and priorities
3. Follow conventions in `CONVENTIONS.md` strictly
4. Review `docs/ARCHITECTURE.md` before making structural changes
5. Test all changes - see `docs/TESTS.md` for testing approach
6. Update documentation immediately after making changes
7. Follow the workflow in `docs/HOW_TO_DEV.md`

## Usage

### Basic Calibration

```python
from src.data import HDF5DataLoader
from src.models.least_squares import LeastSquaresModel

# Load observation data
loader = HDF5DataLoader()
data = loader.load_observation('data/reach_observation.hdf5')

# Apply frequency mask (50-130 MHz for cleaner results)
mask = (data.vna_frequencies >= 50e6) & (data.vna_frequencies <= 130e6)
masked_data = loader.apply_frequency_mask(data, mask)

# Fit least squares model
model = LeastSquaresModel()
model.fit(masked_data)

# Get noise wave parameters
params = model.get_parameters()  # Returns u, c, s, NS, L

# Predict antenna temperature
result = model.get_result()
T_ant = result.predicted_temperatures['ant']
```

### Visualization

```python
from src.visualization.calibration_plots import CalibrationPlotter

# Create comprehensive plots
plotter = CalibrationPlotter(output_dir=Path("plots"), save=True)
plotter.create_summary_plot(masked_data, model, result)
```

## Key Features

- **JAX-based**: Fully vectorized operations with GPU acceleration support
- **HDF5 Support**: Efficient loading of REACH observation format
- **Least Squares Calibration**: Extract noise wave parameters (u, c, s, NS, L)
- **Comprehensive Visualization**: Publication-quality plots for analysis
- **Validated Performance**:
  - Calibration sources: RMSE < 0.001K (synthetic data)
  - Antenna prediction: ~5000K (physically accurate)

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test module
pytest tests/test_least_squares.py

# Run with verbose output
pytest -v
```

## Development Status

See `docs/PLAN.md` for detailed progress on features and upcoming work.

### Completed Features ✅
- HDF5 data loading infrastructure
- Base model architecture
- Least squares calibration implementation
- LNA S11 support (critical for accurate calibration)
- Comprehensive visualization suite
- Full test coverage

### Performance Metrics
- **Calibration Accuracy**: RMSE < 0.001K on synthetic calibration sources
- **Processing Speed**: < 1 second for full calibration
- **Memory Usage**: ~500MB for full observation dataset

## Project Structure

```
jaxcal/
├── src/                    # Source code
│   ├── data.py            # Data loading and management
│   ├── models/            # Calibration models
│   │   ├── base.py        # Abstract base model
│   │   ├── least_squares/ # Least squares implementation
│   │   └── io.py          # Model persistence
│   └── visualization/     # Plotting utilities
├── tests/                 # Unit tests
├── examples/              # Usage examples
├── scripts/               # Utility scripts
└── data/                  # Sample datasets
```

## Contributing

See `CONTRIBUTING.md` for guidelines on how to contribute to this project.

## License

MIT License - see LICENSE file for details

## Author

Sam Leeney <samleeney@gmail.com>