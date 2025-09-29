<!-- PERMANENT TEMPLATE STRUCTURE - DO NOT REMOVE -->
# JAX Radiometer Calibration Pipeline (jaxcal)

GPU-accelerated radiometer calibration pipeline for 21cm cosmology using JAX

## Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd jaxcal

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For development

# Run calibration example
python examples/least_squares_calibration.py

# Run tests
python -m pytest tests/ -v
```

## Documentation

<!-- PERMANENT - KEEP THESE LINKS -->
- `docs/PROJECT_GOAL.md` - Project vision and objectives
- `docs/PLAN.md` - Development roadmap and status
- `docs/ARCHITECTURE.md` - System design and structure
- `docs/HOW_TO_DEV.md` - Development workflow
- `CONVENTIONS.md` - Code standards and best practices
- `docs/TESTS.md` - Testing strategy and coverage

## For AI Assistants

<!-- PERMANENT - CRITICAL FOR AI CONTEXT -->
**Read this first!** Essential context for working on this project:
1. Start by reading `docs/PROJECT_GOAL.md` to understand objectives
2. Check `docs/PLAN.md` for current tasks and priorities
3. Follow conventions in `CONVENTIONS.md` strictly
4. Review `docs/ARCHITECTURE.md` before making structural changes
5. Test all changes - see `docs/TESTS.md` for testing approach
6. Update documentation immediately after making changes
7. Follow the workflow in `docs/HOW_TO_DEV.md`

**Initial Setup**: If user has filled in "FILL IN HERE" sections in PROJECT_GOAL.md and PLAN.md, merge their content with the example structure when asked to "optimize" or "merge"

## Usage

```python
from src.data import HDF5DataLoader
from src.models.least_squares.lsq import LeastSquaresModel

# Load calibration data
loader = HDF5DataLoader()
data = loader.load_observation("data/reach_observation.hdf5")

# Apply frequency mask (50-200 MHz)
mask = (data.psd_frequencies >= 50e6) & (data.psd_frequencies <= 200e6)
data_filtered = loader.apply_frequency_mask(data, mask)

# Fit calibration model
model = LeastSquaresModel(regularisation=1e-6)
model.fit(data_filtered)

# Get calibration results
result = model.get_result()
print(f"Noise parameters shape: {result.noise_parameters['u'].shape}")

# Save results
loader.save_results("results/calibration.hdf5", result)
```

## Testing

```bash
# Activate virtual environment first
source venv/bin/activate

# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html --cov-report=term

# Run specific test file
python -m pytest tests/test_least_squares.py -v

# Run with short traceback
python -m pytest tests/ -v --tb=short
```

## Development Status

<!-- PERMANENT - ALWAYS REFERENCE PLAN.MD -->
See `docs/PLAN.md` for detailed progress on features and upcoming work.

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![JAX](https://img.shields.io/badge/JAX-0.4.0%2B-purple)

## Contributing

See `CONTRIBUTING.md` for guidelines on how to contribute to this project.

## License

<!-- EXAMPLE - REPLACE WITH YOUR LICENSE -->
MIT License - see LICENSE file for details

## Author

Sam Mangham