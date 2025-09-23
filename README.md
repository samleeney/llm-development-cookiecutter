<!-- PERMANENT TEMPLATE STRUCTURE - DO NOT REMOVE -->
# {{cookiecutter.project_name}}

{{cookiecutter.project_short_description}}

## Quick Start

<!-- EXAMPLE - REPLACE WITH YOUR PROJECT'S ACTUAL SETUP -->
```bash
# Clone the repository
git clone <repository-url>
cd {{cookiecutter.project_slug}}

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For development

# Run the application
python src/main.py
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

## Usage

<!-- EXAMPLE - REPLACE WITH YOUR PROJECT'S USAGE -->
```python
from myproject import DataProcessor

# Initialize the processor
processor = DataProcessor(config_path="config.yaml")

# Load and process data
data = processor.load_data("input.csv")
results = processor.process(data)

# Export results
processor.export(results, "output.json")
```

## Testing

<!-- EXAMPLE - UPDATE WITH YOUR TEST COMMANDS -->
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_processor.py

# Run in watch mode during development
pytest-watch
```

## Development Status

<!-- PERMANENT - ALWAYS REFERENCE PLAN.MD -->
See `docs/PLAN.md` for detailed progress on features and upcoming work.

<!-- EXAMPLE - CUSTOMIZE YOUR STATUS BADGES -->
![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-85%25-yellow)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)

## Contributing

See `CONTRIBUTING.md` for guidelines on how to contribute to this project.

## License

<!-- EXAMPLE - REPLACE WITH YOUR LICENSE -->
MIT License - see LICENSE file for details

## Author

{{cookiecutter.author_name}} <{{cookiecutter.author_email}}>