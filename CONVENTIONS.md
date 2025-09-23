# CODE CONVENTIONS

<!-- PERMANENT INSTRUCTIONS - DO NOT REMOVE THIS SECTION -->
## How to Use This Document

This document defines coding standards for your project. It ensures consistency across all code, making it easier to read, maintain, and collaborate. Update the example section below with your project-specific conventions while keeping the structure.

---

<!-- PROJECT-SPECIFIC CONVENTIONS - CUSTOMIZE FOR YOUR PROJECT -->

## Python
- **Style**: Black formatter, PEP 8 compliant
- **Line length**: 88 characters (Black default)
- **Naming**:
  - Functions/variables: `snake_case`
  - Classes: `PascalCase`
  - Constants: `UPPER_SNAKE_CASE`
  - Private methods: `_leading_underscore`
- **Docstrings**: Google style, required for all public functions
- **Type hints**: Required for all function signatures
- **Imports**: Sorted with isort (standard library, third-party, local)

## Project Structure
```
project_root/
├── src/           # Source code
│   ├── models/    # Data models
│   ├── services/  # Business logic
│   └── utils/     # Helper functions
├── tests/         # Test files (mirrors src/)
├── data/          # Data files (not in git)
├── docs/          # Documentation
├── scripts/       # Utility scripts
└── configs/       # Configuration files
```

## Git Conventions
- **Branch naming**: `feature/description`, `fix/description`, `docs/description`
- **Commit format**: `type(scope): description`
  - Types: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`
  - Examples:
    - `feat(auth): add password reset functionality`
    - `fix(api): handle null values in response`
    - `docs(readme): update installation instructions`
- **PR titles**: Same as commit format
- **No commits directly to main**

## Testing Standards
- **Coverage**: Minimum 80% overall, 90% for new code
- **Test files**: `test_<module>.py` in tests/ directory
- **Framework**: pytest with fixtures
- **Naming**: `test_<function>_<scenario>_<expected_result>`
- **Organization**:
  - Unit tests: Test individual functions
  - Integration tests: Test component interactions
  - E2E tests: Test complete user flows

## Code Quality
- **Functions**: Maximum 20 lines (prefer smaller)
- **Classes**: Single responsibility principle
- **Files**: Maximum 300 lines
- **Cyclomatic complexity**: Maximum 10
- **No hardcoded values**: Use config files or environment variables
- **Error handling**: Always handle expected errors explicitly
- **Logging**: Use structured logging (not print statements)

## Documentation
- **README**: Keep updated with setup and usage
- **Inline comments**: Only for complex logic
- **Docstrings example**:
```python
def calculate_discount(price: float, discount_percent: float) -> float:
    """Calculate the discounted price.

    Args:
        price: Original price in dollars.
        discount_percent: Discount percentage (0-100).

    Returns:
        Final price after discount.

    Raises:
        ValueError: If discount_percent is not between 0 and 100.
    """
```

## Security
- **No secrets in code**: Use environment variables
- **Input validation**: Validate all user inputs
- **SQL**: Use parameterized queries (never string formatting)
- **Dependencies**: Regular security audits with `safety check`

## Performance
- **Database queries**: Use eager loading to avoid N+1 problems
- **Caching**: Cache expensive computations
- **Async operations**: Use for I/O-bound operations
- **Profiling**: Profile before optimizing