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

<!-- EXAMPLE CONTENT - REPLACE EVERYTHING BELOW WITH YOUR PROJECT SPECIFICS -->

## Test Structure
```
tests/
├── unit/           # Unit tests for individual functions
├── integration/    # Tests for component interactions
├── e2e/            # End-to-end user scenarios
├── fixtures/       # Test data and mocks
└── conftest.py     # Shared pytest fixtures
```

## Running Tests
```bash
# All tests
pytest

# Specific test suites
pytest tests/unit/           # Unit tests only
pytest tests/integration/    # Integration tests only

# With coverage
pytest --cov=src --cov-report=html

# Fast tests only (marked with @pytest.mark.fast)
pytest -m fast

# Watch mode for development
pytest-watch
```

## Test Files

### test_data_loader.py
- Tests data import from various formats
- Validates error handling for corrupt files
- Checks performance with large datasets

### test_auth.py
- User registration and validation
- Login/logout flows
- Token expiration and refresh

### test_api_endpoints.py
- REST API response codes
- Request validation
- Rate limiting

## Coverage Requirements
- Minimum overall: 80%
- Critical paths (auth, payments): 95%
- New code in PRs: 90%

## Testing Best Practices
1. Use descriptive test names: `test_login_with_invalid_password_returns_401`
2. One assertion per test when possible
3. Use fixtures for common setup
4. Mock external services
5. Test both happy paths and edge cases