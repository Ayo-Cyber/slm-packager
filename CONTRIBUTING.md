# Contributing to SLM Packager

Thank you for your interest in contributing to SLM Packager! This document provides guidelines for contributing to the project.

## Development Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/slm-packager.git
cd slm-packager
```

### 2. Install in Development Mode

```bash
# Install with dev dependencies
pip install -e ".[dev]"
```

### 3. Set Up Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Install git hooks
pre-commit install
```

Now the hooks will run automatically before each commit.

## Running Tests

### Run All Tests

```bash
# Run all tests with coverage
pytest --cov=slm_packager --cov-report=term-missing

# Run with verbose output
pytest -v
```

### Run Specific Test Categories

```bash
# Unit tests only (fast)
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# E2E tests (may be slow)
pytest tests/e2e/ -v
```

### Run Tests by Marker

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run only E2E tests
pytest -m e2e

# Skip slow tests
pytest -m "not slow"
```

## Code Quality

### Formatting

We use **black** for code formatting and **isort** for import sorting.

```bash
# Format code
black slm_packager tests

# Sort imports
isort slm_packager tests
```

### Type Checking

We use **mypy** for static type checking (optional in v0.1).

```bash
mypy slm_packager
```

### Run All Quality Checks

```bash
# This is what CI runs
black --check slm_packager tests
isort --check-only slm_packager tests
mypy slm_packager
```

## Writing Tests

### Test Structure

- `tests/unit/` - Fast, isolated unit tests
- `tests/integration/` - CLI and API integration tests
- `tests/e2e/` - End-to-end workflow tests

### Test Guidelines

1. **Use markers** - Mark tests appropriately:
   ```python
   @pytest.mark.unit
   @pytest.mark.integration
   @pytest.mark.e2e
   @pytest.mark.slow  # For tests that take >1 second
   ```

2. **Use fixtures** - Leverage shared fixtures from `conftest.py`

3. **Mock external calls** - Mock HuggingFace downloads, model loading, etc.

4. **Keep tests fast** - Unit tests should run in milliseconds

5. **Test edge cases** - Test both success and failure paths

### Example Test

```python
import pytest
from slm_packager.registry import ModelRegistry

@pytest.mark.unit
class TestMyFeature:
    """Test my new feature."""
    
    def test_valid_input(self, mock_registry_file):
        """Test with valid input."""
        # Arrange
        registry = ModelRegistry()
        
        # Act
        result = registry.get_model("test-model")
        
        # Assert
        assert result is not None
        assert result.name == "Test Model"
    
    def test_invalid_input(self):
        """Test with invalid input."""
        registry = ModelRegistry()
        
        with pytest.raises(ValueError):
            registry.get_model(None)
```

## Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write code
   - Add tests
   - Update documentation

3. **Run tests and quality checks**
   ```bash
   pytest
   black slm_packager tests
   isort slm_packager tests
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add feature: your feature description"
   ```
   
   Pre-commit hooks will run automatically.

5. **Push to GitHub**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request**
   - Go to GitHub and create a PR
   - Describe your changes
   - Link any related issues
   - Wait for CI to pass

## Coding Standards

- Follow **PEP 8** style guide
- Use **type hints** where helpful
- Write **docstrings** for public functions and classes
- Keep functions **focused** and **small**
- Use **descriptive variable names**

## Testing Requirements

All PRs must:
- ✅ Pass all existing tests
- ✅ Add new tests for new features
- ✅ Maintain or improve code coverage (target: ≥70%)
- ✅ Pass all code quality checks (black, isort, mypy)

## Questions?

If you have questions or need help:
- Open an issue on GitHub
- Tag your PR with `help wanted`
- Check existing issues and discussions

---

**Thank you for contributing!** 🎉
