# Tests directory for StableEMRIFisher

This directory contains the test suite for StableEMRIFisher.

## Running Tests

To run all tests:
```bash
pytest tests/
```

To run a specific test file:
```bash
pytest tests/test_stableemriderivative.py -v
```

To run tests with coverage:
```bash
pytest tests/ --cov=stableemrifisher
```

## Test Structure

- `test_stableemriderivative.py`: Tests for the StableEMRIDerivative class
- `test_fisher.py`: Tests for the Fisher matrix calculation (future)
- `test_utils.py`: Tests for utility functions (future)

## Test Configuration

Tests use short waveforms (T=0.01 years) and CPU-only computation for speed and consistency.
