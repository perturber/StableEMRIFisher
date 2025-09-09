# Migration Guide: From pip/conda to uv

This guide helps you migrate your StableEMRIFisher development environment from pip/conda to uv.

## Why UV?

- **Speed**: 10-100x faster than pip
- **Better dependency resolution**: More reliable conflict detection
- **Lock files**: Reproducible builds with `uv.lock`
- **Unified tool**: Package management, virtual environments, and more
- **Compatible**: Works with existing pyproject.toml

## Installation

### Install UV

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.sh | iex"

# Or with pip
pip install uv
```

### Verify Installation

```bash
uv --version
```

## Migration Steps

### 1. Initialize UV in Your Project

```bash
cd StableEMRIFisher
uv init --python 3.12  # or your preferred version
```

### 2. Create Virtual Environment

```bash
# Create and activate venv
uv venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate  # Windows
```

### 3. Install Dependencies

```bash
# Install project in development mode
uv pip install -e ".[docs,dev]"

# For GPU support
uv pip install -e ".[cuda12x,docs,dev]"
```

### 4. Generate Lock File

```bash
uv lock
```

This creates `uv.lock` with exact dependency versions for reproducible installs.

### 5. Update Your Workflow

Replace pip commands with uv equivalents:

| Old (pip/conda) | New (uv) |
|----------------|----------|
| `pip install package` | `uv add package` |
| `pip install -r requirements.txt` | `uv sync` |
| `pip install -e .` | `uv pip install -e .` |
| `python -m pytest` | `uv run pytest` |
| `python script.py` | `uv run python script.py` |

## Common Commands

### Development Workflow

```bash
# Install new dependency
uv add numpy

# Install development dependency
uv add --dev pytest

# Update dependencies
uv lock --upgrade

# Sync environment with lock file
uv sync

# Run commands in environment
uv run pytest
uv run black .
uv run python your_script.py
```

### Environment Management

```bash
# Create environment with specific Python version
uv venv --python 3.11

# Remove environment
rm -rf .venv

# Show environment info
uv pip list
```

### Lock File Operations

```bash
# Generate lock file
uv lock

# Update specific package
uv lock --upgrade-package numpy

# Install from lock file
uv sync

# Check for updates
uv lock --dry-run --upgrade
```

## Convenience Scripts

We've created helper scripts for common tasks:

### Quick Setup
```bash
./scripts/setup-dev.sh
```

### Development Commands
```bash
# Run tests
./scripts/uv-dev.sh test

# Format code
./scripts/uv-dev.sh format

# Build docs
./scripts/uv-dev.sh docs

# Lint code
./scripts/uv-dev.sh lint
```

## IDE Integration

### VS Code

Add to `.vscode/settings.json`:

```json
{
    "python.defaultInterpreterPath": "./.venv/bin/python",
    "python.terminal.activateEnvironment": true
}
```

### PyCharm

1. Go to Settings → Project → Python Interpreter
2. Add Interpreter → Existing Environment
3. Select `.venv/bin/python`

## CI/CD Integration

### GitHub Actions

Use the included `.github/workflows/test.yml` which:
- Installs uv using `astral-sh/setup-uv@v2`
- Caches dependencies based on `uv.lock`
- Runs tests across multiple Python versions

### Read the Docs

Update `docs/requirements.txt` to use uv if needed:

```bash
# Install uv in RTD environment
uv
-e .
```

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure you're in the activated environment
   ```bash
   source .venv/bin/activate
   ```

2. **Missing dependencies**: Sync environment
   ```bash
   uv sync
   ```

3. **Version conflicts**: Check lock file
   ```bash
   uv lock --dry-run
   ```

### Environment Variables

Set in your shell profile:

```bash
# Use uv for faster pip operations
export UV_SYSTEM_PYTHON=1  # Use system Python when no venv active
export UV_CACHE_DIR="$HOME/.cache/uv"  # Custom cache location
```

## Migration Checklist

- [ ] Install uv
- [ ] Create virtual environment with `uv venv`
- [ ] Install dependencies with `uv pip install -e ".[docs,dev]"`
- [ ] Generate lock file with `uv lock`
- [ ] Update scripts to use `uv run`
- [ ] Configure IDE to use `.venv/bin/python`
- [ ] Update CI/CD workflows
- [ ] Test documentation builds
- [ ] Run test suite to verify everything works

## Rollback Plan

If you need to go back to pip/conda:

1. Keep your existing `pyproject.toml` (it's compatible)
2. Delete `.venv/`, `uv.lock`, and `uv.toml`
3. Recreate conda environment or use pip as before

The beauty of uv is that it's fully compatible with existing Python packaging standards!
