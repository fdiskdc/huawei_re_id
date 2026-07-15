# Tests for mRModN_WebAndWx_backend

## Structure

```
tests/
├── conftest.py                     # Shared fixtures
├── unit/
│   ├── test_config.py              # Config class defaults and attributes
│   ├── test_constants.py           # MOD_NAMES, INDEX_TO_NUCLEOTIDE, etc.
│   └── test_paths.py               # Resource file existence checks
├── integration/
│   ├── test_api_routes.py          # Flask route registration and health endpoint
│   └── test_celery_tasks.py        # Celery task names and definitions
└── regression/
    ├── test_model_loading.py       # Model config, checkpoint, architecture
    └── test_linearfold_unchanged.py # LinearFold executable integrity
```

## Running Tests

### Run all tests

```bash
uv run pytest tests/ -v
```

### Run by category

```bash
# Unit tests only
uv run pytest tests/unit/ -v

# Integration tests only
uv run pytest tests/integration/ -v

# Regression tests only
uv run pytest tests/regression/ -v
```

### Run a specific file

```bash
uv run pytest tests/unit/test_config.py -v
```

### Collect tests without running

```bash
uv run pytest tests/ --collect-only
```

## Notes

- Tests import project modules via `sys.path` manipulation (no package structure required).
- Integration tests mock Redis and model loading to run without external services.
- Regression tests require `epoch_040.pt` and `LinearFold/linearfold` to be present.
- The LinearFold hash test captures a baseline on first run; update the expected hash only for intentional upgrades.
