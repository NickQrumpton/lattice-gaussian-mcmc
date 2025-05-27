# Test Suite Documentation

This directory contains a comprehensive test suite for the lattice Gaussian MCMC project. The test suite ensures correctness, robustness, and reproducibility of all components.

## 📁 Test Structure

```
tests/
├── conftest.py              # Pytest configuration and fixtures
├── pytest.ini              # Pytest settings (in project root)
├── unit/                    # Unit tests for individual modules
│   ├── test_lattices.py     # Lattice class tests
│   ├── test_samplers.py     # Sampling algorithm tests
│   ├── test_reduction.py    # Lattice reduction tests
│   ├── test_diagnostics.py  # Diagnostic method tests
│   ├── test_utils.py        # Utility function tests
│   └── test_visualization.py # Plotting utility tests
├── integration/             # Integration and end-to-end tests
│   └── test_full_pipeline.py # Complete pipeline tests
└── README.md               # This documentation
```

## 🚀 Quick Start

### Running All Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=html

# Run tests in parallel
pytest -n auto

# Run only fast tests (exclude slow markers)
pytest -m "not slow"
```

### Running Specific Test Categories

```bash
# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# Statistical tests only
pytest -m statistical

# Numerical accuracy tests
pytest -m numerical

# Edge case tests
pytest -m edge_case

# Reproducibility tests
pytest -m reproducibility
```

### Running Tests for Specific Modules

```bash
# Lattice tests
pytest tests/unit/test_lattices.py

# Sampling algorithm tests
pytest tests/unit/test_samplers.py

# Reduction algorithm tests
pytest tests/unit/test_reduction.py

# Diagnostic tests
pytest tests/unit/test_diagnostics.py

# End-to-end pipeline tests
pytest tests/integration/test_full_pipeline.py
```

## 🏷️ Test Markers

The test suite uses pytest markers to categorize tests:

| Marker | Description | Example Use |
|--------|-------------|-------------|
| `unit` | Unit tests for individual functions | `pytest -m unit` |
| `integration` | Tests for module interactions | `pytest -m integration` |
| `end_to_end` | Full pipeline tests | `pytest -m end_to_end` |
| `slow` | Tests taking > 5 seconds | `pytest -m "not slow"` |
| `statistical` | Tests verifying statistical properties | `pytest -m statistical` |
| `numerical` | Tests for numerical accuracy | `pytest -m numerical` |
| `edge_case` | Edge cases and error conditions | `pytest -m edge_case` |
| `reproducibility` | Deterministic behavior tests | `pytest -m reproducibility` |
| `performance` | Performance and timing tests | `pytest -m performance` |

## 📊 Test Coverage

The test suite aims for comprehensive coverage:

### Module Coverage Targets

- **Lattice Classes**: >95% coverage
  - Abstract base class functionality
  - Identity lattice Z^n implementation
  - Q-ary lattice construction and operations
  - Lattice property validation

- **Sampling Algorithms**: >90% coverage
  - Klein's algorithm correctness and statistics
  - IMHK algorithm convergence and mixing
  - Statistical distribution validation
  - Edge case handling

- **Reduction Algorithms**: >85% coverage
  - LLL reduction correctness and quality
  - BKZ reduction with different block sizes
  - Quality metrics and bounds validation
  - Performance on pathological cases

- **Diagnostics**: >90% coverage
  - Convergence diagnostic accuracy
  - Spectral gap analysis
  - MCMC chain analysis
  - Statistical test validation

- **Utilities**: >95% coverage
  - Discrete Gaussian sampling
  - Jacobi theta functions
  - Numerical stability
  - Helper function correctness

- **Visualization**: >80% coverage
  - Plot generation and formatting
  - Output validation
  - Style consistency

### Current Coverage Status

Run the following to check current coverage:

```bash
pytest --cov=src --cov-report=term-missing
```

Generate HTML coverage report:

```bash
pytest --cov=src --cov-report=html
open tests/coverage_html/index.html
```

## 🧪 Test Categories

### Unit Tests

**Purpose**: Verify individual functions and methods work correctly in isolation.

**Coverage**:
- **Correctness**: Expected outputs for known inputs
- **Numerical Accuracy**: Results within theoretical tolerances
- **Invariants**: Mathematical properties preserved
- **Edge Cases**: Boundary conditions and error handling
- **Statistical Properties**: Random outputs match theory

**Examples**:
```python
def test_klein_statistical_correctness(self, statistical_config):
    """Test Klein sampler produces correct distribution."""
    # Test sample mean, variance, distribution shape
    
def test_lll_lovasz_condition(self, tolerance_config):
    """Test LLL satisfies Lovász condition."""
    # Verify mathematical invariants
```

### Integration Tests

**Purpose**: Verify modules work correctly together.

**Coverage**:
- Data flow between components
- Interface compatibility
- Error propagation
- Performance characteristics

**Examples**:
```python
def test_sampler_with_reduced_lattice(self):
    """Test sampling works after lattice reduction."""
    # Reduction → Sampling → Validation
    
def test_diagnostic_consistency(self):
    """Test diagnostic methods give consistent results."""
    # Multiple diagnostic approaches → Compare results
```

### End-to-End Tests

**Purpose**: Validate complete workflows from start to finish.

**Coverage**:
- Simple lattice pipelines (identity lattice)
- Cryptographic lattice pipelines (NTRU-like)
- Diagnostic pipelines (full MCMC analysis)
- Reproducibility validation
- Golden file regression testing

**Examples**:
```python
def test_identity_lattice_pipeline(self):
    """Complete pipeline: construction → reduction → sampling → diagnostics."""
    
def test_ntru_like_pipeline(self):
    """Cryptographic pipeline with security analysis."""
```

## 🔧 Test Configuration

### Fixtures

Key fixtures defined in `conftest.py`:

| Fixture | Purpose | Usage |
|---------|---------|-------|
| `test_seed` | Reproducible randomness | Deterministic test outcomes |
| `temp_dir` | Temporary file storage | Output validation |
| `simple_2d_basis` | Basic test lattice | Unit testing |
| `pathological_basis` | Edge case testing | Robustness validation |
| `tolerance_config` | Numerical tolerances | Floating-point comparisons |
| `statistical_config` | Test parameters | Sample sizes, confidence levels |
| `performance_config` | Timing limits | Performance validation |

### Configuration Files

**pytest.ini**: Main pytest configuration
- Test discovery patterns
- Marker definitions
- Coverage settings
- Parallel execution
- Warning filters

**conftest.py**: Test fixtures and utilities
- Shared test data
- Utility functions
- Automatic marker assignment
- Random seed management

## 📈 Performance Testing

### Timing Benchmarks

Performance tests validate that algorithms complete within reasonable time limits:

```python
@pytest.mark.performance
def test_lll_performance(self, performance_config):
    """Test LLL reduction completes within time limit."""
    max_time = performance_config['max_time_reduction']
    # Time the operation and assert < max_time
```

### Memory Testing

Memory usage validation for large-scale operations:

```python
def test_large_dataset_plotting(self, performance_config):
    """Test visualization handles large datasets efficiently."""
    # Monitor memory usage and plotting time
```

### Scalability Testing

Dimension scaling analysis:

```python
def test_dimension_scaling(self):
    """Test algorithm performance scaling with dimension."""
    # Test dimensions 2, 3, 4, 5 and analyze scaling
```

## 🔄 Reproducibility

### Deterministic Testing

All tests use fixed random seeds for reproducibility:

```python
@pytest.fixture(scope="session")
def test_seed():
    return 42

def pytest_runtest_setup(item):
    """Set random seed before each test."""
    np.random.seed(42)
```

### Golden File Testing

Critical outputs are validated against reference files:

```python
def test_golden_file_comparison(self, golden_files_config):
    """Compare outputs against reference results."""
    # Run pipeline → Compare with stored results
```

## 📊 Statistical Testing

### Distribution Validation

Tests verify that random outputs follow expected distributions:

```python
def test_discrete_gaussian_profile(self, stat_utils):
    """Test samples follow discrete Gaussian distribution."""
    # Chi-squared goodness of fit test
```

### Convergence Analysis

MCMC convergence validation:

```python
def test_gelman_rubin_diagnostic(self):
    """Test R-hat convergence diagnostic."""
    # Multiple chains → R-hat calculation → Convergence check
```

### Effective Sample Size

Statistical efficiency validation:

```python
def test_effective_sample_size(self, stat_utils):
    """Test ESS computation and reasonable values."""
    # Autocorrelation analysis → ESS calculation
```

## 🐛 Debugging Test Failures

### Common Issues and Solutions

**Statistical Test Failures**:
```bash
# Increase sample size in statistical_config
# Check random seed consistency
# Verify tolerance settings
```

**Numerical Accuracy Failures**:
```bash
# Check tolerance_config values
# Verify input conditioning
# Test with different precision
```

**Performance Test Failures**:
```bash
# Check performance_config limits
# Profile slow operations
# Consider hardware differences
```

**Integration Test Failures**:
```bash
# Check module compatibility
# Verify data flow
# Test components individually
```

### Debugging Commands

```bash
# Verbose output with print statements
pytest -v -s

# Stop on first failure
pytest -x

# Run specific failing test
pytest tests/unit/test_samplers.py::TestKleinSampler::test_statistical_correctness -v

# Debug with pdb
pytest --pdb

# Show test durations
pytest --durations=10
```

### Test Output Analysis

**Coverage Reports**:
- HTML: `tests/coverage_html/index.html`
- Terminal: `--cov-report=term-missing`
- XML: `--cov-report=xml` (for CI)

**Performance Analysis**:
```bash
# Show slowest tests
pytest --durations=0

# Profile memory usage
pytest --profile-svg
```

## 🔄 Continuous Integration

### GitHub Actions Configuration

Create `.github/workflows/tests.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10", "3.11"]
    
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov pytest-xdist
    
    - name: Run tests
      run: |
        pytest --cov=src --cov-report=xml -n auto
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

### Pre-commit Hooks

Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: local
    hooks:
      - id: pytest-fast
        name: pytest-fast
        entry: pytest -m "not slow"
        language: system
        types: [python]
        pass_filenames: false
```

### GitLab CI Configuration

Create `.gitlab-ci.yml`:

```yaml
test:
  image: python:3.9
  before_script:
    - pip install -r requirements.txt
    - pip install pytest pytest-cov pytest-xdist
  script:
    - pytest --cov=src --cov-report=xml -n auto
  coverage: '/TOTAL.+ ([0-9]{1,3}%)/'
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml
```

## 🤝 Contributing New Tests

### Test Writing Guidelines

1. **Descriptive Names**: Use clear, descriptive test names
   ```python
   def test_klein_sampler_statistical_correctness_identity_lattice(self):
   ```

2. **Docstrings**: Document what the test validates
   ```python
   def test_lll_reduction(self):
       """Test LLL reduction preserves lattice and improves quality."""
   ```

3. **Markers**: Add appropriate markers
   ```python
   @pytest.mark.statistical
   @pytest.mark.slow
   def test_mcmc_convergence(self):
   ```

4. **Fixtures**: Use existing fixtures when possible
   ```python
   def test_something(self, identity_lattice_2d, tolerance_config):
   ```

5. **Assertions**: Use informative assertion messages
   ```python
   assert rhat < 1.1, f"Chains not converged: R-hat = {rhat}"
   ```

### Adding New Test Categories

1. **Create Test File**: Follow naming convention `test_*.py`
2. **Add Markers**: Define new markers in `conftest.py`
3. **Update Documentation**: Add to this README
4. **Add to CI**: Include in automated testing

### Test Data Management

- **Small Data**: Include in repository
- **Large Data**: Generate programmatically
- **Golden Files**: Store in `tests/golden/`
- **Temporary Files**: Use `temp_dir` fixture

## 🏆 Test Quality Metrics

### Success Criteria

- **Coverage**: >85% overall, >90% for core modules
- **Performance**: All tests complete within time limits
- **Reproducibility**: All tests deterministic with fixed seeds
- **Reliability**: <1% flaky test rate
- **Documentation**: All test failures easily debuggable

### Monitoring

```bash
# Check test health
pytest --tb=short --quiet

# Performance monitoring
pytest --durations=10 -m performance

# Coverage tracking
pytest --cov=src --cov-fail-under=85
```

### Quality Gates

Before merging code:
1. All tests pass
2. Coverage ≥ 85%
3. No new slow tests without justification
4. Documentation updated if needed
5. Performance regression checks

## 📞 Support

### Getting Help

1. **Documentation**: Read this README and docstrings
2. **Examples**: Check existing tests for patterns
3. **Issues**: Common problems and solutions above
4. **Debug**: Use verbose output and debugging commands

### Reporting Issues

When reporting test failures:
1. Full command and output
2. Environment details (Python version, OS)
3. Minimal reproduction case
4. Expected vs actual behavior

### Performance Issues

For slow tests:
1. Profile with `--durations=0`
2. Check hardware requirements
3. Consider marking as `slow`
4. Optimize if necessary

---

**Last Updated**: 2024
**Test Suite Version**: 1.0
**Coverage Target**: >85%
**Python Versions**: 3.8+