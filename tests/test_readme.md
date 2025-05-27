# Tests for Lattice Gaussian MCMC

This directory contains unit tests and integration tests for the lattice Gaussian MCMC implementation.

## Running Tests

### Unit Tests

To run the NTRU lattice unit tests:

```bash
sage tests/test_ntru.py
```

This will run comprehensive tests for:
- Polynomial arithmetic in ℤ_q[x]/(x^n + 1)
- NTRUSolve algorithm correctness
- Key generation for various parameters
- Gram-Schmidt orthogonalization
- Performance scaling with dimension

### Doctests

To run the doctests embedded in the NTRU module:

```bash
sage tests/run_doctests.sage
```

### Integration Tests

To run integration tests that demonstrate full sampling pipelines:

```bash
sage tests/integration/test_ntru_sampling.sage
```

This will:
- Generate NTRU keys and construct the lattice
- Sample from discrete Gaussian distributions
- Verify samples are in the lattice
- Test CVP solver accuracy
- Compare performance across different parameters
- Generate plots in `results/figures/`

## Test Coverage

The tests cover:

1. **Polynomial Arithmetic** (`test_ntru.py`)
   - Addition and multiplication in quotient rings
   - Modular inversion
   - Conjugate polynomials and norms

2. **NTRU-Specific Operations** (`test_ntru.py`)
   - NTRUSolve algorithm (field norm approach)
   - Key generation with invertibility checks
   - Public key computation

3. **Lattice Operations** (`test_ntru.py`)
   - Basis matrix construction
   - Gram-Schmidt orthogonalization
   - Shortest vector approximation

4. **Cryptographic Parameters** (`test_ntru.py`)
   - n=512 (FALCON-512)
   - n=1024 (FALCON-1024)

5. **Sampling and Applications** (`test_ntru_sampling.sage`)
   - Discrete Gaussian sampling
   - CVP solving accuracy
   - Performance benchmarks

## Expected Output

Successful test runs will show:
- ✓ checkmarks for passed tests
- Performance timings for different parameter sizes
- Generated plots in `results/figures/`
- Performance data in `results/tables/`

## Notes

- Tests with large parameters (n=512, 1024) may take several seconds
- Integration tests generate sample data and plots
- All tests use SageMath's built-in functionality for cryptographic operations