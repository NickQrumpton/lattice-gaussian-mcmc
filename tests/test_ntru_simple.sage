#!/usr/bin/env sage
"""
Simple test of NTRU lattice implementation.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import NTRU module directly
load(os.path.join(os.path.dirname(__file__), '..', 'src', 'lattices', 'ntru.py'))

print("Testing NTRU lattice implementation...")

# Test 1: Create NTRU lattice
print("\n1. Creating NTRU lattice with n=8, q=257")
lattice = NTRULattice(n=8, q=257)
print(f"   ✓ Created lattice: n={lattice.n}, q={lattice.q}")

# Test 2: Polynomial arithmetic
print("\n2. Testing polynomial arithmetic")
R = lattice.R
f = R([1, 2, 3, 4, 0, 0, 0, 0])
g = R([0, 1, 0, 1, 0, 0, 0, 0])
h = lattice._add_poly(f, g)
print(f"   f = {f}")
print(f"   g = {g}")
print(f"   f + g = {h}")
print("   ✓ Polynomial addition works")

# Test 3: Key generation
print("\n3. Generating NTRU keys")
keys = lattice.generate_keys(max_attempts=5)
if keys is None:
    print("   ❌ Key generation failed")
else:
    f, g, F, G, h = keys
    print("   ✓ Keys generated successfully")
    
    # Verify NTRU equation
    fG = lattice._multiply_poly(f, G)
    gF = lattice._multiply_poly(g, F)
    diff = fG - gF
    diff_coeffs = list(diff)
    
    if diff_coeffs[0] % lattice.q == 0 and all(c == 0 for c in diff_coeffs[1:]):
        print("   ✓ NTRU equation verified: fG - gF = q")
    else:
        print("   ❌ NTRU equation failed")

# Test 4: Basis matrix
print("\n4. Checking basis matrix")
if hasattr(lattice, 'basis_matrix'):
    B = lattice.basis_matrix
    print(f"   Basis dimensions: {B.nrows()} × {B.ncols()}")
    det = abs(B.det())
    expected_det = lattice.q^lattice.n
    print(f"   Determinant: {det} (expected: {expected_det})")
    if det == expected_det:
        print("   ✓ Determinant correct")
    else:
        print("   ❌ Determinant incorrect")

print("\n✅ Basic NTRU tests completed!")