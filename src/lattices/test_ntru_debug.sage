#!/usr/bin/env sage
"""Debug NTRU key generation."""

from sage.all import *

# Setup
n, q = 8, 257
sigma = 1.17 * sqrt(q).n()

print(f"Testing NTRU with n={n}, q={q}, sigma={sigma:.2f}")

# Polynomial rings
R.<x> = PolynomialRing(ZZ)
Rmod = R.quotient(x^n + 1)

Rq.<y> = PolynomialRing(GF(q))
Rqmod = Rq.quotient(y^n + 1)

# Test 1: Basic polynomial operations
print("\n1. Testing polynomial operations")
f = Rmod([1, 2, 3, 4, 0, 0, 0, 0])
g = Rmod([0, 1, 0, 1, 0, 0, 0, 0])
h = f + g
print(f"f = {f}")
print(f"g = {g}")
print(f"f + g = {h}")

# Test 2: Modular inversion
print("\n2. Testing modular inversion")
f_coeffs = [2, 1, 0, 0, 0, 0, 0, 0]
f = Rmod(f_coeffs)
f_mod_q = Rqmod([GF(q)(c) for c in f_coeffs])

try:
    f_inv = f_mod_q^(-1)
    print(f"f = {f}")
    print(f"f^(-1) mod q exists")
    
    # Verify
    check = f_mod_q * f_inv
    print(f"f * f^(-1) = {check}")
except:
    print(f"f = {f} is not invertible mod q")

# Test 3: Conjugate
print("\n3. Testing conjugate")
f = Rmod([1, 2, 3, 4, 0, 0, 0, 0])
coeffs = list(f.lift())[:n]
conj_coeffs = [coeffs[0]] + [-coeffs[n - i] for i in range(1, n)]
f_conj = Rmod(conj_coeffs)
print(f"f = {f}")
print(f"f* = {f_conj}")

# Test 4: NTRU solve
print("\n4. Testing NTRU solve")

# Simple f, g
f = Rmod([3, 1, -1, 0, 1, 0, 0, -1])
g = Rmod([1, 2, 0, -1, 0, 1, -1, 0])

# Conjugates
f_coeffs = list(f.lift())[:n]
g_coeffs = list(g.lift())[:n]
f_conj = Rmod([f_coeffs[0]] + [-f_coeffs[n-i] for i in range(1, n)])
g_conj = Rmod([g_coeffs[0]] + [-g_coeffs[n-i] for i in range(1, n)])

# Norms
Nf = f * f_conj
Ng = g * g_conj
nf = ZZ(Nf.lift()[0])
ng = ZZ(Ng.lift()[0])

print(f"N(f) = {nf}")
print(f"N(g) = {ng}")

# Extended GCD
d, u, v = xgcd(nf, ng)
print(f"gcd(N(f), N(g)) = {d}")

# Scale by q
u_scaled = u * q
v_scaled = v * q

# Compute F, G
F = f_conj * Rmod(v_scaled)
G = g_conj * Rmod(u_scaled)

print(f"\nF has degree {F.lift().degree()}")
print(f"G has degree {G.lift().degree()}")

# Verify
check = f * G - g * F
check_coeffs = list(check.lift())
print(f"\nfG - gF = {check_coeffs[0]} + ...")
print(f"Equals q? {check_coeffs[0] == q}")

# Test 5: Random key generation
print("\n5. Testing random key generation")
from sage.stats.distributions.discrete_gaussian_integer import DiscreteGaussianDistributionIntegerSampler

dgauss = DiscreteGaussianDistributionIntegerSampler(sigma)

success = False
for attempt in range(5):
    print(f"\nAttempt {attempt + 1}:")
    
    # Sample f
    f_coeffs = [dgauss() for _ in range(n)]
    f_coeffs[0] += 1
    f = Rmod(f_coeffs)
    print(f"  f coeffs: {f_coeffs}")
    
    # Check invertibility
    f_mod_q = Rqmod([GF(q)(c) for c in f_coeffs])
    try:
        f_inv = f_mod_q^(-1)
        print("  f is invertible mod q ✓")
        
        # Sample g
        g_coeffs = [dgauss() for _ in range(n)]
        g = Rmod(g_coeffs)
        print(f"  g coeffs: {g_coeffs}")
        
        # Try NTRU solve
        try:
            # Conjugates
            f_conj = Rmod([f_coeffs[0]] + [-f_coeffs[n-i] for i in range(1, n)])
            g_conj = Rmod([g_coeffs[0]] + [-g_coeffs[n-i] for i in range(1, n)])
            
            # Norms
            Nf = f * f_conj
            Ng = g * g_conj
            nf = ZZ(Nf.lift()[0])
            ng = ZZ(Ng.lift()[0])
            
            # GCD
            d, u, v = xgcd(nf, ng)
            
            # Scale
            F = f_conj * Rmod(v * q // d if d != 1 else v * q)
            G = g_conj * Rmod(u * q // d if d != 1 else u * q)
            
            # Check
            check = f * G - g * F
            check_coeffs = list(check.lift())
            
            if check_coeffs[0] % q == 0:
                print("  NTRU equation solved ✓")
                success = True
                break
            else:
                print(f"  NTRU check failed: {check_coeffs[0]} mod {q} = {check_coeffs[0] % q}")
        except Exception as e:
            print(f"  NTRU solve error: {e}")
    except:
        print("  f not invertible mod q ✗")

if success:
    print("\n✓ Successfully generated NTRU keys!")
else:
    print("\n✗ Failed to generate keys in 5 attempts")