#!/usr/bin/env sage
"""Debug NTRU implementation step by step."""

from sage.all import *
from sage.stats.distributions.discrete_gaussian_integer import DiscreteGaussianDistributionIntegerSampler

# Parameters
n = 8  # Small for debugging
q = 257
sigma = 1.17 * sqrt(q).n()

print(f"NTRU Debug: n={n}, q={q}, sigma={sigma:.2f}")

# Setup rings
R.<x> = PolynomialRing(ZZ)
Rmod = R.quotient(x^n + 1, 'xbar')

Rq.<y> = PolynomialRing(GF(q))  
Rqmod = Rq.quotient(y^n + 1, 'ybar')

print(f"\nRings created:")
print(f"R = {Rmod}")
print(f"Rq = {Rqmod}")

# Test invertibility of some polynomials
print("\n--- Testing Invertibility ---")

test_polys = [
    [1, 0, 0, 0, 0, 0, 0, 0],  # 1
    [2, 1, 0, 0, 0, 0, 0, 0],  # 2 + x
    [1, 1, 0, 0, 0, 0, 0, 0],  # 1 + x
    [3, 1, -1, 0, 0, 0, 0, 0], # 3 + x - x^2
]

for i, coeffs in enumerate(test_polys):
    f_mod = Rmod(coeffs)
    f_q = Rqmod(coeffs)
    
    try:
        f_inv = f_q^(-1)
        check = f_q * f_inv
        print(f"Poly {i+1}: {coeffs[:4]}... is invertible. Check: {check}")
    except:
        print(f"Poly {i+1}: {coeffs[:4]}... is NOT invertible")

# Test NTRU equation solving
print("\n--- Testing NTRU Equation ---")

# Use small integer polynomials
f_coeffs = [3, 1, -1, 0, 0, 0, 0, 0]
g_coeffs = [1, 1, 0, 0, 0, 0, 0, 0]

f = Rmod(f_coeffs)
g = Rmod(g_coeffs)

print(f"\nf = {f}")
print(f"g = {g}")

# Helper to get padded coefficients
def get_coeffs(poly, length):
    c = list(poly.lift())
    while len(c) < length:
        c.append(0)
    return c[:length]

# Compute conjugates
def conjugate(poly_coeffs):
    conj = [poly_coeffs[0]]
    for i in range(1, n):
        conj.append(-poly_coeffs[(n - i) % n])
    return conj

f_star_coeffs = conjugate(f_coeffs)
g_star_coeffs = conjugate(g_coeffs)

f_star = Rmod(f_star_coeffs)
g_star = Rmod(g_star_coeffs)

print(f"\nf* = {f_star}")
print(f"g* = {g_star}")

# Compute norms
Nf = f * f_star
Ng = g * g_star

nf_coeffs = get_coeffs(Nf, n)
ng_coeffs = get_coeffs(Ng, n)

print(f"\nN(f) = {Nf}")
print(f"N(g) = {Ng}")

nf = ZZ(nf_coeffs[0])
ng = ZZ(ng_coeffs[0])

print(f"\nnf = {nf}, ng = {ng}")

# Extended GCD
d, u, v = xgcd(nf, ng)
print(f"gcd({nf}, {ng}) = {d}")
print(f"u = {u}, v = {v}")
print(f"Check: {u}*{nf} + {v}*{ng} = {u*nf + v*ng}")

# Scale by q
u_scaled = u * q
v_scaled = v * q

print(f"\nScaled: u*q = {u_scaled}, v*q = {v_scaled}")

# Compute F, G
F = f_star * Rmod(v_scaled)
G = g_star * Rmod(u_scaled)

print(f"\nF = {F}")
print(f"G = {G}")

# Verify NTRU equation
check = f * G - g * F
check_coeffs = get_coeffs(check, n)

print(f"\nfG - gF = {check}")
print(f"Constant term: {check_coeffs[0]}")
print(f"Equals q? {check_coeffs[0] == q}")
print(f"Other coeffs zero? {all(c == 0 for c in check_coeffs[1:])}")

# Try with Gaussian sampling
print("\n--- Testing with Gaussian Sampling ---")

dgauss = DiscreteGaussianDistributionIntegerSampler(sigma)

for attempt in range(3):
    print(f"\nAttempt {attempt + 1}:")
    
    # Sample f
    f_coeffs = [dgauss() for _ in range(n)]
    f_coeffs[0] += 1  # Make more likely to be invertible
    
    print(f"  f coeffs: {f_coeffs}")
    
    # Check if invertible
    f_q = Rqmod(f_coeffs)
    try:
        f_inv = f_q^(-1)
        print("  ✓ f is invertible mod q")
        
        # Sample g
        g_coeffs = [dgauss() for _ in range(n)]
        print(f"  g coeffs: {g_coeffs}")
        
        # Do NTRU solve
        f = Rmod(f_coeffs)
        g = Rmod(g_coeffs)
        
        f_star = Rmod(conjugate(f_coeffs))
        g_star = Rmod(conjugate(g_coeffs))
        
        Nf = f * f_star
        Ng = g * g_star
        
        nf = ZZ(get_coeffs(Nf, n)[0])
        ng = ZZ(get_coeffs(Ng, n)[0])
        
        if nf == 0 or ng == 0:
            print("  ✗ Norm is zero!")
            continue
            
        d, u, v = xgcd(nf, ng)
        
        F = f_star * Rmod(v * q // d if d > 1 else v * q)
        G = g_star * Rmod(u * q // d if d > 1 else u * q)
        
        check = f * G - g * F
        check_coeffs = get_coeffs(check, n)
        
        const = check_coeffs[0]
        print(f"  fG - gF constant: {const}")
        print(f"  Divisible by q? {const % q == 0}")
        
        if abs(const) == q and all(c == 0 for c in check_coeffs[1:]):
            print("  ✓ NTRU equation solved!")
        else:
            print("  ✗ NTRU equation failed")
            
    except:
        print("  ✗ f not invertible mod q")