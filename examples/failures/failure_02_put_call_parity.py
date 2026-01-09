"""
Failure Example 02: Put-Call Parity Violation
==============================================

WHAT GOES WRONG
---------------
Black-Scholes implementation that violates put-call parity,
often due to incorrect d1/d2 formulas or sign errors.

WHY IT'S WRONG
--------------
Put-call parity is a fundamental arbitrage relationship [T1]:

    C - P = S*exp(-qT) - K*exp(-rT)

where:
- C = call price
- P = put price
- S = spot price
- K = strike price
- r = risk-free rate
- q = dividend yield
- T = time to maturity

If your BS implementation violates this identity:
1. There's a bug in your code
2. Your prices can be arbitraged
3. Hedges will be wrong
4. Greeks will be incorrect

THE FIX
-------
1. Double-check d1 and d2 formulas
2. Verify the dividend yield term is correct
3. Test put-call parity for all price calculations
4. Use known test cases (Hull textbook examples)

VALIDATION
----------
Put-call parity should hold to numerical precision (< 1e-10).

References
----------
[T1] Hull, J. C. (2018). Options, Futures, and Other Derivatives, 10th Ed.
     Chapter 11: Properties of Stock Options.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


# =============================================================================
# THE WRONG WAY (common errors)
# =============================================================================


def d1_d2_WRONG_v1(S: float, K: float, r: float, q: float, sigma: float, T: float) -> tuple:
    """Wrong d1/d2: missing dividend yield in spot term."""
    # Missing exp(-qT) adjustment!
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def d1_d2_WRONG_v2(S: float, K: float, r: float, q: float, sigma: float, T: float) -> tuple:
    """Wrong d1/d2: sign error in variance term."""
    # Wrong sign: should be + 0.5*sigma^2, not - 0.5*sigma^2
    d1 = (np.log(S / K) + (r - q - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def d1_d2_WRONG_v3(S: float, K: float, r: float, q: float, sigma: float, T: float) -> tuple:
    """Wrong d1/d2: missing sqrt(T) scaling."""
    # Forgot sqrt(T) in denominator!
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / sigma  # Missing sqrt(T)
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def black_scholes_call_WRONG(
    S: float, K: float, r: float, q: float, sigma: float, T: float, error_type: int = 1
) -> float:
    """BS call with wrong d1/d2 - violates put-call parity."""
    if error_type == 1:
        d1, d2 = d1_d2_WRONG_v1(S, K, r, q, sigma, T)
    elif error_type == 2:
        d1, d2 = d1_d2_WRONG_v2(S, K, r, q, sigma, T)
    else:
        d1, d2 = d1_d2_WRONG_v3(S, K, r, q, sigma, T)

    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def black_scholes_put_WRONG(
    S: float, K: float, r: float, q: float, sigma: float, T: float, error_type: int = 1
) -> float:
    """BS put with wrong d1/d2 - violates put-call parity."""
    if error_type == 1:
        d1, d2 = d1_d2_WRONG_v1(S, K, r, q, sigma, T)
    elif error_type == 2:
        d1, d2 = d1_d2_WRONG_v2(S, K, r, q, sigma, T)
    else:
        d1, d2 = d1_d2_WRONG_v3(S, K, r, q, sigma, T)

    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


# =============================================================================
# THE RIGHT WAY
# =============================================================================


def d1_d2_CORRECT(S: float, K: float, r: float, q: float, sigma: float, T: float) -> tuple:
    """Correct d1/d2 formulas [T1]."""
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def black_scholes_call_CORRECT(
    S: float, K: float, r: float, q: float, sigma: float, T: float
) -> float:
    """Correct Black-Scholes call price [T1]."""
    d1, d2 = d1_d2_CORRECT(S, K, r, q, sigma, T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def black_scholes_put_CORRECT(
    S: float, K: float, r: float, q: float, sigma: float, T: float
) -> float:
    """Correct Black-Scholes put price [T1]."""
    d1, d2 = d1_d2_CORRECT(S, K, r, q, sigma, T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def check_put_call_parity(
    S: float,
    K: float,
    r: float,
    q: float,
    T: float,
    call_price: float,
    put_price: float,
    tolerance: float = 1e-10,
) -> tuple[bool, float]:
    """
    Check put-call parity: C - P = S*exp(-qT) - K*exp(-rT) [T1].

    Returns (passes, violation_amount).
    """
    lhs = call_price - put_price
    rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
    violation = abs(lhs - rhs)
    return violation < tolerance, violation


# =============================================================================
# DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    # Hull Example 15.6 parameters
    S = 42.0  # Spot
    K = 40.0  # Strike
    r = 0.10  # Risk-free rate
    q = 0.0  # No dividends
    sigma = 0.20  # Volatility
    T = 0.5  # 6 months

    print("=" * 70)
    print("Failure Example 02: Put-Call Parity Violation")
    print("=" * 70)
    print()
    print("Hull Example 15.6 parameters:")
    print(f"  S = {S}, K = {K}, r = {r:.2%}, q = {q:.2%}, σ = {sigma:.2%}, T = {T}")
    print(f"  Expected: Call ≈ $4.76, Put ≈ $0.81")
    print()

    # Test correct implementation
    call_correct = black_scholes_call_CORRECT(S, K, r, q, sigma, T)
    put_correct = black_scholes_put_CORRECT(S, K, r, q, sigma, T)
    passes, violation = check_put_call_parity(S, K, r, q, T, call_correct, put_correct)

    print("CORRECT Implementation:")
    print(f"  Call = ${call_correct:.6f}")
    print(f"  Put  = ${put_correct:.6f}")
    print(f"  Put-call parity violation = {violation:.2e}")
    print(f"  Status: {'PASS' if passes else 'FAIL'}")
    print()

    # Test each wrong implementation
    error_names = {
        1: "Missing dividend yield in d1",
        2: "Sign error in variance term",
        3: "Missing sqrt(T) scaling",
    }

    print("WRONG Implementations:")
    print("-" * 70)

    for error_type in [1, 2, 3]:
        call_wrong = black_scholes_call_WRONG(S, K, r, q, sigma, T, error_type)
        put_wrong = black_scholes_put_WRONG(S, K, r, q, sigma, T, error_type)
        passes_wrong, violation_wrong = check_put_call_parity(
            S, K, r, q, T, call_wrong, put_wrong, tolerance=0.01
        )

        print(f"\n  Error type {error_type}: {error_names[error_type]}")
        print(f"    Call = ${call_wrong:.6f} (vs correct ${call_correct:.6f})")
        print(f"    Put  = ${put_wrong:.6f} (vs correct ${put_correct:.6f})")
        print(f"    Parity violation = {violation_wrong:.6f}")
        print(f"    Status: {'PASS' if passes_wrong else 'FAIL - PUT-CALL PARITY VIOLATED'}")

    print()
    print("=" * 70)
    print("KEY LESSON:")
    print("  Put-call parity is the first test for any BS implementation.")
    print("  If C - P != S*exp(-qT) - K*exp(-rT), you have a bug.")
    print()
    print("  Common errors:")
    print("    1. Forgetting dividend yield (q) in spot term")
    print("    2. Sign error: using -0.5*sigma^2 instead of +0.5*sigma^2 in d1")
    print("    3. Missing sqrt(T) in the denominator of d1")
    print("=" * 70)
