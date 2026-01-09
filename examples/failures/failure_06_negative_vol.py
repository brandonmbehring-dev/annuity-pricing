"""
Failure Example 06: Edge Cases and Invalid Parameters
======================================================

WHAT GOES WRONG
---------------
Edge cases that crash or produce nonsensical results:
- Negative volatility
- Zero or negative time
- Negative spot or strike
- Extreme parameter values

WHY IT'S WRONG
--------------
Black-Scholes requires specific input domains [T1]:
- S > 0 (spot must be positive)
- K > 0 (strike must be positive)
- T > 0 (time to expiry must be positive)
- sigma > 0 (volatility must be positive)
- -1 < r (rates can be negative but not extremely so)

Violating these constraints leads to:
1. Math errors (log of negative, division by zero)
2. NaN or Inf results
3. Negative prices (impossible)
4. Completely wrong hedging

THE FIX
-------
1. Validate all inputs before calculation
2. Handle edge cases explicitly
3. Use appropriate defaults or raise exceptions
4. Test with boundary conditions

VALIDATION
----------
All inputs should be validated. Invalid inputs should raise clear errors.

References
----------
[T1] Hull, J. C. (2018). Options, Futures, and Other Derivatives, 10th Ed.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
from scipy.stats import norm


# =============================================================================
# THE WRONG WAY (no validation)
# =============================================================================


def black_scholes_call_NO_VALIDATION(
    S: float, K: float, r: float, q: float, sigma: float, T: float
) -> float:
    """
    BS call WITHOUT input validation.

    Will produce NaN, Inf, or crash on invalid inputs.
    """
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


# =============================================================================
# THE RIGHT WAY (with validation)
# =============================================================================


class OptionPricingError(ValueError):
    """Exception for invalid option pricing parameters."""

    pass


def validate_bs_inputs(
    S: float, K: float, r: float, q: float, sigma: float, T: float
) -> None:
    """
    Validate Black-Scholes inputs [T1].

    Raises OptionPricingError with descriptive message for invalid inputs.
    """
    errors = []

    # Spot
    if S <= 0:
        errors.append(f"Spot (S={S}) must be positive")

    # Strike
    if K <= 0:
        errors.append(f"Strike (K={K}) must be positive")

    # Volatility
    if sigma <= 0:
        errors.append(f"Volatility (sigma={sigma}) must be positive")
    elif sigma > 5.0:  # 500% vol is suspicious
        warnings.warn(f"Volatility {sigma:.0%} seems unreasonably high")

    # Time
    if T < 0:
        errors.append(f"Time (T={T}) cannot be negative")
    elif T == 0:
        # At expiry is a valid edge case, but needs special handling
        pass

    # Rate (can be negative in modern markets)
    if r < -0.20:  # -20% seems extreme
        warnings.warn(f"Rate {r:.2%} is very negative, please verify")

    if errors:
        raise OptionPricingError("; ".join(errors))


def black_scholes_call_VALIDATED(
    S: float, K: float, r: float, q: float, sigma: float, T: float
) -> float:
    """
    BS call WITH proper input validation [T1].

    Handles edge cases gracefully and raises clear errors for invalid inputs.
    """
    # Validate inputs
    validate_bs_inputs(S, K, r, q, sigma, T)

    # Handle T=0 edge case (at expiry)
    if T == 0:
        return max(0.0, S - K)

    # Handle extreme ITM (avoid numerical issues)
    if S / K > 10:  # Very deep ITM
        # Call approaches forward - strike
        return max(0.0, S * np.exp(-q * T) - K * np.exp(-r * T))

    # Handle extreme OTM (avoid underflow)
    if S / K < 0.1:  # Very deep OTM
        return 0.0  # Essentially worthless

    # Standard BS formula
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    # Final sanity check
    if math.isnan(price) or math.isinf(price):
        raise OptionPricingError(
            f"Numerical error in BS calculation: price={price}. "
            f"Inputs: S={S}, K={K}, r={r}, q={q}, sigma={sigma}, T={T}"
        )

    return price


# =============================================================================
# DEMONSTRATION
# =============================================================================


def test_edge_case(name: str, S: float, K: float, r: float, q: float, sigma: float, T: float) -> None:
    """Test an edge case with both implementations."""
    print(f"\n{name}")
    print("-" * 60)

    # Try without validation
    try:
        result_no_val = black_scholes_call_NO_VALIDATION(S, K, r, q, sigma, T)
        print(f"  No validation: ${result_no_val:.6f}")
        if math.isnan(result_no_val):
            print("    STATUS: NaN (silent failure)")
        elif math.isinf(result_no_val):
            print("    STATUS: Inf (silent failure)")
        elif result_no_val < 0:
            print("    STATUS: Negative price (impossible)")
    except Exception as e:
        print(f"  No validation: CRASHED - {type(e).__name__}: {e}")

    # Try with validation
    try:
        result_val = black_scholes_call_VALIDATED(S, K, r, q, sigma, T)
        print(f"  With validation: ${result_val:.6f}")
    except OptionPricingError as e:
        print(f"  With validation: REJECTED - {e}")
    except Exception as e:
        print(f"  With validation: ERROR - {type(e).__name__}: {e}")


if __name__ == "__main__":
    print("=" * 70)
    print("Failure Example 06: Edge Cases and Invalid Parameters")
    print("=" * 70)
    print()
    print("Testing various edge cases to show importance of input validation.")

    # Normal case (baseline)
    test_edge_case(
        "1. Normal case (baseline)",
        S=100, K=100, r=0.05, q=0.02, sigma=0.20, T=1.0
    )

    # Negative volatility
    test_edge_case(
        "2. Negative volatility (sigma < 0)",
        S=100, K=100, r=0.05, q=0.02, sigma=-0.20, T=1.0
    )

    # Zero volatility
    test_edge_case(
        "3. Zero volatility (sigma = 0)",
        S=100, K=100, r=0.05, q=0.02, sigma=0.0, T=1.0
    )

    # Negative time
    test_edge_case(
        "4. Negative time (T < 0)",
        S=100, K=100, r=0.05, q=0.02, sigma=0.20, T=-0.5
    )

    # Zero time (at expiry)
    test_edge_case(
        "5. Zero time / At expiry (T = 0)",
        S=100, K=100, r=0.05, q=0.02, sigma=0.20, T=0.0
    )

    # Negative spot
    test_edge_case(
        "6. Negative spot (S < 0)",
        S=-100, K=100, r=0.05, q=0.02, sigma=0.20, T=1.0
    )

    # Zero strike
    test_edge_case(
        "7. Zero strike (K = 0)",
        S=100, K=0, r=0.05, q=0.02, sigma=0.20, T=1.0
    )

    # Extreme volatility
    test_edge_case(
        "8. Extreme volatility (sigma = 1000%)",
        S=100, K=100, r=0.05, q=0.02, sigma=10.0, T=1.0
    )

    # Very deep ITM
    test_edge_case(
        "9. Very deep ITM (S >> K)",
        S=1000, K=1, r=0.05, q=0.02, sigma=0.20, T=1.0
    )

    # Very deep OTM
    test_edge_case(
        "10. Very deep OTM (S << K)",
        S=1, K=1000, r=0.05, q=0.02, sigma=0.20, T=1.0
    )

    print()
    print("=" * 70)
    print("KEY LESSONS:")
    print("  1. ALWAYS validate inputs before calculation")
    print("  2. Handle edge cases explicitly (T=0, deep ITM/OTM)")
    print("  3. Raise clear errors, don't return NaN/Inf silently")
    print("  4. Test boundary conditions in your test suite")
    print("=" * 70)
