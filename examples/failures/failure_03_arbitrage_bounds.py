"""
Failure Example 03: Arbitrage Bounds Violation
===============================================

WHAT GOES WRONG
---------------
Option prices that violate no-arbitrage bounds, such as:
- Call price > Spot price
- Put price > Discounted strike
- Call price < Intrinsic value

WHY IT'S WRONG
--------------
Arbitrage bounds are fundamental constraints from finance theory [T1]:

For a European call:
- C <= S (can't pay more than owning stock)
- C >= max(0, S*exp(-qT) - K*exp(-rT)) (intrinsic value lower bound)

For a European put:
- P <= K*exp(-rT) (can't pay more than getting K at expiry)
- P >= max(0, K*exp(-rT) - S*exp(-qT)) (intrinsic value lower bound)

Violating these bounds means:
1. Implementation has a bug
2. Results are nonsensical
3. Arbitrage opportunities exist in your model
4. Any hedging strategy is invalid

THE FIX
-------
1. Check time scaling (T in years, not days/months)
2. Verify all inputs are positive
3. Check sqrt(T) vs T usage
4. Add bounds checks as validation gates

VALIDATION
----------
All option prices must satisfy no-arbitrage bounds.

References
----------
[T1] Hull, J. C. (2018). Options, Futures, and Other Derivatives, 10th Ed.
     Chapter 11: Properties of Stock Options.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


# =============================================================================
# THE WRONG WAY (causes bound violations)
# =============================================================================


def black_scholes_call_WRONG_time_scaling(
    S: float, K: float, r: float, q: float, sigma: float, T_days: float
) -> float:
    """
    Wrong: Treating T as days without converting to years.

    This causes massive overpricing because sigma*sqrt(T) is huge.
    """
    # WRONG: T is in days but formula expects years!
    T = T_days  # Should be T_days / 365

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def black_scholes_call_WRONG_vol_squared(
    S: float, K: float, r: float, q: float, sigma: float, T: float
) -> float:
    """
    Wrong: Using sigma instead of sigma^2 in d1 numerator.

    This creates incorrect d1 values that can violate bounds.
    """
    # WRONG: should be sigma**2 not sigma
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


# =============================================================================
# THE RIGHT WAY
# =============================================================================


def black_scholes_call_CORRECT(
    S: float, K: float, r: float, q: float, sigma: float, T: float
) -> float:
    """Correct Black-Scholes call price [T1]."""
    if T <= 0:
        return max(0.0, S * np.exp(-q * T) - K)  # Intrinsic at expiry

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def black_scholes_put_CORRECT(
    S: float, K: float, r: float, q: float, sigma: float, T: float
) -> float:
    """Correct Black-Scholes put price [T1]."""
    if T <= 0:
        return max(0.0, K - S * np.exp(-q * T))  # Intrinsic at expiry

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def check_arbitrage_bounds(
    S: float, K: float, r: float, q: float, T: float, call_price: float, put_price: float
) -> dict:
    """
    Check if option prices satisfy no-arbitrage bounds [T1].

    Returns dict with bound checks and violations.
    """
    results = {}

    # Forward price components
    forward_spot = S * np.exp(-q * T)
    discounted_strike = K * np.exp(-r * T)

    # Call bounds
    call_upper = S  # C <= S
    call_lower = max(0, forward_spot - discounted_strike)  # C >= intrinsic

    results["call_upper_bound"] = call_upper
    results["call_lower_bound"] = call_lower
    results["call_upper_violation"] = call_price > call_upper
    results["call_lower_violation"] = call_price < call_lower

    # Put bounds
    put_upper = discounted_strike  # P <= K*exp(-rT)
    put_lower = max(0, discounted_strike - forward_spot)  # P >= intrinsic

    results["put_upper_bound"] = put_upper
    results["put_lower_bound"] = put_lower
    results["put_upper_violation"] = put_price > put_upper
    results["put_lower_violation"] = put_price < put_lower

    results["any_violation"] = any(
        [
            results["call_upper_violation"],
            results["call_lower_violation"],
            results["put_upper_violation"],
            results["put_lower_violation"],
        ]
    )

    return results


# =============================================================================
# DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    # Parameters
    S = 100.0  # Spot
    K = 100.0  # Strike (ATM)
    r = 0.05  # Risk-free rate
    q = 0.02  # Dividend yield
    sigma = 0.20  # Volatility
    T_years = 1.0  # Time in years
    T_days = 365.0  # Same time expressed in days

    print("=" * 70)
    print("Failure Example 03: Arbitrage Bounds Violation")
    print("=" * 70)
    print()
    print("Parameters:")
    print(f"  S = {S}, K = {K}, r = {r:.2%}, q = {q:.2%}, σ = {sigma:.2%}, T = {T_years} year")
    print()

    # Correct implementation
    call_correct = black_scholes_call_CORRECT(S, K, r, q, sigma, T_years)
    put_correct = black_scholes_put_CORRECT(S, K, r, q, sigma, T_years)
    bounds_correct = check_arbitrage_bounds(S, K, r, q, T_years, call_correct, put_correct)

    print("CORRECT Implementation:")
    print(f"  Call = ${call_correct:.4f}")
    print(f"  Put  = ${put_correct:.4f}")
    print(f"  Call bounds: [{bounds_correct['call_lower_bound']:.4f}, {bounds_correct['call_upper_bound']:.4f}]")
    print(f"  Put bounds:  [{bounds_correct['put_lower_bound']:.4f}, {bounds_correct['put_upper_bound']:.4f}]")
    print(f"  Arbitrage violation: {bounds_correct['any_violation']}")
    print()

    # Wrong: time in days instead of years
    print("WRONG Implementation #1: Time in days (not years)")
    print("-" * 70)
    call_wrong_time = black_scholes_call_WRONG_time_scaling(S, K, r, q, sigma, T_days)
    print(f"  Call = ${call_wrong_time:.4f}")
    print(f"  Upper bound (S) = ${S:.4f}")

    if call_wrong_time > S:
        print(f"  VIOLATION: Call (${call_wrong_time:.2f}) > Spot (${S:.2f})")
        print("  This creates a risk-free arbitrage: sell the call, buy the stock!")
    print()

    # Wrong: sigma instead of sigma^2
    print("WRONG Implementation #2: Using sigma instead of sigma^2")
    print("-" * 70)
    call_wrong_vol = black_scholes_call_WRONG_vol_squared(S, K, r, q, sigma, T_years)
    print(f"  Call = ${call_wrong_vol:.4f}")
    print(f"  Correct = ${call_correct:.4f}")
    print(f"  Difference = ${abs(call_wrong_vol - call_correct):.4f}")
    print("  (May not always violate bounds but gives wrong prices)")
    print()

    # Extreme case: very high volatility with time error
    print("EXTREME CASE: High vol + time error")
    print("-" * 70)
    sigma_high = 0.80
    call_extreme = black_scholes_call_WRONG_time_scaling(S, K, r, q, sigma_high, T_days)
    print(f"  With σ={sigma_high:.0%} and T={T_days} days (not years):")
    print(f"  Call = ${call_extreme:.4f}")

    if call_extreme > S:
        excess = call_extreme - S
        print(f"  MASSIVE VIOLATION: Call exceeds spot by ${excess:.2f}")
        print("  This is obviously nonsensical!")

    print()
    print("=" * 70)
    print("KEY LESSON:")
    print("  Always verify option prices against arbitrage bounds:")
    print("    - Call <= Spot")
    print("    - Put <= Discounted Strike")
    print("    - Both >= Intrinsic Value")
    print()
    print("  Most common cause: Time T not in years (e.g., days, months)")
    print("=" * 70)
