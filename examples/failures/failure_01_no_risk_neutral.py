"""
Failure Example 01: No Risk-Neutral Drift
==========================================

WHAT GOES WRONG
---------------
Using real-world (historical) drift mu instead of risk-free rate r
when simulating paths for option pricing.

WHY IT'S WRONG
--------------
Option pricing uses RISK-NEUTRAL valuation [T1]. Under the Q-measure:
- Expected growth rate = r - q (risk-free minus dividend yield)
- NOT historical average returns

Using real-world drift gives wrong prices because:
1. Options are priced by arbitrage, not expectation
2. The martingale property requires r - q drift
3. Historical drift varies; risk-neutral drift is deterministic

THE FIX
-------
Always use risk_free_rate - dividend_yield as GBM drift for pricing.

VALIDATION
----------
With correct drift:
- Put-call parity holds: C - P = S - K*exp(-rT)
- Monte Carlo converges to Black-Scholes
- Price is independent of investor risk preferences

References
----------
[T1] Black, F., & Scholes, M. (1973). The Pricing of Options and Corporate
     Liabilities. Journal of Political Economy, 81(3), 637-654.
[T1] Hull, J. C. (2018). Options, Futures, and Other Derivatives, 10th Ed.
"""

from __future__ import annotations

import numpy as np

# =============================================================================
# THE WRONG WAY
# =============================================================================


def simulate_paths_WRONG(
    spot: float,
    historical_drift: float,  # Using historical returns - WRONG!
    volatility: float,
    time: float,
    n_paths: int,
    n_steps: int,
    seed: int = 42,
) -> np.ndarray:
    """
    Simulate GBM paths using historical drift.

    THIS IS WRONG FOR OPTION PRICING!
    """
    rng = np.random.default_rng(seed)
    dt = time / n_steps

    # Wrong: using historical drift instead of risk-neutral drift
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = spot

    for t in range(n_steps):
        Z = rng.standard_normal(n_paths)
        paths[:, t + 1] = paths[:, t] * np.exp(
            (historical_drift - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * Z
        )

    return paths


def price_call_wrong(
    spot: float,
    strike: float,
    historical_drift: float,
    volatility: float,
    time: float,
    risk_free_rate: float,
    n_paths: int = 100_000,
) -> float:
    """Price call option with wrong drift - gives incorrect results."""
    paths = simulate_paths_WRONG(
        spot=spot,
        historical_drift=historical_drift,  # WRONG!
        volatility=volatility,
        time=time,
        n_paths=n_paths,
        n_steps=252,
    )

    # Payoffs at maturity
    payoffs = np.maximum(paths[:, -1] - strike, 0)

    # Discount back (correctly, but drift was wrong)
    return np.exp(-risk_free_rate * time) * payoffs.mean()


# =============================================================================
# THE RIGHT WAY
# =============================================================================


def simulate_paths_CORRECT(
    spot: float,
    risk_free_rate: float,  # Use risk-free rate
    dividend_yield: float,  # Subtract dividend yield
    volatility: float,
    time: float,
    n_paths: int,
    n_steps: int,
    seed: int = 42,
) -> np.ndarray:
    """
    Simulate GBM paths using risk-neutral drift.

    Under Q-measure: drift = r - q [T1]
    """
    rng = np.random.default_rng(seed)
    dt = time / n_steps

    # Correct: risk-neutral drift
    drift = risk_free_rate - dividend_yield

    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = spot

    for t in range(n_steps):
        Z = rng.standard_normal(n_paths)
        paths[:, t + 1] = paths[:, t] * np.exp(
            (drift - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * Z
        )

    return paths


def price_call_correct(
    spot: float,
    strike: float,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
    time: float,
    n_paths: int = 100_000,
) -> float:
    """Price call option with correct risk-neutral drift."""
    paths = simulate_paths_CORRECT(
        spot=spot,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        volatility=volatility,
        time=time,
        n_paths=n_paths,
        n_steps=252,
    )

    payoffs = np.maximum(paths[:, -1] - strike, 0)
    return np.exp(-risk_free_rate * time) * payoffs.mean()


# =============================================================================
# DEMONSTRATION
# =============================================================================


def black_scholes_call(
    spot: float,
    strike: float,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
    time: float,
) -> float:
    """Analytical Black-Scholes call price for comparison [T1]."""
    from scipy.stats import norm

    d1 = (np.log(spot / strike) + (risk_free_rate - dividend_yield + 0.5 * volatility**2) * time) / (
        volatility * np.sqrt(time)
    )
    d2 = d1 - volatility * np.sqrt(time)

    return spot * np.exp(-dividend_yield * time) * norm.cdf(d1) - strike * np.exp(
        -risk_free_rate * time
    ) * norm.cdf(d2)


if __name__ == "__main__":
    # Market parameters
    S = 100.0  # Spot price
    K = 100.0  # Strike (ATM)
    r = 0.05  # Risk-free rate
    q = 0.02  # Dividend yield
    sigma = 0.20  # Volatility
    T = 1.0  # Time to maturity

    # Historical drift (often higher than risk-free for equities)
    historical_mu = 0.10  # 10% historical average return

    print("=" * 60)
    print("Failure Example 01: No Risk-Neutral Drift")
    print("=" * 60)
    print()
    print("Parameters:")
    print(f"  Spot (S)         = {S}")
    print(f"  Strike (K)       = {K}")
    print(f"  Risk-free (r)    = {r:.2%}")
    print(f"  Dividend (q)     = {q:.2%}")
    print(f"  Volatility (σ)   = {sigma:.2%}")
    print(f"  Time (T)         = {T} year")
    print(f"  Historical drift = {historical_mu:.2%}")
    print()

    # Analytical (correct) price
    bs_price = black_scholes_call(S, K, r, q, sigma, T)

    # Monte Carlo with WRONG drift
    wrong_price = price_call_wrong(S, K, historical_mu, sigma, T, r, n_paths=100_000)

    # Monte Carlo with CORRECT drift
    correct_price = price_call_correct(S, K, r, q, sigma, T, n_paths=100_000)

    print("Results:")
    print("-" * 60)
    print(f"  Black-Scholes (analytical)     = ${bs_price:.4f}")
    print(f"  MC with historical drift       = ${wrong_price:.4f}")
    print(f"  MC with risk-neutral drift     = ${correct_price:.4f}")
    print()

    wrong_error = abs(wrong_price - bs_price)
    correct_error = abs(correct_price - bs_price)

    print("Errors vs Black-Scholes:")
    print(f"  Wrong drift error   = ${wrong_error:.4f} ({wrong_error/bs_price:.2%})")
    print(f"  Correct drift error = ${correct_error:.4f} ({correct_error/bs_price:.2%})")
    print()

    if wrong_error > correct_error * 5:
        print("VALIDATION: Wrong drift gives significantly incorrect price")
        print("            Correct drift matches Black-Scholes within MC variance")
    else:
        print("WARNING: Expected larger error from wrong drift")

    print()
    print("KEY LESSON:")
    print("  Always use r - q as drift for option pricing, NOT historical returns.")
    print("  Option prices are determined by arbitrage, not expectations.")
