"""
Failure Example 05: Monte Carlo Divergence from Black-Scholes
==============================================================

WHAT GOES WRONG
---------------
Monte Carlo prices that don't converge to Black-Scholes for vanilla
European options, indicating implementation bugs.

WHY IT'S WRONG
--------------
For vanilla European options, MC MUST converge to BS [T1]:
- Same underlying model (GBM)
- Same risk-neutral measure
- Same payoff function

Common causes of divergence:
1. Wrong drift (using mu instead of r-q)
2. Wrong discretization (Euler vs exact)
3. Insufficient paths
4. Wrong payoff calculation
5. Forgetting discounting

THE FIX
-------
1. Use exact GBM simulation (log-normal, not Euler)
2. Always use risk-neutral drift (r - q)
3. Use enough paths (typically 100k+)
4. Verify against BS for vanilla options first
5. Use variance reduction techniques

VALIDATION
----------
MC should converge to BS within statistical tolerance (~0.5-1% for 100k paths).

References
----------
[T1] Glasserman, P. (2003). Monte Carlo Methods in Financial Engineering.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


# =============================================================================
# BLACK-SCHOLES REFERENCE
# =============================================================================


def black_scholes_call(S: float, K: float, r: float, q: float, sigma: float, T: float) -> float:
    """Black-Scholes call price for reference [T1]."""
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


# =============================================================================
# THE WRONG WAYS
# =============================================================================


def mc_price_WRONG_euler(
    S: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    n_paths: int = 10_000,
    n_steps: int = 252,
    seed: int = 42,
) -> float:
    """
    WRONG: Euler discretization without Ito correction.

    Euler: S_{t+1} = S_t + mu*S_t*dt + sigma*S_t*dW
    This has O(dt) bias compared to exact solution.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    drift = r - q

    S_t = np.full(n_paths, S)

    for _ in range(n_steps):
        dW = rng.standard_normal(n_paths) * np.sqrt(dt)
        # WRONG: Simple Euler without log transform
        S_t = S_t + drift * S_t * dt + sigma * S_t * dW

    payoffs = np.maximum(S_t - K, 0)
    return np.exp(-r * T) * payoffs.mean()


def mc_price_WRONG_few_paths(
    S: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    n_paths: int = 100,  # Too few!
    seed: int = 42,
) -> float:
    """
    WRONG: Too few paths for reliable convergence.

    Standard error ~ sigma / sqrt(n_paths)
    With 100 paths, SE is 10x larger than with 10,000 paths.
    """
    rng = np.random.default_rng(seed)
    drift = r - q

    # Exact GBM (correct formula, but too few paths)
    Z = rng.standard_normal(n_paths)
    S_T = S * np.exp((drift - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    payoffs = np.maximum(S_T - K, 0)
    return np.exp(-r * T) * payoffs.mean()


def mc_price_WRONG_no_discount(
    S: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    n_paths: int = 100_000,
    seed: int = 42,
) -> float:
    """
    WRONG: Forgetting to discount payoffs.

    Option price is DISCOUNTED expected payoff, not just expected payoff!
    """
    rng = np.random.default_rng(seed)
    drift = r - q

    Z = rng.standard_normal(n_paths)
    S_T = S * np.exp((drift - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    payoffs = np.maximum(S_T - K, 0)
    # WRONG: No discounting!
    return payoffs.mean()


# =============================================================================
# THE RIGHT WAY
# =============================================================================


def mc_price_CORRECT(
    S: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    n_paths: int = 100_000,
    seed: int = 42,
) -> tuple[float, float]:
    """
    CORRECT Monte Carlo pricing [T1].

    Uses exact GBM simulation (log-normal), risk-neutral drift,
    sufficient paths, and proper discounting.

    Returns (price, standard_error).
    """
    rng = np.random.default_rng(seed)
    drift = r - q

    # Exact GBM: log(S_T) ~ N(log(S) + (r-q-0.5*sigma^2)*T, sigma^2*T)
    Z = rng.standard_normal(n_paths)
    S_T = S * np.exp((drift - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    # Payoffs
    payoffs = np.maximum(S_T - K, 0)

    # Discounted expected value
    price = np.exp(-r * T) * payoffs.mean()

    # Standard error
    se = np.exp(-r * T) * payoffs.std() / np.sqrt(n_paths)

    return price, se


# =============================================================================
# DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    # Parameters (Hull Example)
    S = 100.0
    K = 100.0
    r = 0.05
    q = 0.02
    sigma = 0.20
    T = 1.0

    print("=" * 75)
    print("Failure Example 05: Monte Carlo Divergence from Black-Scholes")
    print("=" * 75)
    print()
    print("Parameters:")
    print(f"  S = {S}, K = {K}, r = {r:.2%}, q = {q:.2%}, σ = {sigma:.2%}, T = {T}")
    print()

    # Reference
    bs_price = black_scholes_call(S, K, r, q, sigma, T)
    print(f"Black-Scholes Reference: ${bs_price:.4f}")
    print()

    # Correct MC
    mc_correct, mc_se = mc_price_CORRECT(S, K, r, q, sigma, T)
    mc_error = abs(mc_correct - bs_price)
    mc_error_pct = mc_error / bs_price * 100

    print("CORRECT Monte Carlo (100k paths, exact GBM):")
    print(f"  Price = ${mc_correct:.4f} ± ${mc_se:.4f}")
    print(f"  Error = ${mc_error:.4f} ({mc_error_pct:.2f}%)")
    print(f"  Within 2 SE of BS? {mc_error < 2 * mc_se}")
    print()

    # Wrong implementations
    print("WRONG Implementations:")
    print("-" * 75)

    # Euler discretization
    mc_euler = mc_price_WRONG_euler(S, K, r, q, sigma, T)
    euler_error = abs(mc_euler - bs_price)
    euler_error_pct = euler_error / bs_price * 100
    print(f"\n  1. Euler Discretization (no Ito correction):")
    print(f"     Price = ${mc_euler:.4f}")
    print(f"     Error = ${euler_error:.4f} ({euler_error_pct:.2f}%)")
    if euler_error > mc_error * 2:
        print("     STATUS: Significant bias from discretization error")

    # Too few paths
    mc_few = mc_price_WRONG_few_paths(S, K, r, q, sigma, T)
    few_error = abs(mc_few - bs_price)
    few_error_pct = few_error / bs_price * 100
    print(f"\n  2. Too Few Paths (100 instead of 100k):")
    print(f"     Price = ${mc_few:.4f}")
    print(f"     Error = ${few_error:.4f} ({few_error_pct:.2f}%)")
    print("     STATUS: High variance, unreliable estimate")

    # No discounting
    mc_no_disc = mc_price_WRONG_no_discount(S, K, r, q, sigma, T)
    no_disc_error = abs(mc_no_disc - bs_price)
    no_disc_error_pct = no_disc_error / bs_price * 100
    print(f"\n  3. Forgot Discounting:")
    print(f"     Price = ${mc_no_disc:.4f}")
    print(f"     Error = ${no_disc_error:.4f} ({no_disc_error_pct:.2f}%)")
    print(f"     STATUS: Overpriced by exp(rT) factor = {np.exp(r*T):.4f}")

    print()
    print("=" * 75)
    print("KEY LESSONS:")
    print("  1. Use exact GBM (log-normal), not Euler discretization")
    print("  2. Use sufficient paths (100k+ for < 1% error)")
    print("  3. Always discount payoffs: exp(-rT) * E[payoff]")
    print("  4. Verify against BS for vanilla options before using for exotics")
    print("=" * 75)
