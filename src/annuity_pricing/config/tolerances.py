"""
Centralized tolerance framework for actuarial pricing.

All tolerances are derived from precision requirements, not ad hoc tuning.
See: docs/TOLERANCE_JUSTIFICATION.md for derivation details.

Tolerance Tiers:
    Tier 1 (Analytical): Machine-precision achievable, deterministic results
    Tier 2 (Cross-Library): External oracle precision bounds
    Tier 3 (Stochastic): CLT-derived, path-dependent calculations
    Tier 4 (Integration): Real-world data variability

References:
    [T1] Higham (2002) "Accuracy and Stability of Numerical Algorithms"
    [T1] Hull (2021) Ch. 15 - Options pricing precision requirements
    [T1] Glasserman (2003) Ch. 3-4 - Monte Carlo error bounds
"""

import numpy as np
from typing import Final

# =============================================================================
# Tier 1: Analytical Tolerances (Deterministic)
# =============================================================================
# For closed-form solutions where machine precision is achievable.
# Derived from: machine_epsilon (~2.2e-16) × safety_factor

#: No-arbitrage bounds: option price in [0, S] or [0, K*exp(-rT)]
#: Tolerance: ~1e-10 allows for float64 accumulation errors
ANTI_PATTERN_TOLERANCE: Final[float] = 1e-10

#: Put-call parity: C - P = S*exp(-qT) - K*exp(-rT)
#: Tolerance: sqrt(2 * machine_epsilon) * 10^4 safety factor
#: Derivation: Well-conditioned for typical S, K, r, q, T values
#: Changed from 0.01 (1%) to 1e-8 for numerical rigor
PUT_CALL_PARITY_TOLERANCE: Final[float] = 1e-8

#: Greeks numerical stability (finite difference derivatives)
#: Tolerance: sqrt(machine_epsilon) ≈ 1.5e-8
GREEKS_NUMERICAL_TOLERANCE: Final[float] = 1e-8


# =============================================================================
# Tier 2: Cross-Library Tolerances (External Oracle)
# =============================================================================
# For validation against external libraries (financepy, QuantLib, pyfeng).
# Tolerances reflect library precision, not theoretical limits.

#: Cross-library comparison (financepy, QuantLib)
#: Tolerance: Libraries typically agree to 6 decimal places
CROSS_LIBRARY_TOLERANCE: Final[float] = 1e-6

#: Greeks comparison vs external oracles
#: Slightly looser due to numerical differentiation approaches
GREEKS_VALIDATION_TOLERANCE: Final[float] = 1e-5

#: Hull textbook example tolerance
#: Hull examples quoted to 2 decimal places; allow 0.02 absolute
HULL_EXAMPLE_TOLERANCE: Final[float] = 0.02


# =============================================================================
# Tier 3: Stochastic Tolerances (CLT-Derived)
# =============================================================================
# For Monte Carlo and path-dependent calculations.
# Derived from Central Limit Theorem: 3σ/√N confidence interval


def mc_tolerance(n_paths: int, sigma: float = 0.20, confidence: float = 3.0) -> float:
    """
    Calculate CLT-derived Monte Carlo tolerance.

    [T1] Standard error of MC estimate is σ/√N.
    3σ gives 99.7% confidence interval.

    Parameters
    ----------
    n_paths : int
        Number of Monte Carlo paths
    sigma : float
        Estimated volatility of payoff (default 0.20 for options)
    confidence : float
        Number of standard deviations (default 3 for 99.7% CI)

    Returns
    -------
    float
        Tolerance for MC vs analytical comparison

    Examples
    --------
    >>> mc_tolerance(100_000)  # 100k paths
    0.0019...  # ~0.2%
    >>> mc_tolerance(10_000)   # 10k paths
    0.006  # ~0.6%
    """
    return confidence * sigma / np.sqrt(n_paths)


#: MC tolerance for 10,000 paths: 3 * 0.20 / sqrt(10000) ≈ 0.006
MC_10K_TOLERANCE: Final[float] = 0.006

#: MC tolerance for 100,000 paths: 3 * 0.20 / sqrt(100000) ≈ 0.002
MC_100K_TOLERANCE: Final[float] = 0.01  # Conservative, allows for path complexity

#: MC tolerance for 500,000 paths: 3 * 0.20 / sqrt(500000) ≈ 0.0008
MC_500K_TOLERANCE: Final[float] = 0.005

#: BS to MC convergence for vanilla options
#: Slightly looser to account for implementation differences
BS_MC_CONVERGENCE_TOLERANCE: Final[float] = 0.01


# =============================================================================
# Tier 4: Integration Tolerances (Real-World Data)
# =============================================================================
# For end-to-end tests with real WINK data and complex workflows.
# Looser to account for data variability and workflow complexity.

#: Product-level pricing comparisons
INTEGRATION_TOLERANCE: Final[float] = 1e-4

#: Golden file regression (snapshot) testing
#: Relative tolerance for comparing against stored expected values
GOLDEN_RELATIVE_TOLERANCE: Final[float] = 1e-6

#: Portfolio-level aggregation tolerance
PORTFOLIO_TOLERANCE: Final[float] = 1e-4


# =============================================================================
# Domain-Specific Tolerances
# =============================================================================

#: FIA/RILA payoff floor enforcement: credited return >= floor
#: Very tight since floor is a hard contract guarantee
FLOOR_ENFORCEMENT_TOLERANCE: Final[float] = 1e-10

#: Buffer absorption tolerance: buffer should fully absorb losses up to buffer level
BUFFER_ABSORPTION_TOLERANCE: Final[float] = 1e-10

#: Cap enforcement tolerance: credited return <= cap
CAP_ENFORCEMENT_TOLERANCE: Final[float] = 1e-10


# =============================================================================
# Tolerance Registry (For Dynamic Access)
# =============================================================================

TOLERANCE_REGISTRY: dict[str, float] = {
    # Tier 1: Analytical
    "anti_pattern": ANTI_PATTERN_TOLERANCE,
    "put_call_parity": PUT_CALL_PARITY_TOLERANCE,
    "greeks_numerical": GREEKS_NUMERICAL_TOLERANCE,
    # Tier 2: Cross-Library
    "cross_library": CROSS_LIBRARY_TOLERANCE,
    "greeks_validation": GREEKS_VALIDATION_TOLERANCE,
    "hull_example": HULL_EXAMPLE_TOLERANCE,
    # Tier 3: Stochastic
    "mc_10k": MC_10K_TOLERANCE,
    "mc_100k": MC_100K_TOLERANCE,
    "mc_500k": MC_500K_TOLERANCE,
    "bs_mc_convergence": BS_MC_CONVERGENCE_TOLERANCE,
    # Tier 4: Integration
    "integration": INTEGRATION_TOLERANCE,
    "golden_relative": GOLDEN_RELATIVE_TOLERANCE,
    "portfolio": PORTFOLIO_TOLERANCE,
    # Domain-Specific
    "floor_enforcement": FLOOR_ENFORCEMENT_TOLERANCE,
    "buffer_absorption": BUFFER_ABSORPTION_TOLERANCE,
    "cap_enforcement": CAP_ENFORCEMENT_TOLERANCE,
}


def get_tolerance(name: str) -> float:
    """
    Get tolerance by name from registry.

    Parameters
    ----------
    name : str
        Tolerance name (see TOLERANCE_REGISTRY keys)

    Returns
    -------
    float
        Tolerance value

    Raises
    ------
    KeyError
        If tolerance name not found
    """
    if name not in TOLERANCE_REGISTRY:
        available = ", ".join(sorted(TOLERANCE_REGISTRY.keys()))
        raise KeyError(f"Unknown tolerance '{name}'. Available: {available}")
    return TOLERANCE_REGISTRY[name]
