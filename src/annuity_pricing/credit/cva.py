"""
Credit Valuation Adjustment (CVA) for annuity products.

[T1] CVA = -LGD × Σ EE(t) × PD(t) × DF(t)

where:
- LGD = Loss Given Default (1 - recovery rate)
- EE(t) = Expected Exposure at time t
- PD(t) = Probability of default in period ending at t
- DF(t) = Discount factor to time t

For annuities, we adjust for state guaranty fund coverage:
- Adjusted CVA = CVA × (1 - coverage_ratio)

References
----------
[T1] Hull, J. (2018). Options, Futures, and Other Derivatives. Ch. 24.
[T2] Gregory, J. (2015). The xVA Challenge. Wiley.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from annuity_pricing.credit.default_prob import (
    AMBestRating,
    get_annual_pd,
    get_hazard_rate,
    rating_from_string,
)
from annuity_pricing.credit.guaranty_funds import (
    CoverageType,
    calculate_covered_amount,
    get_coverage_ratio,
)


# Industry-standard LGD for insurance companies
# [T2] Higher recovery than typical corporates due to regulatory protection
DEFAULT_INSURANCE_LGD = 0.70  # 70% LGD (30% recovery)


@dataclass
class CVAResult:
    """
    CVA calculation result.

    Attributes
    ----------
    cva_gross : float
        CVA before guaranty fund adjustment
    cva_net : float
        CVA after guaranty fund adjustment (exposure at risk)
    guaranty_adjustment : float
        Reduction in CVA due to guaranty coverage
    expected_exposure : float
        Total expected exposure
    covered_exposure : float
        Exposure covered by guaranty fund
    uncovered_exposure : float
        Exposure at credit risk
    coverage_ratio : float
        Ratio of covered to total exposure
    lgd : float
        Loss given default used
    rating : AMBestRating
        Insurer rating used
    annual_pd : float
        Annual probability of default
    """

    cva_gross: float
    cva_net: float
    guaranty_adjustment: float
    expected_exposure: float
    covered_exposure: float
    uncovered_exposure: float
    coverage_ratio: float
    lgd: float
    rating: AMBestRating
    annual_pd: float


def calculate_exposure_profile(
    principal: float,
    rate: float,
    term_years: int,
    payment_frequency: int = 1,
) -> np.ndarray:
    """
    Calculate expected exposure profile for MYGA.

    [T1] For a fixed annuity, exposure = PV of remaining guaranteed payments.
    At time t, exposure = Principal × (1 + rate)^t × remaining_factor

    Parameters
    ----------
    principal : float
        Initial principal/premium
    rate : float
        Guaranteed rate (decimal)
    term_years : int
        Contract term in years
    payment_frequency : int
        Payments per year (1 = annual, 12 = monthly)

    Returns
    -------
    np.ndarray
        Exposure at each time point (per period)
    """
    periods = term_years * payment_frequency
    exposures = np.zeros(periods)

    for t in range(periods):
        # Remaining value at time t (accumulated value still guaranteed)
        years_elapsed = t / payment_frequency
        years_remaining = term_years - years_elapsed

        # Exposure is PV of remaining guaranteed maturity value
        maturity_value = principal * (1 + rate) ** term_years
        remaining_pv = maturity_value / (1 + rate) ** years_remaining

        exposures[t] = remaining_pv

    return exposures


def calculate_cva(
    exposure: float,
    rating: AMBestRating,
    term_years: int = 1,
    lgd: float = DEFAULT_INSURANCE_LGD,
    risk_free_rate: float = 0.05,
    state: Optional[str] = None,
    coverage_type: CoverageType = CoverageType.ANNUITY_DEFERRED,
) -> CVAResult:
    """
    Calculate Credit Valuation Adjustment for annuity.

    [T1] CVA = LGD × EE × (1 - exp(-h × T))

    where h is hazard rate and T is time horizon.

    Parameters
    ----------
    exposure : float
        Expected exposure (contract value)
    rating : AMBestRating
        Insurer AM Best rating
    term_years : int
        Contract term/horizon in years
    lgd : float
        Loss given default (default 70%)
    risk_free_rate : float
        Risk-free rate for discounting (default 5%)
    state : str, optional
        State for guaranty fund adjustment (if None, no adjustment)
    coverage_type : CoverageType
        Type of coverage for guaranty limits

    Returns
    -------
    CVAResult
        CVA calculation results

    Examples
    --------
    >>> result = calculate_cva(
    ...     exposure=250000,
    ...     rating=AMBestRating.A,
    ...     term_years=5,
    ...     state="TX"
    ... )
    >>> result.cva_net  # CVA after guaranty adjustment
    """
    if exposure <= 0:
        raise ValueError(f"CRITICAL: exposure must be > 0. Got: {exposure}")
    if term_years < 1:
        raise ValueError(f"CRITICAL: term_years must be >= 1. Got: {term_years}")
    if not 0 < lgd <= 1:
        raise ValueError(f"CRITICAL: lgd must be in (0, 1]. Got: {lgd}")

    # Get hazard rate and annual PD
    annual_pd = get_annual_pd(rating)
    hazard_rate = get_hazard_rate(rating)

    # Cumulative default probability over term
    # [T1] P(default by T) = 1 - exp(-h × T)
    cum_pd = 1 - np.exp(-hazard_rate * term_years)

    # Average discount factor (simplified: use midpoint)
    avg_discount = np.exp(-risk_free_rate * term_years / 2)

    # Gross CVA (before guaranty adjustment)
    # [T1] CVA ≈ LGD × EE × PD × DF
    cva_gross = lgd * exposure * cum_pd * avg_discount

    # Apply guaranty fund adjustment if state provided
    if state is not None:
        covered = calculate_covered_amount(exposure, state, coverage_type)
        coverage_ratio = get_coverage_ratio(exposure, state, coverage_type)
        uncovered = exposure - covered

        # CVA only applies to uncovered portion
        cva_net = cva_gross * (1 - coverage_ratio)
        guaranty_adjustment = cva_gross - cva_net
    else:
        covered = 0.0
        uncovered = exposure
        coverage_ratio = 0.0
        cva_net = cva_gross
        guaranty_adjustment = 0.0

    return CVAResult(
        cva_gross=cva_gross,
        cva_net=cva_net,
        guaranty_adjustment=guaranty_adjustment,
        expected_exposure=exposure,
        covered_exposure=covered,
        uncovered_exposure=uncovered,
        coverage_ratio=coverage_ratio,
        lgd=lgd,
        rating=rating,
        annual_pd=annual_pd,
    )


def calculate_cva_term_structure(
    exposure_profile: np.ndarray,
    rating: AMBestRating,
    lgd: float = DEFAULT_INSURANCE_LGD,
    risk_free_rate: float = 0.05,
    periods_per_year: int = 1,
) -> float:
    """
    Calculate CVA with full exposure term structure.

    [T1] CVA = LGD × Σ_t EE(t) × [Q(t) - Q(t-1)] × DF(t)

    where Q(t) is cumulative default probability.

    Parameters
    ----------
    exposure_profile : np.ndarray
        Expected exposure at each period
    rating : AMBestRating
        Insurer rating
    lgd : float
        Loss given default
    risk_free_rate : float
        Risk-free rate
    periods_per_year : int
        Number of periods per year

    Returns
    -------
    float
        CVA value
    """
    hazard_rate = get_hazard_rate(rating)
    n_periods = len(exposure_profile)

    cva = 0.0
    for t in range(n_periods):
        # Time in years
        time_years = (t + 1) / periods_per_year
        time_prev = t / periods_per_year

        # Incremental default probability in period
        q_t = 1 - np.exp(-hazard_rate * time_years)
        q_prev = 1 - np.exp(-hazard_rate * time_prev)
        incremental_pd = q_t - q_prev

        # Discount factor
        df = np.exp(-risk_free_rate * time_years)

        # CVA contribution from this period
        cva += lgd * exposure_profile[t] * incremental_pd * df

    return cva


def calculate_credit_adjusted_price(
    base_price: float,
    rating: AMBestRating,
    term_years: int = 1,
    lgd: float = DEFAULT_INSURANCE_LGD,
    risk_free_rate: float = 0.05,
    state: Optional[str] = None,
    coverage_type: CoverageType = CoverageType.ANNUITY_DEFERRED,
) -> float:
    """
    Calculate credit-adjusted price for annuity.

    [T1] Credit-adjusted price = Base price - CVA

    Parameters
    ----------
    base_price : float
        Base price without credit adjustment
    rating : AMBestRating
        Insurer rating (or rating string)
    term_years : int
        Contract term
    lgd : float
        Loss given default
    risk_free_rate : float
        Risk-free rate
    state : str, optional
        State for guaranty adjustment
    coverage_type : CoverageType
        Type of coverage

    Returns
    -------
    float
        Credit-adjusted price

    Examples
    --------
    >>> # $100k annuity from A-rated insurer
    >>> adj_price = calculate_credit_adjusted_price(
    ...     base_price=100000,
    ...     rating=AMBestRating.A,
    ...     term_years=5,
    ...     state="CA"
    ... )
    """
    cva_result = calculate_cva(
        exposure=base_price,
        rating=rating,
        term_years=term_years,
        lgd=lgd,
        risk_free_rate=risk_free_rate,
        state=state,
        coverage_type=coverage_type,
    )

    return base_price - cva_result.cva_net


def calculate_credit_spread(
    rating: AMBestRating,
    lgd: float = DEFAULT_INSURANCE_LGD,
) -> float:
    """
    Calculate implied credit spread for rating.

    [T1] Credit spread ≈ hazard_rate × LGD

    Parameters
    ----------
    rating : AMBestRating
        Insurer rating
    lgd : float
        Loss given default

    Returns
    -------
    float
        Implied credit spread (decimal, per year)

    Examples
    --------
    >>> spread = calculate_credit_spread(AMBestRating.A)
    >>> print(f"{spread * 10000:.1f} bps")  # basis points
    """
    hazard_rate = get_hazard_rate(rating)
    return hazard_rate * lgd
