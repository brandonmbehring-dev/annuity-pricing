"""
State guaranty association coverage for life insurance and annuities.

[T2] Based on NOLHGA (National Organization of Life & Health Insurance
Guaranty Associations) published limits and state-specific statutes.

Key coverage limits (typical):
- Life insurance death benefits: $300,000
- Annuity benefits (deferred): $250,000
- Annuity benefits (in payout): $300,000
- Group annuities: $5,000,000

References
----------
[T2] NOLHGA. "How You're Protected."
     https://nolhga.com/policyholders/how-youre-protected/
[T2] State guaranty association websites and statutes
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional


class CoverageType(Enum):
    """Types of insurance/annuity coverage."""

    LIFE_DEATH_BENEFIT = "life_death_benefit"
    LIFE_CASH_VALUE = "life_cash_value"
    ANNUITY_DEFERRED = "annuity_deferred"
    ANNUITY_PAYOUT = "annuity_payout"
    ANNUITY_STRUCTURED_SETTLEMENT = "annuity_ssa"
    GROUP_ANNUITY = "group_annuity"
    HEALTH = "health"


@dataclass(frozen=True)
class GuarantyFundCoverage:
    """
    State guaranty fund coverage limits.

    Attributes
    ----------
    state : str
        Two-letter state code (e.g., "CA", "NY", "TX")
    life_death_benefit : float
        Maximum coverage for life insurance death benefits
    life_cash_value : float
        Maximum coverage for life insurance cash surrender value
    annuity_deferred : float
        Maximum coverage for deferred annuity benefits
    annuity_payout : float
        Maximum coverage for annuity in payout status
    annuity_ssa : float
        Maximum coverage for structured settlement annuities
    group_annuity : float
        Maximum coverage for group/unallocated annuities
    health : float
        Maximum coverage for health insurance (if applicable)
    coverage_percentage : float
        Percentage of benefits covered (default 100%, CA uses 80%)
    """

    state: str
    life_death_benefit: float
    life_cash_value: float
    annuity_deferred: float
    annuity_payout: float
    annuity_ssa: float
    group_annuity: float
    health: float
    coverage_percentage: float = 1.0  # 100%


# Standard NOLHGA limits (used as default for most states)
_STANDARD_LIMITS = GuarantyFundCoverage(
    state="DEFAULT",
    life_death_benefit=300_000,
    life_cash_value=100_000,
    annuity_deferred=250_000,
    annuity_payout=300_000,
    annuity_ssa=250_000,
    group_annuity=5_000_000,
    health=500_000,
    coverage_percentage=1.0,
)


# State-specific limits where they differ from standard
# [T2] Based on NOLHGA and state guaranty association websites
STATE_GUARANTY_LIMITS: Dict[str, GuarantyFundCoverage] = {
    # California - 80% of benefits
    "CA": GuarantyFundCoverage(
        state="CA",
        life_death_benefit=300_000,
        life_cash_value=100_000,
        annuity_deferred=250_000,
        annuity_payout=300_000,
        annuity_ssa=250_000,
        group_annuity=5_000_000,
        health=668_205,  # Inflation-adjusted as of 2024
        coverage_percentage=0.80,  # California covers 80%
    ),
    # New York - higher health limits
    "NY": GuarantyFundCoverage(
        state="NY",
        life_death_benefit=500_000,
        life_cash_value=100_000,
        annuity_deferred=500_000,
        annuity_payout=500_000,
        annuity_ssa=500_000,
        group_annuity=5_000_000,
        health=500_000,
        coverage_percentage=1.0,
    ),
    # Minnesota - higher SSA limits
    "MN": GuarantyFundCoverage(
        state="MN",
        life_death_benefit=300_000,
        life_cash_value=100_000,
        annuity_deferred=250_000,
        annuity_payout=300_000,
        annuity_ssa=410_000,  # Higher for SSA and 10yr+ certain
        group_annuity=5_000_000,
        health=500_000,
        coverage_percentage=1.0,
    ),
    # North Carolina - higher SSA limits
    "NC": GuarantyFundCoverage(
        state="NC",
        life_death_benefit=300_000,
        life_cash_value=100_000,
        annuity_deferred=300_000,
        annuity_payout=300_000,
        annuity_ssa=1_000_000,  # $1M for SSA
        group_annuity=5_000_000,
        health=500_000,
        coverage_percentage=1.0,
    ),
    # Texas - standard limits
    "TX": GuarantyFundCoverage(
        state="TX",
        life_death_benefit=300_000,
        life_cash_value=100_000,
        annuity_deferred=250_000,
        annuity_payout=300_000,
        annuity_ssa=250_000,
        group_annuity=5_000_000,
        health=500_000,
        coverage_percentage=1.0,
    ),
    # Florida - standard limits
    "FL": GuarantyFundCoverage(
        state="FL",
        life_death_benefit=300_000,
        life_cash_value=100_000,
        annuity_deferred=250_000,
        annuity_payout=300_000,
        annuity_ssa=250_000,
        group_annuity=5_000_000,
        health=500_000,
        coverage_percentage=1.0,
    ),
    # Washington - higher annuity limits
    "WA": GuarantyFundCoverage(
        state="WA",
        life_death_benefit=500_000,
        life_cash_value=100_000,
        annuity_deferred=500_000,
        annuity_payout=500_000,
        annuity_ssa=500_000,
        group_annuity=5_000_000,
        health=500_000,
        coverage_percentage=1.0,
    ),
    # New Jersey - standard limits
    "NJ": GuarantyFundCoverage(
        state="NJ",
        life_death_benefit=500_000,
        life_cash_value=100_000,
        annuity_deferred=500_000,
        annuity_payout=500_000,
        annuity_ssa=500_000,
        group_annuity=5_000_000,
        health=500_000,
        coverage_percentage=1.0,
    ),
}


def get_state_coverage(state: str) -> GuarantyFundCoverage:
    """
    Get guaranty fund coverage limits for a state.

    [T2] Returns state-specific limits if available, else standard limits.

    Parameters
    ----------
    state : str
        Two-letter state code (e.g., "CA", "NY")

    Returns
    -------
    GuarantyFundCoverage
        Coverage limits for the state

    Examples
    --------
    >>> coverage = get_state_coverage("CA")
    >>> coverage.annuity_deferred
    250000
    >>> coverage.coverage_percentage
    0.8
    """
    state = state.upper().strip()

    if state in STATE_GUARANTY_LIMITS:
        return STATE_GUARANTY_LIMITS[state]

    # Return standard limits with state code
    return GuarantyFundCoverage(
        state=state,
        life_death_benefit=_STANDARD_LIMITS.life_death_benefit,
        life_cash_value=_STANDARD_LIMITS.life_cash_value,
        annuity_deferred=_STANDARD_LIMITS.annuity_deferred,
        annuity_payout=_STANDARD_LIMITS.annuity_payout,
        annuity_ssa=_STANDARD_LIMITS.annuity_ssa,
        group_annuity=_STANDARD_LIMITS.group_annuity,
        health=_STANDARD_LIMITS.health,
        coverage_percentage=_STANDARD_LIMITS.coverage_percentage,
    )


def calculate_covered_amount(
    benefit_amount: float,
    state: str,
    coverage_type: CoverageType,
) -> float:
    """
    Calculate amount covered by state guaranty fund.

    [T2] Returns minimum of benefit and state limit, adjusted by coverage %.

    Parameters
    ----------
    benefit_amount : float
        Total benefit/contract value
    state : str
        Two-letter state code
    coverage_type : CoverageType
        Type of coverage

    Returns
    -------
    float
        Amount covered by guaranty fund

    Examples
    --------
    >>> calculate_covered_amount(300000, "CA", CoverageType.ANNUITY_DEFERRED)
    200000.0  # 80% of $250k limit
    >>> calculate_covered_amount(100000, "TX", CoverageType.ANNUITY_DEFERRED)
    100000.0  # Full amount (under limit)
    """
    coverage = get_state_coverage(state)

    # Get applicable limit
    limit_map = {
        CoverageType.LIFE_DEATH_BENEFIT: coverage.life_death_benefit,
        CoverageType.LIFE_CASH_VALUE: coverage.life_cash_value,
        CoverageType.ANNUITY_DEFERRED: coverage.annuity_deferred,
        CoverageType.ANNUITY_PAYOUT: coverage.annuity_payout,
        CoverageType.ANNUITY_STRUCTURED_SETTLEMENT: coverage.annuity_ssa,
        CoverageType.GROUP_ANNUITY: coverage.group_annuity,
        CoverageType.HEALTH: coverage.health,
    }

    limit = limit_map[coverage_type]

    # Apply limit
    capped = min(benefit_amount, limit)

    # Apply coverage percentage
    return capped * coverage.coverage_percentage


def calculate_uncovered_amount(
    benefit_amount: float,
    state: str,
    coverage_type: CoverageType,
) -> float:
    """
    Calculate amount NOT covered by state guaranty fund.

    This is the amount exposed to insurer credit risk.

    Parameters
    ----------
    benefit_amount : float
        Total benefit/contract value
    state : str
        Two-letter state code
    coverage_type : CoverageType
        Type of coverage

    Returns
    -------
    float
        Amount at credit risk (not covered by guaranty)

    Examples
    --------
    >>> calculate_uncovered_amount(500000, "TX", CoverageType.ANNUITY_DEFERRED)
    250000.0  # $500k - $250k limit
    >>> calculate_uncovered_amount(100000, "TX", CoverageType.ANNUITY_DEFERRED)
    0.0  # Fully covered
    """
    covered = calculate_covered_amount(benefit_amount, state, coverage_type)
    return max(0, benefit_amount - covered)


def get_coverage_ratio(
    benefit_amount: float,
    state: str,
    coverage_type: CoverageType,
) -> float:
    """
    Get ratio of benefit covered by guaranty fund.

    Parameters
    ----------
    benefit_amount : float
        Total benefit/contract value
    state : str
        Two-letter state code
    coverage_type : CoverageType
        Type of coverage

    Returns
    -------
    float
        Coverage ratio (0 to 1)
    """
    if benefit_amount <= 0:
        return 0.0

    covered = calculate_covered_amount(benefit_amount, state, coverage_type)
    return covered / benefit_amount
