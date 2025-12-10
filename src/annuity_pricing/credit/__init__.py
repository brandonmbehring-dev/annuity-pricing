"""
Credit risk and CVA (Credit Valuation Adjustment) module.

Provides:
- AM Best rating → probability of default mapping
- State guaranty fund coverage limits
- CVA calculation for annuity products
- Credit-adjusted pricing

[T2] Based on:
- AM Best Impairment Rate and Rating Transition Study (1977-2023)
- NOLHGA state guaranty association limits
- Industry-standard CVA methodology

See: docs/knowledge/domain/credit_risk.md (to be created)
"""

from annuity_pricing.credit.default_prob import (
    AMBestRating,
    RatingPD,
    get_annual_pd,
    get_cumulative_pd,
    rating_from_string,
)
from annuity_pricing.credit.guaranty_funds import (
    GuarantyFundCoverage,
    get_state_coverage,
    calculate_covered_amount,
)
from annuity_pricing.credit.cva import (
    CVAResult,
    calculate_cva,
    calculate_credit_adjusted_price,
)

__all__ = [
    # Default probability
    "AMBestRating",
    "RatingPD",
    "get_annual_pd",
    "get_cumulative_pd",
    "rating_from_string",
    # Guaranty funds
    "GuarantyFundCoverage",
    "get_state_coverage",
    "calculate_covered_amount",
    # CVA
    "CVAResult",
    "calculate_cva",
    "calculate_credit_adjusted_price",
]
