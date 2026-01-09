"""
Data Loaders - Phase 10.

Enhanced data loading for:
- Yield curves (construction, interpolation, Nelson-Siegel)
- Mortality tables (SOA 2012 IAM, Gompertz, improvements)

See: docs/CROSS_VALIDATION_MATRIX.md
"""

from .mortality import (
    MortalityLoader,
    MortalityTable,
    calculate_annuity_pv,
    compare_life_expectancy,
)
from .yield_curve import (
    InterpolationMethod,
    NelsonSiegelParams,
    YieldCurve,
    YieldCurveLoader,
    calculate_duration,
    fit_nelson_siegel,
)

__all__ = [
    # Yield Curve
    "YieldCurve",
    "YieldCurveLoader",
    "NelsonSiegelParams",
    "InterpolationMethod",
    "fit_nelson_siegel",
    "calculate_duration",
    # Mortality
    "MortalityTable",
    "MortalityLoader",
    "compare_life_expectancy",
    "calculate_annuity_pv",
]
