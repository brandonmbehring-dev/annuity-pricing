"""
Behavioral modeling for annuity pricing.

Phase 7 deliverables:
- Dynamic lapse (moneyness-based)
- GLWB withdrawal utilization
- Per-policy and % of AV expenses

Phase H deliverables (SOA calibration):
- SOA 2006 calibrated surrender curves
- SOA 2018 GLWB utilization curves
- Duration, age, and ITM-based behavior

See: docs/knowledge/domain/dynamic_lapse.md
See: docs/knowledge/domain/glwb_mechanics.md
See: docs/assumptions/BEHAVIOR_CALIBRATION.md
"""

# Original Phase 7 models (hardcoded parameters)
from .dynamic_lapse import (
    DynamicLapseModel,
    LapseAssumptions,
    LapseResult,
    CalibrationSource,
)
from .withdrawal import (
    WithdrawalModel,
    WithdrawalAssumptions,
    WithdrawalResult,
    UtilizationCalibration,
)
from .expenses import (
    ExpenseModel,
    ExpenseAssumptions,
    ExpenseResult,
)

# Phase H: SOA-calibrated models
from .dynamic_lapse import (
    SOADynamicLapseModel,
    SOALapseAssumptions,
    SOALapseResult,
)
from .withdrawal import (
    SOAWithdrawalModel,
    SOAWithdrawalAssumptions,
    SOAWithdrawalResult,
)

# SOA benchmark data
from .soa_benchmarks import (
    SOA_2006_SURRENDER_BY_DURATION_7YR_SC,
    SOA_2006_SC_CLIFF_EFFECT,
    SOA_2006_SC_CLIFF_MULTIPLIER,
    SOA_2006_FULL_SURRENDER_BY_AGE,
    SOA_2006_PARTIAL_WITHDRAWAL_BY_AGE,
    SOA_2006_POST_SC_DECAY,
    SOA_2018_GLWB_UTILIZATION_BY_DURATION,
    SOA_2018_GLWB_UTILIZATION_BY_AGE,
    SOA_2018_ITM_SENSITIVITY,
    SOA_2018_ITM_VS_NOT_ITM_BY_AGE,
    SOA_KEY_INSIGHTS,
    DATA_QUALITY_NOTES,
)

# Calibration utilities
from .calibration import (
    interpolate_surrender_by_duration,
    get_sc_cliff_multiplier,
    get_post_sc_decay_factor,
    interpolate_surrender_by_age,
    interpolate_utilization_by_duration,
    interpolate_utilization_by_age,
    get_itm_sensitivity_factor,
    get_itm_sensitivity_factor_continuous,
    combined_utilization,
    get_surrender_curve,
    get_utilization_curve,
)

__all__ = [
    # Original Dynamic Lapse (hardcoded)
    "DynamicLapseModel",
    "LapseAssumptions",
    "LapseResult",
    "CalibrationSource",
    # Original Withdrawal (hardcoded)
    "WithdrawalModel",
    "WithdrawalAssumptions",
    "WithdrawalResult",
    "UtilizationCalibration",
    # Expenses
    "ExpenseModel",
    "ExpenseAssumptions",
    "ExpenseResult",
    # SOA-Calibrated Dynamic Lapse
    "SOADynamicLapseModel",
    "SOALapseAssumptions",
    "SOALapseResult",
    # SOA-Calibrated Withdrawal
    "SOAWithdrawalModel",
    "SOAWithdrawalAssumptions",
    "SOAWithdrawalResult",
    # SOA Benchmark Data
    "SOA_2006_SURRENDER_BY_DURATION_7YR_SC",
    "SOA_2006_SC_CLIFF_EFFECT",
    "SOA_2006_SC_CLIFF_MULTIPLIER",
    "SOA_2006_FULL_SURRENDER_BY_AGE",
    "SOA_2006_PARTIAL_WITHDRAWAL_BY_AGE",
    "SOA_2006_POST_SC_DECAY",
    "SOA_2018_GLWB_UTILIZATION_BY_DURATION",
    "SOA_2018_GLWB_UTILIZATION_BY_AGE",
    "SOA_2018_ITM_SENSITIVITY",
    "SOA_2018_ITM_VS_NOT_ITM_BY_AGE",
    "SOA_KEY_INSIGHTS",
    "DATA_QUALITY_NOTES",
    # Calibration Utilities
    "interpolate_surrender_by_duration",
    "get_sc_cliff_multiplier",
    "get_post_sc_decay_factor",
    "interpolate_surrender_by_age",
    "interpolate_utilization_by_duration",
    "interpolate_utilization_by_age",
    "get_itm_sensitivity_factor",
    "get_itm_sensitivity_factor_continuous",
    "combined_utilization",
    "get_surrender_curve",
    "get_utilization_curve",
]
