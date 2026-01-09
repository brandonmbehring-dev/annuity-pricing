"""
Stress Testing Framework for Annuity Pricing - Phase I.

[T2] Provides historical crisis scenarios, sensitivity analysis,
reverse stress testing, and comprehensive reporting.

Phase I.1: Historical Stress Testing ✅
- 7 historical crisis definitions (2000-2022)
- Full monthly profiles for path-dependent analysis
- ORSA standard scenarios (Moderate/Severe/Extreme)

Phase I.2: Sensitivity Analysis ✅
- One-at-a-time (OAT) parameter sweeps
- Tornado diagram generation

Phase I.3: Reverse Stress Testing ✅
- Bisection search for breaking points
- Reserve exhaustion and RBC breach targets

Phase I.4: Reporting ✅
- Markdown executive summaries
- JSON structured output for automation

Design: Wrapper pattern - StressTestRunner wraps VM-21/VM-22
calculators without modifying them.

See: docs/stress_testing/HISTORICAL_SCENARIOS.md
See: ~/.claude/plans/rippling-spinning-ripple.md (Phase I details)
"""

# Historical crisis definitions
from .historical import (
    # Collections
    ALL_HISTORICAL_CRISES,
    CRISIS_2000_DOTCOM,
    # Individual crisis constants
    CRISIS_2008_GFC,
    CRISIS_2011_EURO_DEBT,
    CRISIS_2015_CHINA,
    CRISIS_2018_Q4,
    CRISIS_2020_COVID,
    CRISIS_2022_RATES,
    CrisisProfile,
    HistoricalCrisis,
    RecoveryType,
    get_crisis_by_name,
    get_crisis_summary,
)

# Metrics and results
from .metrics import (
    SeverityLevel,
    StressMetrics,
    StressTestSummary,
    calculate_percentiles,
    calculate_reserve_delta,
    classify_severity,
)

# Reporting (Phase I.4)
from .reporting import (
    ReportConfig,
    StressTestReporter,
    generate_quick_summary,
    generate_stress_report,
)

# Reverse stress testing (Phase I.3)
from .reverse import (
    ALL_PREDEFINED_TARGETS,
    DEFAULT_SEARCH_RANGES,
    RBC_BREACH_200,
    RBC_BREACH_300,
    # Predefined targets
    RESERVE_EXHAUSTION,
    RESERVE_INCREASE_50,
    RESERVE_INCREASE_100,
    RESERVE_NEGATIVE,
    SOLVENCY_BREACH,
    BreachCondition,
    ReverseStressReport,
    ReverseStressResult,
    ReverseStressTarget,
    ReverseStressTester,
    # Functions
    create_custom_target,
    find_reserve_exhaustion_point,
    format_reverse_stress_result,
    format_reverse_stress_summary,
    format_reverse_stress_table,
    quick_reverse_stress,
)

# Runner (wrapper pattern)
from .runner import (
    StressTestConfig,
    StressTestResult,
    StressTestRunner,
    quick_stress_test,
    stress_single_scenario,
)

# Stress scenarios
from .scenarios import (
    # Collections and utilities
    ALL_ORSA_SCENARIOS,
    ORSA_EXTREMELY_ADVERSE,
    # ORSA standard scenarios
    ORSA_MODERATE_ADVERSE,
    ORSA_SEVERELY_ADVERSE,
    ScenarioType,
    StressScenario,
    create_custom_scenario,
    crisis_to_scenario,
)

# Sensitivity analysis (Phase I.2)
from .sensitivity import (
    SensitivityAnalyzer,
    SensitivityDirection,
    SensitivityParameter,
    SensitivityResult,
    TornadoData,
    format_sensitivity_result,
    format_tornado_summary,
    format_tornado_table,
    get_default_sensitivity_parameters,
    quick_sensitivity_analysis,
)

__all__ = [
    # Historical crises
    "HistoricalCrisis",
    "CrisisProfile",
    "RecoveryType",
    "CRISIS_2008_GFC",
    "CRISIS_2020_COVID",
    "CRISIS_2000_DOTCOM",
    "CRISIS_2011_EURO_DEBT",
    "CRISIS_2015_CHINA",
    "CRISIS_2018_Q4",
    "CRISIS_2022_RATES",
    "ALL_HISTORICAL_CRISES",
    "get_crisis_by_name",
    "get_crisis_summary",
    # Scenarios
    "StressScenario",
    "ScenarioType",
    "ORSA_MODERATE_ADVERSE",
    "ORSA_SEVERELY_ADVERSE",
    "ORSA_EXTREMELY_ADVERSE",
    "ALL_ORSA_SCENARIOS",
    "crisis_to_scenario",
    "create_custom_scenario",
    # Metrics
    "StressMetrics",
    "StressTestSummary",
    "SeverityLevel",
    "calculate_reserve_delta",
    "calculate_percentiles",
    "classify_severity",
    # Runner
    "StressTestRunner",
    "StressTestConfig",
    "StressTestResult",
    "quick_stress_test",
    "stress_single_scenario",
    # Sensitivity Analysis (Phase I.2)
    "SensitivityParameter",
    "SensitivityResult",
    "TornadoData",
    "SensitivityDirection",
    "SensitivityAnalyzer",
    "get_default_sensitivity_parameters",
    "format_sensitivity_result",
    "format_tornado_table",
    "format_tornado_summary",
    "quick_sensitivity_analysis",
    # Reverse Stress Testing (Phase I.3)
    "BreachCondition",
    "ReverseStressTarget",
    "ReverseStressResult",
    "ReverseStressReport",
    "ReverseStressTester",
    "RESERVE_EXHAUSTION",
    "RESERVE_NEGATIVE",
    "RBC_BREACH_200",
    "RBC_BREACH_300",
    "SOLVENCY_BREACH",
    "RESERVE_INCREASE_50",
    "RESERVE_INCREASE_100",
    "ALL_PREDEFINED_TARGETS",
    "create_custom_target",
    "DEFAULT_SEARCH_RANGES",
    "format_reverse_stress_result",
    "format_reverse_stress_table",
    "format_reverse_stress_summary",
    "quick_reverse_stress",
    "find_reserve_exhaustion_point",
    # Reporting (Phase I.4)
    "ReportConfig",
    "StressTestReporter",
    "generate_stress_report",
    "generate_quick_summary",
]
