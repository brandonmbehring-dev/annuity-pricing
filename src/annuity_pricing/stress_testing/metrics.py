"""
Stress Test Metrics and Results - Phase I.

[T2] Provides metrics calculation, severity classification,
and result aggregation for stress testing.

Design: Immutable dataclasses for results, pure functions for calculations.

See: docs/stress_testing/STRESS_TESTING_GUIDE.md
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


class SeverityLevel(Enum):
    """
    Severity classification for stress test results.

    Based on reserve impact relative to base case.
    """

    LOW = "low"  # Reserve increase < 5%
    MEDIUM = "medium"  # 5% <= increase < 15%
    HIGH = "high"  # 15% <= increase < 30%
    CRITICAL = "critical"  # increase >= 30%


@dataclass(frozen=True)
class StressMetrics:
    """
    Metrics for a single stress scenario result.

    [T2] Captures the reserve impact of applying a stress scenario.

    Attributes
    ----------
    scenario_name : str
        Name of the applied scenario
    base_reserve : float
        Reserve without stress (baseline)
    stressed_reserve : float
        Reserve under stress
    reserve_delta : float
        Absolute change (stressed - base)
    reserve_delta_pct : float
        Percentage change ((stressed - base) / base)
    severity : SeverityLevel
        Classification of impact severity
    passed_gates : bool
        Whether stress test passes validation gates
    gate_failures : Tuple[str, ...]
        List of failed validation gates (if any)
    additional_metrics : Dict[str, float]
        Optional additional metrics (CTE, VaR, etc.)
    """

    scenario_name: str
    base_reserve: float
    stressed_reserve: float
    reserve_delta: float
    reserve_delta_pct: float
    severity: SeverityLevel
    passed_gates: bool
    gate_failures: Tuple[str, ...] = ()
    additional_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class StressTestSummary:
    """
    Summary of stress test results across multiple scenarios.

    [T2] Aggregates results and provides percentile distribution.

    Attributes
    ----------
    n_scenarios : int
        Number of scenarios tested
    n_passed : int
        Number passing all gates
    n_failed : int
        Number failing at least one gate
    worst_scenario : str
        Name of scenario with highest reserve impact
    worst_reserve_delta_pct : float
        Reserve impact of worst scenario
    best_scenario : str
        Name of scenario with lowest reserve impact
    best_reserve_delta_pct : float
        Reserve impact of best scenario
    percentiles : Dict[int, float]
        Reserve delta percentiles (5th, 25th, 50th, 75th, 95th)
    severity_counts : Dict[SeverityLevel, int]
        Count by severity level
    base_reserve : float
        Baseline reserve (no stress)
    """

    n_scenarios: int
    n_passed: int
    n_failed: int
    worst_scenario: str
    worst_reserve_delta_pct: float
    best_scenario: str
    best_reserve_delta_pct: float
    percentiles: Dict[int, float]
    severity_counts: Dict[SeverityLevel, int]
    base_reserve: float


# =============================================================================
# Classification Functions
# =============================================================================


def classify_severity(reserve_delta_pct: float) -> SeverityLevel:
    """
    Classify severity based on reserve impact percentage.

    Parameters
    ----------
    reserve_delta_pct : float
        Reserve change as percentage (0.15 = 15% increase)

    Returns
    -------
    SeverityLevel
        Classification

    Examples
    --------
    >>> classify_severity(0.03)
    <SeverityLevel.LOW: 'low'>
    >>> classify_severity(0.10)
    <SeverityLevel.MEDIUM: 'medium'>
    >>> classify_severity(0.25)
    <SeverityLevel.HIGH: 'high'>
    >>> classify_severity(0.40)
    <SeverityLevel.CRITICAL: 'critical'>
    """
    abs_delta = abs(reserve_delta_pct)

    if abs_delta < 0.05:
        return SeverityLevel.LOW
    elif abs_delta < 0.15:
        return SeverityLevel.MEDIUM
    elif abs_delta < 0.30:
        return SeverityLevel.HIGH
    else:
        return SeverityLevel.CRITICAL


def calculate_reserve_delta(
    base_reserve: float,
    stressed_reserve: float,
) -> Tuple[float, float]:
    """
    Calculate absolute and percentage reserve delta.

    Parameters
    ----------
    base_reserve : float
        Baseline reserve
    stressed_reserve : float
        Reserve under stress

    Returns
    -------
    Tuple[float, float]
        (absolute_delta, percentage_delta)

    Raises
    ------
    ValueError
        If base_reserve <= 0

    Examples
    --------
    >>> delta, pct = calculate_reserve_delta(100_000, 120_000)
    >>> delta
    20000.0
    >>> pct
    0.2
    """
    if base_reserve <= 0:
        raise ValueError(f"base_reserve must be > 0, got {base_reserve}")

    delta = stressed_reserve - base_reserve
    pct = delta / base_reserve

    return float(delta), float(pct)


def calculate_percentiles(
    values: List[float],
    percentiles: Tuple[int, ...] = (5, 25, 50, 75, 95),
) -> Dict[int, float]:
    """
    Calculate percentiles of a distribution.

    Parameters
    ----------
    values : List[float]
        Values to calculate percentiles for
    percentiles : Tuple[int, ...]
        Percentile levels to calculate (default: 5, 25, 50, 75, 95)

    Returns
    -------
    Dict[int, float]
        Mapping from percentile level to value

    Raises
    ------
    ValueError
        If values is empty

    Examples
    --------
    >>> vals = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
    >>> percs = calculate_percentiles(vals)
    >>> percs[50]  # Median
    0.2
    """
    if not values:
        raise ValueError("Cannot calculate percentiles of empty list")

    arr = np.array(values)
    return {p: float(np.percentile(arr, p)) for p in percentiles}


# =============================================================================
# Metrics Creation
# =============================================================================


def create_stress_metrics(
    scenario_name: str,
    base_reserve: float,
    stressed_reserve: float,
    validation_gates: Optional[Dict[str, bool]] = None,
    additional_metrics: Optional[Dict[str, float]] = None,
) -> StressMetrics:
    """
    Create stress metrics from reserve values.

    Parameters
    ----------
    scenario_name : str
        Name of the applied scenario
    base_reserve : float
        Reserve without stress
    stressed_reserve : float
        Reserve under stress
    validation_gates : Optional[Dict[str, bool]]
        Gate name -> passed (True/False). If None, assumes all pass.
    additional_metrics : Optional[Dict[str, float]]
        Additional metrics to include (CTE, VaR, etc.)

    Returns
    -------
    StressMetrics
        Complete metrics object

    Examples
    --------
    >>> metrics = create_stress_metrics(
    ...     scenario_name="2008_gfc",
    ...     base_reserve=100_000,
    ...     stressed_reserve=145_000,
    ...     validation_gates={"reserve_positive": True, "solvency": True}
    ... )
    >>> metrics.severity
    <SeverityLevel.CRITICAL: 'critical'>
    """
    delta, delta_pct = calculate_reserve_delta(base_reserve, stressed_reserve)
    severity = classify_severity(delta_pct)

    # Process validation gates
    if validation_gates is None:
        passed_gates = True
        gate_failures: Tuple[str, ...] = ()
    else:
        gate_failures = tuple(name for name, passed in validation_gates.items() if not passed)
        passed_gates = len(gate_failures) == 0

    return StressMetrics(
        scenario_name=scenario_name,
        base_reserve=base_reserve,
        stressed_reserve=stressed_reserve,
        reserve_delta=delta,
        reserve_delta_pct=delta_pct,
        severity=severity,
        passed_gates=passed_gates,
        gate_failures=gate_failures,
        additional_metrics=additional_metrics or {},
    )


def create_summary(
    metrics_list: List[StressMetrics],
) -> StressTestSummary:
    """
    Create summary from list of stress test metrics.

    Parameters
    ----------
    metrics_list : List[StressMetrics]
        Results from individual scenarios

    Returns
    -------
    StressTestSummary
        Aggregated summary

    Raises
    ------
    ValueError
        If metrics_list is empty or base reserves differ
    """
    if not metrics_list:
        raise ValueError("Cannot create summary from empty metrics list")

    # Validate all use same base reserve
    base_reserves = {m.base_reserve for m in metrics_list}
    if len(base_reserves) > 1:
        raise ValueError(
            f"Inconsistent base reserves in metrics: {base_reserves}. "
            "All scenarios should use same baseline."
        )

    base_reserve = metrics_list[0].base_reserve

    # Calculate aggregates
    n_passed = sum(1 for m in metrics_list if m.passed_gates)
    n_failed = len(metrics_list) - n_passed

    # Find worst/best by reserve delta percentage
    sorted_by_impact = sorted(metrics_list, key=lambda m: m.reserve_delta_pct, reverse=True)
    worst = sorted_by_impact[0]
    best = sorted_by_impact[-1]

    # Calculate percentiles
    deltas = [m.reserve_delta_pct for m in metrics_list]
    percentiles = calculate_percentiles(deltas)

    # Count by severity
    severity_counts: Dict[SeverityLevel, int] = {level: 0 for level in SeverityLevel}
    for m in metrics_list:
        severity_counts[m.severity] += 1

    return StressTestSummary(
        n_scenarios=len(metrics_list),
        n_passed=n_passed,
        n_failed=n_failed,
        worst_scenario=worst.scenario_name,
        worst_reserve_delta_pct=worst.reserve_delta_pct,
        best_scenario=best.scenario_name,
        best_reserve_delta_pct=best.reserve_delta_pct,
        percentiles=percentiles,
        severity_counts=severity_counts,
        base_reserve=base_reserve,
    )


# =============================================================================
# Validation Gates
# =============================================================================


def check_reserve_positive(stressed_reserve: float) -> bool:
    """Check that stressed reserve remains positive."""
    return stressed_reserve > 0


def check_solvency_ratio(
    assets: float,
    liabilities: float,
    min_ratio: float = 1.0,
) -> bool:
    """
    Check solvency ratio (assets / liabilities).

    Parameters
    ----------
    assets : float
        Total assets
    liabilities : float
        Total liabilities (including stressed reserve)
    min_ratio : float
        Minimum required ratio (default 1.0 = 100%)

    Returns
    -------
    bool
        True if solvency ratio >= min_ratio
    """
    if liabilities <= 0:
        return True  # No liabilities = solvent
    return (assets / liabilities) >= min_ratio


def check_rbc_ratio(
    total_adjusted_capital: float,
    authorized_control_level: float,
    min_ratio: float = 2.0,
) -> bool:
    """
    Check Risk-Based Capital ratio.

    Parameters
    ----------
    total_adjusted_capital : float
        Total Adjusted Capital (TAC)
    authorized_control_level : float
        Authorized Control Level RBC
    min_ratio : float
        Minimum RBC ratio (default 2.0 = 200%)

    Returns
    -------
    bool
        True if RBC ratio >= min_ratio
    """
    if authorized_control_level <= 0:
        return True  # Cannot calculate RBC
    return (total_adjusted_capital / authorized_control_level) >= min_ratio


def check_reserve_increase_limit(
    reserve_delta_pct: float,
    max_increase: float = 0.50,
) -> bool:
    """
    Check that reserve increase is within acceptable limit.

    Parameters
    ----------
    reserve_delta_pct : float
        Reserve percentage change
    max_increase : float
        Maximum acceptable increase (default 0.50 = 50%)

    Returns
    -------
    bool
        True if reserve increase <= max_increase
    """
    return reserve_delta_pct <= max_increase


# =============================================================================
# Utility Functions
# =============================================================================


def format_metrics_row(metrics: StressMetrics) -> str:
    """
    Format metrics as a table row string.

    Parameters
    ----------
    metrics : StressMetrics
        Metrics to format

    Returns
    -------
    str
        Formatted row for display
    """
    status = "PASS" if metrics.passed_gates else "FAIL"
    return (
        f"| {metrics.scenario_name:20s} | "
        f"{metrics.reserve_delta_pct * 100:+7.1f}% | "
        f"{metrics.severity.value:8s} | "
        f"{status:4s} |"
    )


def format_summary(summary: StressTestSummary) -> str:
    """
    Format summary as a multi-line string.

    Parameters
    ----------
    summary : StressTestSummary
        Summary to format

    Returns
    -------
    str
        Formatted summary for display
    """
    lines = [
        "=" * 60,
        "STRESS TEST SUMMARY",
        "=" * 60,
        f"Scenarios Tested: {summary.n_scenarios}",
        f"Passed: {summary.n_passed} | Failed: {summary.n_failed}",
        f"Base Reserve: ${summary.base_reserve:,.0f}",
        "",
        f"Worst Case: {summary.worst_scenario} ({summary.worst_reserve_delta_pct * 100:+.1f}%)",
        f"Best Case: {summary.best_scenario} ({summary.best_reserve_delta_pct * 100:+.1f}%)",
        "",
        "Percentile Distribution (Reserve Delta %):",
        f"  5th: {summary.percentiles.get(5, 0) * 100:+.1f}%",
        f"  25th: {summary.percentiles.get(25, 0) * 100:+.1f}%",
        f"  50th: {summary.percentiles.get(50, 0) * 100:+.1f}%",
        f"  75th: {summary.percentiles.get(75, 0) * 100:+.1f}%",
        f"  95th: {summary.percentiles.get(95, 0) * 100:+.1f}%",
        "",
        "Severity Distribution:",
        f"  LOW: {summary.severity_counts.get(SeverityLevel.LOW, 0)}",
        f"  MEDIUM: {summary.severity_counts.get(SeverityLevel.MEDIUM, 0)}",
        f"  HIGH: {summary.severity_counts.get(SeverityLevel.HIGH, 0)}",
        f"  CRITICAL: {summary.severity_counts.get(SeverityLevel.CRITICAL, 0)}",
        "=" * 60,
    ]
    return "\n".join(lines)
