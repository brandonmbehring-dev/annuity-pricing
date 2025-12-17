# Stress Testing Guide

**Last Updated**: 2025-12-12

This guide covers the stress testing framework for indexed annuity portfolios.

## Overview

The stress testing module allows you to:
- Apply historical crisis scenarios to portfolios
- Measure impact on option values and reserves
- Run custom shock scenarios
- Generate risk reports

## Quick Start

```python
from annuity_pricing.stress_testing.historical import CRISIS_2008_GFC
from annuity_pricing.stress_testing.runner import StressTestRunner, StressTestConfig

# Configure stress test
config = StressTestConfig(
    include_historical=True,
    include_orsa=False,
    parallel=False,
)

# Run stress tests
runner = StressTestRunner(calculator, config)
result = runner.run()

# Print summary
print(result.summary)
```

For a complete example, see `examples/04_stress_testing.py`.

---

## Historical Scenarios

### Available Crises [T2]

All parameters empirically calibrated from market data:

| Crisis | Period | Equity | VIX Peak | Rates |
|--------|--------|--------|----------|-------|
| 2008 GFC | Oct 2007 - Mar 2009 | -56.8% | 80.9 | -254 bp |
| 2020 COVID | Feb - Mar 2020 | -31.3% | 82.7 | -138 bp |
| 2000 Dot-Com | Mar 2000 - Oct 2002 | -49.2% | ~45 | -221 bp |
| 2022 Rate Shock | Jan - Oct 2022 | -25.4% | 36.4 | +232 bp |
| 2011 Euro Crisis | May - Oct 2011 | -21.6% | 48.0 | -145 bp |
| 2015 China Deval | Aug 2015 | -12.4% | 40.7 | -43 bp |
| 2018 Q4 Selloff | Oct - Dec 2018 | -20.2% | 36.0 | -60 bp |

### Using Historical Scenarios

```python
from annuity_pricing.stress_testing.historical import (
    CRISIS_2008_GFC,
    CRISIS_2020_COVID,
    ALL_HISTORICAL_CRISES,
)

# Access scenario details
print(f"GFC equity shock: {CRISIS_2008_GFC.equity_shock:.1%}")
print(f"VIX peak: {CRISIS_2008_GFC.vix_peak}")

# Get monthly profiles
for profile in CRISIS_2008_GFC.profile:
    print(f"Month {profile.month}: equity={profile.equity_cumulative:.1%}")
```

### Crisis Properties

Each `HistoricalCrisis` contains:

| Property | Type | Description |
|----------|------|-------------|
| `name` | str | Identifier (e.g., "2008_gfc") |
| `display_name` | str | Human-readable name |
| `start_date` | str | YYYY-MM format |
| `end_date` | str | YYYY-MM format |
| `equity_shock` | float | Peak-to-trough (negative decimal) |
| `rate_shock` | float | Rate change (decimal, e.g., -0.0254) |
| `vix_peak` | float | Maximum VIX level |
| `duration_months` | int | Months to trough |
| `recovery_months` | int | Months to pre-crisis level |
| `recovery_type` | RecoveryType | V, U, or L-shaped |
| `profile` | Tuple[CrisisProfile] | Monthly evolution |
| `notes` | str | Context |

---

## Custom Scenarios

### Creating a Shock Scenario

```python
from annuity_pricing.stress_testing.scenarios import StressScenario

# Simple instantaneous shock
custom_shock = StressScenario(
    name="custom_severe",
    display_name="Custom Severe Downturn",
    equity_shock=-0.40,      # -40% equity
    rate_shock=-0.02,        # -200bp rates
    vol_shock=0.50,          # +50 points implied vol
    correlation_shock=0.2,   # Correlation increases
)

# Apply to market params
stressed_market = apply_scenario(baseline_market, custom_shock)
```

### ORSA Standard Scenarios

The framework includes ORSA-aligned scenarios:

```python
from annuity_pricing.stress_testing.scenarios import ALL_ORSA_SCENARIOS

for scenario in ALL_ORSA_SCENARIOS:
    print(f"{scenario.name}: equity={scenario.equity_shock:.1%}")
```

---

## Running Stress Tests

### Configuration Options

```python
from annuity_pricing.stress_testing.runner import StressTestConfig

config = StressTestConfig(
    include_historical=True,     # Include 7 historical crises
    include_orsa=True,           # Include ORSA scenarios
    custom_scenarios=(),         # Add custom scenarios
    max_reserve_increase=0.50,   # Validation gate (50%)
    parallel=False,              # Multiprocessing
    n_workers=None,              # Auto-detect workers
    verbose=False,               # Progress output
)
```

### Result Structure

```python
result = runner.run()

# Summary statistics
print(f"Max reserve increase: {result.summary.max_reserve_increase:.1%}")
print(f"Worst scenario: {result.summary.worst_scenario_name}")

# Individual metrics
for metric in result.metrics:
    print(f"{metric.scenario_name}: change={metric.reserve_change_pct:.1%}")
```

---

## Metrics and Thresholds

### Severity Levels

| Level | Reserve Increase | Action |
|-------|------------------|--------|
| LOW | < 10% | Monitor |
| MEDIUM | 10-25% | Review |
| HIGH | 25-50% | Escalate |
| CRITICAL | > 50% | Immediate action |

### Validation Gates

The framework includes automatic checks:

```python
from annuity_pricing.stress_testing.metrics import (
    check_reserve_positive,
    check_reserve_increase_limit,
)

# These are called automatically by runner
# HALTs on violation (per CLAUDE.md)
```

---

## Portfolio Analysis

### Running on Product Portfolio

```python
from annuity_pricing import FIAProduct, RILAProduct, MarketParams

# Create portfolio
portfolio = [
    FIAProduct(...),
    RILAProduct(...),
]

# Price under baseline and stressed conditions
baseline_pvs = price_portfolio(portfolio, baseline_market)
stressed_pvs = price_portfolio(portfolio, stressed_market)

# Calculate impact
for product, baseline, stressed in zip(portfolio, baseline_pvs, stressed_pvs):
    change_pct = (stressed - baseline) / baseline
    print(f"{product.product_name}: {change_pct:+.1%}")
```

See `examples/04_stress_testing.py` for complete implementation.

---

## Sensitivity Analysis

### Single Factor Sensitivities

```python
from annuity_pricing.stress_testing.sensitivity import SensitivityAnalyzer

analyzer = SensitivityAnalyzer(pricer)

# Equity sensitivity
equity_results = analyzer.equity_sensitivity(
    product,
    shock_range=(-0.30, 0.30),
    n_points=11,
)

# Volatility sensitivity
vol_results = analyzer.volatility_sensitivity(
    product,
    vol_range=(0.10, 0.50),
    n_points=9,
)
```

### Multi-Factor Analysis

```python
# Equity + Vol combined
results = analyzer.multi_factor_sensitivity(
    product,
    factors=["equity", "volatility"],
    ranges=[(-0.30, 0.30), (0.10, 0.50)],
)
```

---

## Reverse Stress Testing

Find scenarios that breach thresholds:

```python
from annuity_pricing.stress_testing.reverse import ReverseStressTester

tester = ReverseStressTester(pricer)

# What causes 50% reserve increase?
scenario = tester.find_threshold_scenario(
    product,
    target_metric="reserve_increase",
    threshold=0.50,
)

print(f"Equity required: {scenario.equity_shock:.1%}")
print(f"Vol required: {scenario.vol_level:.1%}")
```

---

## Best Practices

### 1. Always Include Historical

Historical scenarios are empirically calibrated and provide realistic stress:

```python
config = StressTestConfig(include_historical=True)
```

### 2. Validate Against Limits

Use validation gates to catch extreme results:

```python
config = StressTestConfig(max_reserve_increase=0.50)
```

### 3. Document Assumptions

Tag all assumptions with knowledge tiers:

```python
# [T3] Assuming correlation increases 20% in stress
correlation_shock = 0.20
```

### 4. Run Regularly

Integrate stress tests into CI/CD:

```yaml
# .github/workflows/ci.yml
- name: Run stress tests
  run: python -m annuity_pricing.stress_testing.runner --quick
```

---

## Data Sources

| Data | Source | Update Frequency |
|------|--------|------------------|
| S&P 500 | Yahoo Finance (^GSPC) | Historical only |
| 10Y Treasury | FRED (DGS10) | Historical only |
| VIX | Yahoo Finance (^VIX) | Historical only |

---

## References

- [T1] RMS, "Historical Crisis Calibration", Risk Management Solutions
- [T1] Basel III, "Stress Testing Principles", BIS
- [T2] SOA, "Variable Annuity Stress Testing", Society of Actuaries
