# Policyholder Behavior Calibration - SOA Data

**Version**: 1.0 | **Status**: Phase H Complete | **Tier**: [T2] Empirical

---

## Overview

This document describes the calibration of policyholder behavior models using empirical data from Society of Actuaries (SOA) experience studies.

### Data Sources

| Source | Year | Coverage | Key Tables |
|--------|------|----------|------------|
| SOA Deferred Annuity Persistency Study | 2006 | 2000-2004 | Tables 5, 6, 8 |
| SOA VA GLB Utilization Study | 2018 | 2015 experience | Tables 1-17, 1-18, Figure 1-43, 1-44 |

### Implementation Files

```
src/annuity_pricing/behavioral/
├── soa_benchmarks.py     # Extracted SOA data tables
├── calibration.py        # Interpolation functions
├── dynamic_lapse.py      # SOADynamicLapseModel
└── withdrawal.py         # SOAWithdrawalModel
```

---

## 1. Surrender Rate Calibration

### 1.1 Duration-Based Surrender (SOA 2006 Table 6)

**Source**: SOA 2006 Deferred Annuity Persistency Study, Table 6
**Title**: "Contract Surrender Rates by Contract Year, 7-Year SC Schedule"

| Contract Year | Surrender Rate | Notes |
|---------------|----------------|-------|
| 1 | 1.4% | Lowest (SC penalty) |
| 2 | 2.3% | |
| 3 | 2.8% | |
| 4 | 3.2% | |
| 5 | 3.7% | |
| 6 | 4.3% | |
| 7 | 5.3% | Last SC year |
| **8** | **11.2%** | **POST-SC CLIFF** |
| 9 | 8.2% | Decline after cliff |
| 10 | 7.7% | |
| 11+ | 6.7% | Ultimate rate |

**Key Insight**: Surrender rate jumps 2.1x from year 7 (5.3%) to year 8 (11.2%) when SC expires.

### 1.2 SC Cliff Effect (SOA 2006 Table 5)

**Source**: SOA 2006 Deferred Annuity Persistency Study, Table 5
**Title**: "Surrender Rates by Years in SC Period"

| Years to SC Expiration | Surrender Rate | Multiplier |
|------------------------|----------------|------------|
| 3+ years remaining | 2.6% | 1.00 (baseline) |
| 2 years remaining | 4.9% | 1.88 |
| 1 year remaining | 5.8% | 2.23 |
| **At expiration** | **14.4%** | **2.48** |
| 1 year after | 11.1% | 1.91 |
| 2 years after | 9.8% | 1.69 |
| 3+ years after | 8.6% | 1.48 |

**Cliff Multiplier**: 14.4% / 5.8% = **2.48x** at SC expiration

### 1.3 Implementation: SOADynamicLapseModel

```python
from annuity_pricing.behavioral import SOADynamicLapseModel, SOALapseAssumptions

model = SOADynamicLapseModel(SOALapseAssumptions(
    surrender_charge_length=7,     # 7-year SC schedule
    use_duration_curve=True,       # Use SOA Table 6
    use_sc_cliff_effect=True,      # Apply Table 5 multiplier
    moneyness_sensitivity=1.0,     # ITM reduces lapse
))

result = model.calculate_lapse(
    gwb=100_000,           # Guarantee value
    av=80_000,             # Account value (ITM)
    duration=7,            # Last SC year
    years_to_sc_end=0,     # At expiration
)
# result.base_rate = 0.053 (from Table 6)
# result.sc_cliff_factor = 2.48 (from Table 5)
# result.moneyness_factor < 1.0 (ITM reduces lapse)
```

---

## 2. GLWB Utilization Calibration

### 2.1 Utilization by Duration (SOA 2018 Table 1-17)

**Source**: SOA 2018 VA GLB Utilization Study, Table 1-17
**Title**: "GLWB Utilization by Year Issued"

| Duration (Years) | Utilization Rate | Notes |
|------------------|------------------|-------|
| 1 | 11.1% | Low initial |
| 2 | 17.7% | |
| 3 | 19.9% | |
| 4 | 20.5% | |
| 5 | 21.5% | |
| 6 | 23.3% | |
| 7 | 25.6% | |
| 8 | 36.5% | Acceleration |
| 9 | 45.9% | |
| 10 | 51.8% | |
| 11 | 53.6% | Near peak |

**Key Insight**: Utilization ramps from 11% → 54% over 10 years (4.9x increase).

### 2.2 Utilization by Age (SOA 2018 Table 1-18)

**Source**: SOA 2018 VA GLB Utilization Study, Table 1-18
**Title**: "GLWB Utilization by Current Age and Year Issued" (2008 cohort)

| Age Band | Utilization Rate | Notes |
|----------|------------------|-------|
| Under 60 | ~5% | Very low |
| 60-64 | ~16% | |
| 65-69 | ~32% | Post-retirement |
| 70-74 | ~59% | RMD influence |
| 75-79 | ~65% | Peak |
| 80+ | ~63% | Slight decline |

**Key Insight**: Utilization peaks around age 75-79, not 80+ (declining health may reduce withdrawals).

### 2.3 ITM Sensitivity (SOA 2018 Figure 1-44)

**Source**: SOA 2018 VA GLB Utilization Study, Figure 1-44
**Title**: "Withdrawal by Degree of ITM"

| Moneyness (GWB/AV) | Sensitivity Factor | Relative to ATM |
|--------------------|-------------------|-----------------|
| ≤ 100% (Not ITM) | 1.00 | Baseline |
| 100-125% | 1.39 | +39% |
| 125-150% | 1.79 | +79% |
| > 150% | 2.11 | +111% |

**Key Insight**: Deep ITM (>150%) more than doubles withdrawal utilization.

### 2.4 Implementation: SOAWithdrawalModel

```python
from annuity_pricing.behavioral import SOAWithdrawalModel, SOAWithdrawalAssumptions

model = SOAWithdrawalModel(SOAWithdrawalAssumptions(
    use_duration_curve=True,       # SOA Table 1-17
    use_age_curve=True,            # SOA Table 1-18
    use_itm_sensitivity=True,      # SOA Figure 1-44
    combination_method='multiplicative',
))

result = model.calculate_withdrawal(
    gwb=150_000,           # Guarantee value
    av=100_000,            # Account value
    withdrawal_rate=0.05,  # 5% GLWB rate
    duration=8,            # Year 8
    age=72,                # Age 72
)
# result.duration_utilization = 0.365 (36.5%)
# result.age_utilization = 0.59 (59%)
# result.itm_factor = 1.79 (125-150% ITM)
# result.utilization_rate = combined effect
```

---

## 3. Combination Methodology

### 3.1 Duration × Age × ITM Combination

The model combines three effects multiplicatively:

```
utilization = duration_util × (age_util / reference_age_util) × ITM_factor
```

Where:
- `duration_util`: From SOA 2018 Table 1-17
- `age_util`: From SOA 2018 Table 1-18
- `reference_age_util`: Age 67 utilization (32%) as baseline
- `ITM_factor`: From SOA 2018 Figure 1-44

**Rationale**: Duration is the primary driver, with age and ITM providing multiplicative adjustments.

### 3.2 Alternative: Additive Combination

```
utilization = ((duration_util + age_util) / 2) × ITM_factor
```

Use `combination_method='additive'` for this approach.

---

## 4. Data Limitations

### 4.1 Data Age
- **SOA 2006**: Data from 2000-2004 (20+ years old)
- **SOA 2018**: Data from 2015 experience (9 years old)

**Impact**: Market conditions, product designs, and policyholder demographics have evolved.

### 4.2 Product Type Mismatch
- **SOA 2006**: Deferred annuities (not specifically FIA/RILA)
- **SOA 2018**: Variable annuities with GLWB

**Impact**: FIA/RILA behavior may differ from traditional deferred annuities and VAs.

### 4.3 Missing Granularity
- No surrender data by gender, premium size, or distribution channel
- Limited ITM sensitivity data for surrender (mostly for utilization)
- No behavioral data specific to buffer/floor products

### 4.4 Mitigation
1. Use SOA data as starting point, not gospel
2. Apply sensitivity analysis to key parameters
3. Collect company-specific experience when available
4. Update calibration as newer SOA studies are released

---

## 5. Validation

### 5.1 Test Coverage

| Test File | Purpose | Count |
|-----------|---------|-------|
| `test_soa_benchmarks.py` | Data integrity | ~30 tests |
| `test_calibration.py` | Interpolation functions | ~40 tests |
| `test_soa_dynamic_lapse.py` | Lapse model | ~30 tests |
| `test_soa_withdrawal.py` | Withdrawal model | ~35 tests |
| `test_behavior_vs_soa.py` | Cross-validation | ~25 tests |

### 5.2 Validation Targets

| Metric | SOA Value | Tolerance |
|--------|-----------|-----------|
| Year 1 surrender | 1.4% | ±0.1% |
| Year 8 surrender (cliff) | 11.2% | ±0.1% |
| SC cliff multiplier | 2.48x | ±0.1x |
| Year 1 utilization | 11.1% | ±0.1% |
| Year 10 utilization | 51.8% | ±0.1% |
| Age 72 utilization | 59% | ±2% |
| Deep ITM factor | 2.11x | ±0.1x |

---

## 6. Usage Examples

### 6.1 Pricing with SOA-Calibrated Behavior

```python
from annuity_pricing.behavioral import (
    SOADynamicLapseModel,
    SOALapseAssumptions,
    SOAWithdrawalModel,
    SOAWithdrawalAssumptions,
)

# Initialize models
lapse_model = SOADynamicLapseModel(SOALapseAssumptions())
withdrawal_model = SOAWithdrawalModel(SOAWithdrawalAssumptions())

# Simulate policy year 8 (post-SC cliff)
lapse = lapse_model.calculate_lapse(
    gwb=100_000, av=90_000, duration=8, years_to_sc_end=-1
)
withdrawal = withdrawal_model.calculate_withdrawal(
    gwb=100_000, av=90_000, withdrawal_rate=0.05, duration=8, age=72
)

print(f"Lapse rate: {lapse.lapse_rate:.1%}")
print(f"Utilization: {withdrawal.utilization_rate:.1%}")
```

### 6.2 Path-Based Simulation

```python
import numpy as np

# Simulate 10-year path
gwb_path = np.full(10, 100_000)
av_path = np.array([100_000, 95_000, 90_000, 85_000, 82_000,
                    80_000, 78_000, 76_000, 74_000, 72_000])
ages = np.arange(65, 75)

lapse_rates = lapse_model.calculate_path_lapses(
    gwb_path=gwb_path,
    av_path=av_path,
    start_duration=1,
    surrender_charge_length=7,
)

withdrawal_amounts = withdrawal_model.calculate_path_withdrawals(
    gwb_path=gwb_path,
    av_path=av_path,
    ages=ages,
    withdrawal_rate=0.05,
    start_duration=1,
)
```

---

## 7. References

### Primary Sources
1. **SOA (2006)** "Deferred Annuity Persistency Study: Survey of Participant Companies"
   - Tables 5, 6, 8 used for surrender calibration
   - URL: soa.org (free PDF)

2. **SOA/LIMRA (2018)** "Variable Annuity Guaranteed Living Benefit Utilization Study"
   - Tables 1-17, 1-18, Figures 1-43, 1-44 used for utilization calibration
   - URL: soa.org/limra (free PDF)

### Related Documentation
- `docs/knowledge/domain/dynamic_lapse.md` - Theoretical framework
- `docs/knowledge/domain/glwb_mechanics.md` - GLWB product mechanics
- `src/annuity_pricing/behavioral/soa_benchmarks.py` - Raw data

---

## Appendix A: Full Data Tables

### A.1 SOA 2006 Surrender by Duration (7-Year SC)

```python
SOA_2006_SURRENDER_BY_DURATION_7YR_SC = {
    1: 0.014,   # 1.4%
    2: 0.023,   # 2.3%
    3: 0.028,   # 2.8%
    4: 0.032,   # 3.2%
    5: 0.037,   # 3.7%
    6: 0.043,   # 4.3%
    7: 0.053,   # 5.3%
    8: 0.112,   # 11.2% (cliff)
    9: 0.082,   # 8.2%
    10: 0.077,  # 7.7%
    11: 0.067,  # 6.7% (ultimate)
}
```

### A.2 SOA 2018 GLWB Utilization by Duration

```python
SOA_2018_GLWB_UTILIZATION_BY_DURATION = {
    1: 0.111,   # 11.1%
    2: 0.177,   # 17.7%
    3: 0.199,   # 19.9%
    4: 0.205,   # 20.5%
    5: 0.215,   # 21.5%
    6: 0.233,   # 23.3%
    7: 0.256,   # 25.6%
    8: 0.365,   # 36.5%
    9: 0.459,   # 45.9%
    10: 0.518,  # 51.8%
    11: 0.536,  # 53.6%
}
```

### A.3 SOA 2018 ITM Sensitivity

```python
SOA_2018_ITM_SENSITIVITY = {
    'not_itm': 1.00,        # ≤100% moneyness
    'itm_100_125': 1.39,    # 100-125%
    'itm_125_150': 1.79,    # 125-150%
    'itm_150_plus': 2.11,   # >150%
}
```

---

*Document generated as part of Phase H: Behavior Calibration*
*Last updated: 2025-12-09*
