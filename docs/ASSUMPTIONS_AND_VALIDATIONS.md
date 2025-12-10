# Assumptions and Validations

**Last Updated**: 2025-12-09

This document provides a single source of truth for all simplifying assumptions in the annuity-pricing codebase, their justifications, and validation status.

---

## Knowledge Tier Key

| Tier | Meaning | Validation Required |
|------|---------|---------------------|
| **[T1]** | Academically established | Reference citation |
| **[T2]** | Empirically derived | Data source citation |
| **[T3]** | Simplifying assumption | Explicit justification |

---

## Option Pricing Assumptions

### Black-Scholes Model [T1]

| Assumption | Justification | Validation |
|------------|--------------|------------|
| Geometric Brownian Motion | Standard industry practice | Hull (2021) Ch. 14 |
| Constant volatility | Simplification; see Heston/SABR for stochastic vol | Validated against financepy |
| Continuous dividend yield | Approximation for index products | Put-call parity tests pass |
| Log-normal returns | Foundation of BS model | Anti-pattern tests |
| No transaction costs | Standard for pricing theory | N/A (model limitation) |

**Validation Tests**:
- `tests/validation/test_bs_known_answers.py` (Hull examples)
- `tests/anti_patterns/test_put_call_parity.py`
- `tests/anti_patterns/test_arbitrage_bounds.py`

### Monte Carlo Simulation [T1]

| Assumption | Justification | Validation |
|------------|--------------|------------|
| 100,000 paths default | Convergence to <1% error | MC convergence tests |
| Antithetic variates | Variance reduction | Glasserman (2003) §4.2 |
| 252 steps/year (daily) | Industry convention | N/A |
| 12 steps/year (monthly FIA) | Product design match | `test_asian_cap_benchmark.py` |

**Validation Tests**:
- `tests/validation/test_mc_convergence.py`
- `tests/validation/test_asian_cap_benchmark.py`

---

## FIA Pricing Assumptions

### Crediting Methods [T1/T3]

| Assumption | Tier | Justification | Validation |
|------------|------|--------------|------------|
| Capped call = call spread | [T1] | Standard replication | financepy benchmark |
| Participation = scaled ATM call | [T1] | Standard replication | financepy benchmark |
| Floor = 0% minimum | [T1] | Product design | `test_floor_enforcement.py` |
| Option budget = 3% default | [T3] | Industry typical | User-configurable |
| Discounted annual hedge spend | [T1] | Time value of money | `test_fia_option_budget_scales_with_term` |

**Validation Tests**:
- `tests/validation/test_fia_vs_financepy.py` (24 tests)
- `tests/anti_patterns/test_floor_enforcement.py`

### Monthly-Averaging FIA [T1]

| Assumption | Justification | Validation |
|------------|--------------|------------|
| 12 observations/year | Monthly crediting design | `test_asian_cap_benchmark.py` |
| Arithmetic average | Product specifications | Hull Ch. 26 reference |
| Average < P2P value | Volatility smoothing effect | `test_monthly_average_lower_value_than_point_to_point` |

---

## RILA Pricing Assumptions

### Buffer Protection [T1]

| Assumption | Justification | Validation |
|------------|--------------|------------|
| Buffer = put spread | Long ATM put - Short OTM put | QuantLib benchmark |
| Buffer strike = S×(1-buffer_rate) | Standard formulation | `test_rila_vs_quantlib.py` |
| Dollar-for-dollar after exhaustion | Product design | `test_buffer_mechanics.py` |

### Floor Protection [T1]

| Assumption | Justification | Validation |
|------------|--------------|------------|
| Floor = OTM put | Single put at floor strike | QuantLib benchmark |
| Floor strike = S×(1-floor_rate) | Standard formulation | `test_rila_vs_quantlib.py` |
| No protection above floor | Product design | `test_buffer_vs_floor.py` |

### Breakeven Calculation [T1]

| Assumption | Justification | Validation |
|------------|--------------|------------|
| Buffer breakeven = -buffer_rate | First point of full protection | `test_buffer_breakeven_at_left_edge` |
| Floor breakeven = 0% | Any negative return = loss | `test_floor_breakeven_at_zero` |

**Validation Tests**:
- `tests/validation/test_rila_vs_quantlib.py` (26 tests)
- `tests/anti_patterns/test_buffer_mechanics.py`
- `tests/anti_patterns/test_buffer_vs_floor.py`

---

## MYGA Pricing Assumptions

### Present Value [T1]

| Assumption | Justification | Validation |
|------------|--------------|------------|
| Continuous discounting | Standard actuarial practice | N/A |
| Flat yield curve | Simplification | User can provide custom rates |
| No credit adjustment | Separate from base pricing | Credit module available |

**Validation Tests**:
- `tests/unit/test_products_myga.py`
- `tests/validation/test_pv_discounting.py`

---

## Behavioral Assumptions

### Dynamic Lapse [T2/T3]

| Assumption | Tier | Justification | Validation |
|------------|------|--------------|------------|
| Moneyness-based lapse | [T2] | SOA 2006/2018 studies | `test_behavior_vs_soa.py` |
| Base lapse 5% default | [T3] | Industry typical | User-configurable |
| Shock multiplier formula | [T2] | SOA calibration | `test_soa_dynamic_lapse.py` |

### Withdrawal Utilization [T2/T3]

| Assumption | Tier | Justification | Validation |
|------------|------|--------------|------------|
| Age-based withdrawal rate | [T2] | SOA studies | `test_soa_withdrawal.py` |
| Maximum = GLWB rate | [T1] | Product design | Unit tests |

---

## Regulatory Assumptions

### VM-21 [T1/T3]

| Assumption | Tier | Justification | Validation |
|------------|------|--------------|------------|
| CTE70 threshold | [T1] | NAIC requirement | `test_vm21.py` |
| Vasicek rate model | [T1] | Standard stochastic rates | Unit tests |
| Correlated scenarios | [T1] | Cholesky decomposition | Unit tests |

### VM-22 [T1/T3]

| Assumption | Tier | Justification | Validation |
|------------|------|--------------|------------|
| NPR/DR/SR reserves | [T1] | NAIC requirement | `test_vm22.py` |
| SET/SST tests | [T1] | Standard Scenario | Unit tests |

---

## Validation Summary

### External Oracle Coverage

| Component | Oracle | Tolerance | Tests |
|-----------|--------|-----------|-------|
| Black-Scholes | financepy | <1% | 17+ tests |
| FIA capped call | financepy | <1% | 24 tests |
| RILA buffer/floor | QuantLib | <1% | 26 tests |
| Heston MC | QuantLib | <1% | Multiple tests |
| SABR vol | QuantLib | 0% exact | Multiple tests |
| Yield curves | QuantLib | <0.01% | Multiple tests |

### Anti-Pattern Coverage

| Anti-Pattern | Test File | Tests |
|--------------|-----------|-------|
| No-arbitrage bounds | `test_arbitrage_bounds.py` | 12+ |
| Put-call parity | `test_put_call_parity.py` | 13+ |
| Floor enforcement | `test_floor_enforcement.py` | 7+ |
| Buffer mechanics | `test_buffer_mechanics.py` | 9+ |
| Buffer vs floor | `test_buffer_vs_floor.py` | 8+ |
| Spread rate HALT | `test_spread_rate_halt.py` | 14 |

---

## Test Summary

| Category | Count |
|----------|-------|
| Unit tests | ~1300 |
| Integration tests | ~100 |
| Anti-pattern tests | ~85 |
| Validation tests | ~200 |
| **Total** | **~1686** |

---

## References

- Black, F., & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities.
- Hull, J. C. (2021). Options, Futures, and Other Derivatives (11th ed.).
- Glasserman, P. (2003). Monte Carlo Methods in Financial Engineering.
- SOA (2006). Annuity Surrender Behavior Study.
- SOA (2018). Variable Annuity Policyholder Behavior Study.
- NAIC VM-21/VM-22 Valuation Manual.
