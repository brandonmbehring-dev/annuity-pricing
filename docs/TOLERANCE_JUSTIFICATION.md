# Tolerance Justification

**Location**: `src/annuity_pricing/config/tolerances.py`
**Last Updated**: 2025-12-09

---

## Philosophy

Tolerances must be **derived from precision requirements**, not tuned to make tests pass.

> "A tolerance of 1% for put-call parity suggests code was tuned to pass tests, not derived from precision requirements." — Codex Audit

---

## Tier 1: Analytical Tolerances

For deterministic, closed-form solutions where machine precision is achievable.

### `PUT_CALL_PARITY_TOLERANCE = 1e-8`

**Previous value**: 0.01 (1%) — **10,000x too loose**

**Derivation**:
- Put-call parity: `C - P = S×e^(-qT) - K×e^(-rT)`
- Well-conditioned for typical values (S~100, K~100, r~0.05, T~1)
- Machine epsilon: `ε ≈ 2.2e-16`
- Safety factor: `sqrt(ε) × 10^4 ≈ 1.5e-8`
- **Chosen value: 1e-8**

**Reference**: Hull (2021) Ch. 15 — parity should hold to <0.01% for analytical pricing.

**Impact**: A 0.5% parity violation would silently pass with 1% tolerance. For a $1M portfolio, that's $5,000 undetected.

### `ANTI_PATTERN_TOLERANCE = 1e-10`

**Derivation**:
- No-arbitrage bounds: `0 ≤ C ≤ S` and `0 ≤ P ≤ K×e^(-rT)`
- These are hard mathematical bounds, not approximations
- Allow for float64 accumulation across multiple operations
- **Chosen value: 1e-10**

**Reference**: [T1] No-arbitrage is a fundamental theorem, not an approximation.

### `GREEKS_NUMERICAL_TOLERANCE = 1e-8`

**Derivation**:
- Finite difference Greeks involve subtracting nearly-equal numbers
- Catastrophic cancellation risk: `(f(x+h) - f(x)) / h`
- Optimal `h ≈ sqrt(ε) × x ≈ 1e-8` for float64
- **Chosen value: 1e-8**

**Reference**: Higham (2002) "Accuracy and Stability of Numerical Algorithms", Ch. 1.

---

## Tier 2: Cross-Library Tolerances

For validation against external oracles (financepy, QuantLib, pyfeng).

### `CROSS_LIBRARY_TOLERANCE = 1e-6`

**Derivation**:
- External libraries typically agree to 6 decimal places
- Differences arise from:
  - Numerical integration methods
  - Root-finding convergence criteria
  - Day count conventions
- **Chosen value: 1e-6**

**Empirical validation**: financepy vs QuantLib vs our BS match to 1e-6 in all tested cases.

### `HULL_EXAMPLE_TOLERANCE = 0.02`

**Derivation**:
- Hull textbook examples are quoted to 2 decimal places
- Example 15.6: C = 4.76, P = 0.81
- Our implementation should match to within rounding
- **Chosen value: 0.02 (absolute)**

**Reference**: Hull (2021) "Options, Futures, and Other Derivatives", 11th Ed.

---

## Tier 3: Stochastic Tolerances

For Monte Carlo and path-dependent calculations. Derived from CLT.

### Formula: `tolerance = 3σ / √N`

**Derivation**:
- MC estimator: `Ê[f] = (1/N) Σ f(Xᵢ)`
- Standard error: `SE = σ / √N`
- 3σ confidence interval covers 99.7% of estimates
- **Formula: 3 × σ / √N**

**Reference**: Glasserman (2003) "Monte Carlo Methods in Financial Engineering", Ch. 3-4.

### Specific Values

| Paths | Formula | Value | Notes |
|-------|---------|-------|-------|
| 10,000 | 3×0.20/√10000 | 0.006 | Quick tests |
| 100,000 | 3×0.20/√100000 | 0.002 | Standard MC |
| 500,000 | 3×0.20/√500000 | 0.0008 | High precision |

**Conservative adjustment**: We use slightly looser values (0.01, 0.005) to account for:
- Path complexity beyond GBM
- Variance in payoff estimates
- Antithetic variate imperfect correlation

### `BS_MC_CONVERGENCE_TOLERANCE = 0.01`

**Derivation**:
- For vanilla options, MC should converge to BS
- 100k paths: theoretical tolerance ~0.2%
- Allow 1% to account for implementation differences
- **Chosen value: 0.01**

---

## Tier 4: Integration Tolerances

For end-to-end tests with real data and complex workflows.

### `INTEGRATION_TOLERANCE = 1e-4`

**Derivation**:
- Product-level comparisons involve multiple calculations
- Error accumulates: pricing + Greeks + validation
- Allow 0.01% relative error for realistic workflows
- **Chosen value: 1e-4**

### `GOLDEN_RELATIVE_TOLERANCE = 1e-6`

**Derivation**:
- Golden file comparisons should be tight
- Detects any unintentional numerical drift
- Matches cross-library precision tier
- **Chosen value: 1e-6**

---

## Domain-Specific Tolerances

### Payoff Enforcement (`1e-10`)

**Applies to**: `FLOOR_ENFORCEMENT`, `BUFFER_ABSORPTION`, `CAP_ENFORCEMENT`

**Derivation**:
- Contract guarantees are legally binding
- Floor = -10% means client NEVER loses more than 10%
- No tolerance acceptable for violations
- Allow only float representation error
- **Chosen value: 1e-10**

---

## Migration from Old Tolerances

| Old Location | Old Value | New Constant | New Value | Change |
|--------------|-----------|--------------|-----------|--------|
| `settings.py:put_call_parity_tolerance` | 0.01 | `PUT_CALL_PARITY_TOLERANCE` | 1e-8 | 10^6x tighter |
| `settings.py:bs_mc_tolerance` | 0.01 | `BS_MC_CONVERGENCE_TOLERANCE` | 0.01 | Same |
| `settings.py:arbitrage_tolerance` | 1e-6 | `ANTI_PATTERN_TOLERANCE` | 1e-10 | 10^4x tighter |
| Hardcoded in adapters | 0.01 | `CROSS_LIBRARY_TOLERANCE` | 1e-6 | 10^4x tighter |

---

## Test Categories by Tolerance

| Test Type | Tolerance | Files |
|-----------|-----------|-------|
| Anti-pattern | `ANTI_PATTERN_TOLERANCE` (1e-10) | `tests/anti_patterns/*` |
| Put-call parity | `PUT_CALL_PARITY_TOLERANCE` (1e-8) | `test_put_call_parity.py` |
| External validation | `CROSS_LIBRARY_TOLERANCE` (1e-6) | `tests/validation/*` |
| Hull examples | `HULL_EXAMPLE_TOLERANCE` (0.02) | `test_bs_known_answers.py` |
| MC convergence | `MC_100K_TOLERANCE` (0.01) | `test_mc_convergence.py` |
| Golden regression | `GOLDEN_RELATIVE_TOLERANCE` (1e-6) | `tests/golden/*` |
| Integration | `INTEGRATION_TOLERANCE` (1e-4) | `tests/integration/*` |

---

## References

1. **Higham, N.J. (2002)** "Accuracy and Stability of Numerical Algorithms", SIAM, 2nd Ed.
2. **Hull, J.C. (2021)** "Options, Futures, and Other Derivatives", Pearson, 11th Ed.
3. **Glasserman, P. (2003)** "Monte Carlo Methods in Financial Engineering", Springer.
4. **IEEE 754** Double precision floating point standard.
