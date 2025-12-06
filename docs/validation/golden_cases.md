# Golden Test Cases

Reference values used for validation.

## Black-Scholes (Hull Example 15.6)

**Parameters**:
- S = 42 (spot)
- K = 40 (strike)
- r = 10% (rate)
- σ = 20% (volatility)
- T = 0.5 years

**Expected Call Price**: **4.76** [T1]

```python
from annuity_pricing.options.pricing.black_scholes import black_scholes_call

price = black_scholes_call(
    spot=42, strike=40, rate=0.10, dividend=0,
    volatility=0.20, time_to_expiry=0.5
)
assert abs(price - 4.76) < 0.02
```

## Greeks (ATM Call)

**Parameters**: S=K=100, r=5%, σ=20%, T=1, q=0

| Greek | Expected | Tolerance |
|-------|----------|-----------|
| Delta | 0.6368 | ±0.01 |
| Gamma | 0.0188 | ±0.001 |
| Vega | 37.52 (per 100% vol) | ±0.5 |
| Theta | -6.41 (per year) | ±0.5 |

## Monte Carlo Convergence

**Test**: ATM European Call (S=K=100, r=5%, σ=20%, T=1)
**BS Price**: 10.45

| Paths | Expected Stderr | MC Range |
|-------|-----------------|----------|
| 10,000 | ~0.09 | 10.45 ± 0.18 |
| 100,000 | ~0.03 | 10.45 ± 0.06 |

## Put-Call Parity

$$C - P = S - K \cdot e^{-rT}$$

For Hull 15.6:
- C = 4.76
- P = 0.81
- C - P = 3.95
- S - K·e^(-rT) = 42 - 40·e^(-0.05) = 3.95 ✓

## Validation Notebooks

- [Black-Scholes vs financepy](https://github.com/bbehring/annuity-pricing/blob/main/notebooks/validation/options/black_scholes_vs_financepy.ipynb)
- [Monte Carlo vs BS](https://github.com/bbehring/annuity-pricing/blob/main/notebooks/validation/options/monte_carlo_vs_pyfeng.ipynb)
- [PV Correction Derivation](https://github.com/bbehring/annuity-pricing/blob/main/notebooks/validation/pv_correction_derivation.ipynb)
