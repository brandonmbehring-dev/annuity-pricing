# Put-Spread Buffer Replication

**Tier**: L2 (Full Derivation)
**Domain**: RILA Pricing
**Prerequisites**: Black-Scholes, option payoffs
**Source**: [T1] Hull (2021), Structured Products literature

---

## RILA Buffer Mechanics

### Definition [T1]

A **buffer** protects against the first X% of index losses. Beyond the buffer, the client absorbs further losses.

**Example** (10% buffer, -15% index return):
- Buffer absorbs: first 10%
- Client loss: 15% - 10% = 5%

### Payoff Function [T1]

For buffer level $B$ (e.g., $B = 0.10$ for 10% buffer):

$$\text{Payoff}(R) = \begin{cases}
\min(R, C) & R \geq 0 \\
0 & -B \leq R < 0 \\
R + B & R < -B
\end{cases}$$

Where:
- $R$: Index return
- $C$: Cap rate
- $B$: Buffer level

---

## Replication with Put Spread [T1]

### Insight

The buffer protection is equivalent to:
- **Long** ATM put (100% strike)
- **Short** OTM put at $(1 - B)$ strike

This creates protection between strikes, with losses below the lower strike passed through.

### Mathematical Derivation

**ATM Put payoff** (strike = $S_0$):
$$P_{ATM} = \max(S_0 - S_T, 0)$$

**OTM Put payoff** (strike = $S_0(1-B)$):
$$P_{OTM} = \max(S_0(1-B) - S_T, 0)$$

**Put spread payoff**:
$$P_{spread} = P_{ATM} - P_{OTM}$$

### Payoff Analysis

| Index Return | $S_T$ | ATM Put | OTM Put | Spread | Protection |
|-------------|-------|---------|---------|--------|------------|
| +10% | 110 | 0 | 0 | 0 | No loss |
| 0% | 100 | 0 | 0 | 0 | No loss |
| -5% | 95 | 5 | 0 | 5 | Full protection |
| -10% | 90 | 10 | 0 | 10 | Full protection |
| -15% | 85 | 15 | 5 | 10 | Partial (client loses 5) |
| -25% | 75 | 25 | 15 | 10 | Partial (client loses 15) |

**Interpretation**: The put spread pays out losses up to $B$, capping protection at the buffer level.

---

## Pricing the Buffer [T1]

### Fair Value

The buffer value equals the put spread value:

$$V_{buffer} = P_{ATM} - P_{OTM}$$

Using Black-Scholes:

$$V_{buffer} = BS_{put}(S_0, S_0, r, q, \sigma, T) - BS_{put}(S_0, S_0(1-B), r, q, \sigma, T)$$

### Example Calculation

Parameters:
- $S_0 = 100$
- $B = 0.10$ (10% buffer)
- $r = 0.05$, $q = 0.02$, $\sigma = 0.20$, $T = 1.0$

```python
from annuity_pricing.options.pricing.black_scholes import black_scholes_put

# ATM put (strike = 100)
p_atm = black_scholes_put(100, 100, 0.05, 0.02, 0.20, 1.0)
# p_atm ≈ 5.57

# OTM put (strike = 90)
p_otm = black_scholes_put(100, 90, 0.05, 0.02, 0.20, 1.0)
# p_otm ≈ 2.01

# Buffer value
buffer_value = p_atm - p_otm
# buffer_value ≈ 3.56 (3.56% of notional)
```

---

## Buffer vs Floor [T1]

### Critical Distinction

| Feature | Buffer | Floor |
|---------|--------|-------|
| Protection mechanism | Absorbs first X% of loss | Sets minimum return |
| Replication | Put spread | Long put |
| Client exposure | Beyond buffer | None below floor |
| Cost | Lower | Higher |
| Risk profile | Tail risk exposure | No tail risk |

### Payoff Comparison (-15% return)

| Protection Type | Level | Client Return |
|----------------|-------|---------------|
| 10% Buffer | First 10% absorbed | -5% |
| 0% Floor | Minimum 0% | 0% |

**Key**: Buffer and floor are NOT interchangeable. A 10% buffer does NOT provide the same protection as a -10% floor.

---

## Greeks for Buffer Position

### Delta

$$\Delta_{buffer} = \Delta_{ATM\ put} - \Delta_{OTM\ put}$$

Since puts have negative delta:
- ATM put: $\Delta \approx -0.5$
- OTM put: $\Delta \approx -0.2$ (depends on moneyness)
- Buffer: $\Delta \approx -0.3$

### Gamma

Concentrated around the strikes, especially ATM.

### Vega

$$\mathcal{V}_{buffer} = \mathcal{V}_{ATM} - \mathcal{V}_{OTM}$$

Higher vega than a single put since we're long the higher-vega ATM put.

---

## RILA Product Pricing

### Full RILA Structure

A RILA with buffer $B$ and cap $C$ combines:
1. **Zero-coupon bond** (principal protection)
2. **Capped call** (upside participation)
3. **Short buffer protection** (investor sells protection)

### Pricing Formula

$$V_{RILA} = PV_{principal} + V_{capped\ call} - V_{buffer}$$

The insurer:
- Pays for the capped call (gives upside to client)
- Receives buffer value (client accepts downside risk beyond buffer)

---

## Implementation Notes

### Module Location

```
src/annuity_pricing/options/payoffs/rila.py  # Buffer payoff definitions
src/annuity_pricing/products/rila.py         # RILA pricing
```

### Key Functions

| Function | Purpose |
|----------|---------|
| `buffer_payoff()` | Calculate buffer protection payoff |
| `price_buffer_protection()` | Value buffer as put spread |
| `RILAPricer.price()` | Full RILA pricing |

### Validation

Buffer replication should match:
1. Direct payoff simulation (MC)
2. Put spread analytical (BS)

Tolerance: < 1% relative error

---

## References

- Hull, J. (2021). *Options, Futures, and Other Derivatives*. Chapter 11 (Trading Strategies).
- SEC RILA Final Rule (2024). Form N-4 disclosure requirements.
- Boyle, P. & Tian, Y. (2008). EIA design from investor perspective.
