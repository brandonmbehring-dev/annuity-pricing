# Pricing RILA Products

```{note}
This guide is under active development. Core functionality is implemented and tested.
See the [API Reference](../api/products.md) for current capabilities.
```

## Overview

RILA (Registered Index-Linked Annuity) products offer market participation with downside protection via buffers or floors.

## Buffer vs Floor

| Protection | Small Loss (-5%) | Large Loss (-25%) |
|------------|------------------|-------------------|
| 10% Buffer | 0% (absorbed) | -15% (25-10) |
| 10% Floor | -5% (pass-through) | -10% (capped) |

**Buffer**: Better for small losses
**Floor**: Better for catastrophic losses

## Payoff Formulas

### Buffer Payoff [T1]

The buffer absorbs the **first X%** of losses:

```
if index_return >= 0:
    payoff = min(index_return, cap)     # Upside capped
elif index_return >= -buffer:
    payoff = 0                          # Buffer absorbs loss
else:
    payoff = index_return + buffer      # Client absorbs excess
```

**Replication**: Buffer = Long Put(K=100%) - Long Put(K=100%-buffer)

### Floor Payoff [T1]

The floor limits **maximum loss** to X%:

```
if index_return >= 0:
    payoff = min(index_return, cap)     # Upside capped
else:
    payoff = max(index_return, -floor)  # Floor limits loss
```

**Replication**: Floor = Long Put(K=100%-floor)

## Greeks

```python
from annuity_pricing.products.rila import RILAPricer, MarketParams
from annuity_pricing.data.schemas import RILAProduct

# Market setup
market = MarketParams(
    spot=100.0,
    risk_free_rate=0.05,
    dividend_yield=0.02,
    volatility=0.20,
)

# Product definition
rila = RILAProduct(
    company_name="Example Life",
    product_name="10% Buffer S&P",
    product_group="RILA",
    status="current",
    buffer_rate=0.10,
    buffer_modifier="Losses Covered Up To",
    cap_rate=0.15,
    index_used="S&P 500",
)

pricer = RILAPricer(market_params=market)
greeks = pricer.calculate_greeks(rila, term_years=1.0, premium=100_000)

print(f"Delta: {greeks.delta:.4f}")
print(f"Vega: {greeks.vega:.4f}")
```

## See Also

- {doc}`pricing_fia` — FIA option budgeting
- {doc}`../api/products` — Full API reference
