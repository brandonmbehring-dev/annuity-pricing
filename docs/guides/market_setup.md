# Market Setup

```{todo}
This guide is under development.
```

## Overview

Setting up market parameters for pricing calculations.

## Yield Curve Loading

```python
from annuity_pricing.loaders.yield_curve import YieldCurveLoader

loader = YieldCurveLoader()
curve = loader.load_treasury_curve()

# Get rates at different maturities
rate_1y = curve.rate(1.0)
rate_5y = curve.rate(5.0)
rate_10y = curve.rate(10.0)
```

## Mortality Tables

```python
from annuity_pricing.loaders.mortality import MortalityLoader

loader = MortalityLoader()
table = loader.load_soa_2012_iam()

# Get mortality rate at age 65
qx_65 = table.qx(65)
```

## Market Parameters

```python
from annuity_pricing.products.fia import MarketParams

market = MarketParams(
    spot=4500.0,           # Index level
    risk_free_rate=0.045,  # Risk-free rate
    dividend_yield=0.015,  # Dividend yield
    volatility=0.18,       # Implied volatility
)
```

## See Also

- {doc}`getting_started` — Quick start guide
- {doc}`../api/loaders` — Loader API reference
