---
paths:
  - "src/**"
---

## Source Code Structure

```
src/annuity_pricing/
├── adapters/                 # External library integrations (financepy, QuantLib, pyfeng)
├── behavioral/               # Policyholder behavior models (lapse, withdrawal, expenses)
├── competitive/              # Rate positioning, rankings, spreads
├── config/settings.py        # Frozen dataclass config
├── credit/                   # CVA, default probability, guaranty funds
├── data/                     # WINK loader, market data, schemas
├── glwb/                     # GLWB/GMxB modeling (GWB tracker, path sim, rollup)
├── loaders/                  # Mortality tables, yield curves
├── options/
│   ├── payoffs/              # FIA crediting, RILA buffer/floor
│   ├── pricing/              # BS, Heston, SABR
│   └── simulation/           # GBM paths, Monte Carlo engine
├── products/                 # MYGA/FIA/RILA pricers + registry
├── rate_setting/             # Rate recommendations
├── regulatory/               # VM-21, VM-22, NAIC scenarios
├── stress_testing/           # Historical, reverse, sensitivity
├── validation/gates.py       # HALT/PASS validation gates
└── valuation/myga_pv.py      # MYGA present value
```

FIA and RILA valuation logic is embedded in their respective pricers (`products/fia.py`, `products/rila.py`).
