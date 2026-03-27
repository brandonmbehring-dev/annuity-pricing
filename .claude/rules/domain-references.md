---
paths:
  - "docs/**"
  - "wink-research-archive/**"
---

## Domain Reference Index

### Quick Lookup by Topic

| Topic | Quick Ref | Derivation | Paper |
|-------|-----------|------------|-------|
| GLWB/GMWB | `glwb_mechanics.md` | `glwb_pde.md` | `bauer_kling_russ_2008.md` |
| RILA | `rila_mechanics.md` | - | `sec_rila_final_rule_2024.md` |
| FIA/EIA | `crediting_methods.md` | - | `boyle_tian_2008.md` |
| Monte Carlo | - | `monte_carlo.md` | `glasserman_2003_monte_carlo.md` |
| Black-Scholes | `option_pricing.md` | `bs_greeks.md` | `black_scholes_1973.md` |
| Buffer/Floor | `buffer_floor.md` | - | `sec_rila_investor_testing_2023.md` |

### Validation Packages

| Language | Package | Validates |
|----------|---------|-----------|
| Python | financepy | BS pricing, Greeks |
| Python | QuantLib | Yield curves, bonds |
| Python | pyfeng | SABR, Heston |
| Julia | MortalityTables.jl | SOA tables |
| Julia | LifeContingencies.jl | Life-contingent values |
| R | lifecontingencies | Life actuarial math |

### Key Reference Documents

| Document | Location |
|----------|----------|
| METHODOLOGY.md | Project root |
| CURRENT_WORK.md | Project root |
| ROADMAP.md | Project root |
| Cross-Validation Matrix | `docs/CROSS_VALIDATION_MATRIX.md` |
| WINK Data Dictionary | `wink-research-archive/data-dictionary/WINK_DATA_DICTIONARY.md` |
| Product Guide | `wink-research-archive/product-guides/ANNUITY_PRODUCT_GUIDE.md` |

### Option Modeling Tiers

| Level | Approach | When to Use |
|-------|----------|-------------|
| L1: Empirical | WINK pattern analysis | Competitive positioning |
| L2: Black-Scholes | Closed-form pricing | Fair value, Greeks |
| L3: Monte Carlo | Path simulation | Path-dependent payoffs |
