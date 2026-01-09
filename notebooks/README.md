# Notebook Curriculum

A tiered learning path for annuity pricing concepts.

## Learning Path

| Tier | Focus | Audience | Time |
|------|-------|----------|------|
| **Tier 0** | Onboarding | New users | ~30 min |
| **Tier 1** | Core Concepts | All users | ~1 hour |
| **Tier 2** | Product Mechanics | Practitioners | ~1 hour |
| **Tier 3** | Integration | Advanced | ~45 min |

## Notebooks by Tier

### Tier 0: Onboarding
| Notebook | Topic | Key Concepts |
|----------|-------|--------------|
| `00_pricing_foundations.ipynb` | First steps | Risk-neutral intro, library overview |

### Tier 1: Core Concepts
| Notebook | Topic | Key Concepts |
|----------|-------|--------------|
| `01_risk_neutral_valuation.ipynb` | Q-measure | μ vs r drift, arbitrage-free pricing |
| `02_put_call_parity.ipynb` | Arbitrage | C - P = S·e^(-qT) - K·e^(-rT) |
| `03_arbitrage_bounds.ipynb` | Price constraints | max(0, S-K) ≤ C ≤ S |

### Tier 2: Product Mechanics
| Notebook | Topic | Key Concepts |
|----------|-------|--------------|
| `05_fia_crediting_methods.ipynb` | FIA payoffs | Cap, participation, spread, 0% floor |
| `06_rila_buffer_vs_floor.ipynb` | RILA protection | Buffer absorbs first X%, floor caps max loss |
| `08_monte_carlo_convergence.ipynb` | MC validation | SE ~ 1/√n, path count guidelines |

### Tier 3: Integration
| Notebook | Topic | Key Concepts |
|----------|-------|--------------|
| `10_end_to_end_pricing.ipynb` | Complete workflow | Market → Product → Price → Validate |
| `12_cross_library_validation.ipynb` | External validation | Hull examples, financepy comparison |

## Prerequisites

```bash
pip install annuity-pricing[dev]
```

For cross-library validation (Tier 3):
```bash
pip install annuity-pricing[validation]
```

## Running Notebooks

```bash
# Single notebook
jupyter notebook notebooks/tier_1_core_concepts/01_risk_neutral_valuation.ipynb

# Validate all (CI)
jupyter nbconvert --to notebook --execute notebooks/**/*.ipynb
```

## Related Resources

- **Failure Examples**: `examples/failures/` — Common mistakes with fixes
- **Domain Knowledge**: `docs/knowledge/domain/` — Reference documentation
- **Anti-Pattern Tests**: `tests/anti_patterns/` — Bug prevention tests
