# Annuity Pricing Examples

This directory contains runnable examples demonstrating key actuarial use cases.

## Quick Start

```bash
# Ensure you're in the project root with venv activated
source venv/bin/activate

# Run any example
python examples/01_fair_rates.py
```

## Examples Overview

| Example | Description | Key Concepts |
|---------|-------------|--------------|
| `01_fair_rates.py` | Fair cap/participation rate calculation | Option budget, BS pricing |
| `02_competitive_positioning.py` | Rate percentile analysis | Market data, spreads |
| `03_glwb_valuation.py` | GLWB guarantee pricing | Monte Carlo, path-dependent |
| `04_stress_testing.py` | Portfolio stress testing | GFC 2008 scenario |

## Running in CI Mode

All examples support a `--ci` flag for non-interactive execution:

```bash
python examples/01_fair_rates.py --ci    # Fewer paths, no plots
python examples/04_stress_testing.py --ci
```

CI mode:
- Reduces Monte Carlo paths for faster execution
- Skips interactive plots (matplotlib)
- Used by GitHub Actions for smoke tests

## Example Details

### 01: Fair Cap/Participation Rates

Calculates "fair" cap and participation rates given an option budget.

**Question answered**: *"Given a 3% option budget, what cap rate is fair?"*

```bash
python examples/01_fair_rates.py --budget 0.03 --vol 0.18
```

Output includes:
- Fair cap rate
- Fair participation rate
- Volatility sensitivity analysis

### 02: Competitive Positioning

Analyzes where a product ranks in the competitive landscape.

**Question answered**: *"How does our 4.5% MYGA rate compare to the market?"*

```bash
python examples/02_competitive_positioning.py --rate 0.045
```

Output includes:
- Rate percentile (0-100)
- Rank among competitors
- Spread over Treasury
- Rate recommendations for target percentiles

### 03: GLWB Valuation

Prices Guaranteed Lifetime Withdrawal Benefits using Monte Carlo simulation.

**Question answered**: *"What is the guarantee cost for this GLWB product?"*

```bash
python examples/03_glwb_valuation.py --paths 10000
```

Output includes:
- Guarantee cost as % of premium
- Probability of ruin (account exhaustion)
- Sensitivity to rollup rate and age

### 04: Portfolio Stress Testing

Stress tests a portfolio of indexed annuities under the GFC 2008 scenario.

**Question answered**: *"How would our portfolio perform in another financial crisis?"*

```bash
python examples/04_stress_testing.py --paths 10000
python examples/04_stress_testing.py --save-report  # Save markdown report
```

Output includes:
- Product-by-product PV changes
- Portfolio-level impact
- Risk interpretation

## Creating Your Own Examples

Use these examples as templates. Key patterns:

1. **Add `sys.path.insert(0, "src")`** at top for script execution
2. **Support `--ci` flag** for CI-friendly mode
3. **Use realistic defaults** that demonstrate the concept
4. **Include interpretation** to explain results

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `pip install -e .` from project root |
| Slow execution | Use `--ci` flag or reduce `--paths` |
| Matplotlib errors | Install matplotlib: `pip install matplotlib` |
