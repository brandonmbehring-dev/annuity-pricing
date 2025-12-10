# Competitive Analysis Quick Reference

**Tier**: [T2] Empirical (WINK data) | **Source**: WINK Data Dictionary, Internal Analysis

---

## Overview

Competitive analysis determines where a product's rate falls within the market distribution. This module supports rate positioning, company rankings, and spread-over-Treasury calculations.

---

## Rate Positioning [T2]

### Percentile Calculation

```python
percentile = (count_below / total_count) * 100
```

Where:
- `count_below`: Number of comparable products with rates below the target rate
- `total_count`: Total number of comparable products

### Quartile Classification

| Quartile | Percentile Range | Label |
|----------|-----------------|-------|
| 1 | 75-100 | Top Quartile |
| 2 | 50-75 | Above Median |
| 3 | 25-50 | Below Median |
| 4 | 0-25 | Bottom Quartile |

**Note**: Higher percentile = more competitive rate.

---

## Spread Over Treasury [T1]

### Definition

```
Spread = Product_Rate - Treasury_Rate
```

For annuities, spreads are typically calculated against the matching Treasury duration:

| Product Term | Treasury Benchmark |
|-------------|-------------------|
| 3-year | 3-year Treasury |
| 5-year | 5-year Treasury |
| 7-year | 7-year Treasury |
| 10-year | 10-year Treasury |

### Interpretation [T2]

| Spread Range | Interpretation |
|-------------|----------------|
| > 150 bps | Aggressive pricing (potential margin squeeze) |
| 100-150 bps | Competitive |
| 50-100 bps | Conservative |
| < 50 bps | Uncompetitive |

---

## Company Rankings [T2]

### Ranking Metrics

1. **Rate Rank**: Direct comparison of offered rates
2. **Spread Rank**: Comparison of spreads over Treasury
3. **Consistency Rank**: Historical rate stability

### Filtering Dimensions

| Dimension | Values | WINK Column |
|-----------|--------|-------------|
| Product Type | MYGA, FIA, RILA | `product_type` |
| Term | 3, 5, 7, 10 years | `surrender_years` |
| Premium Band | $10K, $100K, $250K | `premium_band` |
| Channel | Direct, IMO, Bank | `distribution_channel` |

---

## WINK Data Integration [T2]

### Key Columns for Competitive Analysis

| Column | Purpose |
|--------|---------|
| `credit_rate` | Declared rate for MYGA |
| `cap_rate` | Cap for FIA/RILA |
| `participation_rate` | Participation for FIA/RILA |
| `spread_rate` | Spread for spread-crediting FIA |
| `buffer_rate` | Buffer level for RILA |
| `floor_rate` | Floor level for FIA/RILA |

### Data Quality Checks

Before analysis, validate:
1. No negative rates (except spread, which can be 0)
2. Participation rates typically 0-200%
3. Cap rates typically 0-20%
4. Buffer rates typically 5-30%

---

## Implementation Notes

### Module Location

```
src/annuity_pricing/competitive/
├── positioning.py   # PositionResult, percentile calculations
├── rankings.py      # Company/product rankings
└── spreads.py       # Spread over Treasury calculations
```

### Key Functions

| Function | Module | Purpose |
|----------|--------|---------|
| `calculate_position()` | positioning.py | Get percentile for a rate |
| `get_distribution_stats()` | positioning.py | Get min/max/mean/median |
| `rank_companies()` | rankings.py | Rank companies by metric |
| `calculate_spread()` | spreads.py | Compute spread over Treasury |

---

## Example Usage

```python
from annuity_pricing.competitive.positioning import calculate_position

# Calculate where 4.5% falls in 5-year MYGA market
result = calculate_position(
    rate=0.045,
    product_type="MYGA",
    term_years=5,
    wink_data=df
)

print(f"Percentile: {result.percentile}th")
print(f"Position: {result.position_label}")
# Output: Percentile: 72nd, Position: Above Median
```

---

## References

- WINK Data Dictionary: `wink-research-archive/data-dictionary/WINK_DATA_DICTIONARY.md`
- Product Guide: `wink-research-archive/ANNUITY_PRODUCT_GUIDE.md`
