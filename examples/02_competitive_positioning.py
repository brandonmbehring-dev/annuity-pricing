#!/usr/bin/env python3
"""
Competitive Positioning Analysis Demo.

This example demonstrates how to analyze where a product sits in the
competitive landscape using rate percentiles and Treasury spreads.

Key Concepts:
- Rate Percentile: Where does our rate rank vs competitors? (0-100)
- Spread over Treasury: Excess yield over risk-free rate (basis points)
- Recommend Rate: What rate needed for target percentile?

Usage:
    python examples/02_competitive_positioning.py          # Full demo
    python examples/02_competitive_positioning.py --ci     # CI mode

See Also:
    - docs/guides/pricing_myga.md
    - docs/knowledge/domain/competitive_analysis.md
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add src to path if running as script
sys.path.insert(0, "src")

from annuity_pricing import MYGAPricer, MYGAProduct


def create_sample_market_data() -> pd.DataFrame:
    """
    Create sample MYGA market data for demonstration.

    In production, this would come from WINK data:
        from annuity_pricing.data.loader import load_wink_data
        df = load_wink_data()
        myga_df = df[df['productGroup'] == 'MYGA']

    Returns
    -------
    pd.DataFrame
        Sample market data with fixedRate, guaranteeDuration, status
    """
    # Sample 5-year MYGA rates (representative of 2024 market)
    data = {
        "company": [
            "Company A",
            "Company B",
            "Company C",
            "Company D",
            "Company E",
            "Company F",
            "Company G",
            "Company H",
            "Company I",
            "Company J",
            "Company K",
            "Company L",
        ],
        "fixedRate": [
            0.048,  # 4.80%
            0.047,  # 4.70%
            0.046,  # 4.60%
            0.045,  # 4.50%
            0.045,  # 4.50%
            0.044,  # 4.40%
            0.043,  # 4.30%
            0.042,  # 4.20%
            0.041,  # 4.10%
            0.040,  # 4.00%
            0.039,  # 3.90%
            0.038,  # 3.80%
        ],
        "guaranteeDuration": [5] * 12,  # All 5-year products
        "productGroup": ["MYGA"] * 12,
        "status": ["current"] * 12,
    }
    return pd.DataFrame(data)


def analyze_competitive_position(
    product: MYGAProduct,
    market_data: pd.DataFrame,
    treasury_rate: float = 0.042,
) -> dict:
    """
    Perform full competitive analysis for a product.

    Parameters
    ----------
    product : MYGAProduct
        Product to analyze
    market_data : pd.DataFrame
        Market data with comparable products
    treasury_rate : float
        Treasury yield for spread calculation (5Y Treasury ~4.2% in late 2024)

    Returns
    -------
    dict
        Analysis results
    """
    pricer = MYGAPricer()

    # Get competitive position
    position = pricer.competitive_position(
        product=product,
        market_data=market_data,
        duration_match=True,
        duration_tolerance=1,
    )

    # Calculate spread over Treasury
    spread_bps = pricer.calculate_spread_over_treasury(
        product=product,
        treasury_rate=treasury_rate,
    )

    return {
        "product_rate": product.fixed_rate,
        "percentile": position.percentile,
        "rank": position.rank,
        "total_products": position.total_products,
        "spread_bps": spread_bps,
        "treasury_rate": treasury_rate,
    }


def recommend_rates_for_targets(
    market_data: pd.DataFrame,
    guarantee_duration: int = 5,
    target_percentiles: list[float] = [25, 50, 75, 90],
) -> dict[int, float]:
    """
    Recommend rates to achieve target competitive percentiles.

    Parameters
    ----------
    market_data : pd.DataFrame
        Market data with comparable products
    guarantee_duration : int
        Product duration
    target_percentiles : list[float]
        Target percentile ranks

    Returns
    -------
    dict[int, float]
        {percentile: recommended_rate}
    """
    pricer = MYGAPricer()
    recommendations = {}

    for pct in target_percentiles:
        rate = pricer.recommend_rate(
            target_percentile=pct,
            market_data=market_data,
            guarantee_duration=guarantee_duration,
        )
        recommendations[pct] = rate

    return recommendations


def print_analysis(analysis: dict, product_name: str) -> None:
    """Print competitive analysis results."""
    print("\n" + "=" * 60)
    print(f"COMPETITIVE ANALYSIS: {product_name}")
    print("=" * 60)

    print(f"\nProduct Rate: {analysis['product_rate']:.2%}")
    print(f"\nMarket Position:")
    print(f"  Percentile:  {analysis['percentile']:.0f}th percentile")
    print(f"  Rank:        #{analysis['rank']} of {analysis['total_products']} products")

    print(f"\nTreasury Spread:")
    print(f"  5Y Treasury: {analysis['treasury_rate']:.2%}")
    print(f"  Spread:      {analysis['spread_bps']:.0f} basis points")

    # Interpretation
    print("\nInterpretation:")
    if analysis["percentile"] >= 75:
        print("  -> Very competitive rate (top quartile)")
    elif analysis["percentile"] >= 50:
        print("  -> Above average rate (top half)")
    elif analysis["percentile"] >= 25:
        print("  -> Below average rate (bottom half)")
    else:
        print("  -> Uncompetitive rate (bottom quartile)")


def print_recommendations(recommendations: dict[int, float]) -> None:
    """Print rate recommendations table."""
    print("\n" + "=" * 60)
    print("RATE RECOMMENDATIONS BY TARGET PERCENTILE")
    print("=" * 60)
    print("\n  Target Percentile    Recommended Rate")
    print("  " + "-" * 40)

    for pct, rate in sorted(recommendations.items()):
        label = ""
        if pct == 50:
            label = " (median)"
        elif pct == 75:
            label = " (competitive)"
        elif pct == 90:
            label = " (aggressive)"
        print(f"       {pct:3d}th              {rate:.2%}{label}")


def plot_market_distribution(market_data: pd.DataFrame, our_rate: float) -> None:
    """Plot market rate distribution (requires matplotlib)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nNote: matplotlib not installed, skipping plot")
        return

    rates = market_data["fixedRate"].dropna() * 100  # Convert to %

    fig, ax = plt.subplots(figsize=(10, 6))

    # Histogram of market rates
    ax.hist(rates, bins=10, edgecolor="black", alpha=0.7, label="Market rates")

    # Our rate
    ax.axvline(
        our_rate * 100,
        color="red",
        linewidth=2,
        linestyle="--",
        label=f"Our rate ({our_rate:.2%})",
    )

    ax.set_xlabel("Fixed Rate (%)", fontsize=12)
    ax.set_ylabel("Number of Products", fontsize=12)
    ax.set_title("MYGA Market Rate Distribution (5-Year)", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("examples/competitive_distribution.png", dpi=150)
    print("\nPlot saved to: examples/competitive_distribution.png")
    plt.show()


def main() -> None:
    """Run competitive positioning demo."""
    parser = argparse.ArgumentParser(description="Competitive Positioning Demo")
    parser.add_argument("--ci", action="store_true", help="CI mode (no interactive plots)")
    parser.add_argument(
        "--rate", type=float, default=0.045, help="Our product rate (default: 0.045)"
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("MYGA COMPETITIVE POSITIONING DEMO")
    print("=" * 60)
    print("\nThis demo analyzes where a product ranks in the competitive")
    print("landscape using rate percentiles and Treasury spreads.")

    # Create sample market data (in production, load from WINK)
    market_data = create_sample_market_data()
    print(f"\nLoaded {len(market_data)} comparable products")

    # Define our product
    our_product = MYGAProduct(
        company_name="Our Company",
        product_name="5-Year MYGA",
        product_group="MYGA",
        status="current",
        fixed_rate=args.rate,
        guarantee_duration=5,
    )

    # Analyze competitive position
    analysis = analyze_competitive_position(
        product=our_product,
        market_data=market_data,
        treasury_rate=0.042,  # 5Y Treasury
    )
    print_analysis(analysis, our_product.product_name)

    # Get rate recommendations
    recommendations = recommend_rates_for_targets(market_data)
    print_recommendations(recommendations)

    # Plot distribution (only if not CI mode)
    if not args.ci:
        plot_market_distribution(market_data, args.rate)

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
