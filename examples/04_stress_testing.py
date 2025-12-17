#!/usr/bin/env python3
"""
Portfolio Stress Testing Demo - GFC 2008 Scenario.

This example demonstrates stress testing a portfolio of indexed annuity products
using the 2008 Global Financial Crisis scenario.

Key Concepts:
- Historical stress scenarios apply calibrated market shocks
- GFC 2008: -57% equity, VIX spike to 81, rates down 254bp
- Impact analysis: how do option values change under stress?

See: docs/stress_testing/STRESS_TESTING_GUIDE.md
See: docs/knowledge/domain/stress_testing.md

Usage:
    python examples/04_stress_testing.py          # Full demo
    python examples/04_stress_testing.py --ci     # CI mode (fewer paths)
"""

import argparse
import sys
from dataclasses import dataclass
from datetime import date

# Add src to path if running as script
sys.path.insert(0, "src")

from annuity_pricing import (
    FIAPricer,
    FIAProduct,
    RILAPricer,
    RILAProduct,
    MarketParams,
)
from annuity_pricing.stress_testing.historical import CRISIS_2008_GFC, HistoricalCrisis


@dataclass
class StressedResult:
    """Results for a single product under stress."""

    product_name: str
    product_type: str
    baseline_pv: float
    stressed_pv: float
    pv_change: float
    pv_change_pct: float
    baseline_vol: float
    stressed_vol: float


@dataclass
class PortfolioStressReport:
    """Complete stress test report for a portfolio."""

    scenario_name: str
    equity_shock: float
    vol_shock: float
    rate_shock: float
    results: list[StressedResult]
    total_baseline_pv: float
    total_stressed_pv: float
    total_pv_change: float
    total_pv_change_pct: float


def create_sample_portfolio() -> list[FIAProduct | RILAProduct]:
    """
    Create a sample portfolio of indexed annuity products.

    Returns a mix of FIA and RILA products with different features
    to show varied stress responses.
    """
    products = [
        # FIA with 10% cap - conservative
        FIAProduct(
            company_name="Alpha Insurance",
            product_name="FIA 10% Cap",
            product_group="FIA",
            status="current",
            cap_rate=0.10,
            index_used="S&P 500",
            term_years=1,
        ),
        # FIA with participation - more aggressive
        FIAProduct(
            company_name="Beta Life",
            product_name="FIA 80% Participation",
            product_group="FIA",
            status="current",
            participation_rate=0.80,
            index_used="S&P 500",
            term_years=1,
        ),
        # RILA with 10% buffer
        RILAProduct(
            company_name="Gamma Annuity",
            product_name="RILA 10% Buffer",
            product_group="RILA",
            status="current",
            buffer_rate=0.10,
            buffer_modifier="Losses Covered Up To",
            cap_rate=0.15,
            index_used="S&P 500",
            term_years=6,
        ),
        # RILA with 20% buffer - more protection
        RILAProduct(
            company_name="Delta Insurance",
            product_name="RILA 20% Buffer",
            product_group="RILA",
            status="current",
            buffer_rate=0.20,
            buffer_modifier="Losses Covered Up To",
            cap_rate=0.12,
            index_used="S&P 500",
            term_years=6,
        ),
        # RILA with floor protection
        RILAProduct(
            company_name="Epsilon Life",
            product_name="RILA -10% Floor",
            product_group="RILA",
            status="current",
            buffer_rate=0.10,
            buffer_modifier="Losses Covered After",  # Floor
            cap_rate=0.20,
            index_used="S&P 500",
            term_years=6,
        ),
    ]
    return products


def create_baseline_market() -> MarketParams:
    """Create baseline (pre-stress) market parameters."""
    return MarketParams(
        spot=4500.0,  # S&P 500 level
        risk_free_rate=0.045,  # 4.5% (pre-GFC rates)
        dividend_yield=0.02,
        volatility=0.18,  # Normal VIX ~18
    )


def create_stressed_market(
    baseline: MarketParams,
    crisis: HistoricalCrisis,
) -> MarketParams:
    """
    Apply historical crisis shocks to market parameters.

    Parameters
    ----------
    baseline : MarketParams
        Pre-stress market conditions
    crisis : HistoricalCrisis
        Historical crisis to apply

    Returns
    -------
    MarketParams
        Stressed market conditions
    """
    # Apply shocks
    stressed_spot = baseline.spot * (1 + crisis.equity_shock)
    stressed_rate = max(0.001, baseline.risk_free_rate + crisis.rate_shock)

    # VIX peak → implied volatility
    # VIX ≈ annualized 30-day implied vol × 100
    # VIX of 80 → implied vol of ~0.80
    stressed_vol = crisis.vix_peak / 100

    return MarketParams(
        spot=stressed_spot,
        risk_free_rate=stressed_rate,
        dividend_yield=baseline.dividend_yield,
        volatility=stressed_vol,
    )


def price_portfolio(
    products: list[FIAProduct | RILAProduct],
    market: MarketParams,
    n_paths: int = 10000,
    seed: int | None = 42,
) -> dict[str, float]:
    """
    Price all products in portfolio under given market conditions.

    Returns dictionary mapping product name to present value.
    """
    results = {}

    fia_pricer = FIAPricer(market_params=market, n_mc_paths=n_paths, seed=seed)
    rila_pricer = RILAPricer(market_params=market, n_mc_paths=n_paths, seed=seed)

    for product in products:
        if isinstance(product, FIAProduct):
            result = fia_pricer.price(product)
        else:
            result = rila_pricer.price(product)

        results[product.product_name] = result.present_value

    return results


def run_stress_test(
    products: list[FIAProduct | RILAProduct],
    crisis: HistoricalCrisis,
    n_paths: int = 10000,
    seed: int | None = 42,
) -> PortfolioStressReport:
    """
    Run stress test on portfolio using historical crisis scenario.

    Parameters
    ----------
    products : list
        Portfolio of FIA/RILA products
    crisis : HistoricalCrisis
        Historical crisis scenario to apply
    n_paths : int
        Monte Carlo paths for pricing
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    PortfolioStressReport
        Complete stress test results
    """
    baseline_market = create_baseline_market()
    stressed_market = create_stressed_market(baseline_market, crisis)

    # Price under both scenarios
    baseline_prices = price_portfolio(products, baseline_market, n_paths, seed)
    stressed_prices = price_portfolio(products, stressed_market, n_paths, seed)

    # Build individual results
    results = []
    for product in products:
        name = product.product_name
        baseline_pv = baseline_prices[name]
        stressed_pv = stressed_prices[name]
        pv_change = stressed_pv - baseline_pv
        pv_change_pct = pv_change / baseline_pv if baseline_pv != 0 else 0

        results.append(
            StressedResult(
                product_name=name,
                product_type=product.product_group,
                baseline_pv=baseline_pv,
                stressed_pv=stressed_pv,
                pv_change=pv_change,
                pv_change_pct=pv_change_pct,
                baseline_vol=baseline_market.volatility,
                stressed_vol=stressed_market.volatility,
            )
        )

    # Calculate totals
    total_baseline = sum(baseline_prices.values())
    total_stressed = sum(stressed_prices.values())
    total_change = total_stressed - total_baseline
    total_change_pct = total_change / total_baseline if total_baseline != 0 else 0

    return PortfolioStressReport(
        scenario_name=crisis.display_name,
        equity_shock=crisis.equity_shock,
        vol_shock=(stressed_market.volatility - baseline_market.volatility),
        rate_shock=crisis.rate_shock,
        results=results,
        total_baseline_pv=total_baseline,
        total_stressed_pv=total_stressed,
        total_pv_change=total_change,
        total_pv_change_pct=total_change_pct,
    )


def print_scenario_header(crisis: HistoricalCrisis) -> None:
    """Print scenario information header."""
    print("\n" + "=" * 70)
    print(f"SCENARIO: {crisis.display_name}")
    print("=" * 70)
    print(f"\n  Period:         {crisis.start_date} to {crisis.end_date}")
    print(f"  Duration:       {crisis.duration_months} months to trough")
    print(f"  Recovery:       {crisis.recovery_months} months to pre-crisis level")
    print(f"\n  Equity Shock:   {crisis.equity_shock:.1%}")
    print(f"  Rate Change:    {crisis.rate_shock * 10000:.0f} basis points")
    print(f"  VIX Peak:       {crisis.vix_peak:.1f}")
    print(f"\n  Notes: {crisis.notes}")


def print_stress_results(report: PortfolioStressReport) -> None:
    """Print stress test results table."""
    print("\n" + "=" * 70)
    print("PORTFOLIO STRESS TEST RESULTS")
    print("=" * 70)

    print(f"\n  Scenario: {report.scenario_name}")
    print(f"  Equity:   {report.equity_shock:.1%}")
    print(f"  Vol:      +{report.vol_shock:.1%} (baseline → stressed)")
    print(f"  Rates:    {report.rate_shock * 10000:+.0f} bp")

    print("\n  {:<25} {:>10} {:>12} {:>12} {:>10}".format(
        "Product", "Type", "Baseline PV", "Stressed PV", "Change"
    ))
    print("  " + "-" * 70)

    for r in report.results:
        print("  {:<25} {:>10} {:>11,.0f} {:>11,.0f} {:>+9.1%}".format(
            r.product_name[:25],
            r.product_type,
            r.baseline_pv,
            r.stressed_pv,
            r.pv_change_pct,
        ))

    print("  " + "-" * 70)
    print("  {:<25} {:>10} {:>11,.0f} {:>11,.0f} {:>+9.1%}".format(
        "TOTAL",
        "",
        report.total_baseline_pv,
        report.total_stressed_pv,
        report.total_pv_change_pct,
    ))


def print_interpretation(report: PortfolioStressReport) -> None:
    """Print interpretation of stress test results."""
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    print("\n  1. Portfolio Impact:")
    print(f"     - Total portfolio PV changes by {report.total_pv_change_pct:+.1%}")

    # Find most/least impacted
    sorted_results = sorted(report.results, key=lambda r: r.pv_change_pct, reverse=True)
    most_positive = sorted_results[0]
    most_negative = sorted_results[-1]

    print(f"\n  2. Best performing: {most_positive.product_name}")
    print(f"     - Change: {most_positive.pv_change_pct:+.1%}")
    if most_positive.product_type == "FIA":
        print("     - FIA benefits from vol spike (higher option value)")

    print(f"\n  3. Worst performing: {most_negative.product_name}")
    print(f"     - Change: {most_negative.pv_change_pct:+.1%}")
    if most_negative.product_type == "RILA":
        print("     - RILA buffer cost increases with vol (more expensive protection)")

    print("\n  Key Insights:")
    print("  - Vol spike (+63%): Increases option values (vega effect)")
    print("  - Equity drop (-57%): Reduces intrinsic value")
    print("  - FIA options: Vol effect often dominates -> positive PV change")
    print("  - RILA buffers: Higher vol = more expensive -> larger loss for insurer")


def generate_markdown_report(report: PortfolioStressReport) -> str:
    """Generate markdown report for stress test results."""
    lines = [
        f"# Stress Test Report: {report.scenario_name}",
        "",
        f"**Generated:** {date.today().isoformat()}",
        "",
        "## Scenario Parameters",
        "",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| Equity Shock | {report.equity_shock:.1%} |",
        f"| Volatility Shock | +{report.vol_shock:.1%} |",
        f"| Rate Change | {report.rate_shock * 10000:+.0f} bp |",
        "",
        "## Portfolio Results",
        "",
        "| Product | Type | Baseline PV | Stressed PV | Change |",
        "|---------|------|-------------|-------------|--------|",
    ]

    for r in report.results:
        lines.append(
            f"| {r.product_name} | {r.product_type} | "
            f"${r.baseline_pv:,.0f} | ${r.stressed_pv:,.0f} | "
            f"{r.pv_change_pct:+.1%} |"
        )

    lines.extend([
        f"| **TOTAL** | | **${report.total_baseline_pv:,.0f}** | "
        f"**${report.total_stressed_pv:,.0f}** | "
        f"**{report.total_pv_change_pct:+.1%}** |",
        "",
        "## Summary",
        "",
        f"Under the {report.scenario_name} scenario:",
        f"- Total portfolio present value changes from ${report.total_baseline_pv:,.0f} to ${report.total_stressed_pv:,.0f}",
        f"- Net change: ${report.total_pv_change:+,.0f} ({report.total_pv_change_pct:+.1%})",
        "",
    ])

    return "\n".join(lines)


def main() -> None:
    """Run stress testing demo."""
    parser = argparse.ArgumentParser(description="Portfolio Stress Testing Demo")
    parser.add_argument("--ci", action="store_true", help="CI mode (fewer paths)")
    parser.add_argument("--paths", type=int, default=10000, help="MC paths (default: 10000)")
    parser.add_argument("--save-report", action="store_true", help="Save markdown report")
    args = parser.parse_args()

    # Use fewer paths in CI mode
    n_paths = 1000 if args.ci else args.paths
    seed = 42

    print("\n" + "=" * 60)
    print("PORTFOLIO STRESS TESTING DEMO")
    print("=" * 60)
    print("\nThis demo stress tests a portfolio of indexed annuity products")
    print("using the 2008 Global Financial Crisis scenario.")
    print(f"\nSimulation settings: {n_paths:,} MC paths, seed={seed}")

    # Create portfolio
    portfolio = create_sample_portfolio()
    print(f"\nPortfolio: {len(portfolio)} products")
    for p in portfolio:
        print(f"  - {p.product_name} ({p.product_group})")

    # Print scenario details
    print_scenario_header(CRISIS_2008_GFC)

    # Run stress test
    print("\nRunning stress test...")
    report = run_stress_test(portfolio, CRISIS_2008_GFC, n_paths=n_paths, seed=seed)

    # Print results
    print_stress_results(report)
    print_interpretation(report)

    # Save markdown report if requested
    if args.save_report:
        md_report = generate_markdown_report(report)
        report_path = "examples/stress_test_report.md"
        with open(report_path, "w") as f:
            f.write(md_report)
        print(f"\nReport saved to: {report_path}")

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
