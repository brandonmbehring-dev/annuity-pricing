#!/usr/bin/env python3
"""
GLWB Valuation Demo.

This example demonstrates how to price Guaranteed Lifetime Withdrawal Benefits
(GLWBs) using path-dependent Monte Carlo simulation.

Key Concepts:
- GLWB provides guaranteed income for life, continuing even if the account
  value is exhausted (ruin).
- Pricing requires Monte Carlo simulation because the payoff depends on the
  entire path of account values, not just terminal value.
- Guarantee cost = E[PV(insurer payments when AV exhausted)]

See: docs/knowledge/domain/glwb_mechanics.md
See: docs/references/L3/bauer_kling_russ_2008.md

Usage:
    python examples/03_glwb_valuation.py          # Full demo
    python examples/03_glwb_valuation.py --ci     # CI mode (no interactive)
"""

import argparse
import sys
from dataclasses import dataclass

# Add src to path if running as script
sys.path.insert(0, "src")

from annuity_pricing import GLWBPricer, GLWBProduct, GLWBPricingResult


@dataclass
class GLWBSensitivityResult:
    """Results from sensitivity analysis."""

    parameter: str
    value: float
    guarantee_cost: float
    prob_ruin: float
    present_value: float


def create_sample_products() -> list[GLWBProduct]:
    """
    Create sample GLWB products for demonstration.

    Returns a range of products with different features to show how
    guarantee costs vary with product parameters.
    """
    products = [
        # Standard product - baseline
        GLWBProduct(
            company_name="Standard Life",
            product_name="GLWB Standard",
            product_group="GLWB",
            status="current",
            withdrawal_rate=0.05,
            rollup_rate=0.06,
            rollup_type="compound",
            rollup_cap_years=10,
            step_up_frequency=1,
            fee_rate=0.01,
        ),
        # Aggressive product - higher withdrawal, higher rollup
        GLWBProduct(
            company_name="Aggressive Life",
            product_name="GLWB Aggressive",
            product_group="GLWB",
            status="current",
            withdrawal_rate=0.06,
            rollup_rate=0.08,
            rollup_type="compound",
            rollup_cap_years=10,
            step_up_frequency=1,
            fee_rate=0.0125,
        ),
        # Conservative product - lower withdrawal, lower rollup
        GLWBProduct(
            company_name="Conservative Life",
            product_name="GLWB Conservative",
            product_group="GLWB",
            status="current",
            withdrawal_rate=0.04,
            rollup_rate=0.04,
            rollup_type="simple",
            rollup_cap_years=10,
            step_up_frequency=0,  # No ratchet
            fee_rate=0.0075,
        ),
    ]
    return products


def price_glwb_product(
    product: GLWBProduct,
    risk_free_rate: float = 0.04,
    volatility: float = 0.15,
    premium: float = 100_000.0,
    age: int = 65,
    n_paths: int = 10000,
    seed: int | None = 42,
) -> GLWBPricingResult:
    """
    Price a single GLWB product.

    Parameters
    ----------
    product : GLWBProduct
        GLWB product to price
    risk_free_rate : float
        Risk-free rate for discounting
    volatility : float
        Index volatility for GBM simulation
    premium : float
        Initial premium amount
    age : int
        Starting age of annuitant
    n_paths : int
        Number of Monte Carlo paths
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    GLWBPricingResult
        Pricing result with guarantee cost and risk metrics
    """
    pricer = GLWBPricer(
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        n_paths=n_paths,
        seed=seed,
    )

    return pricer.price(
        product=product,
        premium=premium,
        age=age,
    )


def rollup_rate_sensitivity(
    base_product: GLWBProduct,
    rollup_rates: list[float] = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08],
    n_paths: int = 10000,
    seed: int | None = 42,
) -> list[GLWBSensitivityResult]:
    """
    Analyze sensitivity of guarantee cost to rollup rate.

    Higher rollup rate → larger benefit base → higher guarantee cost.

    Note: Each iteration uses a fresh pricer instance to avoid state reuse.
    """
    results = []

    for rollup_rate in rollup_rates:
        # Create fresh pricer for each rollup rate (avoids RNG state issues)
        pricer = GLWBPricer(n_paths=n_paths, seed=seed)

        # Create product with modified rollup rate
        product = GLWBProduct(
            company_name=base_product.company_name,
            product_name=f"Rollup {rollup_rate:.0%}",
            product_group="GLWB",
            status="current",
            withdrawal_rate=base_product.withdrawal_rate,
            rollup_rate=rollup_rate,
            rollup_type=base_product.rollup_type,
            rollup_cap_years=base_product.rollup_cap_years,
            step_up_frequency=base_product.step_up_frequency,
            fee_rate=base_product.fee_rate,
        )

        result = pricer.price(product, premium=100_000, age=65)
        results.append(
            GLWBSensitivityResult(
                parameter="rollup_rate",
                value=rollup_rate,
                guarantee_cost=result.guarantee_cost,
                prob_ruin=result.prob_ruin,
                present_value=result.present_value,
            )
        )

    return results


def age_sensitivity(
    product: GLWBProduct,
    ages: list[int] = [55, 60, 65, 70, 75],
    n_paths: int = 10000,
    seed: int | None = 42,
) -> list[GLWBSensitivityResult]:
    """
    Analyze sensitivity of guarantee cost to starting age.

    Younger age → longer potential payout period → higher guarantee cost.

    Note: Each iteration uses a fresh pricer instance to avoid state reuse.
    """
    results = []

    for age in ages:
        # Create fresh pricer for each age (avoids RNG state issues)
        pricer = GLWBPricer(n_paths=n_paths, seed=seed)
        result = pricer.price(product, premium=100_000, age=age)
        results.append(
            GLWBSensitivityResult(
                parameter="age",
                value=float(age),
                guarantee_cost=result.guarantee_cost,
                prob_ruin=result.prob_ruin,
                present_value=result.present_value,
            )
        )

    return results


def print_product_comparison(products: list[GLWBProduct], results: list[GLWBPricingResult]) -> None:
    """Print comparison of multiple GLWB products."""
    print("\n" + "=" * 80)
    print("GLWB PRODUCT COMPARISON")
    print("=" * 80)

    print("\n  {:<25} {:>12} {:>12} {:>12} {:>12}".format(
        "Product", "Guarantee", "Prob Ruin", "PV Guarantee", "Fee Rate"
    ))
    print("  " + "-" * 75)

    for product, result in zip(products, results):
        print("  {:<25} {:>11.2%} {:>11.2%} {:>11,.0f} {:>11.2%}".format(
            product.product_name[:25],
            result.guarantee_cost,
            result.prob_ruin,
            result.present_value,
            product.fee_rate,
        ))

    print("\n  Notes:")
    print("  - Guarantee cost = % of premium that covers the lifetime income guarantee")
    print("  - Prob ruin = probability account value exhausted before death")
    print("  - PV Guarantee = present value of expected insurer payments")


def print_sensitivity_table(results: list[GLWBSensitivityResult], title: str) -> None:
    """Print sensitivity analysis results as a table."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    param = results[0].parameter
    if param == "rollup_rate":
        print("\n  Rollup Rate    Guarantee Cost    Prob Ruin    PV Guarantee")
        print("  " + "-" * 60)
        for r in results:
            print(f"      {r.value:5.1%}          {r.guarantee_cost:6.2%}          {r.prob_ruin:5.2%}       {r.present_value:>10,.0f}")
        # Note about known limitation
        print("\n  NOTE: Rollup rate sensitivity may show limited variation due to")
        print("        known implementation limitation in path_sim module.")
    else:
        print("\n  Age            Guarantee Cost    Prob Ruin    PV Guarantee")
        print("  " + "-" * 60)
        for r in results:
            print(f"      {int(r.value):3d}            {r.guarantee_cost:6.2%}          {r.prob_ruin:5.2%}       {r.present_value:>10,.0f}")


def print_detailed_result(product: GLWBProduct, result: GLWBPricingResult) -> None:
    """Print detailed pricing result for a single product."""
    print("\n" + "=" * 60)
    print(f"DETAILED PRICING: {product.product_name}")
    print("=" * 60)

    print("\nProduct Features:")
    print(f"  Withdrawal Rate:      {product.withdrawal_rate:.1%}")
    print(f"  Rollup Rate:          {product.rollup_rate:.1%} ({product.rollup_type})")
    print(f"  Rollup Cap:           {product.rollup_cap_years} years")
    print(f"  Step-up Frequency:    {'Annual' if product.step_up_frequency > 0 else 'None'}")
    print(f"  Annual Fee:           {product.fee_rate:.2%}")

    print("\nPricing Results:")
    print(f"  Guarantee Cost:       {result.guarantee_cost:.2%} of premium")
    print(f"  PV of Guarantee:      ${result.present_value:,.0f}")
    print(f"  Probability of Ruin:  {result.prob_ruin:.2%}")
    if result.mean_ruin_year > 0:
        print(f"  Mean Ruin Year:       {result.mean_ruin_year:.1f}")

    print("\nSimulation Details:")
    print(f"  Monte Carlo Paths:    {result.n_paths:,}")
    if result.details:
        if "standard_error" in result.details:
            print(f"  Standard Error:       ${result.details['standard_error']:,.0f}")

    # Interpretation
    print("\nInterpretation:")
    if result.guarantee_cost < 0.05:
        print("  -> Low guarantee cost: conservative product or high fees cover risk")
    elif result.guarantee_cost < 0.15:
        print("  -> Moderate guarantee cost: balanced risk/reward for insurer")
    else:
        print("  -> High guarantee cost: aggressive product features increase insurer risk")


def main() -> None:
    """Run GLWB valuation demo."""
    parser = argparse.ArgumentParser(description="GLWB Valuation Demo")
    parser.add_argument("--ci", action="store_true", help="CI mode (fewer paths)")
    parser.add_argument("--paths", type=int, default=10000, help="Number of MC paths (default: 10000)")
    args = parser.parse_args()

    # Use fewer paths in CI mode for speed
    n_paths = 1000 if args.ci else args.paths
    seed = 42  # Reproducibility

    print("\n" + "=" * 60)
    print("GLWB VALUATION DEMO")
    print("=" * 60)
    print("\nThis demo shows how to price Guaranteed Lifetime Withdrawal Benefits")
    print("using path-dependent Monte Carlo simulation.")
    print(f"\nSimulation settings: {n_paths:,} paths, seed={seed}")

    # Create sample products
    products = create_sample_products()

    # Price all products
    print("\nPricing sample products...")
    results = []
    for product in products:
        result = price_glwb_product(product, n_paths=n_paths, seed=seed)
        results.append(result)

    # Print product comparison
    print_product_comparison(products, results)

    # Detailed result for standard product
    print_detailed_result(products[0], results[0])

    # Sensitivity analysis: rollup rate
    print("\nRunning rollup rate sensitivity analysis...")
    rollup_sensitivity = rollup_rate_sensitivity(products[0], n_paths=n_paths, seed=seed)
    print_sensitivity_table(rollup_sensitivity, "ROLLUP RATE SENSITIVITY")
    print("\n  Insight: Higher rollup rate -> larger benefit base -> higher guarantee cost")

    # Sensitivity analysis: starting age
    print("\nRunning age sensitivity analysis...")
    age_sens = age_sensitivity(products[0], n_paths=n_paths, seed=seed)
    print_sensitivity_table(age_sens, "STARTING AGE SENSITIVITY")
    print("\n  Insight: Younger age -> longer potential payout -> higher guarantee cost")

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
