#!/usr/bin/env python3
"""
Generate GLWB golden vectors for Julia cross-validation.

Phase 0 of Julia port plan - creates reference values at monthly timesteps
for validating the Julia GLWB implementation against Python.

Output: tests/references/glwb_golden_monthly.csv
"""

import csv
from dataclasses import asdict
from pathlib import Path

import numpy as np

from annuity_pricing.glwb.path_sim import GLWBPathSimulator, GLWBPricingResult
from annuity_pricing.glwb.gwb_tracker import GWBConfig, RollupType


def generate_golden_vectors(
    output_path: Path,
    n_paths: int = 10000,
    seed: int = 42,
) -> None:
    """
    Generate GLWB golden vectors for Julia cross-validation.

    Creates test cases covering:
    - Different ages (55, 60, 65, 70)
    - Different rollup rates (3%, 5%, 7%)
    - Different volatilities (15%, 20%, 25%)
    - Different withdrawal rates (4%, 5%, 6%)
    - Both annual and monthly timesteps

    Parameters
    ----------
    output_path : Path
        Output CSV file path
    n_paths : int
        Number of MC paths per test case
    seed : int
        Random seed for reproducibility
    """
    print(f"Generating GLWB golden vectors with {n_paths} paths, seed={seed}")
    print()

    # Test case parameters
    ages = [55, 60, 65, 70]
    rollup_rates = [0.03, 0.05, 0.07]
    volatilities = [0.15, 0.20, 0.25]
    withdrawal_rates = [0.04, 0.05, 0.06]
    timesteps = [1, 12]  # Annual and monthly

    # Fixed parameters
    premium = 100_000
    r = 0.04  # Risk-free rate
    max_age = 100

    results = []

    for steps_per_year in timesteps:
        timestep_name = "annual" if steps_per_year == 1 else "monthly"
        print(f"Generating {timestep_name} timestep vectors...")

        for rollup_rate in rollup_rates:
            for withdrawal_rate in withdrawal_rates:
                config = GWBConfig(
                    rollup_type=RollupType.COMPOUND,
                    rollup_rate=rollup_rate,
                    rollup_cap_years=10,
                    ratchet_enabled=True,
                    withdrawal_rate=withdrawal_rate,
                    fee_rate=0.01,
                )

                for age in ages:
                    for sigma in volatilities:
                        # Create simulator with specific seed
                        sim = GLWBPathSimulator(
                            config,
                            n_paths=n_paths,
                            seed=seed,
                            steps_per_year=steps_per_year,
                        )

                        # Price with simple behavioral model (100% utilization)
                        result = sim.price(
                            premium=premium,
                            age=age,
                            r=r,
                            sigma=sigma,
                            max_age=max_age,
                            use_behavioral_models=False,
                            utilization_rate=1.0,
                        )

                        # Record test case
                        record = {
                            # Input parameters
                            "premium": premium,
                            "age": age,
                            "r": r,
                            "sigma": sigma,
                            "max_age": max_age,
                            "rollup_type": "compound",
                            "rollup_rate": rollup_rate,
                            "rollup_cap_years": 10,
                            "withdrawal_rate": withdrawal_rate,
                            "fee_rate": 0.01,
                            "steps_per_year": steps_per_year,
                            "n_paths": n_paths,
                            "seed": seed,
                            # Output values
                            "price": result.price,
                            "guarantee_cost": result.guarantee_cost,
                            "mean_payoff": result.mean_payoff,
                            "std_payoff": result.std_payoff,
                            "standard_error": result.standard_error,
                            "prob_ruin": result.prob_ruin,
                            "mean_ruin_year": result.mean_ruin_year,
                            "prob_lapse": result.prob_lapse,
                            "mean_lapse_year": result.mean_lapse_year,
                        }
                        results.append(record)

        print(f"  Generated {len([r for r in results if r['steps_per_year'] == steps_per_year])} {timestep_name} test cases")

    # Write to CSV
    print()
    print(f"Writing {len(results)} test cases to {output_path}")

    fieldnames = list(results[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("Done!")

    # Print summary statistics
    print()
    print("Summary of monthly timestep golden vectors:")
    monthly_results = [r for r in results if r["steps_per_year"] == 12]
    prices = [r["price"] for r in monthly_results]
    costs = [r["guarantee_cost"] for r in monthly_results]
    prob_ruins = [r["prob_ruin"] for r in monthly_results]

    print(f"  Price range: ${min(prices):,.2f} - ${max(prices):,.2f}")
    print(f"  Guarantee cost range: {min(costs):.2%} - {max(costs):.2%}")
    print(f"  P(ruin) range: {min(prob_ruins):.2%} - {max(prob_ruins):.2%}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate GLWB golden vectors")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/references/glwb_golden_monthly.csv"),
        help="Output CSV file path",
    )
    parser.add_argument(
        "--n-paths",
        type=int,
        default=10000,
        help="Number of MC paths per test case",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()
    generate_golden_vectors(args.output, args.n_paths, args.seed)
