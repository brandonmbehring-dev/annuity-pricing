#!/usr/bin/env python3
"""
Generate synthetic annuity product data mimicking WINK distributions.

Phase 0 of Julia port plan - creates reference values for CI testing
without requiring licensed WINK data.

Output: tests/references/synthetic_rates.csv

WINK Distribution Targets (from WINK_DATA_DICTIONARY.md):
- MYGA: 20% (fixedRate 3-6%, guaranteeDuration 3-10yr)
- FIA: 55% (capRate 5-15%, participationRate 20-120%)
- RILA: 25% (bufferRate 10-100%, capRate 10-25%)
"""

import csv
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np

# AM Best ratings with approximate market distribution
AM_BEST_RATINGS = ["A++", "A+", "A", "A-", "B++", "B+"]
AM_BEST_WEIGHTS = [0.15, 0.35, 0.30, 0.15, 0.04, 0.01]

# Company names (synthetic)
COMPANIES = [
    "Alpha Life Insurance",
    "Beta Annuity Corp",
    "Gamma Financial",
    "Delta Insurance Co",
    "Epsilon Life",
    "Zeta Retirement Services",
    "Eta Annuity Group",
    "Theta Life Insurance",
    "Iota Financial Services",
    "Kappa Retirement Corp",
]

# Indices used for FIA/RILA (weighted by WINK prevalence)
INDICES = ["S&P 500", "Russell 2000", "NASDAQ-100", "MSCI EAFE", "DJIA"]
INDEX_WEIGHTS = [0.45, 0.20, 0.15, 0.12, 0.08]

# Crediting methods for FIA
CREDITING_METHODS = [
    "Annual Point to Point",
    "Monthly Average",
    "Monthly Point to Point",
    "Performance Triggered",
]
CREDITING_WEIGHTS = [0.55, 0.20, 0.15, 0.10]

# Product type mappings
PRODUCT_TYPE_IDS = {"MYGA": 3.0, "FIA": 1.0, "RILA": 4.0}
PRODUCT_TYPE_NAMES = {
    "MYGA": "Multi-Year Guarantee",
    "FIA": "Fixed Indexed",
    "RILA": "Registered Index-Linked",
}
RATE_TYPES = {"MYGA": "Fixed", "FIA": "Indexed", "RILA": "Structured"}


def generate_synthetic_rates(
    n_products: int = 200,
    seed: int = 42,
    output_path: Optional[Path] = None,
) -> list[dict]:
    """
    Generate synthetic annuity product data mimicking WINK distributions.

    Parameters
    ----------
    n_products : int
        Total number of products to generate
    seed : int
        Random seed for reproducibility
    output_path : Path, optional
        Output CSV path. If None, returns records without writing.

    Returns
    -------
    list[dict]
        List of product records
    """
    rng = np.random.default_rng(seed)

    # WINK-mimicking product mix
    # Excluding IVA (42%) and FA (1%) which aren't in scope
    # Rebalanced: MYGA ~20%, FIA ~55%, RILA ~25%
    n_myga = int(n_products * 0.20)
    n_fia = int(n_products * 0.55)
    n_rila = n_products - n_myga - n_fia

    print(f"Generating {n_products} synthetic products (seed={seed})")
    print(f"  MYGA: {n_myga} ({n_myga / n_products:.1%})")
    print(f"  FIA:  {n_fia} ({n_fia / n_products:.1%})")
    print(f"  RILA: {n_rila} ({n_rila / n_products:.1%})")
    print()

    records = []

    # Generate MYGA products
    for i in range(n_myga):
        records.append(_generate_myga_record(rng, i))

    # Generate FIA products
    for i in range(n_fia):
        records.append(_generate_fia_record(rng, n_myga + i))

    # Generate RILA products
    for i in range(n_rila):
        records.append(_generate_rila_record(rng, n_myga + n_fia + i))

    # Shuffle to mix product types
    rng.shuffle(records)

    # Add sequential IDs after shuffling
    for i, record in enumerate(records):
        record["rateID"] = i + 1
        record["productID"] = i + 1

    if output_path:
        _write_csv(records, output_path)

    return records


def _generate_base_record(rng: np.random.Generator, idx: int, product_group: str) -> dict:
    """Generate base fields common to all product types."""
    company_idx = rng.integers(0, len(COMPANIES))

    return {
        # Product classification
        "productGroup": product_group,
        "productTypeID": PRODUCT_TYPE_IDS[product_group],
        "productTypeName": PRODUCT_TYPE_NAMES[product_group],
        "productName": f"Synthetic {product_group} Product {idx}",
        "productMarketStatus": "Actively Marketed",
        "rateType": RATE_TYPES[product_group],
        "status": rng.choice(["current", "historic"], p=[0.85, 0.15]),
        # Company info
        "companyID": float(company_idx + 1),
        "companyName": COMPANIES[company_idx],
        "amBestRating": rng.choice(AM_BEST_RATINGS, p=AM_BEST_WEIGHTS),
        # Temporal
        "effectiveDate": date(
            2020 + rng.integers(0, 5),
            rng.integers(1, 13),
            rng.integers(1, 29),
        ).isoformat(),
        "date": date.today().isoformat(),
        # Surrender
        "surrChargeDuration": str(rng.choice([0, 5, 6, 7, 10], p=[0.25, 0.20, 0.15, 0.30, 0.10])),
        "mva": rng.choice(["Y", "N"], p=[0.39, 0.61]),
        # Metadata
        "_synthetic": True,
    }


def _generate_myga_record(rng: np.random.Generator, idx: int) -> dict:
    """Generate MYGA-specific fields matching WINK distributions."""
    record = _generate_base_record(rng, idx, "MYGA")

    # MYGA-specific: guaranteeDuration is always populated (100% in WINK)
    guarantee_duration = rng.choice([3, 5, 7, 10], p=[0.15, 0.40, 0.30, 0.15])

    # fixedRate: WINK mean 2.95%, median 2.7%, range 0.5%-51%
    # Using realistic distribution centered around 3-5%
    fixed_rate = rng.normal(0.042, 0.008)
    fixed_rate = np.clip(fixed_rate, 0.025, 0.065)

    # effectiveYield: WINK mean 3.06%, median 2.85%
    effective_yield = fixed_rate + rng.uniform(-0.003, 0.005)
    effective_yield = np.clip(effective_yield, 0.020, 0.060)

    record.update(
        {
            # MYGA core fields
            "fixedRate": round(fixed_rate, 4),
            "effectiveYield": round(effective_yield, 4),
            "guaranteeDuration": float(guarantee_duration),
            "termYears": guarantee_duration,
            # Premium band (WINK has tiered pricing)
            "premiumBand": rng.choice(["0", "100", "250"], p=[0.40, 0.35, 0.25]),
            "ratesBand": rng.choice(["< $100k", "$100k", "$250k+"], p=[0.40, 0.35, 0.25]),
            # MGSV fields (31% populated for MYGA in WINK)
            "mgsvBaseRate": 0.875 if rng.random() < 0.31 else None,
            "mgsvRate": round(rng.uniform(0.01, 0.03), 4) if rng.random() < 0.31 else None,
            "mgsvRateUpperBound": round(rng.uniform(0.03, 0.05), 4) if rng.random() < 0.31 else None,
            # FIA/RILA fields as None
            "capRate": None,
            "participationRate": None,
            "spreadRate": None,
            "performanceTriggeredRate": None,
            "bufferRate": None,
            "bufferModifier": None,
            "indexUsed": None,
            "indexingMethod": None,
            "indexCreditingFrequency": None,
            "bonusRate": None,
        }
    )
    return record


def _generate_fia_record(rng: np.random.Generator, idx: int) -> dict:
    """Generate FIA-specific fields matching WINK distributions."""
    record = _generate_base_record(rng, idx, "FIA")

    # Crediting method (weighted by WINK prevalence)
    crediting_method = rng.choice(CREDITING_METHODS, p=CREDITING_WEIGHTS)
    term_years = rng.choice([1, 2, 3, 5, 6], p=[0.45, 0.15, 0.15, 0.15, 0.10])

    # Index (weighted by WINK prevalence)
    index_used = rng.choice(INDICES, p=INDEX_WEIGHTS)

    # Initialize rate fields
    cap_rate = None
    participation_rate = None
    spread_rate = None
    trigger_rate = None

    # Different crediting structures based on method
    if crediting_method == "Performance Triggered":
        # 6% of FIA in WINK
        trigger_rate = round(rng.uniform(0.03, 0.08), 4)
    elif rng.random() < 0.42:
        # Cap-based (42% have cap in WINK)
        cap_rate = round(rng.uniform(0.05, 0.15), 4)
        participation_rate = 1.0
    else:
        # Participation-based (86% have participation in WINK)
        participation_rate = round(rng.uniform(0.30, 1.20), 2)
        # 7% have spread
        if rng.random() < 0.07:
            spread_rate = round(rng.uniform(0.005, 0.03), 4)

    # Bonus rate (low prevalence)
    bonus_rate = round(rng.uniform(0.01, 0.05), 4) if rng.random() < 0.10 else None

    record.update(
        {
            # FIA core fields
            "indexUsed": index_used,
            "indexingMethod": crediting_method,
            "indexCreditingFrequency": rng.choice(["Annual", "Daily", "Biennial"], p=[0.70, 0.20, 0.10]),
            "termYears": term_years,
            "capRate": cap_rate,
            "participationRate": participation_rate,
            "spreadRate": spread_rate,
            "performanceTriggeredRate": trigger_rate,
            "bonusRate": bonus_rate,
            # MGSV fields (99% populated for FIA in WINK)
            "mgsvBaseRate": 0.875,
            "mgsvRate": round(rng.uniform(0.01, 0.03), 4),
            "mgsvRateUpperBound": round(rng.uniform(0.03, 0.05), 4),
            # MYGA fields as None
            "fixedRate": None,
            "effectiveYield": None,
            "guaranteeDuration": None,
            "premiumBand": None,
            "ratesBand": None,
            # RILA fields as None
            "bufferRate": None,
            "bufferModifier": None,
        }
    )
    return record


def _generate_rila_record(rng: np.random.Generator, idx: int) -> dict:
    """Generate RILA-specific fields matching WINK distributions."""
    record = _generate_base_record(rng, idx, "RILA")

    # Buffer vs Floor (70% buffer, 30% floor in WINK)
    is_buffer = rng.random() < 0.70

    # Term years (RILA typically 1, 3, 6 years)
    term_years = rng.choice([1, 3, 6], p=[0.30, 0.40, 0.30])

    # Index (weighted)
    index_used = rng.choice(INDICES, p=INDEX_WEIGHTS)

    # Buffer/floor rates (WINK mean 14.9%)
    # Include 100% buffer as edge case (1% of records)
    if rng.random() < 0.01:
        buffer_rate = 1.0  # 100% buffer - full protection edge case
    else:
        buffer_rate = rng.choice([0.10, 0.15, 0.20, 0.25], p=[0.35, 0.35, 0.20, 0.10])

    # Cap rate (69% populated, higher than FIA due to RILA structure)
    cap_rate = round(rng.uniform(0.10, 0.25), 4) if rng.random() < 0.69 else None

    # Participation rate (81% populated)
    participation_rate = round(rng.uniform(0.80, 1.20), 2) if rng.random() < 0.81 else None

    record.update(
        {
            # RILA core fields
            "indexUsed": index_used,
            "indexingMethod": "Annual Point to Point",
            "indexCreditingFrequency": "Annual",
            "termYears": term_years,
            "bufferRate": buffer_rate,
            "bufferModifier": "Losses Covered Up To" if is_buffer else "Losses Covered After",
            "capRate": cap_rate,
            "participationRate": participation_rate,
            # Trigger rate (17% in RILA)
            "performanceTriggeredRate": round(rng.uniform(0.03, 0.08), 4) if rng.random() < 0.17 else None,
            "spreadRate": None,
            "bonusRate": None,
            # MGSV fields (3% populated for RILA in WINK)
            "mgsvBaseRate": 0.875 if rng.random() < 0.03 else None,
            "mgsvRate": round(rng.uniform(0.01, 0.03), 4) if rng.random() < 0.03 else None,
            "mgsvRateUpperBound": None,
            # MYGA fields as None
            "fixedRate": None,
            "effectiveYield": None,
            "guaranteeDuration": None,
            "premiumBand": None,
            "ratesBand": None,
        }
    )
    return record


def _write_csv(records: list[dict], output_path: Path) -> None:
    """Write records to CSV with consistent column ordering."""
    # Define column order (matches WINK data dictionary organization)
    column_order = [
        # Product classification
        "rateID",
        "productID",
        "productGroup",
        "productTypeID",
        "productTypeName",
        "productName",
        "productMarketStatus",
        "rateType",
        "status",
        # Company info
        "companyID",
        "companyName",
        "amBestRating",
        # Rate fields
        "fixedRate",
        "capRate",
        "participationRate",
        "spreadRate",
        "performanceTriggeredRate",
        "effectiveYield",
        "bonusRate",
        # Buffer/protection
        "bufferRate",
        "bufferModifier",
        # MGSV
        "mgsvBaseRate",
        "mgsvRate",
        "mgsvRateUpperBound",
        # Index/crediting
        "indexUsed",
        "indexingMethod",
        "indexCreditingFrequency",
        # Duration
        "guaranteeDuration",
        "termYears",
        "surrChargeDuration",
        "mva",
        # Premium
        "premiumBand",
        "ratesBand",
        # Temporal
        "effectiveDate",
        "date",
        # Metadata
        "_synthetic",
    ]

    # Ensure all columns exist in records
    for record in records:
        for col in column_order:
            if col not in record:
                record[col] = None

    print(f"Writing {len(records)} records to {output_path}")

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=column_order)
        writer.writeheader()
        writer.writerows(records)

    print("Done!")


def print_summary(records: list[dict]) -> None:
    """Print summary statistics for generated data."""
    print()
    print("=" * 60)
    print("Summary Statistics")
    print("=" * 60)

    # Product mix
    groups = {}
    for r in records:
        g = r["productGroup"]
        groups[g] = groups.get(g, 0) + 1

    print("\nProduct Mix:")
    for g, count in sorted(groups.items()):
        print(f"  {g}: {count} ({count / len(records):.1%})")

    # Rate statistics by product
    print("\nRate Statistics:")

    # MYGA
    myga = [r for r in records if r["productGroup"] == "MYGA"]
    fixed_rates = [r["fixedRate"] for r in myga if r["fixedRate"]]
    if fixed_rates:
        print(f"  MYGA fixedRate: min={min(fixed_rates):.2%}, max={max(fixed_rates):.2%}, mean={np.mean(fixed_rates):.2%}")

    # FIA
    fia = [r for r in records if r["productGroup"] == "FIA"]
    cap_rates = [r["capRate"] for r in fia if r["capRate"]]
    part_rates = [r["participationRate"] for r in fia if r["participationRate"]]
    if cap_rates:
        print(f"  FIA capRate: min={min(cap_rates):.2%}, max={max(cap_rates):.2%}, mean={np.mean(cap_rates):.2%}")
    if part_rates:
        print(f"  FIA participationRate: min={min(part_rates):.0%}, max={max(part_rates):.0%}, mean={np.mean(part_rates):.0%}")

    # RILA
    rila = [r for r in records if r["productGroup"] == "RILA"]
    buffer_rates = [r["bufferRate"] for r in rila if r["bufferRate"]]
    if buffer_rates:
        print(f"  RILA bufferRate: min={min(buffer_rates):.0%}, max={max(buffer_rates):.0%}, mean={np.mean(buffer_rates):.1%}")
        # Check for 100% buffer edge case
        full_buffers = [r for r in buffer_rates if r >= 0.99]
        if full_buffers:
            print(f"    (includes {len(full_buffers)} 100% buffer edge cases)")

    # Buffer modifier distribution
    buffer_types = {}
    for r in rila:
        mod = r.get("bufferModifier")
        if mod:
            buffer_types[mod] = buffer_types.get(mod, 0) + 1

    if buffer_types:
        print("\n  RILA Protection Types:")
        for mod, count in buffer_types.items():
            label = "Buffer" if "Up To" in mod else "Floor"
            print(f"    {label}: {count} ({count / len(rila):.0%})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic annuity rates")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/references/synthetic_rates.csv"),
        help="Output CSV file path",
    )
    parser.add_argument(
        "--n-products",
        type=int,
        default=200,
        help="Number of products to generate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()
    records = generate_synthetic_rates(args.n_products, args.seed, args.output)
    print_summary(records)
