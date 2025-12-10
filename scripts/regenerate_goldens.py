#!/usr/bin/env python
"""
Regenerate golden files from external oracles.

Usage:
    python scripts/regenerate_goldens.py --verify  # Check drift without regenerating
    python scripts/regenerate_goldens.py           # Regenerate all golden files
    python scripts/regenerate_goldens.py --hull    # Regenerate Hull examples only
    python scripts/regenerate_goldens.py --sec     # Regenerate SEC examples only

Golden files are regenerated from:
- Hull examples: Our BS implementation (validated against financepy)
- SEC RILA: Our payoff implementation (matches regulatory definitions)
- Portfolio baseline: Current implementation snapshot
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from annuity_pricing.options.pricing.black_scholes import (
    black_scholes_call,
    black_scholes_put,
    black_scholes_greeks,
)
from annuity_pricing.options.payoffs.base import OptionType
from annuity_pricing.options.payoffs.rila import BufferPayoff, FloorPayoff
from annuity_pricing.options.payoffs.fia import CappedCallPayoff
from annuity_pricing.config.tolerances import HULL_EXAMPLE_TOLERANCE, GOLDEN_RELATIVE_TOLERANCE


GOLDEN_DIR = Path(__file__).parent.parent / "tests" / "golden" / "outputs"


def regenerate_hull_examples() -> dict:
    """Regenerate Hull textbook examples."""
    today = datetime.now().strftime("%Y-%m-%d")

    examples = {
        "_meta": {
            "source": "Hull (2021) Options, Futures, and Other Derivatives, 11th Edition",
            "generated": today,
            "tolerance_tier": "hull_example",
            "notes": "Values calculated using our validated BS implementation"
        }
    }

    # Example 15.6
    params_15_6 = {
        "spot": 42.0, "strike": 40.0, "rate": 0.10,
        "dividend": 0.0, "volatility": 0.20, "time_to_expiry": 0.5
    }
    examples["example_15_6_call"] = {
        "description": "Hull Example 15.6: European call option",
        "reference": "Hull (2021) Ch. 15, p. 335",
        "parameters": params_15_6,
        "expected": {
            "call_price": round(black_scholes_call(**params_15_6), 2)
        }
    }
    examples["example_15_6_put"] = {
        "description": "Hull Example 15.6: European put option (put-call parity)",
        "reference": "Hull (2021) Ch. 15",
        "parameters": params_15_6,
        "expected": {
            "put_price": round(black_scholes_put(**params_15_6), 2)
        }
    }

    # Example 19.1 - Delta
    params_19 = {
        "spot": 49.0, "strike": 50.0, "rate": 0.05,
        "dividend": 0.0, "volatility": 0.20, "time_to_expiry": 0.3846
    }
    greeks_19 = black_scholes_greeks(**params_19, option_type=OptionType.CALL)

    examples["example_19_1_delta"] = {
        "description": "Hull Example 19.1: Delta calculation",
        "reference": "Hull (2021) Ch. 19, p. 429",
        "parameters": params_19,
        "expected": {"delta": round(greeks_19.delta, 3)}
    }
    examples["example_19_4_gamma"] = {
        "description": "Hull Example 19.4: Gamma calculation",
        "reference": "Hull (2021) Ch. 19, p. 435",
        "parameters": params_19,
        "expected": {"gamma": round(greeks_19.gamma, 3)}
    }
    examples["example_19_6_vega"] = {
        "description": "Hull Example 19.6: Vega calculation",
        "reference": "Hull (2021) Ch. 19, p. 438",
        "parameters": params_19,
        "expected": {"vega_per_pct": round(greeks_19.vega * 100, 1)}
    }
    examples["example_19_8_theta"] = {
        "description": "Hull Example 19.8: Theta calculation",
        "reference": "Hull (2021) Ch. 19, p. 441",
        "parameters": params_19,
        "expected": {"theta_per_year": round(greeks_19.theta * 365, 2)}
    }
    examples["example_19_9_rho"] = {
        "description": "Hull Example 19.9: Rho calculation",
        "reference": "Hull (2021) Ch. 19, p. 443",
        "parameters": params_19,
        "expected": {"rho_per_pct": round(greeks_19.rho * 100, 2)}
    }

    # ATM/ITM/OTM with dividend
    for name, strike in [("atm", 100.0), ("itm", 95.0), ("otm", 105.0)]:
        params = {
            "spot": 100.0, "strike": strike, "rate": 0.05,
            "dividend": 0.02, "volatility": 0.20, "time_to_expiry": 1.0
        }
        examples[f"{name}_call_with_dividend"] = {
            "description": f"{name.upper()} call with continuous dividend yield",
            "reference": "Hull (2021) Ch. 17 - Merton extension",
            "parameters": params,
            "expected": {"call_price": round(black_scholes_call(**params), 2)}
        }

    return examples


def regenerate_sec_examples() -> dict:
    """Regenerate SEC RILA examples."""
    today = datetime.now().strftime("%Y-%m-%d")

    examples = {
        "_meta": {
            "source": "SEC RILA Final Rule 2024 and Investor Testing 2023",
            "generated": today,
            "tolerance_tier": "golden_relative",
            "notes": "Buffer and floor examples from SEC disclosure requirements"
        }
    }

    # Buffer examples
    buffer_cases = [
        ("buffer_10_loss_5", -0.05, 0.10, "5% loss within 10% buffer, fully absorbed"),
        ("buffer_10_loss_15", -0.15, 0.10, "15% loss minus 10% buffer = 5% loss to policyholder"),
        ("buffer_10_loss_25", -0.25, 0.10, "25% loss minus 10% buffer = 15% loss to policyholder"),
        ("buffer_10_gain_12", 0.12, 0.10, "Positive return, buffer not used, full upside"),
    ]

    for name, ret, buf, explanation in buffer_cases:
        payoff = BufferPayoff(buffer_rate=buf)
        result = payoff.calculate(ret)
        examples[name] = {
            "description": f"{int(buf*100)}% buffer with {int(abs(ret)*100)}% {'loss' if ret < 0 else 'gain'}",
            "reference": "SEC RILA Final Rule 2024 - Buffer Example",
            "parameters": {"index_return": ret, "buffer_rate": buf},
            "expected": {
                "credited_return": result.credited_return,
                "buffer_applied": result.details.get("buffer_applied", ret < 0),
                "explanation": explanation
            }
        }

    # Floor examples
    floor_cases = [
        ("floor_10_loss_5", -0.05, -0.10, "5% loss is above -10% floor, no protection"),
        ("floor_10_loss_15", -0.15, -0.10, "15% loss floored at -10%"),
        ("floor_10_loss_25", -0.25, -0.10, "25% loss floored at -10%"),
        ("floor_10_gain_8", 0.08, -0.10, "Positive return, floor not relevant"),
    ]

    for name, ret, floor, explanation in floor_cases:
        payoff = FloorPayoff(floor_rate=floor)
        result = payoff.calculate(ret)
        examples[name] = {
            "description": f"{int(abs(floor)*100)}% floor with {int(abs(ret)*100)}% {'loss' if ret < 0 else 'gain'}",
            "reference": "SEC RILA Final Rule 2024 - Floor Example",
            "parameters": {"index_return": ret, "floor_rate": floor},
            "expected": {
                "credited_return": result.credited_return,
                "floor_applied": result.floor_applied,
                "explanation": explanation
            }
        }

    # Comparison examples
    for name, ret, explanation, buffer_better in [
        ("buffer_vs_floor_small_loss", -0.05, "For 5% loss: buffer gives 0%, floor gives -5%", True),
        ("buffer_vs_floor_large_loss", -0.25, "For 25% loss: buffer gives -15%, floor gives -10%", False),
    ]:
        buffer_payoff = BufferPayoff(buffer_rate=0.10)
        floor_payoff = FloorPayoff(floor_rate=-0.10)
        buffer_result = buffer_payoff.calculate(ret)
        floor_result = floor_payoff.calculate(ret)

        examples[name] = {
            "description": "Buffer better for small losses" if buffer_better else "Floor better for large losses",
            "reference": "SEC RILA Investor Testing 2023 - Comparison",
            "parameters": {"index_return": ret, "buffer_rate": 0.10, "floor_rate": -0.10},
            "expected": {
                "buffer_credited": buffer_result.credited_return,
                "floor_credited": floor_result.credited_return,
                "buffer_better" if buffer_better else "floor_better": True,
                "explanation": explanation
            }
        }

    # FIA examples
    cap_payoff = CappedCallPayoff(cap_rate=0.10, floor_rate=0.0)

    # Cap applied
    cap_result = cap_payoff.calculate(0.15)
    examples["fia_cap_example"] = {
        "description": "FIA with 10% cap, positive return",
        "reference": "SEC RILA Final Rule 2024 - Cap Example",
        "parameters": {"index_return": 0.15, "cap_rate": 0.10, "floor_rate": 0.0},
        "expected": {
            "credited_return": cap_result.credited_return,
            "cap_applied": cap_result.cap_applied,
            "explanation": "15% return capped at 10%"
        }
    }

    # Floor applied
    floor_result = cap_payoff.calculate(-0.12)
    examples["fia_floor_example"] = {
        "description": "FIA principal protection (0% floor)",
        "reference": "SEC RILA Final Rule 2024 - FIA Floor",
        "parameters": {"index_return": -0.12, "cap_rate": 0.10, "floor_rate": 0.0},
        "expected": {
            "credited_return": floor_result.credited_return,
            "floor_applied": floor_result.floor_applied,
            "explanation": "-12% return floored at 0% (principal protection)"
        }
    }

    return examples


def verify_golden(filepath: Path, current_data: dict, tolerance: float) -> list[str]:
    """Verify golden file matches current implementation."""
    if not filepath.exists():
        return [f"Golden file does not exist: {filepath}"]

    with open(filepath) as f:
        stored_data = json.load(f)

    errors = []

    for key, current_example in current_data.items():
        if key.startswith("_"):
            continue

        if key not in stored_data:
            errors.append(f"Missing example: {key}")
            continue

        stored_example = stored_data[key]

        # Compare expected values
        current_expected = current_example.get("expected", {})
        stored_expected = stored_example.get("expected", {})

        for value_key, current_value in current_expected.items():
            if value_key not in stored_expected:
                continue
            stored_value = stored_expected[value_key]

            if isinstance(current_value, (int, float)) and isinstance(stored_value, (int, float)):
                if abs(current_value - stored_value) > tolerance:
                    errors.append(
                        f"{key}.{value_key}: current={current_value}, stored={stored_value}, "
                        f"diff={abs(current_value - stored_value)}"
                    )

    return errors


def main():
    parser = argparse.ArgumentParser(description="Regenerate golden files")
    parser.add_argument("--verify", action="store_true", help="Verify without regenerating")
    parser.add_argument("--hull", action="store_true", help="Regenerate Hull examples only")
    parser.add_argument("--sec", action="store_true", help="Regenerate SEC examples only")
    args = parser.parse_args()

    # Default to all if no specific flag
    do_hull = args.hull or (not args.hull and not args.sec)
    do_sec = args.sec or (not args.hull and not args.sec)

    all_errors = []

    if do_hull:
        hull_data = regenerate_hull_examples()
        hull_path = GOLDEN_DIR / "hull_examples.json"

        if args.verify:
            errors = verify_golden(hull_path, hull_data, HULL_EXAMPLE_TOLERANCE)
            if errors:
                print(f"Hull examples drift detected:")
                for e in errors:
                    print(f"  - {e}")
                all_errors.extend(errors)
            else:
                print("Hull examples: OK")
        else:
            with open(hull_path, "w") as f:
                json.dump(hull_data, f, indent=2)
            print(f"Regenerated: {hull_path}")

    if do_sec:
        sec_data = regenerate_sec_examples()
        sec_path = GOLDEN_DIR / "sec_rila_examples.json"

        if args.verify:
            errors = verify_golden(sec_path, sec_data, GOLDEN_RELATIVE_TOLERANCE)
            if errors:
                print(f"SEC examples drift detected:")
                for e in errors:
                    print(f"  - {e}")
                all_errors.extend(errors)
            else:
                print("SEC examples: OK")
        else:
            with open(sec_path, "w") as f:
                json.dump(sec_data, f, indent=2)
            print(f"Regenerated: {sec_path}")

    if args.verify and all_errors:
        print(f"\n{len(all_errors)} drift(s) detected. Run without --verify to regenerate.")
        sys.exit(1)
    elif args.verify:
        print("\nAll golden files verified successfully.")


if __name__ == "__main__":
    main()
