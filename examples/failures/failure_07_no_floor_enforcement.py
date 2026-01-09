"""
Failure Example 07: No Floor Enforcement in FIA
================================================

WHAT GOES WRONG
---------------
FIA (Fixed Indexed Annuity) products that allow negative credited returns
when the floor should enforce a minimum of 0%.

WHY IT'S WRONG
--------------
FIA products have a guaranteed 0% floor [T1]:
- Principal is protected (floor = 0%)
- Credited return = max(0, capped index return)
- Client NEVER loses principal due to index performance

Common mistakes:
1. Forgetting to apply the floor
2. Applying floor before cap (wrong order)
3. Using floor incorrectly for spreads/participation

A negative FIA credit means:
1. Contract violation (product design error)
2. Legal liability
3. Wrong option valuation (FIA = capped call, not put spread)

THE FIX
-------
ALWAYS apply floor LAST in FIA calculations:
1. Get index return
2. Apply cap/participation/spread
3. Apply floor: max(0, adjusted_return)

VALIDATION
----------
FIA credited return should NEVER be negative.

References
----------
[T1] SOA/LIMRA Annuity Study, FIA Product Design
[T1] NAIC Model Regulation for Index-Linked Annuities
"""

from __future__ import annotations


# =============================================================================
# THE WRONG WAYS
# =============================================================================


def fia_capped_payoff_WRONG_no_floor(
    index_return: float, cap: float
) -> float:
    """
    WRONG: Cap without floor enforcement.

    Negative returns pass through uncapped!
    """
    # WRONG: Only caps upside, doesn't floor downside
    return min(index_return, cap)


def fia_participation_payoff_WRONG_no_floor(
    index_return: float, participation_rate: float
) -> float:
    """
    WRONG: Participation rate without floor.

    Negative returns are multiplied by participation!
    """
    # WRONG: Applies participation to negative returns too
    return index_return * participation_rate


def fia_spread_payoff_WRONG_no_floor(
    index_return: float, spread: float
) -> float:
    """
    WRONG: Spread without floor.

    Can result in negative credited returns!
    """
    # WRONG: Spread can push positive return negative
    return index_return - spread


def fia_payoff_WRONG_floor_before_cap(
    index_return: float, cap: float
) -> float:
    """
    WRONG: Applying floor BEFORE cap.

    This is wrong because floor should be applied to final credited amount.
    """
    # WRONG ORDER: floor first, then cap
    floored = max(0, index_return)
    return min(floored, cap)  # This happens to work but is conceptually wrong


# =============================================================================
# THE RIGHT WAY
# =============================================================================


def fia_capped_payoff_CORRECT(
    index_return: float, cap: float
) -> float:
    """
    CORRECT FIA capped payoff [T1].

    Payoff = max(0, min(index_return, cap))

    Order: cap first, then floor
    """
    # Apply cap
    capped = min(index_return, cap)
    # Apply 0% floor (ALWAYS)
    return max(0, capped)


def fia_participation_payoff_CORRECT(
    index_return: float, participation_rate: float
) -> float:
    """
    CORRECT FIA participation payoff [T1].

    Payoff = max(0, index_return * participation_rate)
    """
    # Apply participation rate only to positive returns
    if index_return <= 0:
        return 0.0  # Floor kicks in
    return index_return * participation_rate


def fia_spread_payoff_CORRECT(
    index_return: float, spread: float
) -> float:
    """
    CORRECT FIA spread payoff [T1].

    Payoff = max(0, index_return - spread)

    Spread is subtracted from positive returns, floor protects against negative.
    """
    return max(0, index_return - spread)


def fia_compound_payoff_CORRECT(
    index_return: float,
    cap: float | None = None,
    participation_rate: float = 1.0,
    spread: float = 0.0,
) -> float:
    """
    CORRECT FIA compound payoff with multiple crediting features [T1].

    Order of operations:
    1. Apply participation rate
    2. Apply spread
    3. Apply cap
    4. Apply floor (LAST)
    """
    credited = index_return

    # 1. Participation rate
    credited = credited * participation_rate

    # 2. Spread
    credited = credited - spread

    # 3. Cap (if specified)
    if cap is not None:
        credited = min(credited, cap)

    # 4. Floor (ALWAYS LAST)
    credited = max(0, credited)

    return credited


# =============================================================================
# DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    # Test scenarios
    test_returns = [0.15, 0.10, 0.05, 0.02, 0.00, -0.05, -0.10, -0.20]
    cap = 0.08  # 8% cap
    participation = 0.80  # 80% participation
    spread = 0.02  # 2% spread

    print("=" * 85)
    print("Failure Example 07: No Floor Enforcement in FIA")
    print("=" * 85)
    print()
    print("FIA products have a 0% floor - credited return can NEVER be negative.")
    print()

    # Test capped payoff
    print("1. CAPPED PAYOFF (cap = 8%)")
    print("-" * 85)
    print(f"{'Index Return':>15} | {'Correct':>10} | {'Wrong (no floor)':>18} | {'Violation?':>12}")
    print("-" * 85)

    for ret in test_returns:
        correct = fia_capped_payoff_CORRECT(ret, cap)
        wrong = fia_capped_payoff_WRONG_no_floor(ret, cap)
        violation = "YES!" if wrong < 0 else ""

        print(f"{ret:>15.2%} | {correct:>10.2%} | {wrong:>18.2%} | {violation:>12}")

    print()

    # Test participation payoff
    print("2. PARTICIPATION PAYOFF (rate = 80%)")
    print("-" * 85)
    print(f"{'Index Return':>15} | {'Correct':>10} | {'Wrong (no floor)':>18} | {'Violation?':>12}")
    print("-" * 85)

    for ret in test_returns:
        correct = fia_participation_payoff_CORRECT(ret, participation)
        wrong = fia_participation_payoff_WRONG_no_floor(ret, participation)
        violation = "YES!" if wrong < 0 else ""

        print(f"{ret:>15.2%} | {correct:>10.2%} | {wrong:>18.2%} | {violation:>12}")

    print()

    # Test spread payoff
    print("3. SPREAD PAYOFF (spread = 2%)")
    print("-" * 85)
    print(f"{'Index Return':>15} | {'Correct':>10} | {'Wrong (no floor)':>18} | {'Violation?':>12}")
    print("-" * 85)

    for ret in test_returns:
        correct = fia_spread_payoff_CORRECT(ret, spread)
        wrong = fia_spread_payoff_WRONG_no_floor(ret, spread)
        violation = "YES!" if wrong < 0 else ""

        print(f"{ret:>15.2%} | {correct:>10.2%} | {wrong:>18.2%} | {violation:>12}")

    print()
    print("=" * 85)
    print("CRITICAL: Negative FIA credits are IMPOSSIBLE by product design.")
    print()
    print("Why this matters for pricing:")
    print("  - FIA embedded option = capped call option (with 0 floor)")
    print("  - Payoff: max(0, min(index_return, cap))")
    print("  - This is NOT a put spread (that's RILA with buffer)")
    print()
    print("Testing recommendation:")
    print("  - Add property test: assert all FIA credits >= 0")
    print("  - Test with extreme negative returns")
    print("  - Verify floor is applied AFTER other adjustments")
    print("=" * 85)
