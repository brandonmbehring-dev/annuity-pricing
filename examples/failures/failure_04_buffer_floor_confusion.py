"""
Failure Example 04: Buffer vs Floor Confusion
==============================================

WHAT GOES WRONG
---------------
Mixing up buffer and floor protection mechanics in RILA products,
leading to completely wrong payoffs and hedge positions.

WHY IT'S WRONG
--------------
Buffer and Floor are DIFFERENT protection types [T1]:

BUFFER: Insurer absorbs the FIRST X% of losses
- Client exposed to losses BEYOND the buffer
- Example: 10% buffer, -15% index return -> Client loses 5%
- Replication: Long ATM put - Short OTM put (put spread)

FLOOR: Insurer covers losses BEYOND X%
- Client exposed to FIRST X% of losses (up to floor level)
- Example: -10% floor, -15% index return -> Client loses 10%
- Replication: Long OTM put

Common confusion:
- Using floor mechanics when buffer is specified
- Using buffer mechanics when floor is specified
- Getting the put strikes wrong for hedging

THE FIX
-------
1. Buffer = insurer absorbs FIRST X% (client protected at bottom)
2. Floor = client protected at SOME level (client exposed at top)
3. Double-check option replication logic
4. Test with extreme scenarios

VALIDATION
----------
- Buffer: Any loss <= buffer should result in 0% loss to client
- Floor: Loss should never exceed floor level (in absolute terms)

References
----------
[T1] SEC Release No. IC-34708 (RILA Product Standards)
[T1] FINRA Regulatory Notice 22-08
"""

from __future__ import annotations


# =============================================================================
# THE WRONG WAY
# =============================================================================


def rila_payoff_WRONG_buffer_as_floor(
    index_return: float, buffer: float, cap: float
) -> float:
    """
    WRONG: Treats buffer like a floor.

    This incorrectly limits losses at the buffer level instead of
    having the buffer absorb first losses.
    """
    # WRONG: This is floor logic, not buffer logic!
    if index_return < 0:
        # Incorrectly limiting loss to buffer level
        credited = max(index_return, -buffer)
    else:
        credited = min(index_return, cap)

    return credited


def rila_payoff_WRONG_floor_as_buffer(
    index_return: float, floor: float, cap: float
) -> float:
    """
    WRONG: Treats floor like a buffer.

    This incorrectly absorbs first losses instead of limiting max loss.
    """
    # WRONG: This is buffer logic, not floor logic!
    if index_return < 0:
        # Incorrectly absorbing first losses
        loss = abs(index_return)
        if loss <= floor:
            credited = 0.0
        else:
            credited = -(loss - floor)
    else:
        credited = min(index_return, cap)

    return credited


# =============================================================================
# THE RIGHT WAY
# =============================================================================


def rila_buffer_payoff_CORRECT(
    index_return: float, buffer: float, cap: float
) -> float:
    """
    CORRECT buffer payoff [T1].

    Buffer absorbs FIRST X% of losses.
    Client exposed to losses beyond buffer.

    Replication: Long ATM put - Short (ATM - buffer) put
    """
    if index_return >= 0:
        # Gains capped
        return min(index_return, cap)
    else:
        # Losses: buffer absorbs first X%
        loss = abs(index_return)
        if loss <= buffer:
            # Buffer absorbs entire loss
            return 0.0
        else:
            # Client bears loss beyond buffer
            return -(loss - buffer)


def rila_floor_payoff_CORRECT(
    index_return: float, floor: float, cap: float
) -> float:
    """
    CORRECT floor payoff [T1].

    Floor limits maximum loss to X%.
    Client exposed to first X% of losses.

    Replication: Long (ATM - floor) OTM put
    """
    if index_return >= 0:
        # Gains capped
        return min(index_return, cap)
    else:
        # Losses: floor limits maximum loss
        return max(index_return, -floor)


# =============================================================================
# DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    # Test scenarios
    test_returns = [0.15, 0.05, 0.0, -0.05, -0.10, -0.15, -0.25, -0.40]
    buffer = 0.10  # 10% buffer
    floor = 0.10  # 10% floor
    cap = 0.12  # 12% cap

    print("=" * 80)
    print("Failure Example 04: Buffer vs Floor Confusion")
    print("=" * 80)
    print()
    print("Parameters:")
    print(f"  Buffer = {buffer:.0%} (absorbs first 10% of losses)")
    print(f"  Floor  = -{floor:.0%} (limits max loss to 10%)")
    print(f"  Cap    = {cap:.0%}")
    print()

    # Show the difference
    print("BUFFER Protection (absorbs FIRST X% of losses):")
    print("-" * 80)
    print(f"{'Index Return':>15} | {'Correct':>12} | {'Wrong (floor logic)':>18} | {'Match?':>8}")
    print("-" * 80)

    for ret in test_returns:
        correct = rila_buffer_payoff_CORRECT(ret, buffer, cap)
        wrong = rila_payoff_WRONG_buffer_as_floor(ret, buffer, cap)
        match = "Yes" if abs(correct - wrong) < 0.0001 else "NO!"

        print(f"{ret:>15.2%} | {correct:>12.2%} | {wrong:>18.2%} | {match:>8}")

    print()
    print("FLOOR Protection (limits MAX loss to X%):")
    print("-" * 80)
    print(f"{'Index Return':>15} | {'Correct':>12} | {'Wrong (buffer logic)':>18} | {'Match?':>8}")
    print("-" * 80)

    for ret in test_returns:
        correct = rila_floor_payoff_CORRECT(ret, floor, cap)
        wrong = rila_payoff_WRONG_floor_as_buffer(ret, floor, cap)
        match = "Yes" if abs(correct - wrong) < 0.0001 else "NO!"

        print(f"{ret:>15.2%} | {correct:>12.2%} | {wrong:>18.2%} | {match:>8}")

    print()
    print("=" * 80)
    print("CRITICAL DIFFERENCE:")
    print()
    print("  Index Return: -25%")
    print(f"  BUFFER (10%): Client loses {-rila_buffer_payoff_CORRECT(-0.25, buffer, cap):.0%}")
    print(f"                (Buffer absorbed first 10%, client bears remaining 15%)")
    print(f"  FLOOR (-10%): Client loses {-rila_floor_payoff_CORRECT(-0.25, floor, cap):.0%}")
    print(f"                (Client's loss capped at floor level)")
    print()
    print("  This is a 5% difference in client loss!")
    print()
    print("HEDGING IMPLICATIONS:")
    print("  Buffer: Requires a PUT SPREAD (long ATM put, short OTM put)")
    print("  Floor:  Requires a single OTM PUT")
    print()
    print("  Using wrong hedge = unhedged exposure to the difference!")
    print("=" * 80)
