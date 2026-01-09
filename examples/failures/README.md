# Failure-Driven Learning Examples

This directory contains pedagogical examples demonstrating common pricing mistakes
and how to avoid them. Each example follows the WHAT/WHY/FIX/VALIDATION pattern.

## Why Failure Examples?

Learning from failures is more effective than memorizing correct implementations.
These examples show:

1. **What goes wrong** when common mistakes are made
2. **Why it's wrong** from theoretical and practical perspectives
3. **How to fix it** with correct implementations
4. **How to validate** to catch these errors in testing

## The Examples

| File | Error Type | Severity | Key Lesson |
|------|------------|----------|------------|
| `failure_01_no_risk_neutral.py` | Wrong GBM drift | Critical | Use r-q, not historical mu |
| `failure_02_put_call_parity.py` | BS implementation bugs | Critical | Test C-P = S-Ke^(-rT) |
| `failure_03_arbitrage_bounds.py` | Impossible prices | Critical | Call <= Spot, always |
| `failure_04_buffer_floor_confusion.py` | Product mechanics | High | Buffer != Floor |
| `failure_05_mc_divergence.py` | MC implementation | High | Verify against BS |
| `failure_06_negative_vol.py` | Edge case handling | Medium | Validate all inputs |
| `failure_07_no_floor_enforcement.py` | FIA product design | High | FIA credit >= 0 always |

## Running the Examples

Each example is a standalone script that demonstrates the failure:

```bash
# Run all examples
for f in examples/failures/failure_*.py; do
    echo "Running $f"
    python "$f"
    echo ""
done

# Run a specific example
python examples/failures/failure_01_no_risk_neutral.py
```

## Pattern Structure

Each example follows this structure:

```python
"""
Failure Example NN: Title
=========================

WHAT GOES WRONG
---------------
Description of the error

WHY IT'S WRONG
--------------
Theoretical explanation with [T1] citations

THE FIX
-------
Correct implementation approach

VALIDATION
----------
How to test for this error

References
----------
[T1] Academic citations
"""

# Wrong implementation
def something_WRONG(...):
    ...

# Correct implementation
def something_CORRECT(...):
    ...

# Demonstration
if __name__ == "__main__":
    # Show both implementations and compare
    ...
```

## Integration with Test Suite

These examples are validated by the anti-pattern test suite:

```bash
# Run anti-pattern tests
pytest tests/anti_patterns/ -v
```

The tests in `tests/anti_patterns/` automatically catch the errors demonstrated here.

## Knowledge Tiers

All claims are tagged with knowledge tiers:

- **[T1] Academically Validated**: Based on peer-reviewed literature
- **[T2] Empirically Validated**: Verified against external libraries
- **[T3] Assumptions**: Working assumptions requiring sensitivity analysis

## Contributing

When adding new failure examples:

1. Follow the WHAT/WHY/FIX/VALIDATION pattern
2. Include both wrong and correct implementations
3. Add runnable demonstration code
4. Add corresponding anti-pattern test
5. Include proper [T1]/[T2]/[T3] citations
