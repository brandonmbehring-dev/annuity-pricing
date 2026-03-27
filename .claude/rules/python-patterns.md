---
paths:
  - "**/*.py"
---

## Python Code Patterns

### Error Handling (NEVER fail silently)

```python
# CORRECT:
if data.empty:
    raise ValueError(
        f"CRITICAL: Empty data in {function_name}. "
        f"Expected: DataFrame with data. Got: Empty."
    )

# WRONG (silent failure):
if data.empty:
    return pd.DataFrame()  # PROHIBITED
```

### Type Hints (Required)

```python
def price_call(
    spot: float, strike: float, rate: float, vol: float, time: float
) -> float:
    """Price European call using Black-Scholes [T1]."""
    ...
```

### Docstrings (NumPy style with tier tags)

```python
def price_buffer_payoff(
    index_return: float, buffer: float, cap: float
) -> float:
    """
    Calculate RILA buffer payoff.

    [T1] Buffer absorbs first X% of losses.
    See: docs/knowledge/domain/buffer_floor.md

    Parameters
    ----------
    index_return : float
        Index return over the term (decimal, e.g., -0.15 for -15%)
    buffer : float
        Buffer level (decimal, e.g., 0.10 for 10% buffer)
    cap : float
        Maximum return cap (decimal, e.g., 0.12 for 12% cap)

    Returns
    -------
    float
        Credited return (decimal)

    Examples
    --------
    >>> price_buffer_payoff(-0.05, 0.10, 0.12)  # -5% return, 10% buffer
    0.0  # Buffer absorbs the loss
    """
```
