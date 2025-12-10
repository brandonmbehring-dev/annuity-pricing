# Monte Carlo Methods for Option Pricing

**Tier**: L2 (Full Derivation)
**Domain**: Options Simulation
**Prerequisites**: Stochastic calculus, GBM, risk-neutral pricing
**Source**: [T1] Glasserman (2003), *Monte Carlo Methods in Financial Engineering*

---

## Geometric Brownian Motion (GBM)

### Stochastic Differential Equation [T1]

Under risk-neutral measure:

$$dS_t = (r - q) S_t \, dt + \sigma S_t \, dW_t$$

Where:
- $S_t$: Asset price at time $t$
- $r$: Risk-free rate
- $q$: Dividend yield
- $\sigma$: Volatility
- $W_t$: Standard Brownian motion

### Exact Solution [T1]

$$S_T = S_0 \exp\left[\left(r - q - \frac{\sigma^2}{2}\right)T + \sigma \sqrt{T} Z\right]$$

Where $Z \sim N(0, 1)$.

### Discrete Time Steps [T1]

For path generation with $n$ steps and $\Delta t = T/n$:

$$S_{t+\Delta t} = S_t \exp\left[\left(r - q - \frac{\sigma^2}{2}\right)\Delta t + \sigma \sqrt{\Delta t} Z_t\right]$$

---

## Monte Carlo Pricing

### European Option [T1]

$$V_0 = e^{-rT} \mathbb{E}^Q[\text{Payoff}(S_T)]$$

Estimated by:

$$\hat{V} = e^{-rT} \frac{1}{N} \sum_{i=1}^{N} \text{Payoff}(S_T^{(i)})$$

### Standard Error [T1]

$$SE = \frac{\hat{\sigma}}{\sqrt{N}}$$

Where $\hat{\sigma}$ is the sample standard deviation of discounted payoffs.

### Convergence Rate [T1]

MC error decreases as $O(N^{-1/2})$. To halve the error, quadruple the paths.

---

## Variance Reduction Techniques

### Antithetic Variates [T1]

For each random draw $Z$, also use $-Z$:

$$\hat{V}_{AV} = e^{-rT} \frac{1}{N} \sum_{i=1}^{N/2} \frac{\text{Payoff}(S_T^{Z_i}) + \text{Payoff}(S_T^{-Z_i})}{2}$$

**Variance reduction**: Up to 50% for monotonic payoffs.

### Control Variates [T1]

Use known quantity (e.g., forward price) to reduce variance:

$$\hat{V}_{CV} = \hat{V} - \beta(\hat{F} - F)$$

Where:
- $\hat{F}$: Sample mean of terminal values
- $F = S_0 e^{(r-q)T}$: Theoretical forward
- $\beta$: Optimal coefficient (estimated from pilot run)

---

## Path-Dependent Options

### Asian Options [T1]

For arithmetic average:

$$\text{Payoff} = \max\left(\frac{1}{n}\sum_{i=1}^{n} S_{t_i} - K, 0\right)$$

**Critical**: The averaging frequency must match product terms:
- **Monthly average**: 12 observations for 1-year term
- **Weekly average**: 52 observations
- **Daily average**: 252 observations

### Observation Schedule Design

**Current Implementation Issue**: The MC engine defaults to `n_steps=252` regardless of payoff type, causing monthly-average FIAs to average 252 points instead of 12.

**Recommended Design Pattern** (for future implementation):

```python
@dataclass(frozen=True)
class ObservationSchedule:
    """
    Defines observation points for path-dependent payoffs.

    Attributes
    ----------
    frequency : Literal["monthly", "quarterly", "weekly", "daily", "custom"]
        Standard frequency or custom
    n_observations : int
        Number of observation points
    observation_times : tuple[float, ...]
        Exact observation times as fraction of year
    """
    frequency: str
    n_observations: int
    observation_times: tuple[float, ...]

    @classmethod
    def monthly(cls, term_years: float = 1.0) -> "ObservationSchedule":
        """Monthly observations (12 per year)."""
        n = int(12 * term_years)
        times = tuple(i / 12 for i in range(1, n + 1))
        return cls("monthly", n, times)

    @classmethod
    def daily(cls, term_years: float = 1.0) -> "ObservationSchedule":
        """Daily observations (252 per year)."""
        n = int(252 * term_years)
        times = tuple(i / 252 for i in range(1, n + 1))
        return cls("daily", n, times)
```

**Integration with Payoffs**:

```python
class BasePayoff(ABC):
    @property
    def observation_schedule(self) -> Optional[ObservationSchedule]:
        """Return required observation schedule, or None for terminal-only."""
        return None

class MonthlyAveragePayoff(BasePayoff):
    @property
    def observation_schedule(self) -> ObservationSchedule:
        return ObservationSchedule.monthly(self.term_years)
```

**Integration with MC Engine**:

```python
def price_with_payoff(params, payoff, n_paths):
    schedule = payoff.observation_schedule
    if schedule is not None:
        n_steps = schedule.n_observations
        # Generate paths at observation times only
    else:
        n_steps = 252  # Default daily for path generation
```

---

## Convergence Testing [T1]

### Against Closed-Form Solutions

For vanilla European options, MC should converge to Black-Scholes:

$$|\hat{V}_{MC} - V_{BS}| < 3 \cdot SE$$

With 99.7% confidence.

### Convergence Diagnostics

| Paths | Expected Error (rel) | Notes |
|-------|---------------------|-------|
| 1,000 | ~3% | Quick tests only |
| 10,000 | ~1% | Standard accuracy |
| 100,000 | ~0.3% | High accuracy |
| 1,000,000 | ~0.1% | Research quality |

---

## Implementation Notes

### Module Location

```
src/annuity_pricing/options/simulation/
├── gbm.py          # GBM path generation
├── monte_carlo.py  # MC pricing engine
└── heston_paths.py # Heston stochastic vol paths
```

### Key Functions

| Function | Purpose |
|----------|---------|
| `generate_gbm_paths()` | Generate GBM paths with antithetic variates |
| `MonteCarloEngine.price_with_payoff()` | Price any payoff via MC |
| `calculate_standard_error()` | Compute MC standard error |

### Current Workaround (Pre-ObservationSchedule)

For monthly-average FIAs, explicitly pass `n_steps=12`:

```python
# In FIAPricer._calculate_expected_credit()
if isinstance(payoff, MonthlyAveragePayoff):
    result = engine.price_with_payoff(params, payoff, n_steps=12)
else:
    result = engine.price_with_payoff(params, payoff)  # default 252
```

---

## References

- Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*. Springer. Chapters 3-4.
- Hull, J. (2021). *Options, Futures, and Other Derivatives*. Chapter 26 (Asian options).
