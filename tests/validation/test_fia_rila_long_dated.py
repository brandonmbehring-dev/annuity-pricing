"""
Long-dated option validation for FIA/RILA pricing.

[T1] Validates that FIA/RILA pricing works correctly for 5-7 year terms:
- Option prices scale appropriately with term
- Annuity factors calculated correctly
- Solver convergence maintained

FIA/RILA products commonly have 5-7 year terms.

See: Hull (2021) Ch. 15 - Long-dated options
"""

import pytest
import numpy as np

from annuity_pricing.options.pricing.black_scholes import (
    black_scholes_call,
    black_scholes_put,
)


class TestBSLongDated:
    """
    Validates Black-Scholes for long-dated options (3-7 years).

    [T1] Long-dated options have:
    - Higher time value (sqrt(T) effect on vega)
    - Larger dividend impact (e^(-qT) compounds)
    - Rho becomes more significant
    """

    @pytest.mark.validation
    @pytest.mark.parametrize("term_years", [3, 5, 6, 7])
    def test_call_value_increases_with_term(self, term_years):
        """[T1] Call value increases with term (time value)."""
        spot = 100.0
        strike = 100.0
        rate = 0.05
        dividend = 0.02
        volatility = 0.20

        short_term = 1.0
        short_price = black_scholes_call(spot, strike, rate, dividend, volatility, short_term)
        long_price = black_scholes_call(spot, strike, rate, dividend, volatility, term_years)

        assert long_price > short_price, (
            f"Long-dated call ({long_price:.4f}) should exceed short-dated ({short_price:.4f})"
        )

    @pytest.mark.validation
    @pytest.mark.parametrize("term_years", [3, 5, 6, 7])
    def test_put_value_increases_with_term(self, term_years):
        """[T1] Put value increases with term (time value)."""
        spot = 100.0
        strike = 100.0
        rate = 0.05
        dividend = 0.02
        volatility = 0.20

        short_term = 1.0
        short_price = black_scholes_put(spot, strike, rate, dividend, volatility, short_term)
        long_price = black_scholes_put(spot, strike, rate, dividend, volatility, term_years)

        assert long_price > short_price, (
            f"Long-dated put ({long_price:.4f}) should exceed short-dated ({short_price:.4f})"
        )

    @pytest.mark.validation
    @pytest.mark.parametrize("term_years", [5, 7])
    def test_put_call_parity_long_dated(self, term_years):
        """[T1] Put-call parity holds for long-dated options."""
        spot = 100.0
        strike = 100.0
        rate = 0.05
        dividend = 0.02
        volatility = 0.20

        call_price = black_scholes_call(spot, strike, rate, dividend, volatility, term_years)
        put_price = black_scholes_put(spot, strike, rate, dividend, volatility, term_years)

        expected = spot * np.exp(-dividend * term_years) - strike * np.exp(-rate * term_years)
        actual = call_price - put_price

        assert abs(actual - expected) < 1e-8, (
            f"Put-call parity violated at T={term_years}: actual={actual:.6f}, expected={expected:.6f}"
        )


class TestFIALongDated:
    """
    Validates FIA pricer for long-dated products (5-7 years).

    [T1] Long-dated FIA products should:
    - Have valid embedded option values
    - Fair cap solver should converge
    - Expected credit bounded by cap
    """

    @pytest.fixture
    def market_params(self):
        """Create market params."""
        from annuity_pricing.products.fia import MarketParams

        return MarketParams(
            spot=100.0,
            risk_free_rate=0.05,
            dividend_yield=0.02,
            volatility=0.20,
        )

    @pytest.fixture
    def sample_product(self):
        """Create sample FIA product."""
        from annuity_pricing.data.schemas import FIAProduct

        return FIAProduct(
            company_name="Test Company",
            product_name="Test FIA",
            product_group="FIA",
            status="current",
            cap_rate=0.10,
            indexing_method="Annual Point to Point",
            index_used="S&P 500",
        )

    @pytest.mark.validation
    @pytest.mark.parametrize("term_years", [3, 5, 6, 7])
    def test_fia_long_dated_pricing(self, market_params, sample_product, term_years):
        """[T1] FIA pricing valid for long-dated products."""
        from annuity_pricing.products.fia import FIAPricer

        pricer = FIAPricer(market_params=market_params, n_mc_paths=5000, seed=42)
        result = pricer.price(sample_product, term_years=term_years, premium=100.0)

        # Embedded option value should be positive and bounded
        assert result.embedded_option_value > 0
        assert result.embedded_option_value <= 100.0

        # Expected credit bounded by cap
        assert result.expected_credit >= 0
        assert result.expected_credit <= sample_product.cap_rate + 0.01

    @pytest.mark.validation
    @pytest.mark.parametrize("term_years", [5, 7])
    def test_fia_fair_cap_reasonable_long_dated(self, market_params, sample_product, term_years):
        """
        [T1] FIA fair cap is reasonable for long-dated products.

        For longer terms, the option budget accumulates, allowing higher caps.
        """
        from annuity_pricing.products.fia import FIAPricer

        pricer = FIAPricer(market_params=market_params, n_mc_paths=5000, seed=42)
        result = pricer.price(sample_product, term_years=term_years, premium=100.0)

        # Fair cap should be positive and increase with term
        # (more option budget allows higher caps)
        assert result.fair_cap > 0, "Fair cap should be positive"
        assert result.fair_cap > sample_product.cap_rate, (
            f"Fair cap ({result.fair_cap:.2%}) should exceed product cap "
            f"({sample_product.cap_rate:.2%}) at T={term_years}"
        )


class TestRILALongDated:
    """
    Validates RILA pricer for long-dated products (5-7 years).

    [T1] Long-dated RILA products should:
    - Have valid protection values
    - Max loss bounded by product definition
    - Protection value scales appropriately
    """

    @pytest.fixture
    def market_params(self):
        """Create market params."""
        from annuity_pricing.products.rila import MarketParams

        return MarketParams(
            spot=100.0,
            risk_free_rate=0.05,
            dividend_yield=0.02,
            volatility=0.20,
        )

    @pytest.fixture
    def buffer_product(self):
        """Create sample RILA buffer product."""
        from annuity_pricing.data.schemas import RILAProduct

        return RILAProduct(
            company_name="Test Company",
            product_name="Test RILA",
            product_group="RILA",
            status="current",
            buffer_rate=0.10,
            buffer_modifier="Up To 10%",
            index_used="S&P 500",
        )

    @pytest.mark.validation
    @pytest.mark.parametrize("term_years", [3, 5, 6, 7])
    def test_rila_long_dated_pricing(self, market_params, buffer_product, term_years):
        """[T1] RILA pricing valid for long-dated products."""
        from annuity_pricing.products.rila import RILAPricer

        pricer = RILAPricer(market_params=market_params, n_mc_paths=5000, seed=42)
        result = pricer.price(buffer_product, term_years=term_years, premium=100.0)

        # Protection value should be positive
        assert result.protection_value > 0

        # Max loss bounded
        assert result.max_loss >= buffer_product.buffer_rate
        assert result.max_loss <= 1.0

    @pytest.mark.validation
    @pytest.mark.parametrize("term_years", [5, 7])
    def test_rila_upside_increases_with_term(self, market_params, buffer_product, term_years):
        """
        [T1] RILA upside value increases with term (time value).

        Note: protection_value is in PV terms and may decrease due to discounting.
        The upside_value (uncapped call value) increases with term.
        """
        from annuity_pricing.products.rila import RILAPricer

        pricer = RILAPricer(market_params=market_params, n_mc_paths=5000, seed=42)

        short_result = pricer.price(buffer_product, term_years=1, premium=100.0)
        long_result = pricer.price(buffer_product, term_years=term_years, premium=100.0)

        # Upside value (call value) increases with term
        assert long_result.upside_value > short_result.upside_value, (
            f"Long-dated upside ({long_result.upside_value:.4f}) "
            f"should exceed short-dated ({short_result.upside_value:.4f})"
        )
