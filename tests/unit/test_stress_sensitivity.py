"""
Tests for Sensitivity Analysis - Phase I.2.

[T2] Verifies OAT parameter sweeps and tornado diagram data generation.

See: docs/stress_testing/STRESS_TESTING_GUIDE.md
"""

import pytest

from annuity_pricing.stress_testing.sensitivity import (
    SensitivityParameter,
    SensitivityResult,
    SensitivityDirection,
    TornadoData,
    SensitivityAnalyzer,
    get_default_sensitivity_parameters,
    format_sensitivity_result,
    format_tornado_table,
    format_tornado_summary,
    quick_sensitivity_analysis,
)


class TestSensitivityParameter:
    """Tests for SensitivityParameter dataclass."""

    def test_create_basic_parameter(self) -> None:
        """Should create parameter with required fields."""
        param = SensitivityParameter(
            name="equity_shock",
            display_name="Equity Shock",
            base_value=-0.30,
            range_down=-0.60,
            range_up=-0.10,
        )
        assert param.name == "equity_shock"
        assert param.base_value == -0.30

    def test_default_unit(self) -> None:
        """Should have empty string as default unit."""
        param = SensitivityParameter(
            name="test",
            display_name="Test",
            base_value=1.0,
            range_down=0.5,
            range_up=2.0,
        )
        assert param.unit == ""

    def test_with_unit(self) -> None:
        """Should accept custom unit."""
        param = SensitivityParameter(
            name="rate_shock",
            display_name="Rate Shock",
            base_value=-0.0100,
            range_down=-0.0200,
            range_up=0.0100,
            unit="bps",
        )
        assert param.unit == "bps"

    def test_parameter_is_frozen(self) -> None:
        """Parameter should be immutable."""
        param = SensitivityParameter(
            name="test",
            display_name="Test",
            base_value=1.0,
            range_down=0.5,
            range_up=2.0,
        )
        with pytest.raises(AttributeError):
            param.base_value = 2.0  # type: ignore


class TestSensitivityResult:
    """Tests for SensitivityResult dataclass."""

    def test_create_result(self) -> None:
        """Should create result with all fields."""
        result = SensitivityResult(
            parameter="equity_shock",
            display_name="Equity Shock",
            base_value=-0.30,
            base_reserve=100_000,
            down_value=-0.60,
            down_reserve=124_000,
            up_value=-0.10,
            up_reserve=108_000,
            down_delta_pct=0.24,
            up_delta_pct=0.08,
            sensitivity_width=0.16,
        )
        assert result.parameter == "equity_shock"
        assert result.sensitivity_width == 0.16

    def test_negative_sensitivity_width_raises(self) -> None:
        """Should raise for negative sensitivity width."""
        with pytest.raises(ValueError, match="sensitivity_width must be >= 0"):
            SensitivityResult(
                parameter="test",
                display_name="Test",
                base_value=0.0,
                base_reserve=100_000,
                down_value=0.0,
                down_reserve=100_000,
                up_value=0.0,
                up_reserve=100_000,
                down_delta_pct=0.0,
                up_delta_pct=0.0,
                sensitivity_width=-0.01,
            )

    def test_result_is_frozen(self) -> None:
        """Result should be immutable."""
        result = SensitivityResult(
            parameter="test",
            display_name="Test",
            base_value=0.0,
            base_reserve=100_000,
            down_value=0.0,
            down_reserve=100_000,
            up_value=0.0,
            up_reserve=100_000,
            down_delta_pct=0.0,
            up_delta_pct=0.0,
            sensitivity_width=0.0,
        )
        with pytest.raises(AttributeError):
            result.sensitivity_width = 1.0  # type: ignore


class TestTornadoData:
    """Tests for TornadoData dataclass."""

    def test_create_tornado_data(self) -> None:
        """Should create tornado data with results."""
        results = [
            SensitivityResult(
                parameter="equity_shock",
                display_name="Equity Shock",
                base_value=-0.30,
                base_reserve=100_000,
                down_value=-0.60,
                down_reserve=124_000,
                up_value=-0.10,
                up_reserve=108_000,
                down_delta_pct=0.24,
                up_delta_pct=0.08,
                sensitivity_width=0.16,
            ),
            SensitivityResult(
                parameter="rate_shock",
                display_name="Rate Shock",
                base_value=-0.0100,
                base_reserve=100_000,
                down_value=-0.0200,
                down_reserve=110_000,
                up_value=0.0100,
                up_reserve=80_000,
                down_delta_pct=0.10,
                up_delta_pct=-0.20,
                sensitivity_width=0.30,
            ),
        ]
        tornado = TornadoData(
            results=results,
            base_reserve=100_000,
            scenario_name="Test Scenario",
        )
        assert tornado.n_parameters == 2
        assert tornado.base_reserve == 100_000

    def test_results_sorted_by_width(self) -> None:
        """Results should be sorted by sensitivity_width descending."""
        results = [
            SensitivityResult(
                parameter="small",
                display_name="Small",
                base_value=0.0,
                base_reserve=100_000,
                down_value=0.0,
                down_reserve=100_000,
                up_value=0.0,
                up_reserve=100_000,
                down_delta_pct=0.0,
                up_delta_pct=0.0,
                sensitivity_width=0.05,
            ),
            SensitivityResult(
                parameter="large",
                display_name="Large",
                base_value=0.0,
                base_reserve=100_000,
                down_value=0.0,
                down_reserve=100_000,
                up_value=0.0,
                up_reserve=100_000,
                down_delta_pct=0.0,
                up_delta_pct=0.0,
                sensitivity_width=0.20,
            ),
        ]
        tornado = TornadoData(
            results=results,
            base_reserve=100_000,
            scenario_name="Test",
        )
        # Should be sorted: large first, then small
        assert tornado.results[0].parameter == "large"
        assert tornado.results[1].parameter == "small"

    def test_most_sensitive_parameter(self) -> None:
        """Should return most sensitive parameter name."""
        results = [
            SensitivityResult(
                parameter="large",
                display_name="Large",
                base_value=0.0,
                base_reserve=100_000,
                down_value=0.0,
                down_reserve=100_000,
                up_value=0.0,
                up_reserve=100_000,
                down_delta_pct=0.0,
                up_delta_pct=0.0,
                sensitivity_width=0.20,
            ),
        ]
        tornado = TornadoData(
            results=results,
            base_reserve=100_000,
            scenario_name="Test",
        )
        assert tornado.most_sensitive_parameter == "large"

    def test_least_sensitive_parameter(self) -> None:
        """Should return least sensitive parameter name."""
        results = [
            SensitivityResult(
                parameter="large",
                display_name="Large",
                base_value=0.0,
                base_reserve=100_000,
                down_value=0.0,
                down_reserve=100_000,
                up_value=0.0,
                up_reserve=100_000,
                down_delta_pct=0.0,
                up_delta_pct=0.0,
                sensitivity_width=0.20,
            ),
            SensitivityResult(
                parameter="small",
                display_name="Small",
                base_value=0.0,
                base_reserve=100_000,
                down_value=0.0,
                down_reserve=100_000,
                up_value=0.0,
                up_reserve=100_000,
                down_delta_pct=0.0,
                up_delta_pct=0.0,
                sensitivity_width=0.05,
            ),
        ]
        tornado = TornadoData(
            results=results,
            base_reserve=100_000,
            scenario_name="Test",
        )
        assert tornado.least_sensitive_parameter == "small"

    def test_empty_results_properties(self) -> None:
        """Empty results should return None for properties."""
        tornado = TornadoData(
            results=[],
            base_reserve=100_000,
            scenario_name="Empty",
        )
        assert tornado.most_sensitive_parameter is None
        assert tornado.least_sensitive_parameter is None
        assert tornado.n_parameters == 0


class TestGetDefaultSensitivityParameters:
    """Tests for get_default_sensitivity_parameters function."""

    def test_returns_five_parameters(self) -> None:
        """Should return 5 default parameters."""
        params = get_default_sensitivity_parameters()
        assert len(params) == 5

    def test_parameter_names(self) -> None:
        """Should have correct parameter names."""
        params = get_default_sensitivity_parameters()
        names = {p.name for p in params}
        expected = {
            "equity_shock",
            "rate_shock",
            "vol_shock",
            "lapse_multiplier",
            "withdrawal_multiplier",
        }
        assert names == expected

    def test_custom_base_values(self) -> None:
        """Should accept custom base values."""
        params = get_default_sensitivity_parameters(
            base_equity_shock=-0.50,
            base_rate_shock=-0.0200,
        )
        equity_param = next(p for p in params if p.name == "equity_shock")
        rate_param = next(p for p in params if p.name == "rate_shock")
        assert equity_param.base_value == -0.50
        assert rate_param.base_value == -0.0200

    def test_equity_shock_range(self) -> None:
        """Equity shock should have reasonable range."""
        params = get_default_sensitivity_parameters()
        equity_param = next(p for p in params if p.name == "equity_shock")
        assert equity_param.range_down == -0.60  # -60%
        assert equity_param.range_up == -0.10  # -10%

    def test_vol_shock_range(self) -> None:
        """Vol shock should have reasonable range."""
        params = get_default_sensitivity_parameters()
        vol_param = next(p for p in params if p.name == "vol_shock")
        assert vol_param.range_down == 1.0  # 1x (no shock)
        assert vol_param.range_up == 4.0  # 4x


class TestSensitivityAnalyzer:
    """Tests for SensitivityAnalyzer class."""

    def test_create_analyzer(self) -> None:
        """Should create analyzer."""
        analyzer = SensitivityAnalyzer()
        assert analyzer is not None

    def test_run_single_parameter(self) -> None:
        """Should run analysis for single parameter."""
        analyzer = SensitivityAnalyzer()
        param = SensitivityParameter(
            name="equity_shock",
            display_name="Equity Shock",
            base_value=-0.30,
            range_down=-0.60,
            range_up=-0.10,
        )
        base_params = {
            "equity_shock": -0.30,
            "rate_shock": -0.01,
            "vol_shock": 2.0,
            "lapse_multiplier": 1.0,
            "withdrawal_multiplier": 1.0,
        }
        result = analyzer.run_single_parameter(param, base_params, 100_000)
        assert result.parameter == "equity_shock"
        assert result.base_reserve > 0
        assert result.sensitivity_width >= 0

    def test_run_oat_returns_tornado_data(self) -> None:
        """Should return TornadoData from OAT analysis."""
        analyzer = SensitivityAnalyzer()
        tornado = analyzer.run_oat(base_reserve=100_000)
        assert isinstance(tornado, TornadoData)
        assert tornado.n_parameters == 5  # Default parameters
        assert tornado.base_reserve == 100_000

    def test_run_oat_custom_parameters(self) -> None:
        """Should accept custom parameters."""
        analyzer = SensitivityAnalyzer()
        params = [
            SensitivityParameter(
                name="equity_shock",
                display_name="Equity Shock",
                base_value=-0.30,
                range_down=-0.50,
                range_up=-0.10,
            ),
        ]
        tornado = analyzer.run_oat(
            base_reserve=100_000,
            parameters=params,
        )
        assert tornado.n_parameters == 1
        assert tornado.results[0].parameter == "equity_shock"

    def test_run_oat_custom_base_values(self) -> None:
        """Should use custom base values."""
        analyzer = SensitivityAnalyzer()
        tornado = analyzer.run_oat(
            base_reserve=100_000,
            base_equity_shock=-0.50,
            scenario_name="Custom Scenario",
        )
        assert tornado.scenario_name == "Custom Scenario"

    def test_equity_more_sensitive_than_lapse(self) -> None:
        """Equity shock should typically be more sensitive than lapse."""
        analyzer = SensitivityAnalyzer()
        tornado = analyzer.run_oat(base_reserve=100_000)

        equity_result = next(r for r in tornado.results if r.parameter == "equity_shock")
        lapse_result = next(r for r in tornado.results if r.parameter == "lapse_multiplier")

        # Equity typically has larger impact than lapse
        assert equity_result.sensitivity_width > lapse_result.sensitivity_width

    def test_results_have_positive_width(self) -> None:
        """All results should have non-negative sensitivity width."""
        analyzer = SensitivityAnalyzer()
        tornado = analyzer.run_oat(base_reserve=100_000)

        for result in tornado.results:
            assert result.sensitivity_width >= 0

    def test_custom_impact_function(self) -> None:
        """Should accept custom impact function."""

        def custom_impact(
            base_reserve: float,
            equity_shock: float,
            rate_shock: float,
            vol_shock: float,
            lapse_multiplier: float,
            withdrawal_multiplier: float,
        ) -> float:
            # Simple custom model: only equity matters
            return base_reserve * (1 - equity_shock)

        analyzer = SensitivityAnalyzer(impact_function=custom_impact)
        tornado = analyzer.run_oat(base_reserve=100_000)

        # In custom model, equity shock should be most sensitive
        assert tornado.most_sensitive_parameter == "equity_shock"


class TestFormatSensitivityResult:
    """Tests for format_sensitivity_result function."""

    def test_format_basic_result(self) -> None:
        """Should format result as table row."""
        result = SensitivityResult(
            parameter="equity_shock",
            display_name="Equity Shock",
            base_value=-0.30,
            base_reserve=100_000,
            down_value=-0.60,
            down_reserve=124_000,
            up_value=-0.10,
            up_reserve=108_000,
            down_delta_pct=0.24,
            up_delta_pct=0.08,
            sensitivity_width=0.16,
        )
        formatted = format_sensitivity_result(result)
        assert "Equity Shock" in formatted
        assert "+24.0%" in formatted
        assert "+8.0%" in formatted
        assert "16.0%" in formatted

    def test_format_negative_delta(self) -> None:
        """Should show negative deltas correctly."""
        result = SensitivityResult(
            parameter="rate_shock",
            display_name="Rate Shock",
            base_value=-0.0100,
            base_reserve=100_000,
            down_value=-0.0200,
            down_reserve=110_000,
            up_value=0.0100,
            up_reserve=80_000,
            down_delta_pct=0.10,
            up_delta_pct=-0.20,
            sensitivity_width=0.30,
        )
        formatted = format_sensitivity_result(result)
        assert "-20.0%" in formatted


class TestFormatTornadoTable:
    """Tests for format_tornado_table function."""

    def test_format_table(self) -> None:
        """Should format tornado as Markdown table."""
        results = [
            SensitivityResult(
                parameter="equity_shock",
                display_name="Equity Shock",
                base_value=-0.30,
                base_reserve=100_000,
                down_value=-0.60,
                down_reserve=124_000,
                up_value=-0.10,
                up_reserve=108_000,
                down_delta_pct=0.24,
                up_delta_pct=0.08,
                sensitivity_width=0.16,
            ),
        ]
        tornado = TornadoData(
            results=results,
            base_reserve=100_000,
            scenario_name="Test Scenario",
        )
        formatted = format_tornado_table(tornado)
        assert "### Sensitivity Analysis: Test Scenario" in formatted
        assert "Base Reserve: $100,000" in formatted
        assert "| Parameter" in formatted
        assert "Equity Shock" in formatted
        assert "Most sensitive" in formatted


class TestFormatTornadoSummary:
    """Tests for format_tornado_summary function."""

    def test_format_summary(self) -> None:
        """Should format brief summary."""
        results = [
            SensitivityResult(
                parameter="large",
                display_name="Large Impact",
                base_value=0.0,
                base_reserve=100_000,
                down_value=0.0,
                down_reserve=100_000,
                up_value=0.0,
                up_reserve=100_000,
                down_delta_pct=0.0,
                up_delta_pct=0.0,
                sensitivity_width=0.20,
            ),
            SensitivityResult(
                parameter="small",
                display_name="Small Impact",
                base_value=0.0,
                base_reserve=100_000,
                down_value=0.0,
                down_reserve=100_000,
                up_value=0.0,
                up_reserve=100_000,
                down_delta_pct=0.0,
                up_delta_pct=0.0,
                sensitivity_width=0.05,
            ),
        ]
        tornado = TornadoData(
            results=results,
            base_reserve=100_000,
            scenario_name="Test",
        )
        summary = format_tornado_summary(tornado)
        assert "2 parameters" in summary
        assert "Large Impact" in summary
        assert "most impactful" in summary
        assert "Small Impact" in summary
        assert "least impactful" in summary

    def test_empty_results_summary(self) -> None:
        """Should handle empty results."""
        tornado = TornadoData(
            results=[],
            base_reserve=100_000,
            scenario_name="Empty",
        )
        summary = format_tornado_summary(tornado)
        assert "No sensitivity results" in summary


class TestQuickSensitivityAnalysis:
    """Tests for quick_sensitivity_analysis function."""

    def test_quick_analysis(self) -> None:
        """Should run quick analysis."""
        tornado = quick_sensitivity_analysis(base_reserve=100_000)
        assert isinstance(tornado, TornadoData)
        assert tornado.n_parameters == 5

    def test_custom_shocks(self) -> None:
        """Should accept custom shock values."""
        tornado = quick_sensitivity_analysis(
            base_reserve=100_000,
            equity_shock=-0.50,
            rate_shock=-0.0200,
            vol_shock=3.0,
        )
        assert tornado.n_parameters == 5


class TestSensitivityDirection:
    """Tests for SensitivityDirection enum."""

    def test_up_value(self) -> None:
        """UP should have value 'up'."""
        assert SensitivityDirection.UP.value == "up"

    def test_down_value(self) -> None:
        """DOWN should have value 'down'."""
        assert SensitivityDirection.DOWN.value == "down"

    def test_both_value(self) -> None:
        """BOTH should have value 'both'."""
        assert SensitivityDirection.BOTH.value == "both"


class TestImpactModel:
    """Tests for the default impact model."""

    def test_equity_shock_increases_reserve(self) -> None:
        """Negative equity shock should increase reserve."""
        analyzer = SensitivityAnalyzer()
        base_params = {
            "equity_shock": 0.0,
            "rate_shock": 0.0,
            "vol_shock": 1.0,
            "lapse_multiplier": 1.0,
            "withdrawal_multiplier": 1.0,
        }
        base_reserve = analyzer._calculate_reserve(100_000, base_params)

        stressed_params = base_params.copy()
        stressed_params["equity_shock"] = -0.30
        stressed_reserve = analyzer._calculate_reserve(100_000, stressed_params)

        assert stressed_reserve > base_reserve

    def test_rate_decrease_increases_reserve(self) -> None:
        """Rate decrease should increase reserve."""
        analyzer = SensitivityAnalyzer()
        base_params = {
            "equity_shock": 0.0,
            "rate_shock": 0.0,
            "vol_shock": 1.0,
            "lapse_multiplier": 1.0,
            "withdrawal_multiplier": 1.0,
        }
        base_reserve = analyzer._calculate_reserve(100_000, base_params)

        stressed_params = base_params.copy()
        stressed_params["rate_shock"] = -0.0100  # -100 bps
        stressed_reserve = analyzer._calculate_reserve(100_000, stressed_params)

        assert stressed_reserve > base_reserve

    def test_vol_increase_increases_reserve(self) -> None:
        """Vol increase should increase reserve."""
        analyzer = SensitivityAnalyzer()
        base_params = {
            "equity_shock": 0.0,
            "rate_shock": 0.0,
            "vol_shock": 1.0,
            "lapse_multiplier": 1.0,
            "withdrawal_multiplier": 1.0,
        }
        base_reserve = analyzer._calculate_reserve(100_000, base_params)

        stressed_params = base_params.copy()
        stressed_params["vol_shock"] = 2.0  # 2x vol
        stressed_reserve = analyzer._calculate_reserve(100_000, stressed_params)

        assert stressed_reserve > base_reserve

    def test_lapse_increase_decreases_reserve(self) -> None:
        """Higher lapses should decrease reserve (fewer policies)."""
        analyzer = SensitivityAnalyzer()
        base_params = {
            "equity_shock": 0.0,
            "rate_shock": 0.0,
            "vol_shock": 1.0,
            "lapse_multiplier": 1.0,
            "withdrawal_multiplier": 1.0,
        }
        base_reserve = analyzer._calculate_reserve(100_000, base_params)

        stressed_params = base_params.copy()
        stressed_params["lapse_multiplier"] = 2.0  # 2x lapses
        stressed_reserve = analyzer._calculate_reserve(100_000, stressed_params)

        assert stressed_reserve < base_reserve

    def test_reserve_never_negative(self) -> None:
        """Reserve should never go negative."""
        analyzer = SensitivityAnalyzer()
        extreme_params = {
            "equity_shock": 0.0,  # No equity shock
            "rate_shock": 0.50,  # +500 bps (very high)
            "vol_shock": 0.1,  # Low vol
            "lapse_multiplier": 10.0,  # Extreme lapses
            "withdrawal_multiplier": 0.0,
        }
        reserve = analyzer._calculate_reserve(100_000, extreme_params)
        assert reserve >= 0.01  # Minimum enforced
