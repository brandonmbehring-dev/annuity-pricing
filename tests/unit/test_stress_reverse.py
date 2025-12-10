"""
Tests for Reverse Stress Testing - Phase I.3.

[T2] Verifies bisection search algorithm and breaking point detection.

See: docs/stress_testing/STRESS_TESTING_GUIDE.md
"""

import pytest

from annuity_pricing.stress_testing.reverse import (
    ReverseStressTarget,
    ReverseStressResult,
    ReverseStressReport,
    ReverseStressTester,
    BreachCondition,
    RESERVE_EXHAUSTION,
    RESERVE_NEGATIVE,
    RBC_BREACH_200,
    RBC_BREACH_300,
    SOLVENCY_BREACH,
    RESERVE_INCREASE_50,
    RESERVE_INCREASE_100,
    ALL_PREDEFINED_TARGETS,
    DEFAULT_SEARCH_RANGES,
    create_custom_target,
    format_reverse_stress_result,
    format_reverse_stress_table,
    format_reverse_stress_summary,
    quick_reverse_stress,
    find_reserve_exhaustion_point,
)


class TestBreachCondition:
    """Tests for BreachCondition enum."""

    def test_below_value(self) -> None:
        """BELOW should have value 'below'."""
        assert BreachCondition.BELOW.value == "below"

    def test_above_value(self) -> None:
        """ABOVE should have value 'above'."""
        assert BreachCondition.ABOVE.value == "above"


class TestReverseStressTarget:
    """Tests for ReverseStressTarget dataclass."""

    def test_create_target(self) -> None:
        """Should create target with all fields."""
        target = ReverseStressTarget(
            target_type="test",
            threshold=100.0,
            description="Test target",
            breach_condition=BreachCondition.BELOW,
            metric_name="reserve",
        )
        assert target.target_type == "test"
        assert target.threshold == 100.0

    def test_is_breached_below(self) -> None:
        """Should detect breach when value below threshold."""
        target = ReverseStressTarget(
            target_type="test",
            threshold=100.0,
            description="Test",
            breach_condition=BreachCondition.BELOW,
        )
        assert target.is_breached(50.0) is True
        assert target.is_breached(150.0) is False
        assert target.is_breached(100.0) is False  # Exactly at threshold not breached

    def test_is_breached_above(self) -> None:
        """Should detect breach when value above threshold."""
        target = ReverseStressTarget(
            target_type="test",
            threshold=100.0,
            description="Test",
            breach_condition=BreachCondition.ABOVE,
        )
        assert target.is_breached(150.0) is True
        assert target.is_breached(50.0) is False
        assert target.is_breached(100.0) is False  # Exactly at threshold not breached

    def test_target_is_frozen(self) -> None:
        """Target should be immutable."""
        target = ReverseStressTarget(
            target_type="test",
            threshold=100.0,
            description="Test",
            breach_condition=BreachCondition.BELOW,
        )
        with pytest.raises(AttributeError):
            target.threshold = 200.0  # type: ignore


class TestPredefinedTargets:
    """Tests for predefined targets."""

    def test_reserve_exhaustion(self) -> None:
        """RESERVE_EXHAUSTION should be defined correctly."""
        assert RESERVE_EXHAUSTION.target_type == "reserve_exhaustion"
        assert RESERVE_EXHAUSTION.threshold == 0.0
        assert RESERVE_EXHAUSTION.breach_condition == BreachCondition.BELOW

    def test_reserve_negative(self) -> None:
        """RESERVE_NEGATIVE should be defined correctly."""
        assert RESERVE_NEGATIVE.target_type == "reserve_negative"
        assert RESERVE_NEGATIVE.threshold == 0.01
        assert RESERVE_NEGATIVE.breach_condition == BreachCondition.BELOW

    def test_rbc_breach_200(self) -> None:
        """RBC_BREACH_200 should be defined correctly."""
        assert RBC_BREACH_200.target_type == "rbc_breach_200"
        assert RBC_BREACH_200.threshold == 2.0
        assert RBC_BREACH_200.breach_condition == BreachCondition.BELOW

    def test_rbc_breach_300(self) -> None:
        """RBC_BREACH_300 should be defined correctly."""
        assert RBC_BREACH_300.target_type == "rbc_breach_300"
        assert RBC_BREACH_300.threshold == 3.0
        assert RBC_BREACH_300.breach_condition == BreachCondition.BELOW

    def test_solvency_breach(self) -> None:
        """SOLVENCY_BREACH should be defined correctly."""
        assert SOLVENCY_BREACH.target_type == "solvency_breach"
        assert SOLVENCY_BREACH.threshold == 1.0
        assert SOLVENCY_BREACH.breach_condition == BreachCondition.BELOW

    def test_reserve_increase_50(self) -> None:
        """RESERVE_INCREASE_50 should be defined correctly."""
        assert RESERVE_INCREASE_50.target_type == "reserve_increase_50"
        assert RESERVE_INCREASE_50.threshold == 0.50
        assert RESERVE_INCREASE_50.breach_condition == BreachCondition.ABOVE

    def test_reserve_increase_100(self) -> None:
        """RESERVE_INCREASE_100 should be defined correctly."""
        assert RESERVE_INCREASE_100.target_type == "reserve_increase_100"
        assert RESERVE_INCREASE_100.threshold == 1.00
        assert RESERVE_INCREASE_100.breach_condition == BreachCondition.ABOVE

    def test_all_predefined_targets_count(self) -> None:
        """ALL_PREDEFINED_TARGETS should have 7 targets."""
        assert len(ALL_PREDEFINED_TARGETS) == 7


class TestCreateCustomTarget:
    """Tests for create_custom_target function."""

    def test_create_below_target(self) -> None:
        """Should create target with 'below' condition."""
        target = create_custom_target(
            target_type="custom",
            threshold=50_000,
            description="Custom test",
            breach_condition="below",
        )
        assert target.target_type == "custom"
        assert target.breach_condition == BreachCondition.BELOW

    def test_create_above_target(self) -> None:
        """Should create target with 'above' condition."""
        target = create_custom_target(
            target_type="custom",
            threshold=0.75,
            description="Custom test",
            breach_condition="above",
            metric_name="reserve_delta_pct",
        )
        assert target.breach_condition == BreachCondition.ABOVE
        assert target.metric_name == "reserve_delta_pct"


class TestReverseStressResult:
    """Tests for ReverseStressResult dataclass."""

    def test_create_result(self) -> None:
        """Should create result with all fields."""
        result = ReverseStressResult(
            target=RESERVE_EXHAUSTION,
            parameter_name="equity_shock",
            breaking_point=-0.85,
            iterations=15,
            converged=True,
            breached=True,
            base_value=-0.30,
            search_range=(-1.0, 0.0),
            final_metric_value=0.0,
        )
        assert result.parameter_name == "equity_shock"
        assert result.breaking_point == -0.85
        assert result.converged is True

    def test_parameter_delta(self) -> None:
        """Should calculate parameter delta correctly."""
        result = ReverseStressResult(
            target=RESERVE_EXHAUSTION,
            parameter_name="equity_shock",
            breaking_point=-0.85,
            iterations=15,
            converged=True,
            breached=True,
            base_value=-0.30,
            search_range=(-1.0, 0.0),
        )
        assert result.parameter_delta == pytest.approx(-0.55, rel=0.01)

    def test_parameter_delta_pct(self) -> None:
        """Should calculate percentage delta correctly."""
        result = ReverseStressResult(
            target=RESERVE_EXHAUSTION,
            parameter_name="equity_shock",
            breaking_point=-0.60,
            iterations=15,
            converged=True,
            breached=True,
            base_value=-0.30,
            search_range=(-1.0, 0.0),
        )
        # Delta = -0.60 - (-0.30) = -0.30
        # Pct = -0.30 / |-0.30| = -1.0 = -100%
        assert result.parameter_delta_pct == pytest.approx(-1.0, rel=0.01)

    def test_not_breached_result(self) -> None:
        """Should handle not breached case."""
        result = ReverseStressResult(
            target=RESERVE_EXHAUSTION,
            parameter_name="equity_shock",
            breaking_point=None,
            iterations=0,
            converged=True,
            breached=False,
            base_value=-0.30,
            search_range=(-1.0, 0.0),
        )
        assert result.breached is False
        assert result.parameter_delta is None


class TestReverseStressReport:
    """Tests for ReverseStressReport dataclass."""

    def test_create_report(self) -> None:
        """Should create report from results."""
        result1 = ReverseStressResult(
            target=RESERVE_INCREASE_50,
            parameter_name="equity_shock",
            breaking_point=-0.65,
            iterations=12,
            converged=True,
            breached=True,
            base_value=-0.30,
            search_range=(-1.0, 0.0),
        )
        result2 = ReverseStressResult(
            target=RESERVE_INCREASE_50,
            parameter_name="rate_shock",
            breaking_point=-0.035,
            iterations=10,
            converged=True,
            breached=True,
            base_value=-0.01,
            search_range=(-0.05, 0.05),
        )
        report = ReverseStressReport(
            results={
                ("reserve_increase_50", "equity_shock"): result1,
                ("reserve_increase_50", "rate_shock"): result2,
            },
            base_reserve=100_000,
        )
        assert report.targets_tested == 1
        assert report.parameters_tested == 2

    def test_get_result(self) -> None:
        """Should retrieve specific result."""
        result1 = ReverseStressResult(
            target=RESERVE_INCREASE_50,
            parameter_name="equity_shock",
            breaking_point=-0.65,
            iterations=12,
            converged=True,
            breached=True,
            base_value=-0.30,
            search_range=(-1.0, 0.0),
        )
        report = ReverseStressReport(
            results={("reserve_increase_50", "equity_shock"): result1},
            base_reserve=100_000,
        )
        retrieved = report.get_result("reserve_increase_50", "equity_shock")
        assert retrieved is not None
        assert retrieved.breaking_point == -0.65

    def test_get_breached_results(self) -> None:
        """Should return only breached results."""
        result1 = ReverseStressResult(
            target=RESERVE_INCREASE_50,
            parameter_name="equity_shock",
            breaking_point=-0.65,
            iterations=12,
            converged=True,
            breached=True,
            base_value=-0.30,
            search_range=(-1.0, 0.0),
        )
        result2 = ReverseStressResult(
            target=RESERVE_EXHAUSTION,
            parameter_name="equity_shock",
            breaking_point=None,
            iterations=0,
            converged=True,
            breached=False,
            base_value=-0.30,
            search_range=(-1.0, 0.0),
        )
        report = ReverseStressReport(
            results={
                ("reserve_increase_50", "equity_shock"): result1,
                ("reserve_exhaustion", "equity_shock"): result2,
            },
            base_reserve=100_000,
        )
        breached = report.get_breached_results()
        assert len(breached) == 1
        assert breached[0].target.target_type == "reserve_increase_50"


class TestReverseStressTester:
    """Tests for ReverseStressTester class."""

    def test_create_tester(self) -> None:
        """Should create tester."""
        tester = ReverseStressTester()
        assert tester is not None

    def test_find_breaking_point_converges(self) -> None:
        """Bisection should converge within max iterations."""
        tester = ReverseStressTester()
        result = tester.find_breaking_point(
            target=RESERVE_INCREASE_50,
            parameter="equity_shock",
            search_range=(-1.0, 0.0),
            base_reserve=100_000,
        )
        assert result.converged is True
        assert result.iterations <= 25

    def test_find_breaking_point_breached(self) -> None:
        """Should find breaking point when target can be breached."""
        tester = ReverseStressTester()
        result = tester.find_breaking_point(
            target=RESERVE_INCREASE_50,
            parameter="equity_shock",
            search_range=(-1.0, 0.0),
            base_reserve=100_000,
        )
        assert result.breached is True
        assert result.breaking_point is not None
        # Breaking point should be more negative than base
        assert result.breaking_point < 0

    def test_find_breaking_point_not_breached(self) -> None:
        """Should handle case where target cannot be breached."""
        tester = ReverseStressTester()
        # RESERVE_EXHAUSTION is hard to reach with small equity shocks
        result = tester.find_breaking_point(
            target=RESERVE_EXHAUSTION,
            parameter="equity_shock",
            search_range=(-0.10, 0.0),  # Small range
            base_reserve=100_000,
        )
        # May or may not breach depending on model
        if not result.breached:
            assert result.breaking_point is None

    def test_already_breached_at_base(self) -> None:
        """Should detect if already breached at base."""
        # Create custom target that's already breached
        target = create_custom_target(
            target_type="low_threshold",
            threshold=200_000,  # Higher than any stressed reserve
            description="Always breached",
            breach_condition="below",
        )
        tester = ReverseStressTester()
        result = tester.find_breaking_point(
            target=target,
            parameter="equity_shock",
            search_range=(-1.0, 0.0),
            base_reserve=100_000,
        )
        assert result.breached is True
        assert result.iterations == 0  # No search needed

    def test_bisection_convergence_iterations(self) -> None:
        """Bisection should converge in O(log(range/tolerance)) iterations."""
        tester = ReverseStressTester()
        result = tester.find_breaking_point(
            target=RESERVE_INCREASE_50,
            parameter="equity_shock",
            search_range=(-1.0, 0.0),
            base_reserve=100_000,
            tolerance=0.001,
        )
        # For range=1.0 and tolerance=0.001, expect ~10 iterations
        # log2(1000) ≈ 10
        assert result.iterations <= 15

    def test_find_multiple_breaking_points(self) -> None:
        """Should find breaking points for multiple combinations."""
        tester = ReverseStressTester()
        report = tester.find_multiple_breaking_points(
            targets=[RESERVE_INCREASE_50],
            parameters=["equity_shock", "rate_shock"],
            search_ranges=DEFAULT_SEARCH_RANGES,
            base_reserve=100_000,
        )
        assert report.targets_tested == 1
        assert report.parameters_tested == 2

    def test_custom_impact_function(self) -> None:
        """Should accept custom impact function."""

        def custom_impact(
            base_reserve: float,
            equity_shock: float = 0.0,
            **kwargs,
        ) -> float:
            # Linear model: reserve = base * (1 - equity_shock)
            return base_reserve * (1 - equity_shock)

        tester = ReverseStressTester(impact_function=custom_impact)
        result = tester.find_breaking_point(
            target=RESERVE_INCREASE_50,
            parameter="equity_shock",
            search_range=(-1.0, 0.0),
            base_reserve=100_000,
        )
        assert result.breached is True


class TestDefaultSearchRanges:
    """Tests for DEFAULT_SEARCH_RANGES."""

    def test_has_equity_shock(self) -> None:
        """Should have equity_shock range."""
        assert "equity_shock" in DEFAULT_SEARCH_RANGES
        low, high = DEFAULT_SEARCH_RANGES["equity_shock"]
        assert low < high

    def test_has_rate_shock(self) -> None:
        """Should have rate_shock range."""
        assert "rate_shock" in DEFAULT_SEARCH_RANGES

    def test_has_vol_shock(self) -> None:
        """Should have vol_shock range."""
        assert "vol_shock" in DEFAULT_SEARCH_RANGES

    def test_has_lapse_multiplier(self) -> None:
        """Should have lapse_multiplier range."""
        assert "lapse_multiplier" in DEFAULT_SEARCH_RANGES

    def test_has_withdrawal_multiplier(self) -> None:
        """Should have withdrawal_multiplier range."""
        assert "withdrawal_multiplier" in DEFAULT_SEARCH_RANGES


class TestFormatReverseStressResult:
    """Tests for format_reverse_stress_result function."""

    def test_format_breached_result(self) -> None:
        """Should format breached result."""
        result = ReverseStressResult(
            target=RESERVE_INCREASE_50,
            parameter_name="equity_shock",
            breaking_point=-0.65,
            iterations=12,
            converged=True,
            breached=True,
            base_value=-0.30,
            search_range=(-1.0, 0.0),
        )
        formatted = format_reverse_stress_result(result)
        assert "reserve_increase_50" in formatted
        assert "equity_shock" in formatted
        assert "-0.6500" in formatted
        assert "12 iterations" in formatted

    def test_format_not_breached_result(self) -> None:
        """Should format not breached result."""
        result = ReverseStressResult(
            target=RESERVE_EXHAUSTION,
            parameter_name="equity_shock",
            breaking_point=None,
            iterations=0,
            converged=True,
            breached=False,
            base_value=-0.30,
            search_range=(-1.0, 0.0),
        )
        formatted = format_reverse_stress_result(result)
        assert "N/A" in formatted
        assert "Not breached" in formatted


class TestFormatReverseStressTable:
    """Tests for format_reverse_stress_table function."""

    def test_format_table(self) -> None:
        """Should format report as Markdown table."""
        result = ReverseStressResult(
            target=RESERVE_INCREASE_50,
            parameter_name="equity_shock",
            breaking_point=-0.65,
            iterations=12,
            converged=True,
            breached=True,
            base_value=-0.30,
            search_range=(-1.0, 0.0),
        )
        report = ReverseStressReport(
            results={("reserve_increase_50", "equity_shock"): result},
            base_reserve=100_000,
        )
        formatted = format_reverse_stress_table(report)
        assert "### Reverse Stress Testing Results" in formatted
        assert "Base Reserve: $100,000" in formatted
        assert "| Target" in formatted
        assert "Breached" in formatted


class TestFormatReverseStressSummary:
    """Tests for format_reverse_stress_summary function."""

    def test_format_summary_with_breaches(self) -> None:
        """Should format summary with breaches."""
        result = ReverseStressResult(
            target=RESERVE_INCREASE_50,
            parameter_name="equity_shock",
            breaking_point=-0.65,
            iterations=12,
            converged=True,
            breached=True,
            base_value=-0.30,
            search_range=(-1.0, 0.0),
        )
        report = ReverseStressReport(
            results={("reserve_increase_50", "equity_shock"): result},
            base_reserve=100_000,
        )
        summary = format_reverse_stress_summary(report)
        assert "1 breach scenarios" in summary

    def test_format_summary_no_breaches(self) -> None:
        """Should format summary without breaches."""
        result = ReverseStressResult(
            target=RESERVE_EXHAUSTION,
            parameter_name="equity_shock",
            breaking_point=None,
            iterations=0,
            converged=True,
            breached=False,
            base_value=-0.30,
            search_range=(-1.0, 0.0),
        )
        report = ReverseStressReport(
            results={("reserve_exhaustion", "equity_shock"): result},
            base_reserve=100_000,
        )
        summary = format_reverse_stress_summary(report)
        assert "no scenarios that breach" in summary


class TestQuickReverseStress:
    """Tests for quick_reverse_stress function."""

    def test_quick_reverse_stress(self) -> None:
        """Should run quick reverse stress test."""
        report = quick_reverse_stress(base_reserve=100_000)
        assert isinstance(report, ReverseStressReport)
        assert report.base_reserve == 100_000

    def test_custom_targets(self) -> None:
        """Should accept custom targets."""
        report = quick_reverse_stress(
            base_reserve=100_000,
            targets=[RESERVE_INCREASE_50],
            parameters=["equity_shock"],
        )
        assert report.targets_tested == 1
        assert report.parameters_tested == 1


class TestFindReserveExhaustionPoint:
    """Tests for find_reserve_exhaustion_point function."""

    def test_find_exhaustion_point(self) -> None:
        """Should find reserve exhaustion point."""
        result = find_reserve_exhaustion_point(
            base_reserve=100_000,
            parameter="equity_shock",
        )
        assert isinstance(result, ReverseStressResult)
        assert result.target.target_type == "reserve_exhaustion"

    def test_custom_search_range(self) -> None:
        """Should accept custom search range."""
        result = find_reserve_exhaustion_point(
            base_reserve=100_000,
            parameter="equity_shock",
            search_range=(-0.50, 0.0),
        )
        assert result.search_range == (-0.50, 0.0)


class TestImpactModel:
    """Tests for the default impact model in reverse stress."""

    def test_severe_equity_shock_exhausts_reserve(self) -> None:
        """Very severe equity shock should approach reserve exhaustion."""
        tester = ReverseStressTester()
        # Test with extreme equity shock
        base_params = {
            "equity_shock": 0.0,
            "rate_shock": 0.0,
            "vol_shock": 1.0,
            "lapse_multiplier": 1.0,
            "withdrawal_multiplier": 1.0,
        }
        base_reserve = tester._impact_function(100_000, **base_params)

        extreme_params = base_params.copy()
        extreme_params["equity_shock"] = -1.0  # -100% shock
        extreme_reserve = tester._impact_function(100_000, **extreme_params)

        # Reserve should increase significantly with -100% equity shock
        assert extreme_reserve > base_reserve * 1.5

    def test_rbc_model(self) -> None:
        """Should calculate RBC ratio correctly."""
        tester = ReverseStressTester()
        # RBC = reserve / ACL
        rbc = tester._rbc_function(200_000, authorized_control_level=50_000)
        assert rbc == 4.0  # 400%

    def test_rbc_model_zero_acl(self) -> None:
        """Should handle zero ACL gracefully."""
        tester = ReverseStressTester()
        rbc = tester._rbc_function(200_000, authorized_control_level=0)
        assert rbc == float("inf")
