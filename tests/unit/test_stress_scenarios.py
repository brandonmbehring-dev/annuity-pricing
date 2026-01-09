"""
Tests for Stress Scenarios - Phase I.

[T2] Verifies scenario definitions, ORSA scenarios, and conversion utilities.

See: docs/stress_testing/STRESS_TESTING_GUIDE.md
"""

import pytest

from annuity_pricing.stress_testing.historical import (
    CRISIS_2008_GFC,
)
from annuity_pricing.stress_testing.scenarios import (
    ALL_ORSA_SCENARIOS,
    ORSA_EXTREMELY_ADVERSE,
    ORSA_MODERATE_ADVERSE,
    ORSA_SEVERELY_ADVERSE,
    ScenarioType,
    StressScenario,
    create_custom_scenario,
    crisis_to_scenario,
    get_all_historical_scenarios,
    get_scenario_by_severity,
    scenario_summary,
)


class TestStressScenario:
    """Tests for StressScenario dataclass."""

    def test_create_basic_scenario(self) -> None:
        """Should create scenario with required fields."""
        scenario = StressScenario(
            name="test",
            display_name="Test Scenario",
            equity_shock=-0.20,
            rate_shock=-0.01,
        )
        assert scenario.name == "test"
        assert scenario.equity_shock == -0.20

    def test_default_values(self) -> None:
        """Should have sensible defaults."""
        scenario = StressScenario(
            name="test",
            display_name="Test",
            equity_shock=-0.10,
            rate_shock=0.0,
        )
        assert scenario.vol_shock == 1.0
        assert scenario.lapse_multiplier == 1.0
        assert scenario.withdrawal_multiplier == 1.0
        assert scenario.scenario_type == ScenarioType.CUSTOM

    def test_scenario_is_frozen(self) -> None:
        """Scenarios should be immutable."""
        scenario = StressScenario(
            name="test",
            display_name="Test",
            equity_shock=-0.10,
            rate_shock=0.0,
        )
        with pytest.raises(AttributeError):
            scenario.equity_shock = -0.30  # type: ignore

    def test_negative_vol_shock_raises(self) -> None:
        """Negative vol_shock should raise ValueError."""
        with pytest.raises(ValueError, match="vol_shock must be >= 0"):
            StressScenario(
                name="test",
                display_name="Test",
                equity_shock=-0.10,
                rate_shock=0.0,
                vol_shock=-0.5,
            )

    def test_negative_lapse_multiplier_raises(self) -> None:
        """Negative lapse_multiplier should raise ValueError."""
        with pytest.raises(ValueError, match="lapse_multiplier must be >= 0"):
            StressScenario(
                name="test",
                display_name="Test",
                equity_shock=-0.10,
                rate_shock=0.0,
                lapse_multiplier=-0.5,
            )

    def test_negative_withdrawal_multiplier_raises(self) -> None:
        """Negative withdrawal_multiplier should raise ValueError."""
        with pytest.raises(ValueError, match="withdrawal_multiplier must be >= 0"):
            StressScenario(
                name="test",
                display_name="Test",
                equity_shock=-0.10,
                rate_shock=0.0,
                withdrawal_multiplier=-0.5,
            )


class TestORSAModerateAdverse:
    """Tests for ORSA Moderate Adverse scenario."""

    def test_equity_shock(self) -> None:
        """Moderate adverse should have 15% equity decline."""
        assert ORSA_MODERATE_ADVERSE.equity_shock == -0.15

    def test_rate_shock(self) -> None:
        """Moderate adverse should have -50 bps rate shock."""
        assert ORSA_MODERATE_ADVERSE.rate_shock == -0.0050

    def test_vol_shock(self) -> None:
        """Moderate adverse should have 30% higher vol."""
        assert ORSA_MODERATE_ADVERSE.vol_shock == 1.3

    def test_scenario_type(self) -> None:
        """Should be ORSA type."""
        assert ORSA_MODERATE_ADVERSE.scenario_type == ScenarioType.ORSA


class TestORSASeverelyAdverse:
    """Tests for ORSA Severely Adverse scenario."""

    def test_equity_shock(self) -> None:
        """Severely adverse should have 30% equity decline."""
        assert ORSA_SEVERELY_ADVERSE.equity_shock == -0.30

    def test_rate_shock(self) -> None:
        """Severely adverse should have -100 bps rate shock."""
        assert ORSA_SEVERELY_ADVERSE.rate_shock == -0.0100

    def test_vol_shock(self) -> None:
        """Severely adverse should have double vol."""
        assert ORSA_SEVERELY_ADVERSE.vol_shock == 2.0

    def test_more_severe_than_moderate(self) -> None:
        """Should be more severe than moderate across metrics."""
        assert ORSA_SEVERELY_ADVERSE.equity_shock < ORSA_MODERATE_ADVERSE.equity_shock
        assert ORSA_SEVERELY_ADVERSE.rate_shock < ORSA_MODERATE_ADVERSE.rate_shock
        assert ORSA_SEVERELY_ADVERSE.vol_shock > ORSA_MODERATE_ADVERSE.vol_shock


class TestORSAExtremelyAdverse:
    """Tests for ORSA Extremely Adverse scenario."""

    def test_equity_shock(self) -> None:
        """Extremely adverse should have 50% equity decline."""
        assert ORSA_EXTREMELY_ADVERSE.equity_shock == -0.50

    def test_rate_shock(self) -> None:
        """Extremely adverse should have -200 bps rate shock."""
        assert ORSA_EXTREMELY_ADVERSE.rate_shock == -0.0200

    def test_vol_shock(self) -> None:
        """Extremely adverse should have 4x vol."""
        assert ORSA_EXTREMELY_ADVERSE.vol_shock == 4.0

    def test_most_severe(self) -> None:
        """Should be most severe across all ORSA scenarios."""
        for orsa in ALL_ORSA_SCENARIOS:
            assert ORSA_EXTREMELY_ADVERSE.equity_shock <= orsa.equity_shock
            assert ORSA_EXTREMELY_ADVERSE.rate_shock <= orsa.rate_shock
            assert ORSA_EXTREMELY_ADVERSE.vol_shock >= orsa.vol_shock


class TestAllORSAScenarios:
    """Tests for ALL_ORSA_SCENARIOS collection."""

    def test_has_three_scenarios(self) -> None:
        """Should have exactly 3 ORSA scenarios."""
        assert len(ALL_ORSA_SCENARIOS) == 3

    def test_all_orsa_type(self) -> None:
        """All should be ORSA type."""
        for scenario in ALL_ORSA_SCENARIOS:
            assert scenario.scenario_type == ScenarioType.ORSA

    def test_sorted_by_severity(self) -> None:
        """Should be ordered by increasing severity."""
        shocks = [s.equity_shock for s in ALL_ORSA_SCENARIOS]
        # Each subsequent should be more severe (more negative)
        for i in range(len(shocks) - 1):
            assert shocks[i] >= shocks[i + 1]


class TestCrisisToScenario:
    """Tests for crisis_to_scenario conversion."""

    def test_convert_2008_gfc(self) -> None:
        """Should convert 2008 GFC to scenario."""
        scenario = crisis_to_scenario(CRISIS_2008_GFC)
        assert scenario.equity_shock == CRISIS_2008_GFC.equity_shock
        assert scenario.rate_shock == CRISIS_2008_GFC.rate_shock

    def test_scenario_type_historical(self) -> None:
        """Converted scenario should be HISTORICAL type."""
        scenario = crisis_to_scenario(CRISIS_2008_GFC)
        assert scenario.scenario_type == ScenarioType.HISTORICAL

    def test_source_crisis_recorded(self) -> None:
        """Should record source crisis name."""
        scenario = crisis_to_scenario(CRISIS_2008_GFC)
        assert scenario.source_crisis == "2008_gfc"

    def test_vol_shock_from_vix(self) -> None:
        """Vol shock should be derived from VIX peak."""
        scenario = crisis_to_scenario(CRISIS_2008_GFC)
        expected_vol_shock = CRISIS_2008_GFC.vix_peak / 15.0
        assert abs(scenario.vol_shock - expected_vol_shock) < 0.01

    def test_custom_vol_shock(self) -> None:
        """Should accept custom vol_shock."""
        scenario = crisis_to_scenario(CRISIS_2008_GFC, vol_shock=3.0)
        assert scenario.vol_shock == 3.0

    def test_custom_lapse_multiplier(self) -> None:
        """Should accept custom lapse_multiplier."""
        scenario = crisis_to_scenario(CRISIS_2008_GFC, lapse_multiplier=1.5)
        assert scenario.lapse_multiplier == 1.5

    def test_notes_preserved(self) -> None:
        """Crisis notes should be preserved."""
        scenario = crisis_to_scenario(CRISIS_2008_GFC)
        assert CRISIS_2008_GFC.notes in scenario.notes


class TestCreateCustomScenario:
    """Tests for create_custom_scenario utility."""

    def test_create_basic_custom(self) -> None:
        """Should create custom scenario."""
        scenario = create_custom_scenario(
            name="stagflation",
            display_name="Stagflation Scenario",
            equity_shock=-0.25,
            rate_shock=0.02,
        )
        assert scenario.name == "stagflation"
        assert scenario.scenario_type == ScenarioType.CUSTOM

    def test_custom_with_all_params(self) -> None:
        """Should accept all parameters."""
        scenario = create_custom_scenario(
            name="custom",
            display_name="Custom Test",
            equity_shock=-0.30,
            rate_shock=-0.01,
            vol_shock=2.5,
            lapse_multiplier=1.3,
            withdrawal_multiplier=1.2,
            notes="Test scenario",
        )
        assert scenario.vol_shock == 2.5
        assert scenario.lapse_multiplier == 1.3
        assert scenario.withdrawal_multiplier == 1.2
        assert scenario.notes == "Test scenario"


class TestGetAllHistoricalScenarios:
    """Tests for get_all_historical_scenarios utility."""

    def test_returns_seven_scenarios(self) -> None:
        """Should return 7 scenarios (one per crisis)."""
        scenarios = get_all_historical_scenarios()
        assert len(scenarios) == 7

    def test_all_historical_type(self) -> None:
        """All should be HISTORICAL type."""
        scenarios = get_all_historical_scenarios()
        for scenario in scenarios:
            assert scenario.scenario_type == ScenarioType.HISTORICAL

    def test_custom_multipliers_applied(self) -> None:
        """Should apply custom multipliers to all."""
        scenarios = get_all_historical_scenarios(
            lapse_multiplier=1.5,
            withdrawal_multiplier=1.3,
        )
        for scenario in scenarios:
            assert scenario.lapse_multiplier == 1.5
            assert scenario.withdrawal_multiplier == 1.3


class TestGetScenarioBySeverity:
    """Tests for get_scenario_by_severity utility."""

    def test_filter_mild_shocks(self) -> None:
        """Should filter to mild shocks only."""
        scenarios = get_scenario_by_severity(-0.20, 0.0)
        for s in scenarios:
            assert -0.20 <= s.equity_shock <= 0.0

    def test_filter_severe_shocks(self) -> None:
        """Should filter to severe shocks only."""
        scenarios = get_scenario_by_severity(-1.0, -0.30)
        for s in scenarios:
            assert -1.0 <= s.equity_shock <= -0.30

    def test_empty_for_impossible_range(self) -> None:
        """Should return empty for impossible range."""
        scenarios = get_scenario_by_severity(0.0, 0.5)  # No positive shocks
        assert len(scenarios) == 0


class TestScenarioSummary:
    """Tests for scenario_summary utility."""

    def test_summary_format(self) -> None:
        """Should produce formatted summary string."""
        summary = scenario_summary(ORSA_SEVERELY_ADVERSE)
        assert "ORSA Severely Adverse" in summary
        assert "-30.0%" in summary
        assert "-100bps" in summary
        assert "2.0x" in summary

    def test_positive_rate_shock(self) -> None:
        """Should show positive rate shock correctly."""
        scenario = create_custom_scenario(
            name="rising_rates",
            display_name="Rising Rates",
            equity_shock=-0.10,
            rate_shock=0.02,
        )
        summary = scenario_summary(scenario)
        assert "+200bps" in summary


class TestScenarioType:
    """Tests for ScenarioType enum."""

    def test_historical_value(self) -> None:
        """HISTORICAL should have value 'historical'."""
        assert ScenarioType.HISTORICAL.value == "historical"

    def test_orsa_value(self) -> None:
        """ORSA should have value 'orsa'."""
        assert ScenarioType.ORSA.value == "orsa"

    def test_regulatory_value(self) -> None:
        """REGULATORY should have value 'regulatory'."""
        assert ScenarioType.REGULATORY.value == "regulatory"

    def test_custom_value(self) -> None:
        """CUSTOM should have value 'custom'."""
        assert ScenarioType.CUSTOM.value == "custom"
