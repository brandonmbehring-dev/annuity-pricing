"""
Regulatory Calculations - Phase 9.

[PROTOTYPE] EDUCATIONAL USE ONLY - NOT FOR PRODUCTION REGULATORY FILING
===========================================================================
This module implements simplified NAIC VM-21 and VM-22 calculations for
educational and research purposes. It is NOT suitable for:
- Actual regulatory reserve filings
- Statutory reporting
- Compliance certification

For production regulatory work, you need:
1. Qualified actuarial certification (FSA/MAAA)
2. NAIC-prescribed scenario generators (GOES/AAA ESG)
3. Full policy administration system integration
4. Independent model validation
5. Regulatory approval of methods

See: docs/regulatory/AG43_COMPLIANCE_GAP.md for detailed gap analysis.
===========================================================================

Implements NAIC VM-21 and VM-22 for annuity reserves:
- AG43/VM-21: Variable annuity reserve requirements
- VM-22: Fixed annuity principle-based reserves
- Scenario generation: Economic scenarios for stochastic modeling

See: docs/knowledge/domain/vm21_vm22.md
"""

from .scenarios import (
    AG43Scenarios,
    EconomicScenario,
    EquityParams,
    ScenarioGenerator,
    VasicekParams,
    calculate_scenario_statistics,
    generate_deterministic_scenarios,
)
from .vm21 import (
    PolicyData,
    VM21Calculator,
    VM21Result,
    calculate_cte_levels,
    sensitivity_analysis,
)
from .vm22 import (
    FixedAnnuityPolicy,
    ReserveType,
    StochasticExclusionResult,
    VM22Calculator,
    VM22Result,
    compare_reserve_methods,
    vm22_sensitivity,
)

__all__ = [
    # Scenario Generation
    "ScenarioGenerator",
    "EconomicScenario",
    "AG43Scenarios",
    "VasicekParams",
    "EquityParams",
    "generate_deterministic_scenarios",
    "calculate_scenario_statistics",
    # VM-21
    "VM21Calculator",
    "VM21Result",
    "PolicyData",
    "calculate_cte_levels",
    "sensitivity_analysis",
    # VM-22
    "VM22Calculator",
    "VM22Result",
    "FixedAnnuityPolicy",
    "StochasticExclusionResult",
    "ReserveType",
    "compare_reserve_methods",
    "vm22_sensitivity",
]
