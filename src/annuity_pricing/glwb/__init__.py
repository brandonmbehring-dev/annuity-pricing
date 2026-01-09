"""
GLWB (Guaranteed Lifetime Withdrawal Benefit) Pricing - Phase 8.

Implements path-dependent pricing for GLWB products:
- GWB tracking (rollup, ratchet, reset)
- Path-dependent MC simulation
- GLWB value decomposition

See: docs/knowledge/domain/glwb_mechanics.md
See: docs/references/L3/bauer_kling_russ_2008.md
"""

from .gwb_tracker import (
    GWBConfig,
    GWBState,
    GWBTracker,
    RollupType,
    StepResult,
)
from .path_sim import (
    GLWBPathSimulator,
    GLWBPricingResult,
    PathResult,
)
from .rollup import (
    CompoundRollup,
    RatchetMechanic,
    RollupResult,
    SimpleRollup,
    calculate_rollup_with_cap,
    compare_rollup_methods,
)

__all__ = [
    # GWB Tracker
    "GWBTracker",
    "GWBState",
    "GWBConfig",
    "RollupType",
    "StepResult",
    # Rollup Mechanics
    "SimpleRollup",
    "CompoundRollup",
    "RatchetMechanic",
    "RollupResult",
    "calculate_rollup_with_cap",
    "compare_rollup_methods",
    # Path Simulation
    "GLWBPathSimulator",
    "GLWBPricingResult",
    "PathResult",
]
