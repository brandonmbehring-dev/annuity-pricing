# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - 2025-12-06

### Added

- **Technical Hardening**
  - Wired mortality tables into GLWB via `MortalityLoader`
  - Integrated behavioral models into `path_sim.py`
  - Risk-neutral drift with `RiskNeutralEquityParams`
  - RILA breakeven solver with `brentq`
  - RILA Greeks (`RILAGreeks` dataclass)

- **PyPI Publishability**
  - `py.typed` marker for PEP 561
  - `CITATION.cff` for academic citations
  - GitHub Actions CI/CD workflows
  - Comprehensive pyproject.toml metadata

- **Validation**
  - Adapter tests for financepy, QuantLib, pyfeng
  - Validation notebooks executed with outputs
  - CROSS_VALIDATION_MATRIX.md with tolerances

- **Documentation**
  - Sphinx documentation with Furo theme
  - MyST-NB for notebook integration
  - API autodoc for all modules

### Changed

- Bumped version from 0.1.0 to 0.2.0
- Test count increased from 911 to 948

### Fixed

- FIA/RILA PV formula now correctly discounts full payoff
- Greeks vega/theta scaling conventions documented

## [0.1.0] - 2025-12-05

### Added

- Initial release with Phases 0-10 complete
- MYGA, FIA, RILA, GLWB pricers
- Black-Scholes and Monte Carlo engines
- VM-21/VM-22 regulatory prototypes
- Behavioral models (lapse, withdrawal, expenses)
- Yield curve and mortality loaders
- 911 passing tests
