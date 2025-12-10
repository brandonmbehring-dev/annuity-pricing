"""
Property-based testing using Hypothesis.

This package contains property tests that verify mathematical invariants
hold across randomly generated inputs. These tests are more comprehensive
than parameterized tests because they explore the full input space.

Modules:
    test_option_properties: Black-Scholes invariants (bounds, parity, monotonicity)
    test_payoff_properties: FIA/RILA payoff invariants (floor, cap, buffer)
    test_greeks_properties: Greeks mathematical properties
    test_mc_properties: Monte Carlo convergence properties
"""
