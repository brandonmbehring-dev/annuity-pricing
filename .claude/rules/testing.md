---
paths:
  - "tests/**"
---

## Test-First Development

1. Write test that would catch the bug (from anti_patterns/)
2. Verify test fails (red)
3. Implement code
4. Verify test passes (green)
5. Run all anti-pattern tests
6. Refactor if needed

### Test Categories

```
tests/
├── anti_patterns/        # Bug prevention (MUST pass before commit)
│   ├── test_arbitrage_bounds.py
│   ├── test_put_call_parity.py
│   └── test_floor_enforcement.py
├── validation/           # External verification (Hull examples)
│   ├── test_bs_known_answers.py
│   ├── test_mc_convergence.py
│   └── test_wink_sanity.py
├── unit/                 # Standard unit tests
└── integration/          # End-to-end pricing tests
```

### Pre-Commit Checks

```bash
pytest tests/anti_patterns/ -v  # Bug prevention
pytest tests/validation/ -v     # Known answers
```
