# Security Policy

## Scope

This is a **research and pricing tooling** project, NOT production software. It is designed for:
- Actuarial analysis and research
- Educational demonstrations
- Pricing model development

It is **NOT** designed for:
- Production trading systems
- Real-time pricing in customer-facing applications
- Regulatory compliance calculations without validation

## Supported Versions

| Version | Status |
|---------|--------|
| 0.2.x   | Supported |
| < 0.2   | Unsupported |

## Reporting a Vulnerability

If you discover a security vulnerability, please:

1. **Do NOT** open a public issue
2. Email details to the maintainer directly (or use GitHub's private vulnerability reporting if available)
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### Response Timeline

- **Initial response**: Within 48 hours
- **Assessment**: Within 1 week
- **Fix (if applicable)**: Depends on severity

### What to Expect

1. Acknowledgment of your report
2. Assessment of the vulnerability
3. If applicable: fix development and testing
4. Credit in release notes (unless you prefer anonymity)

## Security Considerations

### Data Handling

This package processes:
- Market data (rates, prices, volatility)
- Product specifications
- Pricing results

It does NOT handle:
- Personally identifiable information (PII)
- Financial account numbers
- Customer data

### API Keys

If you use external data sources (e.g., FRED API), protect your API keys:
- Use environment variables, not hardcoded values
- Never commit `.env` files with real keys
- See `.gitignore` for excluded patterns

### Dependencies

We regularly update dependencies to address known vulnerabilities. Run:
```bash
pip install --upgrade -e ".[dev]"
```

## Known Limitations

1. **Not hardened for adversarial input**: The package assumes well-formed input from trusted sources

2. **No rate limiting**: High-volume requests could exhaust resources

3. **Floating point precision**: Financial calculations use standard IEEE 754 double precision, which has known limitations for exact decimal arithmetic

## Contact

For security concerns, contact the maintainer through private channels rather than public issues.
