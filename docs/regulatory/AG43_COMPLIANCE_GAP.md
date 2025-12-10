# AG43/VM-21/VM-22 Compliance Gap Analysis

**Document Status**: Educational Reference
**Last Updated**: 2025-12-09
**Purpose**: Document gaps between this implementation and NAIC regulatory requirements

---

## Executive Summary

This repository provides **educational/research implementations** of NAIC regulatory calculations. These implementations are:

- **Conceptually correct** for understanding reserve methodologies
- **NOT suitable** for actual regulatory filings
- **NOT NAIC-compliant** in scenario generation or actuarial certification

**Bottom Line**: Production regulatory compliance requires 6-12 months additional work, FSA/MAAA certification, and approximately $15,000+ annually in scenario generator licensing.

---

## 1. What We Have vs. What's Required

### 1.1 VM-21 (Variable Annuity Reserves)

| Component | Our Implementation | NAIC Requirement | Gap |
|-----------|-------------------|------------------|-----|
| **Scenario Generator** | Custom Vasicek + GBM | AAA ESG / GOES | **Critical** |
| **CTE Calculation** | Standard CTE70 formula | CTE70 per AG43 | ✅ Conceptually correct |
| **SSA Calculation** | Simplified stress | Prescribed scenarios | **Medium** |
| **Mortality Tables** | SOA 2012 IAM approximation | Prescribed w/ improvement | **Low** |
| **Policy Modeling** | Single-policy demo | Full inforce portfolio | **High** |
| **Hedging Credit** | Not implemented | CDHS if applicable | **High** |
| **Reinsurance** | Not implemented | Required if applicable | **Medium** |
| **Aggregation** | Single policy | Across all policies | **High** |
| **VM-31 Report** | Not implemented | Required documentation | **High** |

### 1.2 VM-22 (Fixed Annuity PBR)

| Component | Our Implementation | NAIC Requirement | Gap |
|-----------|-------------------|------------------|-----|
| **Scenario Generator** | Custom Vasicek + GBM | AAA ESG → GOES (2026) | **Critical** |
| **SET/SST Tests** | Simplified implementation | Full prescribed tests | **Medium** |
| **Deterministic Reserve** | Basic calculation | Per VM-22 Section 5 | **Medium** |
| **Stochastic Reserve** | CTE70 implementation | Per VM-22 Section 6 | **Medium** |
| **Lapse Assumptions** | Fixed rate input | Company experience + credibility | **High** |
| **Asset Modeling** | Not implemented | ALM matching required | **High** |
| **NGE Modeling** | Not implemented | Non-guaranteed elements | **High** |
| **VM-G Governance** | Not implemented | Corporate governance requirements | **High** |
| **VM-31 Report** | Not implemented | Required documentation | **High** |

### 1.3 Scenario Generation

| Component | Our Implementation | NAIC Requirement | Gap |
|-----------|-------------------|------------------|-----|
| **Interest Rate Model** | Vasicek mean-reversion | NAIC-prescribed calibration | **Critical** |
| **Equity Model** | GBM log-normal | NAIC-prescribed calibration | **Critical** |
| **Correlation** | Cholesky decomposition | NAIC-prescribed structure | **Critical** |
| **Calibration** | Academic defaults | NAIC calibration criteria | **Critical** |
| **Scenario Files** | Self-generated | naic.conning.com files acceptable | **Critical** |
| **Documentation** | Code comments | Full methodology documentation | **Medium** |

---

## 2. Critical Gap: Scenario Generators

### 2.1 Current NAIC Requirements

**AAA Economic Scenario Generator (ESG)**:
- Standard through December 31, 2026
- Provides equity and interest rate scenarios
- Meets NAIC calibration criteria
- Proprietary tool (not freely available)

### 2.2 Future NAIC Requirements

**GOES (Generator of Economic Scenarios)**:
- Mandatory effective December 31, 2026
- Developed by Conning for NAIC
- Stochastic scenarios: interest rates, equity returns, corporate bonds
- Deterministic scenarios for DR calculations
- Scenario files available from naic.conning.com

### 2.3 Why This Matters

Our Vasicek + GBM implementation:
- **Cannot** be used for regulatory filings
- **Does not** meet NAIC calibration criteria
- **Does not** produce identical scenarios to prescribed generators
- **Is useful** for education, research, and prototype testing only

### 2.4 Options for Compliance

1. **Use Pre-Packaged Scenario Files** (Lowest Cost)
   - Download from naic.conning.com (some free, some licensed)
   - Parse into our data structures
   - Estimated effort: 2-4 weeks
   - Cost: Free to ~$5,000/year depending on scenario set

2. **License AAA ESG** (Medium Cost)
   - Full scenario generation capability
   - Estimated cost: $10,000-20,000/year
   - Requires actuarial oversight for proper use

3. **Implement GOES Parser** (Future-Proof)
   - Wait for GOES release and documentation
   - Implement file parser for GOES output
   - Estimated effort: 4-8 weeks when available

---

## 3. Actuarial Certification Requirements

### 3.1 Who Can Sign

VM-21/VM-22 reserve calculations require certification by a **Qualified Actuary**:

- **FSA** (Fellow of the Society of Actuaries) with life experience, OR
- **MAAA** (Member of the American Academy of Actuaries) in good standing

### 3.2 What Certification Entails

The Qualified Actuary must certify:

1. **Methodology appropriateness** for the company's products
2. **Assumption reasonableness** based on company experience
3. **Model validation** - independent review of calculations
4. **Documentation completeness** per VM-31 requirements
5. **Professional responsibility** under Actuarial Standards of Practice

### 3.3 Cost Implications

| Option | Estimated Annual Cost |
|--------|----------------------|
| In-house actuary (share of salary) | $50,000 - $150,000 |
| Consulting actuary (reserve support) | $20,000 - $50,000 |
| Peer review only | $5,000 - $15,000 |

---

## 4. VM-31 Actuarial Report Requirements

### 4.1 Required Sections

VM-31 requires extensive documentation not implemented here:

1. **Executive Summary**
   - Material changes from prior year
   - Key risk drivers

2. **Data Summary**
   - Inforce data quality assessment
   - Data reconciliation

3. **Asset Summary** (VM-22)
   - Supporting asset details
   - ALM methodology

4. **Liability Summary**
   - Product descriptions
   - Reserve methodology by product

5. **Assumptions**
   - Mortality, lapse, expense assumptions
   - Basis for each assumption
   - Sensitivity testing results

6. **Scenario Summary**
   - Scenario generator identification
   - Calibration verification

7. **Reserve Results**
   - By product line
   - By scenario type (DR vs SR)

8. **Model Validation**
   - Validation procedures
   - Key findings

### 4.2 Gap Assessment

| VM-31 Section | Implementation Status |
|---------------|----------------------|
| Executive Summary | Not implemented |
| Data Summary | Not implemented |
| Asset Summary | Not implemented |
| Liability Summary | Basic structure only |
| Assumptions | Hardcoded, no documentation |
| Scenario Summary | Not implemented |
| Reserve Results | Basic output only |
| Model Validation | Not implemented |

---

## 5. Estimated Effort for Full Compliance

### 5.1 Technical Implementation

| Work Item | Effort | Dependencies |
|-----------|--------|--------------|
| GOES/ESG scenario file parser | 4-8 weeks | Scenario file access |
| Full policy data model | 4-6 weeks | — |
| ALM asset modeling | 8-12 weeks | Asset data |
| Lapse/mortality calibration | 4-8 weeks | Company experience |
| VM-31 report generator | 4-6 weeks | All above |
| Model validation framework | 6-10 weeks | All above |
| **Total Technical** | **30-50 weeks** | — |

### 5.2 Non-Technical Requirements

| Item | Timeline | Notes |
|------|----------|-------|
| Scenario generator licensing | 2-4 weeks | Contract negotiation |
| Actuarial engagement | 4-8 weeks | FSA/MAAA availability |
| State regulatory approval | 3-6 months | Varies by state |
| Independent model validation | 8-12 weeks | Third-party review |

### 5.3 Total Timeline

**Optimistic**: 6-9 months with dedicated resources
**Realistic**: 12-18 months for first compliant filing

---

## 6. What This Implementation IS Useful For

### 6.1 Educational Purposes ✅

- Understanding CTE methodology
- Learning VM-21/VM-22 structure
- Demonstrating reserve sensitivity
- Teaching actuarial students

### 6.2 Research Purposes ✅

- Academic papers on reserve methodology
- Comparative analysis of reserve approaches
- Prototyping new features
- Sensitivity analysis frameworks

### 6.3 Internal Analysis ✅

- Approximate reserve estimates
- "What-if" scenario analysis
- Training materials
- Documentation of methodology understanding

### 6.4 NOT Suitable For ❌

- Regulatory filings (state insurance department)
- Statutory financial statements
- NAIC Model Audit Rule compliance
- Risk-Based Capital calculations
- ORSA (Own Risk and Solvency Assessment)

---

## 7. Recommendations

### 7.1 If You Need Educational/Research Tool

**Current implementation is appropriate.** Use for:
- Learning reserve concepts
- Prototyping analysis
- Sensitivity studies
- Academic work

### 7.2 If You Need Production Compliance

**Do not use this implementation for filings.** Instead:

1. **Engage qualified actuary** (FSA/MAAA) immediately
2. **License NAIC-prescribed scenarios** (AAA ESG or GOES when available)
3. **Plan 12-18 month implementation** timeline
4. **Budget $50,000-200,000** for first-year compliance
5. **Establish ongoing governance** per VM-G

### 7.3 If You Want to Extend This Codebase

Consider implementing:

1. **NAIC scenario file parser** (naic.conning.com format)
   - Enables use of prescribed scenarios
   - Removes critical compliance gap
   - Estimated: 4-8 weeks

2. **VM-31 report template generator**
   - Creates documentation structure
   - Still requires actuarial input
   - Estimated: 4-6 weeks

3. **Enhanced policy data model**
   - Support full contract features
   - Required for real-world policies
   - Estimated: 4-6 weeks

---

## 8. References

### 8.1 NAIC Resources

- [VM-21: Requirements for Principle-Based Reserves for Variable Annuities](https://content.naic.org/)
- [VM-22: Requirements for Principle-Based Reserves for Non-Variable Annuities](https://content.naic.org/)
- [VM-31: PBR Actuarial Report Requirements](https://content.naic.org/)
- [VM-G: Corporate Governance Guidance](https://content.naic.org/)
- [NAIC Scenario Files](https://naic.conning.com/)

### 8.2 SOA/AAA Resources

- [AAA Economic Scenario Generator](https://www.actuary.org/)
- [SOA Experience Studies](https://www.soa.org/)

### 8.3 Timeline References

- GOES effective date: December 31, 2026
- VM-22 voluntary adoption: January 1, 2026
- VM-22 mandatory: January 1, 2029

---

## Appendix A: Quick Compliance Checklist

For anyone evaluating whether this implementation meets regulatory needs:

| Requirement | Status | Notes |
|-------------|--------|-------|
| NAIC-prescribed scenarios | ❌ | Custom Vasicek + GBM |
| Qualified Actuary certification | ❌ | N/A (software only) |
| VM-31 documentation | ❌ | Not implemented |
| Model validation | ❌ | Not implemented |
| Full policy modeling | ⚠️ | Single-policy demo only |
| ALM asset modeling | ❌ | Not implemented |
| State regulatory approval | ❌ | N/A (software only) |

**Conclusion**: This implementation has **zero** compliance requirements met.
Use for education and research only.
