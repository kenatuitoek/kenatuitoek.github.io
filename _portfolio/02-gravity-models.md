---
title: "Unpacking Africa's Trade Integration"
excerpt: "PPML-HDFE gravity models quantifying the effects of WTO membership and preferential trade agreements on African bilateral trade flows, using CEPII panel data spanning 1960–2021."
collection: portfolio
category_label: "Econometrics"
status: "Complete"
accent_color: "#5B8DEF"
tagline: "PPML Estimation with High-Dimensional Fixed Effects"
methods:
  - PPML gravity estimation (Santos Silva & Tenreyro, 2006)
  - High-dimensional fixed effects (country-pair, importer-year, exporter-year)
  - Multilateral resistance controls (Anderson & van Wincoop, 2003)
  - PTA heterogeneity decomposition (customs unions, FTAs, partial-scope)
  - Border-year globalisation controls (Herman, 2023)
  - Dynamic lag structures for phased PTA effects
tools:
  - Stata (ppmlhdfe)
  - CEPII gravity database
  - WTO RTA database
tags:
  - gravity model
  - trade
  - PPML
  - panel data
  - Africa
  - WTO
  - preferential trade agreements
  - high-dimensional fixed effects
---

## Overview

This dissertation estimates the effect of WTO membership and preferential trade agreements (PTAs) on African bilateral trade integration using a **PPML-HDFE gravity framework** on a panel of **~960,000 observations** spanning 1960–2021. The central question is whether multilateral (WTO) and regional (PTA) trade regimes act as complements or substitutes in the African context — and whether the aggregate PTA effect masks heterogeneity across agreement types.

## Research question

1. To what extent do WTO commitments and PTA participation improve Africa's trade integration?
2. Do multilateral and preferential agreements act as complements or substitutes in shaping that integration?

---

## Methodology

The baseline specification follows the structural gravity literature:

$$\text{Imports}_{ij,t} = \exp\left(\beta_1 \text{WTO}_{ij,t} + \beta_2 \text{PTA}_{ij,t} + \alpha_{ij} + \gamma_{it} + \delta_{jt}\right) + \varepsilon_{ij,t}$$

where $\alpha_{ij}$ are country-pair fixed effects absorbing all time-invariant bilateral characteristics (distance, contiguity, common language, colonial ties), and $\gamma_{it}$, $\delta_{jt}$ are importer-year and exporter-year fixed effects that control for multilateral resistance terms following Anderson and van Wincoop (2003).

**Why PPML rather than log-linear OLS:**
- Handles zero trade flows, which are common in African bilateral pairs
- Consistent under heteroskedasticity (Santos Silva & Tenreyro, 2006)
- Combined with HDFE, absorbs unobserved heterogeneity and mitigates endogeneity

The WTO variable equals 1 when both the importer and exporter are WTO members in year $t$. The PTA variable equals 1 when both partners share a preferential trade agreement. Country-pair fixed effects absorb all time-invariant bilateral gravity variables, so distance, language, and colonial ties are not estimated as separate coefficients — they are controlled for by construction.

---

## Key results

The specification builds incrementally from naïve pooled regressions to the fully saturated PPML-HDFE model. Without fixed effects, both WTO and PTA coefficients are large and highly significant but severely upward-biased. The preferred specification (PPML-HDFE with full multilateral resistance controls, excluding GFC years) yields:

| Variable | Coefficient | Trade effect (%) | Significance |
|----------|------------|-----------------|--------------|
| WTO (both members) | 0.168 | ≈18% | *** |
| PTA (aggregate) | 0.026 | ≈3% | not significant |

**The central finding:** After properly controlling for multilateral resistance, WTO membership provides a consistent and robust trade premium of approximately 18%. The aggregate PTA dummy, by contrast, is economically small and statistically indistinguishable from zero — but this null result conceals important heterogeneity across agreement types.

---

## Robustness checks

Three structured robustness checks test the stability of the baseline coefficients, progressing from the most direct source of bias to broader macroeconomic and timing concerns.

### 1. PTA heterogeneity

Replacing the single PTA dummy with three categories reveals that only **customs unions** — which involve common external tariff alignment — produce a significant and substantial trade effect:

| Agreement type | Coefficient | Trade effect (%) | Significance |
|----------------|------------|-----------------|--------------|
| Customs union | 0.202 | ≈22% | *** |
| Free trade agreement | 0.011 | ≈1% | not significant |
| Other (partial-scope) | 0.122 | ≈13% | not significant |

The customs union effect amplifies to approximately 27% when coupled with deeper economic integration provisions (EIA). Standalone EIAs without tariff convergence show a strongly negative coefficient (−0.456), suggesting that regulatory harmonisation without a unified external tariff creates compliance costs that deter trade. Throughout all disaggregated specifications, the WTO coefficient remains stable at approximately 19%.

### 2. Globalisation controls

Following Herman (2023), adding border-year interaction terms to absorb continent-wide shifts in trade openness (China's accession, commodity cycles, supply chain reconfigurations) leaves the WTO coefficient virtually unchanged at 0.171. This confirms the multilateral premium is not an artefact of secular globalisation trends.

### 3. Dynamic lag structures

Incorporating two-, four-, and six-year lags of each PTA category tests for phased adjustment effects. The WTO coefficient remains stable at 0.173. Only the four-year customs union lag shows marginal significance, consistent with Baier and Bergstrand's (2007) finding that customs union effects take several years to materialise as members align external tariffs. The six-year FTA lag is marginally negative, suggesting preference erosion over time. All other lags are insignificant, supporting the choice to retain only contemporaneous dummies in the preferred specification.

---

## Data

- **Trade flows:** Bilateral imports from the CEPII gravity database, country-pair level, 1960–2021
- **Policy indicators:** WTO membership dates and PTA indicators from the WTO Regional Trade Agreements Database, disaggregated by agreement type (customs union, FTA, partial-scope, with and without EIA provisions)
- **Fixed effects structure:** Country-pair ($\alpha_{ij}$) absorbs distance, language, contiguity, colonial ties; importer-year ($\gamma_{it}$) and exporter-year ($\delta_{jt}$) absorb multilateral resistance and all time-varying country characteristics
- **Sample:** ~960,000 bilateral observations; ~880,000 in the preferred specification after excluding GFC years

---

## Conclusions

The results support Baldwin's (2006) framework of "multilateralising regionalism": WTO membership provides an independent, stable trade premium of approximately 19% that survives all robustness checks. Among PTAs, only customs unions with deep integration and common external tariff alignment deliver a comparable effect (20–25%). Shallow FTAs and partial-scope agreements show no significant impact once multilateral resistance and globalisation trends are controlled for. The policy implication is that African governments should prioritise deepening existing arrangements into full customs unions — or at minimum, align AfCFTA protocols toward external tariff convergence — rather than proliferating shallow agreements that add compliance costs without measurable trade gains.
