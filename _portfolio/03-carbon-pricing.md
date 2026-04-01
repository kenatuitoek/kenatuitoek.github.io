---
title: "Carbon Pricing & Steel Export Competitiveness"
excerpt: "Fixed-effects panel regression estimating the impact of carbon pricing on iron and steel exports across 36 countries, with product-level heterogeneity."
collection: portfolio
category_label: "Policy / Econometrics"
status: "Complete"
accent_color: "#5B8DEF"
tagline: "Carbon Price Elasticity of EITE Exports"
methods:
  - Fixed-effects OLS (exporter + year FE)
  - Product interaction model (HS72 vs HS73)
  - Exporter-clustered standard errors
  - Trade-weighted carbon price exposure
tools:
  - R
  - ggplot2
  - UN Comtrade (WITS)
  - World Bank Carbon Pricing Dashboard
tags:
  - carbon pricing
  - CBAM
  - EU ETS
  - trade policy
  - climate economics
  - fixed effects
---

## Overview

Empirical analysis of how carbon pricing affects export competitiveness in energy-intensive, trade-exposed (EITE) steel sectors. Estimates the elasticity of iron and steel exports with respect to compliance carbon prices across 36 countries (2022-2024), allowing effects to differ between raw steel (HS 72) and processed steel articles (HS 73). The identification strategy builds through four nested specifications, progressively adding fixed effects and product-level interactions.

---

## Data

- **Trade data**: UN Comtrade bilateral exports at HS 6-digit level for chapters 72 (iron and steel) and 73 (articles of iron/steel), 2022-2024
- **Carbon pricing**: World Bank Compliance Price dataset (1990-2025), covering carbon taxes and ETS schemes. Excludes voluntary offsets
- **Panel structure**: 200 exporter-product-year observations across 36 countries, merged by ISO3 code and year
- **Key variables**: Log export value (thousands USD), average compliance carbon price (USD/tCO2e), ETS and carbon tax prices separately, number of active schemes

---

## Exploratory Analysis

Constructed four visualisations to characterise the data before estimation:

1. **Choropleth map** of average carbon pricing levels across countries (2024), showing concentration of high prices in the EU, North America, and Australasia
2. **Trade-weighted carbon price exposure** comparison: HS72 faces slightly higher exposure ($44.7/tCO2e) than HS73 ($40.9/tCO2e), despite comparable trade volumes (~$955B vs ~$907B)
3. **Scatter plots** of carbon prices against export values (log scale) with fitted regression lines, showing a negative raw relationship for both product categories
4. **Time series** of trade-weighted carbon price exposure (2022-2024): both HS72 and HS73 peaked in 2023 before declining in 2024

Summary statistics split by regime type revealed that countries with both ETS and carbon tax instruments face slightly lower average carbon prices ($61.2) than ETS-only countries ($70.3), but dual-regime countries have lower mean export values.

---

## Methodology

Estimated the following baseline specification with nested controls:

$$y_{ipt} = \alpha + \beta \cdot CP_{it} + \theta \cdot HS73_p + \delta \cdot (CP_{it} \times HS73_p) + \mu_i + \tau_t + \varepsilon_{ipt}$$

Where $y_{ipt}$ is log export value for product $p \in \{HS72, HS73\}$ from country $i$ in year $t$, $CP_{it}$ is the average compliance carbon price, and $\mu_i$, $\tau_t$ are exporter and year fixed effects. Standard errors clustered at the exporter level throughout.

[Download full report (PDF)](/files/carbon_pricing_steel_exports.pdf) | [R notebook](/files/carbon_pricing_code.ipynb)

The four specifications build progressively:

1. **Pooled OLS** — unconditional correlation
2. **+ Exporter FE** — absorbs time-invariant country characteristics (technology, resource endowments, trade infrastructure)
3. **+ Year FE** — controls for common global shocks (commodity prices, exchange rates, pandemic disruptions)
4. **+ Product interaction** — allows carbon price effect to differ between raw steel (HS72) and processed steel articles (HS73)

```r
# Specification 4: Full interaction model
library(fixest)

model_4 <- feols(
  log_export_value ~ carbon_price * hs73_dummy | exporter + year,
  data = panel,
  vcov = ~exporter
)
summary(model_4)
```

---

## Results

| | (1) Pooled | (2) + Country FE | (3) + Year FE | (4) Product Interaction |
|---|---|---|---|---|
| Carbon Price | -0.026*** | 0.002** | 0.000 | 0.002 |
| | (0.004) | (0.001) | (0.001) | (0.003) |
| HS73 (dummy) | | | | 0.007 |
| | | | | (0.302) |
| Carbon Price x HS73 | | | | -0.003 |
| | | | | (0.005) |
| R-squared | 0.126 | 0.959 | 0.960 | 0.963 |
| Observations | 200 | 200 | 200 | 200 |

*Standard errors clustered by exporter. \*\*\* p<0.01, \*\* p<0.05, \* p<0.1.*

**Key findings:**

The pooled OLS estimate (Column 1) shows a significant negative association: a $1/tCO2e increase in carbon price is associated with a 2.6% decline in steel exports. However, this relationship is entirely absorbed by country fixed effects. Once exporter characteristics are controlled for (Column 2), the sign flips to a small positive coefficient, and adding year fixed effects (Column 3) reduces this to effectively zero.

The interaction term in Column 4 is negative (-0.003) but statistically insignificant, providing weak suggestive evidence that processed steel (HS73) may be slightly more sensitive to carbon pricing than raw steel (HS72). The R-squared jumps from 0.126 to 0.963 between Columns 1 and 4, confirming that country-level heterogeneity dominates the variation in steel exports.

---

## Planned Extension

The current analysis establishes a clean baseline but has limitations acknowledged in the paper. The following extensions address each directly:

### Difference-in-differences design

The current fixed-effects approach estimates correlations within countries over time but cannot isolate the causal effect of carbon pricing changes. A proper DiD design exploiting the staggered adoption or price changes of carbon pricing instruments across countries would strengthen the causal claim. This requires extending the panel beyond the current 3-year window.

### Expanded panel and instrumental variables

With only 200 observations across 3 years, the current analysis lacks power to detect small effects, particularly for the interaction term. Extending the panel to 2012-2024 (leveraging the full World Bank dataset which runs from 1990) would increase sample size substantially. An IV strategy using regulatory stringency indices or political variables as instruments for carbon price levels would address the endogeneity concern that carbon prices and trade performance are jointly determined.

### Bilateral trade structure

The current aggregation to exporter-product-year level cannot identify trade diversion (carbon leakage to specific partner countries). Expanding to bilateral flows would enable testing whether exports shift toward importers with lower carbon prices, the core carbon leakage mechanism.

### PPML estimation

Log-linearised OLS drops zero trade flows and can produce inconsistent estimates under heteroskedasticity (Santos Silva and Tenreyro, 2006). Poisson pseudo-maximum likelihood estimation on trade values in levels would address both issues and is the current standard in the gravity model literature.

```r
# PPML with high-dimensional fixed effects
library(fixest)

ppml_model <- fepois(
  export_value ~ carbon_price * hs73_dummy | exporter + year,
  data = panel,
  vcov = ~exporter
)
```

### Multi-sector expansion

The current analysis covers only HS chapters 72 and 73. Extending to other CBAM-relevant sectors (cement, aluminium, fertilisers, electricity, hydrogen) would enable cross-sector comparison of carbon price sensitivity and directly inform CBAM scope design.

---

## Key Takeaways

**From current analysis:**
- The negative pooled OLS relationship between carbon prices and steel exports is entirely driven by cross-country heterogeneity, not within-country variation over time
- Once exporter and year fixed effects are included, there is no statistically significant relationship between carbon prices and export values
- Weak suggestive evidence of differential sensitivity between raw steel (HS72) and processed steel (HS73), but underpowered to detect
- Country-level characteristics explain 96% of export variation, leaving very limited scope for carbon price effects in this specification

**Expected from extension:**
- DiD and IV approaches will provide cleaner causal identification
- Longer panels and bilateral structure will increase power and enable carbon leakage testing
- PPML will address the econometric concerns with log-linear OLS on trade data
- Multi-sector expansion will test whether the null result is steel-specific or generalises across EITE sectors
