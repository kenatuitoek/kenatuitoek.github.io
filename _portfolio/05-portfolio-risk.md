---
title: "Portfolio Risk Modelling"
excerpt: "CAPM estimation, return distribution diagnostics, and non-parametric inference for equity risk analysis, with planned extensions into GARCH volatility and multi-factor decomposition."
collection: portfolio
category_label: "Finance"
status: "In Progress"
accent_color: "#C792EA"
tagline: "Return Distributions & Market Risk Estimation"
methods:
  - CAPM single-factor regression
  - Kernel density estimation
  - Skewness-kurtosis & Shapiro-Wilk normality tests
  - Mann-Whitney U non-parametric test
  - Split-sample volatility comparison
tools:
  - Stata
  - Python (planned)
  - arch (planned)
  - statsmodels (planned)
tags:
  - CAPM
  - distribution diagnostics
  - non-parametric inference
  - equity risk
  - GARCH
  - factor models
pdf_link: "/files/bby_stock_analysis.pdf"
---

## Overview

Statistical analysis of BBY (Best Buy Co., Inc.) weekly log returns against the S&P 500, covering return distribution diagnostics, non-parametric inference, and CAPM market risk estimation. The project follows a deliberate methodological progression: characterise the return distribution, test parametric assumptions, apply appropriate non-parametric alternatives, and estimate systematic risk exposure through a single-factor model.

[Download full report (PDF)](/files/bby_stock_analysis.pdf)

---

## Data

- **Asset**: BBY (Best Buy Co., Inc.) weekly log returns
- **Benchmark**: S&P 500 weekly log returns
- **Period**: 2014 to 2018 (251 weekly observations)
- **Source**: S&P 500 constituent data

---

## Part 1: Return Distribution Diagnostics

Characterised the empirical distribution of BBY returns through summary statistics, percentile analysis, and visual inspection. The time series reveals moderate volatility clustering with a sharp negative outlier in the early period (minimum weekly return of -41.5%), which motivated formal distributional testing.

Split-sample comparison of the first and second halves of the data showed mean returns improving from -0.09% to +0.66% per week, with standard deviation declining from 5.99% to 4.30%, suggesting reduced volatility over the sample period.

### Normality testing

Tested the null hypothesis of normally distributed returns using two complementary approaches:

**Skewness-kurtosis test**: Joint chi-squared statistic of 114.43 decisively rejects normality (p < 0.001). Robustness check excluding extreme values still rejects (chi-squared = 39.98).

**Shapiro-Wilk test**: z-score of 7.351 confirms rejection. Again robust to exclusion of outliers (z = 6.040 after trimming).

Both tests agree: BBY returns are not normally distributed, with significant negative skew and excess kurtosis. This rules out standard parametric inference for comparing return distributions.

### Non-parametric test

Applied a **Mann-Whitney U test** to compare return distributions across the two sample halves. The rank-sum approach is appropriate given the non-normality finding, as it does not depend on distributional assumptions.

Result: p-value of 0.4557, failing to reject the null of equal distributions. Despite visual differences in volatility, the return distributions across the two periods are not statistically distinguishable.

---

## Part 2: CAPM Market Risk Estimation

Estimated the Capital Asset Pricing Model as a single-factor regression:

$$R_i - R_f = \alpha + \beta (R_m - R_f) + \varepsilon_i$$

### Results

| Parameter | Estimate | Std. Error | t-stat | p-value |
|-----------|----------|------------|--------|---------|
| Beta      | 1.124    | 0.185      | 6.07   | 0.000   |
| Alpha     | 0.001    | 0.003      | 0.37   | 0.714   |

- **R-squared**: 12.9% of BBY return variance explained by market movements
- **Root MSE**: 0.049, indicating reasonable model fit
- **Beta test** (H0: beta = 1): p-value of 0.671, failing to reject. BBY is not an aggressive stock; its market sensitivity is statistically indistinguishable from the market portfolio

The insignificant alpha suggests no abnormal return after adjusting for market risk, consistent with weak-form CAPM pricing.

---

## Planned Extension (Python)

The current Stata analysis establishes the distributional and single-factor baseline. The following extensions will upgrade the project toward a more complete risk modelling framework:

### GARCH(1,1) Conditional Volatility

The split-sample comparison showed declining volatility, but this is a coarse measure. GARCH modelling will capture time-varying conditional volatility dynamics and produce forward-looking risk estimates.

```python
from arch import arch_model
import numpy as np

# Fit GARCH(1,1) to weekly returns
garch = arch_model(returns, vol='Garch', p=1, q=1,
                   mean='AR', lags=1, dist='t')
result = garch.fit(disp='off')

# Extract conditional volatility
cond_vol = result.conditional_volatility
annualised_vol = cond_vol * np.sqrt(52)  # weekly data

# Forecast 4-week ahead variance
forecast = result.forecast(horizon=4)
var_4w = forecast.variance.iloc[-1].values
```

**Why this matters**: The normality tests already showed excess kurtosis and fat tails. A Student-t GARCH specification directly accounts for this, producing more realistic volatility estimates and VaR forecasts than the rolling-window approach implied by the split-sample analysis.

### Fama-French 3-Factor Decomposition

The CAPM R-squared of 12.9% leaves 87% of BBY variance unexplained. A multi-factor model will decompose this further:

$$R_i - R_f = \alpha_i + \beta_{\text{MKT}}(R_m - R_f) + \beta_{\text{SMB}} \cdot \text{SMB} + \beta_{\text{HML}} \cdot \text{HML} + \varepsilon_i$$

```python
import statsmodels.api as sm

# Fama-French factors from Kenneth French data library
X = sm.add_constant(ff_factors[['Mkt-RF', 'SMB', 'HML']])
y = returns - ff_factors['RF']

model = sm.OLS(y, X).fit(cov_type='HC1')
print(model.summary())
```

This separates systematic risk into market, size, and value components, and will clarify whether the current alpha estimate survives multi-factor adjustment.

### Bootstrap Inference

The non-normal return distribution motivates non-parametric inference beyond the Mann-Whitney test. Block bootstrap (preserving autocorrelation) will construct confidence intervals for Sharpe ratios and volatility estimates that do not rely on distributional assumptions.

```python
from arch.bootstrap import StationaryBootstrap

bs = StationaryBootstrap(12, returns)  # avg block length 12 weeks

sharpe_ratios = []
for data in bs.bootstrap(1000):
    r = data[0][0]
    sharpe_ratios.append(r.mean() / r.std() * np.sqrt(52))

ci_lower, ci_upper = np.percentile(sharpe_ratios, [2.5, 97.5])
```

---

## Key Takeaways

**From current analysis:**
- BBY returns are not normally distributed (confirmed by both skewness-kurtosis and Shapiro-Wilk tests, robust to outlier exclusion)
- CAPM beta of 1.12 is not statistically different from 1, classifying BBY as a market-neutral rather than aggressive stock
- Market risk explains only 12.9% of return variance, suggesting substantial idiosyncratic risk

**Expected from extension:**
- GARCH conditional volatility will replace the static split-sample comparison with dynamic risk estimates
- Multi-factor decomposition will identify whether size or value exposure explains the remaining 87% of variance
- Bootstrap confidence intervals will provide honest uncertainty quantification for risk metrics
