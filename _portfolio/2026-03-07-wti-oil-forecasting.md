---
title: "ML-Driven WTI Crude Oil Forecasting & Trading System"
excerpt: "Walk-forward validated GBR/RF ensemble for WTI futures trading with volatility-scaled position sizing, achieving +11.2% OOS return and 53.6% directional accuracy across 491 trades."
date: 2026-03-07
collection: portfolio
tags:
  - Machine Learning
  - Quantitative Finance
  - Python
  - Time Series
  - Crude Oil
header:
  teaser: /assets/images/oil-backtest-equity.png
---

## Overview

An end-to-end ML-driven crude oil (WTI CL) forecasting and algorithmic trading system, built from 59 engineered features spanning price momentum, implied volatility (OVX), futures term structure, and the Caldara-Iacoviello Geopolitical Risk Index. The system was validated using walk-forward expanding-window backtesting on actual CME CL futures contracts with realistic transaction costs.

**[Live Backtest Dashboard →](/projects/wti-backtest/backtest_dashboard.html)**

## Key Results (Out-of-Sample)

| Metric | Value |
|--------|-------|
| Total Return | **+11.2%** ($100K → $111.2K) |
| Sharpe Ratio | **0.353** |
| Directional Accuracy | **53.6%** |
| Max Drawdown | **-18.7%** |
| Win Rate | **53.4%** (491 trades) |
| Profit Factor | **1.045** |

## Methodology

**Data:** Bloomberg DSS daily data (Jan 2021 – Feb 2026) for WTI front-month (CL1), second-month (CL2), Brent (CO1), and Oil Implied Volatility (OVX), augmented with the Caldara-Iacoviello monthly GPR index (9 country-level series).

**Feature Engineering (59 features across 6 groups):**
- Price returns & multi-scale lags (1–20 day)
- Rolling volatility, RSI-14, Bollinger %B
- OVX-derived features (lags, z-scores, rolling ratios) — consensus top predictor
- Futures term structure carry signals (CL1–CL2 basis, WTI-Brent spread)
- Geopolitical risk dynamics (GPR z-score, country-level series)
- Calendar seasonality

**Model Architecture:** Weighted ensemble of Gradient Boosted Regressor (40%) and Random Forest (60%), predicting next-day log returns. Walk-forward validation with 3-year initial training window and quarterly retraining.

**Backtesting:** Simulated on CME CL futures ($1,000/point multiplier) with $20 round-trip costs, volatility-targeted position sizing, and half-Kelly capital allocation.

## Key Findings

1. **Momentum dominates:** 10-day and 20-day return features ranked #1 and #2 by GBR importance, consistent with CFA Foundation (2025) findings on commodity futures.
2. **OVX is the top risk signal:** Oil implied volatility features appear 4 times in the top 20, confirming SHAP consensus across recent literature.
3. **GPR captures tail events:** GPR z-score ranked #10 overall — the geopolitical risk index adds value during OPEC decisions and conflict escalation.
4. **High-vol regime outperforms 4.6×:** Average P&L per trade $73 in high-vol vs $16 in low-vol regime, consistent with Omer et al. (2025) findings on tree-based models and mean-reversion.

## Architecture

```
Bloomberg DSS (CL1, CL2, Brent, OVX)
         │
         ▼
  Feature Engineering (59 features)
  ├── Price/Momentum lags
  ├── Volatility & technicals
  ├── OVX implied vol features
  ├── Term structure / carry
  ├── GPR geopolitical risk
  └── Calendar
         │
         ▼
  GBR/RF Weighted Ensemble
  (walk-forward, quarterly retrain)
         │
         ▼
  Position Sizing
  ├── Inverse-vol targeting (15% ann.)
  └── Half-Kelly criterion
         │
         ▼
  CL Futures Execution
  ($20/RT costs, 1-day holding)
```

## Tech Stack

- **Python**: scikit-learn, XGBoost, pandas, NumPy, matplotlib, SHAP
- **Data**: Bloomberg DSS, Caldara-Iacoviello GPR
- **Backtesting**: Custom walk-forward engine with CME CL contract specs
- **Dashboard**: Chart.js, vanilla HTML/CSS/JS (GitHub Pages)

## Repository

[GitHub →](https://github.com/YOUR_USERNAME/wti-oil-forecasting)

The repository contains the full pipeline (`oil_forecasting_pipeline.py`), backtest engine (`backtest_wti_futures.py`), configuration (`config.yaml`), and interactive dashboard.

---

*Research methodology informed by 40+ academic papers from 2024–2026. All results are out-of-sample via walk-forward validation with realistic transaction costs.*
