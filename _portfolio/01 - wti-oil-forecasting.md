---
title: "ML-Driven WTI Crude Oil Forecasting"
excerpt: "Walk-forward validated GBR/RF ensemble for WTI futures trading with volatility-scaled position sizing, achieving +13.8% OOS return and 0.415 Sharpe ratio across 491 trades."
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

An end-to-end ML-driven crude oil (WTI CL) forecasting and algorithmic trading system, built from 66 engineered features spanning price momentum, implied volatility (OVX), futures term structure, and the Caldara-Iacoviello Geopolitical Risk Index. The system was validated using walk-forward expanding-window backtesting on CME CL futures contracts with realistic transaction costs.

## Key Results (Out-of-Sample)

| Metric | Value |
|--------|-------|
| Total Return | **+13.8%** |
| Sharpe Ratio | **0.415** |
| Max Drawdown | **-14.2%** |
| Win Rate | **52.8%** (491 trades) |
| Profit Factor | **1.056** |
| Directional Accuracy | **52.8%** |
| Total Transaction Costs | **$5,620** |

## Methodology

**Data:** Bloomberg DSS daily data (Jan 2021 – Feb 2026), 1,248 observations across 79 columns covering WTI front-month (CL1), second-month (CL2), Brent (CO1), and Oil Implied Volatility (OVX), augmented with the Caldara-Iacoviello monthly GPR index (9 country-level series).

**Feature Engineering (66 features across 6 groups):**
- Price returns & multi-scale lags (1–20 day)
- Rolling volatility, RSI-14, Bollinger %B
- OVX-derived features (lags, z-scores, rolling ratios)
- Futures term structure carry signals (CL1–CL2 basis, WTI-Brent spread)
- Geopolitical risk dynamics (GPR headline, threats/acts sub-indices, 6 country-level series, z-score, momentum, spike detection)
- Calendar seasonality

**Model Architecture:** Weighted ensemble of Gradient Boosted Regressor (60%) and Random Forest (40%), predicting next-day WTI price. Walk-forward validation with 8 expanding-window folds, each training on a growing window and testing on 63-day out-of-sample blocks, producing 492 OOS predictions.

**Backtesting:** Simulated on CME CL futures ($1,000/point multiplier) with $20 round-trip costs and volatility-targeted position sizing. Three position sizing methods were compared (vol-target only, half-Kelly only, and combined vol+Kelly); all produced identical performance, indicating the signal — not the sizing method — drives returns at this scale.

## Key Findings

1. **Momentum dominates:** 10-day and 20-day return features ranked #1 and #2 by GBR importance, consistent with CFA Foundation (2025) findings on commodity futures.
2. **OVX is the top risk signal:** Oil implied volatility features appear 4 times in the top 20, confirming SHAP consensus across recent literature.
3. **GPR captures tail events:** GPR z-score ranked in the top 15 — the geopolitical risk index adds value during OPEC decisions and conflict escalation where price-only models fail.
4. **Regime analysis:** 61 high-volatility trades vs 430 low-volatility trades, with the low-vol regime contributing the bulk of cumulative P&L ($12,970 vs $820), suggesting the model's edge comes from consistent small gains in calm markets rather than crisis alpha.

## Architecture

```
Bloomberg DSS (CL1, CL2, Brent, OVX)  +  Caldara-Iacoviello GPR
         │                                        │
         ▼                                        ▼
  Feature Engineering (66 features)
  ├── Price/Momentum lags
  ├── Volatility & technicals
  ├── OVX implied vol features
  ├── Term structure / carry
  ├── GPR geopolitical risk (headline + 6 country series)
  └── Calendar
         │
         ▼
  GBR (60%) / RF (40%) Weighted Ensemble
  (walk-forward, 8 expanding-window folds)
         │
         ▼
  Backtest Engine
  ├── Vol-targeted position sizing
  ├── $20/RT transaction costs
  └── CME CL contract specs
```

## GPR Integration

The Caldara-Iacoviello Geopolitical Risk Index serves as a proxy for geopolitical supply-disruption risk. The pipeline downloads the monthly GPR export, extracts 9 curated columns (global headline GPR, threats sub-index, acts sub-index, plus country-level series for the US, Saudi Arabia, Russia, Israel, Ukraine, and China), forward-fills to daily frequency, and engineers features including level, 5-day moving average, first-difference shock, 10-day rolling volatility, spike detection (ratio to 20-day MA), and momentum. A Diebold-Mariano test compares forecasting accuracy of the baseline model (without GPR) against the GPR-augmented specification.

```python
# Curated GPR columns: global headline + oil-relevant countries
GPR_COLS = [
    "GPR",        # Global headline index
    "GPRT",       # Threats sub-index
    "GPRA",       # Acts sub-index
    "GPRC_USA",   # US policy/sanctions risk
    "GPRC_SAU",   # Saudi Arabia — OPEC supply
    "GPRC_RUS",   # Russia — producer + sanctions
    "GPRC_ISR",   # Israel — Middle East proxy
    "GPRC_UKR",   # Ukraine — Russia/energy nexus
    "GPRC_CHN",   # China — demand-side risk
]
```

## Planned Extension: NLP Sentiment Signals

The GPR index captures geopolitical risk through newspaper article counts but does not measure market *sentiment* about those events. A natural next step is to replace this coarse proxy with fine-grained sentiment extracted from oil-relevant news headlines using FinBERT or CrudeBERT.

The idea is to score Bloomberg or Reuters headlines daily, flag geopolitically relevant stories (sanctions, supply disruptions, OPEC decisions), and aggregate into daily sentiment features — mean tone, within-day disagreement, headline volume as an attention proxy, and a composite geopolitical risk-sentiment interaction. These would enter the existing GBR/RF ensemble as additional features alongside the current quantitative and GPR inputs, with Diebold-Mariano tests to measure whether text-derived signals improve forecasting accuracy beyond what the GPR index already captures.

## Tech Stack

- **Python**: scikit-learn, pandas, NumPy, matplotlib, SHAP, HuggingFace Transformers (for NLP extension)
- **Data**: Bloomberg DSS, Caldara-Iacoviello GPR
- **Backtesting**: Custom walk-forward engine with CME CL contract specs
- **NLP**: FinBERT (ProsusAI/finbert) for headline sentiment scoring

---

*All results are out-of-sample via 8-fold walk-forward validation with realistic transaction costs. NLP extension pipeline is built; results pending Bloomberg headline data processing.*
