---
title: "Corporate Communication & Systemic Risk Prediction"
excerpt: "Fine-tuning FinBERT on earnings calls from the 10 largest US financial institutions to extract multi-dimensional management tone and test its predictive power for SRISK and ΔCoVaR."
collection: portfolio
category_label: "NLP / Systemic Risk"
status: "In Progress"
accent_color: "#FF6B6B"
tagline: "Transformer-Based Early Warning from Earnings Call Language"
methods:
  - FinBERT fine-tuning (multi-label sentiment)
  - SRISK-weighted sentiment aggregation
  - Panel fixed-effects regression
  - Granger-causality testing
  - Cross-sectional sentiment convergence analysis
  - Out-of-sample forecasting evaluation
tools:
  - Python
  - HuggingFace Transformers
  - PyTorch
  - statsmodels
  - linearmodels (panel)
  - pandas
  - SEC EDGAR API
tags:
  - NLP
  - FinBERT
  - systemic risk
  - earnings calls
  - transformer
  - early warning
  - CoVaR
  - SRISK
---

## Overview

This project applies **transformer-based NLP** to quarterly earnings call transcripts from the 10 largest US financial institutions, extracting multi-dimensional management tone as a predictor of systemic risk. Rather than treating sentiment as a single positive/negative score, the pipeline classifies earnings call language across four dimensions — overall tone, uncertainty, risk-specific language, and forward-looking negativity — and aggregates these into an SRISK-weighted stress sentiment measure. The central empirical question is whether shifts in how bank management *talks* about risk precede movements in established systemic risk measures.

## Research question

Does aggregate corporate communication tone in the financial sector contain **early warning information** about systemic risk (SRISK, ΔCoVaR) beyond what standard market-based indicators capture — and does alignment in management tone across major banks signal stress episodes before they materialise in market data?

---

## Data

- **Corpus**: Quarterly earnings call transcripts from the 10 largest US financial institutions (JPMorgan Chase, Bank of America, Citigroup, Wells Fargo, Goldman Sachs, Morgan Stanley, BNY Mellon, State Street, US Bancorp, MetLife), sourced via SEC EDGAR 8-K filings (~560 transcripts)
- **Period**: 2010–2024, covering post-GFC recovery, taper tantrum (2013), oil crash (2015–16), COVID, and the 2022–23 tightening cycle
- **Systemic risk targets**: SRISK (NYU V-Lab), ΔCoVaR (Adrian & Brunnermeier), Cleveland Fed Systemic Risk Indicators
- **Controls**: VIX, investment-grade and high-yield credit spreads, term spread (10Y–3M), TED spread, aggregate bank leverage

---

## Part 1: Multi-Dimensional Sentiment Extraction

Standard financial sentiment models classify text as positive, negative, or neutral. For earnings calls from financial institutions, this is too coarse — a CEO can be positive about earnings while expressing deep uncertainty about the macro outlook. The pipeline addresses this by fine-tuning FinBERT on a manually labelled subset of earnings call sentences, classified across four dimensions:

| Dimension | What it captures | Example signal |
|-----------|-----------------|----------------|
| **Overall tone** | General positivity/negativity of management language | "Results exceeded expectations" vs "challenging quarter" |
| **Uncertainty** | Hedging, conditional language, lack of visibility | "Difficult to predict", "depends on", "uncertain environment" |
| **Risk-specific** | Explicit references to credit, liquidity, counterparty, or market risk | "Tightening lending standards", "monitoring exposures" |
| **Forward-looking negativity** | Negative outlook statements about future conditions | "Expect headwinds", "anticipate deterioration" |

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load base FinBERT and fine-tune for multi-label classification
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Custom classification head: 4 dimensions, each with 3 classes
# (low / medium / high intensity)
class MultiDimSentiment(torch.nn.Module):
    def __init__(self, base_model, n_dimensions=4, n_classes=3):
        super().__init__()
        self.base = base_model
        hidden_size = base_model.config.hidden_size
        self.heads = torch.nn.ModuleList([
            torch.nn.Linear(hidden_size, n_classes)
            for _ in range(n_dimensions)
        ])

    def forward(self, input_ids, attention_mask):
        outputs = self.base(input_ids=input_ids,
                           attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        return [head(pooled) for head in self.heads]
```

Each sentence in every earnings call transcript receives a 4-dimensional sentiment vector. Firm-quarter scores are computed as the mean of all sentence-level scores within each call, producing a panel of firm × quarter × 4 sentiment dimensions.

---

## Part 2: SRISK-Weighted Aggregation

A simple average of firm-level sentiment would weight JPMorgan and a regional bank equally. Since the research question is about *systemic* risk prediction, aggregation should reflect systemic importance. The pipeline weights each firm's sentiment by its share of total system SRISK:

$$\text{AggSentiment}_t^{(d)} = \sum_{i=1}^{N} w_{i,t} \cdot S_{i,t}^{(d)}$$

where $S_{i,t}^{(d)}$ is firm $i$'s sentiment score in dimension $d$ at quarter $t$, and $w_{i,t}$ is firm $i$'s share of total system SRISK:

$$w_{i,t} = \frac{\text{SRISK}_{i,t}}{\sum_{j=1}^{N} \text{SRISK}_{j,t}}$$

This produces four quarterly time series — one per sentiment dimension — tilted toward the institutions that matter most for system-wide stability.

```python
import pandas as pd

def compute_weighted_sentiment(sentiment_df, srisk_df):
    """
    sentiment_df: firm x quarter x 4 dimensions
    srisk_df: firm x quarter SRISK values
    """
    # Compute SRISK-based weights per quarter
    srisk_totals = srisk_df.groupby('quarter')['srisk'].transform('sum')
    srisk_df['weight'] = srisk_df['srisk'] / srisk_totals

    # Merge and compute weighted sentiment
    merged = sentiment_df.merge(srisk_df[['firm', 'quarter', 'weight']],
                                 on=['firm', 'quarter'])

    dimensions = ['tone', 'uncertainty', 'risk_language', 'fwd_negativity']
    agg_sentiment = {}
    for dim in dimensions:
        agg_sentiment[dim] = (merged.groupby('quarter')
                     .apply(lambda g: (g[dim] * g['weight']).sum()))

    return pd.DataFrame(agg_sentiment)
```

---

## Part 3: Predictive Testing

The core empirical test: does the SRISK-weighted sentiment measure predict next-quarter systemic risk beyond standard controls?

**Panel specification** (firm-level):

$$\text{SRISK}_{i,t+1} = \alpha_i + \beta_1 S_{i,t}^{\text{tone}} + \beta_2 S_{i,t}^{\text{uncertainty}} + \beta_3 S_{i,t}^{\text{risk}} + \beta_4 S_{i,t}^{\text{fwd\_neg}} + \gamma' \mathbf{X}_t + \varepsilon_{i,t+1}$$

where $\alpha_i$ are firm fixed effects and $\mathbf{X}_t$ includes VIX, credit spreads, term spread, TED spread, and aggregate leverage.

**Aggregate specification** (system-level):

$$\Delta\text{SRISK}_{\text{agg},t+1} = \alpha + \boldsymbol{\beta}' \text{Sentiment}_t + \boldsymbol{\gamma}' \mathbf{X}_t + \varepsilon_{t+1}$$

**Out-of-sample evaluation**: Expanding-window forecasts beginning from 2015, comparing RMSE and directional accuracy of models with and without sentiment features.

**Granger-causality**: Bivariate and multivariate Granger tests at 1–4 quarter lags, with Newey-West HAC standard errors.

---

## Part 4: Sentiment Convergence as Early Warning Signal

The novel contribution. Beyond the *level* of aggregate sentiment, the degree of *alignment* across firms contains information. The pipeline measures how similar the 10 banks' tone scores are in each quarter:

$$\text{Dispersion}_t^{(d)} = \sqrt{\frac{1}{N-1} \sum_{i=1}^{N} w_{i,t} \left(S_{i,t}^{(d)} - \overline{S}_t^{(d)}\right)^2}$$

**Hypothesis**: When all 10 bank CEOs start using the same cautious, risk-aware language simultaneously — low dispersion, high convergence — it signals that they are all seeing and reacting to the same underlying threat. This collective positioning is exactly what makes crises systemic rather than idiosyncratic. Conversely, when management tone is diverse across firms (high dispersion), risks are more likely firm-specific and less likely to cascade.

The pipeline tests whether this convergence in each sentiment dimension predicts aggregate SRISK, controlling for the level of aggregate sentiment itself.

```python
import numpy as np

def sentiment_convergence(sentiment_df, srisk_df):
    """Weighted cross-sectional std of sentiment per quarter.
    Low values = high convergence (all banks sound the same)."""
    merged = sentiment_df.merge(
        srisk_df[['firm', 'quarter', 'weight']],
        on=['firm', 'quarter']
    )

    dimensions = ['tone', 'uncertainty', 'risk_language', 'fwd_negativity']
    dispersion = {}

    for dim in dimensions:
        def weighted_std(g):
            avg = (g[dim] * g['weight']).sum()
            var = (g['weight'] * (g[dim] - avg) ** 2).sum()
            return np.sqrt(var)

        dispersion[f'{dim}_dispersion'] = (
            merged.groupby('quarter').apply(weighted_std)
        )

    return pd.DataFrame(dispersion)
```

---

## Expected Contributions

1. **Multi-dimensional financial sentiment**: Moves beyond positive/negative polarity to capture distinct channels (uncertainty, risk awareness, forward-looking negativity) relevant to systemic risk assessment
2. **Systemic importance weighting**: Aggregation reflects institutional significance rather than treating all firms equally
3. **Sentiment convergence as early warning**: Tests whether alignment in corporate risk language across major banks is itself a predictor of systemic stress — a novel signal based on the idea that collective positioning amplifies fragility
4. **Out-of-sample validation**: Avoids the in-sample overfitting that plagues many NLP-finance papers by using expanding-window forecasts

---

## Explore

### Cards to play
- **Regime-conditional analysis**: Test whether the sentiment measure's predictive power is stronger during tightening cycles (when information frictions are more binding) vs easing cycles
- **Comparison with news-based indices**: Benchmark against existing text-based risk measures (e.g., Baker-Bloom-Davis EPU, Fed financial stability vocabulary indices) to quantify incremental information from earnings calls
- **Attention mechanism interpretability**: Extract FinBERT attention weights to identify which phrases and topics drive the sentiment classifications — providing qualitative insight into what management language signals risk

### Opportunities
- Natural complement to the financial contagion network project (market-based contagion) and macro-financial stress testing project (macro-level forecasting) — this adds an **information channel** to the portfolio's risk toolkit
- The fine-tuning pipeline and multi-dimensional classification framework are transferable to other financial text sources (central bank minutes, analyst reports, regulatory filings)
- Sentiment convergence methodology could be applied to any panel of text-derived sentiment — e.g., sovereign risk language across countries, or sectoral earnings tone across industries
