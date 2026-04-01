---
title: "Central Bank Communication & Systemic Risk Prediction"
excerpt: "Applying off-the-shelf FinBERT to FOMC minutes to construct multi-dimensional sentiment indices and test their predictive power for SRISK, ΔCoVaR, and the Cleveland Fed Systemic Risk Indicator."
collection: portfolio
category_label: "NLP / Systemic Risk"
status: "In Progress"
accent_color: "#FF6B6B"
tagline: "Transformer-Based Early Warning from Fed Communication"
methods:
  - FinBERT inference (sentence-level sentiment classification)
  - Entropy-based uncertainty measurement
  - Time-series regression with Newey-West HAC standard errors
  - Granger-causality testing
  - Intra-document sentiment convergence analysis
  - Expanding-window out-of-sample forecasting
tools:
  - Python
  - HuggingFace Transformers
  - PyTorch
  - statsmodels
  - pandas
  - scikit-learn
tags:
  - NLP
  - FinBERT
  - systemic risk
  - FOMC
  - central bank communication
  - transformer
  - early warning
  - CoVaR
  - SRISK
---

## Overview

This project applies **transformer-based NLP** to Federal Open Market Committee (FOMC) minutes — the Federal Reserve's primary post-meeting communication — to construct multi-dimensional sentiment indices and test whether shifts in central bank language predict movements in systemic risk measures. Rather than fine-tuning a custom model, the pipeline uses off-the-shelf FinBERT for sentence-level classification and derives a second uncertainty dimension from the entropy of FinBERT's output distribution, requiring no manual labelling. The central empirical question is whether changes in how the Fed *talks* about financial conditions precede movements in established systemic risk indicators.

## Research question

Does aggregate sentiment in FOMC minutes contain **early warning information** about systemic risk (SRISK, ΔCoVaR, Cleveland Fed Systemic Risk Indicator) beyond what standard market-based indicators capture — and does increasing uniformity of tone *within* FOMC minutes signal stress episodes before they materialise in market data?

---

## Data

- **Corpus**: FOMC meeting minutes, sourced directly from the [Federal Reserve Board website](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm). Minutes are published as clean HTML/text files with consistent formatting, covering 8 scheduled meetings per year plus any inter-meeting actions. No scraping, transcription, or third-party sourcing required (~120 documents).
- **Period**: 2010–2024, covering post-GFC recovery, taper tantrum (2013), oil price collapse (2015–16), COVID-19 crisis, and the 2022–23 tightening cycle
- **Systemic risk targets**:
  - SRISK (NYU V-Lab, weekly firm-level and aggregate estimates, publicly available)
  - ΔCoVaR (estimated via quantile regression on equity returns following Adrian & Brunnermeier 2016)
  - Cleveland Fed Systemic Risk Indicator (weekly, available through October 2025)
- **Controls**: VIX, investment-grade and high-yield credit spreads (ICE BofA indices), term spread (10Y–3M Treasury), TED spread, aggregate bank leverage (Federal Reserve H.8 data)

---

## Part 1: Dual-Dimension Sentiment Extraction

Standard applications of FinBERT classify text along a single positive/neutral/negative axis. For central bank communication about financial stability, this misses a critical dimension: the Fed can be broadly positive about economic conditions while expressing elevated uncertainty about the outlook. The pipeline extracts two complementary dimensions without any fine-tuning or manual labelling.

| Dimension | How it is measured | What it captures |
|-----------|-------------------|-----------------|
| **Tone** | FinBERT predicted class (positive/neutral/negative), scored as net sentiment | Overall directional assessment — is the Fed upbeat or cautious? |
| **Uncertainty** | Shannon entropy of FinBERT's softmax output distribution | Ambiguity in language — high entropy means FinBERT cannot confidently classify a sentence, indicating hedged, conditional, or mixed-signal language |

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

LABELS = ['positive', 'negative', 'neutral']
TONE_MAP = {'positive': 1.0, 'negative': -1.0, 'neutral': 0.0}

def score_sentence(text: str) -> dict:
    """
    Score a single sentence on two dimensions:
    - tone: net sentiment from FinBERT's predicted class
    - uncertainty: Shannon entropy of the softmax distribution

    High entropy indicates FinBERT cannot confidently classify the
    sentence, suggesting hedged or ambiguous language.
    """
    inputs = tokenizer(
        text, return_tensors="pt",
        truncation=True, max_length=512
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).squeeze().numpy()

    predicted_label = LABELS[np.argmax(probs)]
    tone = TONE_MAP[predicted_label]
    entropy = -np.sum(probs * np.log(probs + 1e-10))

    return {
        'tone': tone,
        'uncertainty': entropy,
        'confidence': float(np.max(probs)),
        'label': predicted_label
    }


def score_document(sentences: list[str]) -> dict:
    """
    Score an entire FOMC minutes document.
    Returns mean tone, mean uncertainty, and per-sentence scores.
    """
    scores = [score_sentence(s) for s in sentences]
    return {
        'mean_tone': np.mean([s['tone'] for s in scores]),
        'mean_uncertainty': np.mean([s['uncertainty'] for s in scores]),
        'n_sentences': len(scores),
        'sentence_scores': scores
    }
```

Each FOMC minutes document is split into sentences and scored, producing a meeting-level panel of tone and uncertainty. Meetings are mapped to the subsequent inter-meeting period for alignment with weekly systemic risk data.

---

## Part 2: Time-Series Construction and Alignment

FOMC minutes are released on an irregular schedule (~8 per year), while systemic risk measures are available weekly. The pipeline constructs aligned time series by assigning each FOMC sentiment observation to the inter-meeting window that follows it, then testing predictive power at the meeting-to-meeting frequency.

```python
import pandas as pd

def build_sentiment_series(
    fomc_scores: pd.DataFrame,
    srisk_weekly: pd.DataFrame,
    controls: pd.DataFrame
) -> pd.DataFrame:
    """
    Align FOMC sentiment with systemic risk targets and controls.

    Parameters
    ----------
    fomc_scores : DataFrame with columns [meeting_date, tone, uncertainty]
    srisk_weekly : DataFrame with columns [date, srisk_aggregate]
    controls : DataFrame with columns [date, vix, ig_spread, hy_spread,
               term_spread, ted_spread, leverage]

    Returns
    -------
    DataFrame at FOMC meeting frequency with sentiment, forward-looking
    systemic risk targets, and contemporaneous controls
    """
    fomc = fomc_scores.sort_values('meeting_date').copy()

    # Map each meeting to the next meeting date for target alignment
    fomc['next_meeting'] = fomc['meeting_date'].shift(-1)

    # For each meeting, compute mean SRISK in the inter-meeting window
    results = []
    for _, row in fomc.iterrows():
        if pd.isna(row['next_meeting']):
            continue

        t0 = row['meeting_date']
        t1 = row['next_meeting']

        # Forward-looking target: mean SRISK between this and next meeting
        mask = (srisk_weekly['date'] > t0) & (srisk_weekly['date'] <= t1)
        fwd_srisk = srisk_weekly.loc[mask, 'srisk_aggregate'].mean()

        # Contemporaneous controls: values at meeting date
        ctrl_mask = controls['date'] <= t0
        latest_controls = controls.loc[ctrl_mask].iloc[-1]

        results.append({
            'meeting_date': t0,
            'tone': row['tone'],
            'uncertainty': row['uncertainty'],
            'fwd_srisk': fwd_srisk,
            **latest_controls.to_dict()
        })

    return pd.DataFrame(results)
```

---

## Part 3: Predictive Testing

The core empirical test: does FOMC communication sentiment predict next-period systemic risk beyond standard market-based controls?

**Baseline specification**:

$$\text{SRISK}_{t+1} = \alpha + \beta_1 \, \text{Tone}_t + \beta_2 \, \text{Uncertainty}_t + \boldsymbol{\gamma}' \mathbf{X}_t + \varepsilon_{t+1}$$

where $\mathbf{X}_t$ includes VIX, investment-grade and high-yield credit spreads, term spread, TED spread, and aggregate bank leverage. Standard errors are Newey-West HAC to account for serial correlation.

**Specification variants**:
- Dependent variable alternatives: aggregate SRISK, ΔCoVaR (system-level), Cleveland Fed SRI
- Change specification: $\Delta\text{SRISK}_{t+1}$ as the target to test prediction of *changes* rather than levels
- Interaction: $\text{Tone}_t \times \text{Uncertainty}_t$ to test whether negative tone combined with high uncertainty is a stronger signal than either alone

**Granger-causality**: Bivariate and multivariate Granger tests at 1–4 meeting lags. Given the irregular spacing of FOMC meetings (~6–8 weeks apart), one lag corresponds to roughly one quarter.

**Out-of-sample evaluation**: Expanding-window forecasts beginning from 2015, comparing RMSE and directional accuracy of models with and without sentiment features. The forecast target is the level or change in the systemic risk measure in the inter-meeting window following each FOMC meeting.

```python
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
import numpy as np

def run_predictive_regression(
    df: pd.DataFrame,
    target: str = 'fwd_srisk',
    sentiment_vars: list[str] = ['tone', 'uncertainty'],
    controls: list[str] = ['vix', 'ig_spread', 'hy_spread',
                           'term_spread', 'ted_spread', 'leverage'],
    max_lag_nw: int = 4
) -> dict:
    """
    Run the baseline predictive regression with Newey-West HAC SEs.
    Returns results for the full model and a controls-only benchmark.
    """
    df_clean = df.dropna(subset=[target] + sentiment_vars + controls)

    # Full model: sentiment + controls
    X_full = sm.add_constant(df_clean[sentiment_vars + controls])
    y = df_clean[target]
    model_full = OLS(y, X_full).fit(
        cov_type='HAC', cov_kwds={'maxlags': max_lag_nw}
    )

    # Benchmark: controls only
    X_ctrl = sm.add_constant(df_clean[controls])
    model_ctrl = OLS(y, X_ctrl).fit(
        cov_type='HAC', cov_kwds={'maxlags': max_lag_nw}
    )

    return {
        'full_model': model_full,
        'controls_only': model_ctrl,
        'r2_gain': model_full.rsquared_adj - model_ctrl.rsquared_adj,
        'n_obs': len(df_clean)
    }


def expanding_window_oos(
    df: pd.DataFrame,
    target: str,
    sentiment_vars: list[str],
    controls: list[str],
    first_oos_date: str = '2015-01-01'
) -> dict:
    """
    Expanding-window out-of-sample forecast comparison.
    Compares RMSE and directional accuracy: full model vs controls-only.
    """
    df = df.sort_values('meeting_date').dropna()
    oos_mask = df['meeting_date'] >= first_oos_date

    forecasts_full, forecasts_ctrl, actuals = [], [], []

    for i in df[oos_mask].index:
        train = df.loc[:i-1]
        if len(train) < 20:
            continue

        y_train = train[target]
        X_full_train = sm.add_constant(train[sentiment_vars + controls])
        X_ctrl_train = sm.add_constant(train[controls])

        test_row = df.loc[[i]]
        X_full_test = sm.add_constant(test_row[sentiment_vars + controls])
        X_ctrl_test = sm.add_constant(test_row[controls])

        try:
            pred_full = OLS(y_train, X_full_train).fit().predict(X_full_test)[0]
            pred_ctrl = OLS(y_train, X_ctrl_train).fit().predict(X_ctrl_test)[0]

            forecasts_full.append(pred_full)
            forecasts_ctrl.append(pred_ctrl)
            actuals.append(test_row[target].values[0])
        except Exception:
            continue

    actuals = np.array(actuals)
    forecasts_full = np.array(forecasts_full)
    forecasts_ctrl = np.array(forecasts_ctrl)

    rmse_full = np.sqrt(np.mean((actuals - forecasts_full) ** 2))
    rmse_ctrl = np.sqrt(np.mean((actuals - forecasts_ctrl) ** 2))

    # Directional accuracy
    if len(actuals) > 1:
        actual_dir = np.diff(actuals) > 0
        full_dir = np.diff(forecasts_full) > 0
        ctrl_dir = np.diff(forecasts_ctrl) > 0
        da_full = np.mean(actual_dir == full_dir)
        da_ctrl = np.mean(actual_dir == ctrl_dir)
    else:
        da_full = da_ctrl = np.nan

    return {
        'rmse_full': rmse_full,
        'rmse_controls': rmse_ctrl,
        'rmse_reduction_pct': (rmse_ctrl - rmse_full) / rmse_ctrl * 100,
        'directional_accuracy_full': da_full,
        'directional_accuracy_controls': da_ctrl,
        'n_forecasts': len(actuals)
    }
```

---

## Part 4: Intra-Document Sentiment Convergence as Early Warning

The novel contribution. Beyond the *level* of aggregate sentiment in FOMC minutes, the degree of *uniformity within a single document* contains information. The pipeline measures how consistent the tone is across all sentences (or sections) within a given set of minutes:

$$\text{Dispersion}_t^{(d)} = \sqrt{\frac{1}{N_t - 1} \sum_{s=1}^{N_t} \left(S_{s,t}^{(d)} - \bar{S}_t^{(d)}\right)^2}$$

where $S_{s,t}^{(d)}$ is the score of sentence $s$ in dimension $d$ at meeting $t$, and $N_t$ is the number of sentences.

**Hypothesis**: When FOMC minutes exhibit unusually uniform negative tone — low dispersion combined with negative mean tone — it signals that the committee is collectively focused on a single dominant risk. This unanimity of concern is precisely the condition under which policy responses may be delayed (consensus takes time to form) and under which the identified risk is most likely to be genuinely systemic rather than idiosyncratic. Conversely, high within-document dispersion suggests the committee is discussing a range of issues without convergence on a dominant threat.

A secondary convergence measure operates *across meetings*: declining variation in tone between consecutive FOMC minutes suggests a sustained shift in the committee's assessment, which may predict persistent rather than transitory changes in systemic risk.

```python
import numpy as np
import pandas as pd

def compute_intra_document_dispersion(
    sentence_scores: list[dict]
) -> dict:
    """
    Compute within-document dispersion for a single FOMC minutes.
    Low dispersion + negative tone = uniform concern (potential early warning).

    Parameters
    ----------
    sentence_scores : list of dicts with 'tone' and 'uncertainty' keys

    Returns
    -------
    dict with dispersion and convergence metrics
    """
    tones = np.array([s['tone'] for s in sentence_scores])
    uncertainties = np.array([s['uncertainty'] for s in sentence_scores])

    return {
        'tone_dispersion': np.std(tones, ddof=1),
        'uncertainty_dispersion': np.std(uncertainties, ddof=1),
        'tone_mean': np.mean(tones),
        'uncertainty_mean': np.mean(uncertainties),
        'pct_negative': np.mean(tones < 0),
        'n_sentences': len(sentence_scores)
    }


def compute_inter_meeting_convergence(
    meeting_series: pd.DataFrame,
    window: int = 3
) -> pd.DataFrame:
    """
    Rolling standard deviation of tone and uncertainty across
    consecutive FOMC meetings. Declining values indicate the committee
    is converging on a sustained assessment.

    Parameters
    ----------
    meeting_series : DataFrame with columns [meeting_date, tone, uncertainty]
    window : number of meetings for rolling window

    Returns
    -------
    DataFrame with inter-meeting convergence measures
    """
    df = meeting_series.sort_values('meeting_date').copy()
    df['tone_rolling_std'] = df['tone'].rolling(window).std()
    df['uncertainty_rolling_std'] = df['uncertainty'].rolling(window).std()
    df['tone_rolling_mean'] = df['tone'].rolling(window).mean()
    return df
```

The predictive test adds dispersion and convergence measures to the baseline regression, testing whether within-document uniformity and cross-meeting convergence predict systemic risk beyond the level of sentiment itself.

---

## Expected Contributions

1. **Zero-labelling sentiment pipeline**: Demonstrates that off-the-shelf FinBERT combined with entropy-based uncertainty measurement extracts meaningful multi-dimensional signals from central bank communication without any fine-tuning or manual annotation
2. **Fed communication as systemic risk predictor**: Tests whether FOMC minutes contain forward-looking information about systemic risk measures (SRISK, ΔCoVaR, Cleveland Fed SRI) beyond what market-based indicators provide
3. **Intra-document convergence as early warning**: Proposes and tests a novel signal; uniformity of tone within FOMC minutes as an indicator that the committee has collectively identified a dominant systemic threat, distinct from the average level of sentiment
4. **Out-of-sample validation**: Avoids the in-sample overfitting that afflicts many NLP-finance papers by using expanding-window forecasts from 2015 onward

---

## Considerations

- **FinBERT domain mismatch**: FinBERT is trained on financial news and analyst reports, not central bank communication. Fed language is more formal, uses specific institutional terminology, and tends toward measured understatement. Sentences like "participants noted elevated uncertainty" may register differently than equivalent language in an analyst note. A robustness check using the Loughran-McDonald financial dictionary as an alternative sentiment measure can help assess whether results are FinBERT-specific.
- **Irregular meeting frequency**: FOMC meets ~8 times per year on an irregular schedule, and the time between meetings varies from 3 to 8 weeks. This complicates time-series econometrics; standard lag structures assume regular intervals. The analysis uses meeting-to-meeting frequency rather than calendar time, but this means each "period" covers a different duration, affecting comparability of systemic risk targets across observations.
- **Short time series**: ~120 FOMC meetings over 2010–2024 (approximately 112 after lags). The expanding-window out-of-sample test beginning from 2015 yields roughly 70–75 forecast observations. This is adequate for basic predictive comparisons but limits power for detecting small improvements, particularly during calm periods when both sentiment and systemic risk variation is low.
- **Minutes are backward-looking by construction**: FOMC minutes describe the discussion at a meeting that occurred three weeks prior to publication. Markets have already reacted to the post-meeting statement, press conference, and dot plot. The predictive content of the minutes, if any, must come from nuances of the discussion not captured in the statement, or from the specific language used to characterise risks.
- **Strategic communication**: The Fed is aware that its communications are parsed by market participants and NLP algorithms. Minutes are carefully drafted by Fed staff and reviewed by committee members. Tone may reflect communication strategy as much as genuine risk assessment, and this strategic layer could either strengthen the signal (the Fed deliberately signals concern when warranted) or weaken it (language is smoothed to avoid market disruption).
- **ΔCoVaR estimation choices**: ΔCoVaR is not available as a pre-computed download and must be estimated from equity return data via quantile regression. Methodological choices (quantile level, conditioning variables, estimation window) materially affect the resulting series. These choices are documented and sensitivity-tested.
- **Cleveland Fed SRI discontinued**: The Cleveland Fed Systemic Risk Indicator was released weekly through October 2025 and is no longer updated. Historical data covers the sample period but the series cannot be extended.
- **Single-source text corpus**: The project draws on a single institutional source (the Fed), so the convergence analysis operates *within* documents (uniformity of tone across sentences in a given set of minutes) and *across* consecutive meetings (persistence of tone shifts over time). This captures committee-level consensus formation, a meaningful signal in its own right, but does not identify convergence or herd behaviour across independent institutional actors.
