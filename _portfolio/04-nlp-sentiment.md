---
title: "NLP Sentiment Pipeline"
excerpt: "Multi-model sentiment classification on movie reviews with bag-of-words features, unigram vs bigram comparison, and diagnostic evaluation."
collection: portfolio
category_label: "ML / NLP"
status: "In Progress"
accent_color: "#FFB347"
tagline: "Bag-of-Words Sentiment Classification with Model Comparison"
methods:
  - CountVectorizer (unigram & bigram)
  - Multinomial Naive Bayes
  - Logistic Regression
  - Random Forest
  - Confusion matrix & ROC/AUC analysis
tools:
  - Python
  - scikit-learn
  - NLTK
  - pandas
  - matplotlib
  - wordcloud
tags:
  - NLP
  - classification
  - sentiment analysis
  - scikit-learn
---

## Overview

End-to-end sentiment classification pipeline on a corpus of 5,000 movie reviews with binary sentiment labels and 4-point ratings. The project follows a structured comparison: three classifiers (Naive Bayes, Logistic Regression, Random Forest) evaluated across two feature representations (unigrams vs bigrams) to test whether word-pair context improves classification over individual word frequencies.

---

## Data

- **Corpus**: 5,000 movie reviews with binary sentiment labels (positive/negative) and ordinal ratings (1-4)
- **Class balance**: Near-equal split (slight skew toward negative), sentiment-rating correlation of 0.91
- **Feature extraction**: `CountVectorizer` with English stopword removal, producing 38,178 unique unigram features

---

## Part 1: Exploratory Analysis

Word cloud visualisation of rating-stratified reviews confirmed expected polarity: negative reviews (rating 1) dominated by terms like "worst", "bad", "boring", while positive reviews (rating 4) featured "wonderful", "great", "love". Frequency analysis of the full unigram vocabulary revealed that 7 of the 10 most common words are sentiment-neutral, highlighting the challenge of bag-of-words classification when most high-frequency terms carry no discriminative signal.

---

## Part 2: Model Comparison

All models trained on an 80/20 train-test split using `CountVectorizer` bag-of-words features.

### Unigram results

| Model | Accuracy | F1 (macro) | AUC |
|-------|----------|------------|-----|
| Multinomial Naive Bayes | 83.5% | 0.83 | 0.896 |
| Logistic Regression | 85.0% | 0.85 | - |
| Random Forest (500 trees) | 85.0% | 0.85 | - |

### Bigram results

| Model | Accuracy | F1 (macro) | AUC |
|-------|----------|------------|-----|
| Multinomial Naive Bayes | 80.0% | 0.80 | 0.904 |
| Logistic Regression | 79.0% | 0.79 | - |
| Random Forest (500 trees) | 82.0% | 0.82 | - |

**Key finding**: Unigrams outperform bigrams across all three models. This is counterintuitive since bigrams capture negation and phrase-level context, but the increased dimensionality likely causes overfitting on a 5,000-review corpus. The AUC comparison for Naive Bayes tells a slightly different story (bigram AUC of 0.904 vs unigram 0.896), suggesting that bigrams improve ranking ability even when threshold-dependent accuracy drops.

### Diagnostic evaluation

Confusion matrix analysis revealed an asymmetry in Naive Bayes: higher false negative rate (95) than false positive rate (70), indicating the model is more likely to miss positive sentiment than to hallucinate it. Logistic Regression and Random Forest partially corrected this, achieving more balanced precision-recall trade-offs.

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Unigram bag-of-words
cv_unigram = CountVectorizer(
    lowercase=True,
    stop_words='english',
    ngram_range=(1, 1)
)
X_uni = cv_unigram.fit_transform(film_data['review'])
y = film_data['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X_uni, y, test_size=0.2, random_state=1
)

# Multinomial Naive Bayes
model_mnb = MultinomialNB(alpha=1.0).fit(X_train, y_train)
y_pred_mnb = model_mnb.predict(X_test)
print(metrics.classification_report(y_test, y_pred_mnb))
```

---

## Planned Extension

The current pipeline establishes a clean baseline but has several gaps that limit its value as a portfolio piece. The following extensions address each directly:

### TF-IDF vectorisation

The current `CountVectorizer` treats all word occurrences equally. TF-IDF reweights features by inverse document frequency, dampening high-frequency neutral terms (the exact problem identified in the exploratory analysis where 7/10 top words were non-discriminative). Sublinear TF scaling (`sublinear_tf=True`) further compresses the effect of very frequent terms.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=3,
    stop_words='english'
)
X_tfidf = tfidf.fit_transform(film_data['review'])
```

### SVM classifier

Linear SVM is a natural addition to the model comparison. It often outperforms Naive Bayes and Logistic Regression on high-dimensional text data because of its margin-maximising objective, and will test whether the accuracy ceiling is a model limitation or a feature limitation.

```python
from sklearn.svm import LinearSVC

svm = LinearSVC(C=0.5, class_weight='balanced')
svm.fit(X_train_tfidf, y_train)
```

### Stratified k-fold cross-validation

The current single 80/20 split gives one point estimate of performance. Stratified 5-fold CV will produce mean and standard deviation estimates, giving a much clearer picture of model stability and whether the observed accuracy differences between models are meaningful or within noise.

```python
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=50000, ngram_range=(1,2),
                               sublinear_tf=True, min_df=3)),
    ('clf', LinearSVC(C=0.5, class_weight='balanced'))
])

scores = cross_val_score(pipe, X, y, cv=5, scoring='f1_macro')
print(f"CV F1: {scores.mean():.3f} +/- {scores.std():.3f}")
```

### Systematic error analysis

The current confusion matrices identify aggregate misclassification patterns, but do not inspect individual misclassified reviews. A targeted error analysis will examine what types of reviews each model gets wrong (sarcasm, mixed sentiment, domain-specific language) and whether TF-IDF and SVM resolve failure modes that bag-of-words Naive Bayes cannot.

---

## Key Takeaways

**From current analysis:**
- Unigrams outperform bigrams on accuracy across all three models, likely due to dimensionality overfitting on a 5,000-review corpus
- Naive Bayes shows systematic false-negative bias (misses positive sentiment more than negative)
- Logistic Regression and Random Forest achieve comparable performance (85%), suggesting the accuracy ceiling may be feature-driven rather than model-driven

**Expected from extension:**
- TF-IDF should improve on the bag-of-words baseline by downweighting the non-discriminative high-frequency terms identified in exploratory analysis
- SVM will test whether the 85% ceiling is a model limitation
- Cross-validation will clarify whether the 2% gap between Naive Bayes and the other models is statistically reliable
- Error analysis will identify whether remaining misclassifications are solvable within bag-of-words representations or require richer features
