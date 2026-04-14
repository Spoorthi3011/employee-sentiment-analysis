# Employee Sentiment Analysis

A complete NLP pipeline for analysing employee email sentiment, identifying flight risks, and building a predictive model using Python.

---

## 📋 Project Summary

| Item | Detail |
|------|--------|
| **Dataset** | 2,191 Enron employee emails · 10 employees · Jan 2010 – Dec 2011 |
| **Language** | Python 3.9+ |
| **Core libraries** | pandas · scikit-learn · matplotlib · seaborn |
| **Sentiment method** | Lexicon-based (TextBlob-equivalent) with negation handling |

---

## 🏆 Key Findings

### Top 3 Positive Employees (Overall)
| Rank | Employee | Sentiment Score |
|------|----------|----------------|
| 1 | Lydia Delgado | +113 |
| 2 | Johnny Palmer | +93 |
| 3 | John Arnold | +91 |

### Top 3 Negative Employees (Overall)
| Rank | Employee | Sentiment Score |
|------|----------|----------------|
| 1 | Rhonda Denton | +57 |
| 2 | Kayne Coulter | +62 |
| 3 | Don Baughman | +78 |

> Note: All employees have positive overall scores, but these three have the lowest relative scores.

### 🚨 Flight Risk Employees
Employees who sent **4+ negative messages within a rolling 30-day window**:

| Employee | First Trigger Date | Negatives in Window |
|----------|-------------------|---------------------|
| Don Baughman | 2010-04-26 | 6 |
| Eric Bass | 2011-05-07 | 4 |
| Rhonda Denton | 2010-12-18 | 4 |
| Sally Beck | 2011-08-15 | 4 |

### 📊 Model Performance (Linear Regression)
- **R² Score:** 0.51 — message volume/length features explain ~51% of sentiment score variance
- **RMSE:** ~3.4 | **MAE:** ~2.6
- Adding semantic features (TF-IDF, embeddings) would substantially improve R²

---

## 📁 Project Structure

```
employee_sentiment/
│
├── Employee_Sentiment_Analysis.ipynb   # Main analysis notebook (all 6 tasks)
├── sentiment_analysis.py               # Standalone Python script (same logic)
├── README.md                           # This file
├── .env.example                        # Environment variable template
│
├── data/
│   ├── test_in_.csv                    # Raw input dataset
│   ├── test_labeled.csv                # Dataset with sentiment labels added
│   └── monthly_scores.csv             # Monthly score aggregates per employee
│
└── visualizations/
    ├── 01_sentiment_distribution.png
    ├── 02_messages_per_employee.png
    ├── 03_sentiment_by_employee.png
    ├── 04_monthly_sentiment_trend.png
    ├── 05_polarity_distribution.png
    ├── 06_wordcount_vs_polarity.png
    ├── 07_monthly_score_heatmap.png
    ├── 08_overall_ranking.png
    ├── 09_flight_risk.png
    ├── 10_model_performance.png
    └── 11_residuals.png
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.9 or higher
- pip

### 1. Clone or unzip the project
```bash
# If using git
git clone <your-repo-url>
cd employee_sentiment

# Or unzip
unzip YourName_AI-project-submission.zip
cd employee_sentiment
```

### 2. Create a virtual environment (recommended)
```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Configure environment variables
```bash
cp .env.example .env
# Edit .env if you want to swap in an API key for LLM-based sentiment
```

---

## 🚀 Usage

### Option A — Run the Jupyter Notebook (recommended)
```bash
jupyter notebook Employee_Sentiment_Analysis.ipynb
```
Run all cells in order. Each task is clearly titled and annotated with observations.

### Option B — Run the Python script directly
```bash
python3 sentiment_analysis.py
```
This generates all visualisations in `visualizations/` and saves labeled data to `data/`.

---

## 🔬 Methodology

### Task 1 — Sentiment Labeling
A **lexicon-based analyser** (equivalent to TextBlob's PatternAnalyzer) was implemented:
- Curated vocabulary of ~100 positive and ~80 negative workplace-relevant words
- **Negation handling**: a negation within a 3-word window flips the following word's polarity
- Polarity formula: `score = Σ(word_polarity) / word_count × 10`, clipped to `[-1, 1]`
- Thresholds: Positive > 0.05 · Negative < −0.05 · Neutral otherwise

### Task 2 — EDA
11 charts covering sentiment distribution, per-employee breakdowns, time trends, polarity histograms, and word count analysis.

### Task 3 — Monthly Scoring
Messages are scored (+1 / 0 / −1), grouped by employee × calendar month, and summed. Scores reset each month.

### Task 4 — Ranking
Employees are ranked by cumulative score (descending), with alphabetical tie-breaking as specified.

### Task 5 — Flight Risk
Rolling 30-day window (not calendar month). An employee is flagged if ≥4 negative messages fall within any 30-day span.

### Task 6 — Linear Regression
Features: `message_count`, `avg_word_count`, `avg_msg_length`, `total_word_count` — all independent of the target. 80/20 train-test split. Metrics: R², RMSE, MAE.

---

## 📦 Dependencies

See `requirements.txt`. Core packages:
- `pandas` ≥ 2.0
- `numpy` ≥ 1.24
- `scikit-learn` ≥ 1.3
- `matplotlib` ≥ 3.7
- `seaborn` ≥ 0.12
- `jupyter` ≥ 1.0 (for notebook)


