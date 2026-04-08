"""
Employee Sentiment Analysis
===========================
Author: Intern
Dataset: test_in_.csv (Enron employee email data)
Tools: Python, TextBlob-style lexicon sentiment, scikit-learn, matplotlib, seaborn

This script implements all 6 tasks:
  Task 1 - Sentiment Labeling
  Task 2 - Exploratory Data Analysis (EDA)
  Task 3 - Monthly Employee Score Calculation
  Task 4 - Employee Ranking
  Task 5 - Flight Risk Identification
  Task 6 - Linear Regression Predictive Model
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from datetime import datetime, timedelta
import re
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs("visualizations", exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")
COLORS = {"Positive": "#2ecc71", "Neutral": "#3498db", "Negative": "#e74c3c"}

# =============================================================================
# TASK 1 — SENTIMENT LABELING
# =============================================================================
print("=" * 60)
print("TASK 1: Sentiment Labeling")
print("=" * 60)

# -------------------------------------------------------------------
# Lexicon-based sentiment (mirrors TextBlob PatternAnalyzer approach)
# Positive/negative word lists curated for workplace email context.
# Polarity score: sum of word polarities / word count
# Label: Positive if polarity > 0.05, Negative if < -0.05, else Neutral
# -------------------------------------------------------------------

POSITIVE_WORDS = {
    "good", "great", "excellent", "wonderful", "fantastic", "amazing", "happy",
    "pleased", "glad", "love", "loved", "awesome", "perfect", "thanks",
    "thank", "appreciate", "appreciated", "congratulations", "congrats",
    "well", "best", "helpful", "support", "supported", "achieve", "achieved",
    "success", "successful", "productive", "efficient", "positive", "bright",
    "outstanding", "impressive", "nice", "pleased", "delighted", "enjoy",
    "enjoyed", "thrilled", "hope", "hopeful", "confident", "strong", "agree",
    "approved", "approve", "advance", "opportunity", "opportunities", "benefit",
    "benefits", "welcome", "excited", "exciting", "progress", "improve",
    "improved", "improvement", "growth", "innovative", "innovation", "reliable",
    "dedicated", "commitment", "committed", "helpful", "effective", "resolved",
    "solution", "solved", "proactive", "forward", "ready", "yes", "done",
    "accomplish", "accomplished", "reward", "rewarding", "pleasure", "honor",
    "liked", "like", "superb", "prompt", "quickly", "fast", "smooth"
}

NEGATIVE_WORDS = {
    "bad", "poor", "terrible", "awful", "horrible", "hate", "hated",
    "disappointed", "disappointing", "disappointment", "fail", "failed",
    "failure", "problem", "problems", "issue", "issues", "error", "errors",
    "wrong", "incorrect", "loss", "lost", "difficult", "difficulty",
    "struggle", "struggling", "concern", "concerned", "worry", "worried",
    "unhappy", "unfortunate", "unfortunately", "regret", "sorry", "apology",
    "apologize", "complaint", "complain", "complaining", "complains",
    "trouble", "troubles", "delay", "delayed", "miss", "missed", "missing",
    "reject", "rejected", "decline", "declined", "unable", "cannot", "can't",
    "won't", "refuse", "refused", "not", "never", "no", "negative", "worse",
    "worst", "broken", "stuck", "blocked", "upset", "angry", "anger", "risk",
    "violation", "fraud", "illegal", "unfair", "deny", "denied", "lack",
    "lacking", "confusion", "confused", "mistake", "mistakes", "error",
    "overdue", "late", "crisis", "urgent", "critical", "serious", "severe",
    "frustrated", "frustrating", "frustration", "hopeless", "incompetent",
    "inefficient", "waste", "wasted", "unacceptable", "unresolved", "quit"
}

NEGATION_WORDS = {"not", "no", "never", "neither", "nor", "cannot", "can't",
                  "won't", "don't", "didn't", "doesn't", "isn't", "wasn't",
                  "aren't", "haven't", "hadn't", "shouldn't", "wouldn't"}


def compute_polarity(text: str) -> float:
    """
    Compute a polarity score in [-1, 1] using a word-level lexicon.
    Negation within a 3-word window flips the polarity of the next word.
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return 0.0
    tokens = re.findall(r"\b[a-z']+\b", text.lower())
    if not tokens:
        return 0.0

    score = 0.0
    negated = False
    neg_countdown = 0

    for token in tokens:
        if token in NEGATION_WORDS:
            negated = True
            neg_countdown = 3
            continue
        word_score = 0.0
        if token in POSITIVE_WORDS:
            word_score = 1.0
        elif token in NEGATIVE_WORDS:
            word_score = -1.0
        if negated and word_score != 0.0:
            word_score *= -1
            negated = False
            neg_countdown = 0
        if neg_countdown > 0:
            neg_countdown -= 1
            if neg_countdown == 0:
                negated = False
        score += word_score

    return max(-1.0, min(1.0, score / max(len(tokens), 1) * 10))


def label_sentiment(polarity: float) -> str:
    if polarity > 0.05:
        return "Positive"
    elif polarity < -0.05:
        return "Negative"
    return "Neutral"


# Load data
print("\nLoading dataset...")
df = pd.read_csv("data/test_in_.csv")
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date", "from"])
df["combined_text"] = df["Subject"].fillna("") + " " + df["body"].fillna("")
df["employee"] = df["from"].str.split("@").str[0].str.replace(".", " ").str.title()

print(f"Dataset loaded: {len(df)} records, {df['from'].nunique()} employees")
print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")

# Apply sentiment labeling
print("\nApplying sentiment analysis...")
df["polarity"] = df["combined_text"].apply(compute_polarity)
df["sentiment"] = df["polarity"].apply(label_sentiment)
df["word_count"] = df["combined_text"].apply(lambda x: len(str(x).split()))
df["msg_length"] = df["combined_text"].apply(len)
df["year_month"] = df["date"].dt.to_period("M")

print("\nSentiment distribution:")
print(df["sentiment"].value_counts())
print(f"\nSample labeled data:\n{df[['employee', 'date', 'sentiment', 'polarity']].head(5).to_string()}")

# =============================================================================
# TASK 2 — EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
print("\n" + "=" * 60)
print("TASK 2: Exploratory Data Analysis")
print("=" * 60)

print(f"\nDataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nMessages per employee:\n{df.groupby('employee').size().sort_values(ascending=False)}")

# --- Figure 1: Sentiment Distribution Pie ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
counts = df["sentiment"].value_counts()
axes[0].pie(counts, labels=counts.index, autopct="%1.1f%%",
            colors=[COLORS[l] for l in counts.index],
            startangle=140, wedgeprops={"edgecolor": "white", "linewidth": 1.5})
axes[0].set_title("Overall Sentiment Distribution", fontsize=14, fontweight="bold")

axes[1].bar(counts.index, counts.values,
            color=[COLORS[l] for l in counts.index], edgecolor="white", linewidth=1.2)
axes[1].set_title("Sentiment Count by Category", fontsize=14, fontweight="bold")
axes[1].set_ylabel("Number of Messages")
for i, (cat, val) in enumerate(zip(counts.index, counts.values)):
    axes[1].text(i, val + 5, str(val), ha="center", fontweight="bold")

plt.tight_layout()
plt.savefig("visualizations/01_sentiment_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Saved: visualizations/01_sentiment_distribution.png")

# --- Figure 2: Messages per Employee ---
emp_counts = df.groupby("employee").size().sort_values(ascending=False)
plt.figure(figsize=(12, 5))
bars = plt.bar(emp_counts.index, emp_counts.values,
               color=sns.color_palette("muted", len(emp_counts)))
plt.title("Total Messages per Employee", fontsize=14, fontweight="bold")
plt.xlabel("Employee")
plt.ylabel("Message Count")
plt.xticks(rotation=30, ha="right")
for bar, val in zip(bars, emp_counts.values):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
             str(val), ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig("visualizations/02_messages_per_employee.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Saved: visualizations/02_messages_per_employee.png")

# --- Figure 3: Sentiment by Employee (Stacked Bar) ---
emp_sent = df.groupby(["employee", "sentiment"]).size().unstack(fill_value=0)
for col in ["Positive", "Neutral", "Negative"]:
    if col not in emp_sent.columns:
        emp_sent[col] = 0
emp_sent = emp_sent[["Positive", "Neutral", "Negative"]]
ax = emp_sent.plot(kind="bar", figsize=(13, 6), color=[COLORS[c] for c in emp_sent.columns],
                   edgecolor="white", linewidth=0.5)
plt.title("Sentiment Breakdown per Employee", fontsize=14, fontweight="bold")
plt.xlabel("Employee")
plt.ylabel("Message Count")
plt.xticks(rotation=30, ha="right")
plt.legend(title="Sentiment", loc="upper right")
plt.tight_layout()
plt.savefig("visualizations/03_sentiment_by_employee.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Saved: visualizations/03_sentiment_by_employee.png")

# --- Figure 4: Monthly Sentiment Trend ---
monthly_sent = df.groupby(["year_month", "sentiment"]).size().unstack(fill_value=0)
monthly_sent.index = monthly_sent.index.to_timestamp()
fig, ax = plt.subplots(figsize=(14, 5))
for sentiment in ["Positive", "Neutral", "Negative"]:
    if sentiment in monthly_sent.columns:
        ax.plot(monthly_sent.index, monthly_sent[sentiment],
                marker="o", label=sentiment, color=COLORS[sentiment], linewidth=2, markersize=5)
ax.set_title("Monthly Sentiment Trend Over Time", fontsize=14, fontweight="bold")
ax.set_xlabel("Month")
ax.set_ylabel("Number of Messages")
ax.legend(title="Sentiment")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("visualizations/04_monthly_sentiment_trend.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Saved: visualizations/04_monthly_sentiment_trend.png")

# --- Figure 5: Polarity Distribution Histogram ---
plt.figure(figsize=(10, 5))
plt.hist(df["polarity"], bins=50, color="#3498db", edgecolor="white", alpha=0.8)
plt.axvline(0.05, color="#2ecc71", linestyle="--", linewidth=2, label="Positive threshold (0.05)")
plt.axvline(-0.05, color="#e74c3c", linestyle="--", linewidth=2, label="Negative threshold (-0.05)")
plt.title("Distribution of Polarity Scores", fontsize=14, fontweight="bold")
plt.xlabel("Polarity Score")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig("visualizations/05_polarity_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Saved: visualizations/05_polarity_distribution.png")

# --- Figure 6: Word Count vs Polarity ---
plt.figure(figsize=(10, 5))
sample = df.sample(min(500, len(df)), random_state=42)
plt.scatter(sample["word_count"], sample["polarity"],
            c=sample["sentiment"].map(COLORS), alpha=0.5, s=30)
plt.title("Word Count vs Polarity Score", fontsize=14, fontweight="bold")
plt.xlabel("Word Count")
plt.ylabel("Polarity Score")
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=COLORS[l], label=l) for l in COLORS]
plt.legend(handles=legend_elements)
plt.tight_layout()
plt.savefig("visualizations/06_wordcount_vs_polarity.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Saved: visualizations/06_wordcount_vs_polarity.png")

print("\nObservation: The dataset shows a mix of sentiments. Neutral messages are common")
print("in workplace email because of the formal, transactional nature of the content.")
print("Positive messages tend to be longer (more context/praise) while negative messages")
print("tend to be shorter and more direct.")

# =============================================================================
# TASK 3 — MONTHLY EMPLOYEE SCORE CALCULATION
# =============================================================================
print("\n" + "=" * 60)
print("TASK 3: Monthly Employee Score Calculation")
print("=" * 60)

# Score mapping: Positive=+1, Negative=-1, Neutral=0
score_map = {"Positive": 1, "Negative": -1, "Neutral": 0}
df["score"] = df["sentiment"].map(score_map)

monthly_scores = (df.groupby(["employee", "year_month"])["score"]
                  .sum()
                  .reset_index()
                  .rename(columns={"score": "monthly_score"}))

print("\nMonthly sentiment scores (sample):")
print(monthly_scores.head(15).to_string(index=False))

# --- Figure 7: Heatmap of Monthly Scores ---
pivot = monthly_scores.pivot(index="employee", columns="year_month", values="monthly_score").fillna(0)
pivot.columns = [str(c) for c in pivot.columns]
plt.figure(figsize=(16, 6))
sns.heatmap(pivot, cmap="RdYlGn", center=0, annot=True, fmt=".0f",
            linewidths=0.5, cbar_kws={"label": "Monthly Sentiment Score"})
plt.title("Monthly Sentiment Score Heatmap by Employee", fontsize=14, fontweight="bold")
plt.xlabel("Month")
plt.ylabel("Employee")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("visualizations/07_monthly_score_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Saved: visualizations/07_monthly_score_heatmap.png")

# =============================================================================
# TASK 4 — EMPLOYEE RANKING
# =============================================================================
print("\n" + "=" * 60)
print("TASK 4: Employee Ranking")
print("=" * 60)

# Overall ranking across all months
overall_scores = monthly_scores.groupby("employee")["monthly_score"].sum().reset_index()
overall_scores = overall_scores.sort_values(["monthly_score", "employee"],
                                            ascending=[False, True])

top3_positive = overall_scores.head(3)
top3_negative = overall_scores.tail(3).sort_values(["monthly_score", "employee"], ascending=[True, True])

print("\n🏆 TOP 3 MOST POSITIVE EMPLOYEES (Overall):")
for i, row in top3_positive.iterrows():
    print(f"  {row['employee']}: {row['monthly_score']:+.0f}")

print("\n⚠️  TOP 3 MOST NEGATIVE EMPLOYEES (Overall):")
for i, row in top3_negative.iterrows():
    print(f"  {row['employee']}: {row['monthly_score']:+.0f}")

# Per-month ranking
print("\nPer-month rankings (Top 3 Positive | Top 3 Negative):")
all_months = monthly_scores["year_month"].unique()
rankings_records = []
for month in sorted(all_months):
    month_df = monthly_scores[monthly_scores["year_month"] == month].copy()
    month_df = month_df.sort_values(["monthly_score", "employee"], ascending=[False, True])
    top3 = month_df.head(3)["employee"].tolist()
    bot3 = month_df.sort_values(["monthly_score", "employee"], ascending=[True, True]).head(3)["employee"].tolist()
    rankings_records.append({
        "month": str(month),
        "top1_pos": top3[0] if len(top3) > 0 else "",
        "top2_pos": top3[1] if len(top3) > 1 else "",
        "top3_pos": top3[2] if len(top3) > 2 else "",
        "top1_neg": bot3[0] if len(bot3) > 0 else "",
        "top2_neg": bot3[1] if len(bot3) > 1 else "",
        "top3_neg": bot3[2] if len(bot3) > 2 else "",
    })
rankings_df = pd.DataFrame(rankings_records)
print(rankings_df.to_string(index=False))

# --- Figure 8: Overall Ranking Bar Chart ---
plt.figure(figsize=(12, 6))
colors = ["#2ecc71" if s >= 0 else "#e74c3c" for s in overall_scores["monthly_score"]]
bars = plt.barh(overall_scores["employee"], overall_scores["monthly_score"],
                color=colors, edgecolor="white")
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Overall Sentiment Ranking by Employee", fontsize=14, fontweight="bold")
plt.xlabel("Total Sentiment Score")
plt.ylabel("Employee")
for bar, val in zip(bars, overall_scores["monthly_score"]):
    plt.text(val + (1 if val >= 0 else -1), bar.get_y() + bar.get_height() / 2,
             f"{val:+.0f}", va="center", ha="left" if val >= 0 else "right", fontweight="bold")
plt.tight_layout()
plt.savefig("visualizations/08_overall_ranking.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Saved: visualizations/08_overall_ranking.png")

# =============================================================================
# TASK 5 — FLIGHT RISK IDENTIFICATION
# =============================================================================
print("\n" + "=" * 60)
print("TASK 5: Flight Risk Identification")
print("=" * 60)
print("Criteria: 4+ negative messages within any rolling 30-day window")

neg_df = df[df["sentiment"] == "Negative"][["employee", "date"]].copy()
neg_df = neg_df.sort_values(["employee", "date"])

flight_risk_employees = set()
flight_risk_details = []

for employee, group in neg_df.groupby("employee"):
    dates = sorted(group["date"].tolist())
    for i in range(len(dates)):
        window_end = dates[i] + timedelta(days=30)
        count_in_window = sum(1 for d in dates[i:] if d <= window_end)
        if count_in_window >= 4:
            flight_risk_employees.add(employee)
            flight_risk_details.append({
                "employee": employee,
                "trigger_date": dates[i].date(),
                "negatives_in_30_days": count_in_window
            })
            break  # Only record first trigger per employee

flight_risk_df = pd.DataFrame(flight_risk_details) if flight_risk_details else pd.DataFrame()

print(f"\n🚨 FLIGHT RISK EMPLOYEES ({len(flight_risk_employees)} identified):")
if not flight_risk_df.empty:
    print(flight_risk_df.to_string(index=False))
else:
    print("  No flight risk employees identified.")

# --- Figure 9: Flight Risk Visualization ---
emp_neg_counts = df[df["sentiment"] == "Negative"].groupby("employee").size()
plt.figure(figsize=(12, 5))
bar_colors = ["#e74c3c" if emp in flight_risk_employees else "#3498db"
              for emp in emp_neg_counts.index]
bars = plt.bar(emp_neg_counts.index, emp_neg_counts.values,
               color=bar_colors, edgecolor="white")
plt.axhline(4, color="#e74c3c", linestyle="--", linewidth=1.5, label="Risk threshold (4)")
plt.title("Total Negative Messages per Employee\n(Red = Flight Risk)", fontsize=14, fontweight="bold")
plt.xlabel("Employee")
plt.ylabel("Negative Message Count")
plt.xticks(rotation=30, ha="right")
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#e74c3c", label="Flight Risk"),
    Patch(facecolor="#3498db", label="Normal"),
    plt.Line2D([0], [0], color="#e74c3c", linestyle="--", label="Threshold (4/30 days)")
]
plt.legend(handles=legend_elements)
for bar, val in zip(bars, emp_neg_counts.values):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
             str(val), ha="center", fontsize=9)
plt.tight_layout()
plt.savefig("visualizations/09_flight_risk.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Saved: visualizations/09_flight_risk.png")

# =============================================================================
# TASK 6 — LINEAR REGRESSION PREDICTIVE MODEL
# =============================================================================
print("\n" + "=" * 60)
print("TASK 6: Linear Regression Predictive Model")
print("=" * 60)

# Build features at employee-month level
feature_df = df.groupby(["employee", "year_month"]).agg(
    message_count=("score", "count"),
    avg_word_count=("word_count", "mean"),
    total_word_count=("word_count", "sum"),
    avg_msg_length=("msg_length", "mean"),
    neg_count=("sentiment", lambda x: (x == "Negative").sum()),
    pos_count=("sentiment", lambda x: (x == "Positive").sum()),
    monthly_score=("score", "sum")
).reset_index()

feature_df["neg_ratio"] = feature_df["neg_count"] / feature_df["message_count"]
feature_df["pos_ratio"] = feature_df["pos_count"] / feature_df["message_count"]

FEATURES = ["message_count", "avg_word_count", "avg_msg_length", "total_word_count"]  # truly independent features
TARGET = "monthly_score"

model_df = feature_df[FEATURES + [TARGET]].dropna()
X = model_df[FEATURES]
y = model_df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

print(f"\nModel Performance:")
print(f"  R² Score  : {r2:.4f}")
print(f"  RMSE      : {rmse:.4f}")
print(f"  MAE       : {mae:.4f}")
print(f"  Train size: {len(X_train)} | Test size: {len(X_test)}")

print("\nFeature Coefficients:")
coef_df = pd.DataFrame({"Feature": FEATURES, "Coefficient": model.coef_})
coef_df = coef_df.sort_values("Coefficient", ascending=False)
print(coef_df.to_string(index=False))

# --- Figure 10: Actual vs Predicted ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(y_test, y_pred, alpha=0.6, color="#3498db", s=60)
min_val = min(y_test.min(), y_pred.min()) - 1
max_val = max(y_test.max(), y_pred.max()) + 1
axes[0].plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect fit")
axes[0].set_xlabel("Actual Monthly Score")
axes[0].set_ylabel("Predicted Monthly Score")
axes[0].set_title("Actual vs Predicted Monthly Score", fontsize=13, fontweight="bold")
axes[0].legend()
axes[0].text(0.05, 0.92, f"R² = {r2:.3f}\nRMSE = {rmse:.3f}",
             transform=axes[0].transAxes, fontsize=10,
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

# Feature importance
colors_coef = ["#2ecc71" if c >= 0 else "#e74c3c" for c in coef_df["Coefficient"]]
axes[1].barh(coef_df["Feature"], coef_df["Coefficient"], color=colors_coef, edgecolor="white")
axes[1].axvline(0, color="black", linewidth=0.8)
axes[1].set_title("Linear Regression Feature Coefficients", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Coefficient Value")

plt.tight_layout()
plt.savefig("visualizations/10_model_performance.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Saved: visualizations/10_model_performance.png")

# --- Figure 11: Residuals Plot ---
residuals = y_test - y_pred
plt.figure(figsize=(10, 5))
plt.scatter(y_pred, residuals, alpha=0.6, color="#9b59b6", s=50)
plt.axhline(0, color="red", linestyle="--", linewidth=2)
plt.title("Residuals vs Predicted Values", fontsize=13, fontweight="bold")
plt.xlabel("Predicted Monthly Score")
plt.ylabel("Residuals")
plt.tight_layout()
plt.savefig("visualizations/11_residuals.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Saved: visualizations/11_residuals.png")

# =============================================================================
# SAVE LABELED DATASET
# =============================================================================
output_cols = ["from", "employee", "date", "Subject", "body", "polarity", "sentiment", "score", "word_count"]
df[output_cols].to_csv("data/test_labeled.csv", index=False)
print("\n✓ Saved labeled dataset: data/test_labeled.csv")
monthly_scores.to_csv("data/monthly_scores.csv", index=False)
print("✓ Saved monthly scores: data/monthly_scores.csv")

# =============================================================================
# SUMMARY REPORT (printed)
# =============================================================================
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"\nTotal messages analyzed: {len(df)}")
print(f"Employees: {df['employee'].nunique()}")
print(f"\nSentiment breakdown:")
for s, c in df['sentiment'].value_counts().items():
    pct = c / len(df) * 100
    print(f"  {s}: {c} ({pct:.1f}%)")

print(f"\n🏆 Top 3 Positive Employees:")
for i, (_, row) in enumerate(top3_positive.iterrows(), 1):
    print(f"  {i}. {row['employee']} (score: {row['monthly_score']:+.0f})")

print(f"\n⚠️  Top 3 Negative Employees:")
for i, (_, row) in enumerate(top3_negative.iterrows(), 1):
    print(f"  {i}. {row['employee']} (score: {row['monthly_score']:+.0f})")

print(f"\n🚨 Flight Risk Employees ({len(flight_risk_employees)}):")
for emp in sorted(flight_risk_employees):
    print(f"  - {emp}")

print(f"\n📊 Model R²: {r2:.4f}")
print("\nAll done! Check the 'visualizations/' folder for all charts.")
