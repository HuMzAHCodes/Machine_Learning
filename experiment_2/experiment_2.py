# ============================================================
# Experiment 2: Data Preprocessing with Pandas & Matplotlib
# Part 1 → rawdata.csv  |  Part 2 → Titanic dataset
# ============================================================


import sys
sys.path.insert(0, r"D:\ML-Lab\packages")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split


# ════════════════════════════════════════════════════════════
# PART 1 — rawdata.csv
# ════════════════════════════════════════════════════════════

# ── Task 1: Load and Explore ─────────────────────────────────
raw = pd.read_csv("rawdata.csv")

print("=== rawdata.csv ===")
print("\nFirst 10 rows:")
print(raw.head(10))

print("\n.info():")
raw.info()

print("\n.describe():")
print(raw.describe())


# ── Task 2: Visualize UNCLEAN data ───────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Before Cleaning", fontsize=14)

# Bar chart – Duration for first 20 records
axes[0].bar(range(20), raw["Duration"].head(20))
axes[0].set_title("Duration (first 20 records)")
axes[0].set_xlabel("Record Index")
axes[0].set_ylabel("Duration (min)")

# Line chart – Calories over Date (drop NaN rows for plotting only)
temp = raw.dropna(subset=["Date", "Calories"])
axes[1].plot(temp["Date"], temp["Calories"], marker="o", markersize=3)
axes[1].set_title("Calories over Date")
axes[1].set_xlabel("Date")
axes[1].set_ylabel("Calories")
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("part1_before_cleaning.png", dpi=100)
plt.show()


# ── Task 3: Clean the Data ───────────────────────────────────

# 1. Fill missing Calories with mean
raw["Calories"] = raw["Calories"].fillna(raw["Calories"].mean())

# 2. Forward-fill missing Date values
raw["Date"] = raw["Date"].ffill()

# 3. Convert Date column to proper datetime
raw["Date"] = pd.to_datetime(raw["Date"])

# 4. Replace Duration values above 120 with the median
median_duration = raw["Duration"].median()
raw["Duration"] = raw["Duration"].apply(
    lambda x: median_duration if x > 120 else x
)

# 5. Remove duplicate rows
raw = raw.drop_duplicates()
raw = raw.reset_index(drop=True)

print("\nCleaned data (first 5 rows):")
print(raw.head())


# ── Task 4: Visualize CLEANED data ───────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("After Cleaning", fontsize=14)

axes[0].bar(range(min(20, len(raw))), raw["Duration"].head(20))
axes[0].set_title("Duration (first 20 records)")
axes[0].set_xlabel("Record Index")
axes[0].set_ylabel("Duration (min)")

axes[1].plot(raw["Date"], raw["Calories"], marker="o", markersize=3)
axes[1].set_title("Calories over Date")
axes[1].set_xlabel("Date")
axes[1].set_ylabel("Calories")
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("part1_after_cleaning.png", dpi=100)
plt.show()


# ════════════════════════════════════════════════════════════
# PART 2 — Titanic Dataset
# ════════════════════════════════════════════════════════════

# ── Task 5: Load and Drop Irrelevant Columns ─────────────────
import seaborn as sns

titanic = sns.load_dataset("titanic")   # loads directly from seaborn

# Drop columns specified in the task (in one line)
titanic = titanic.drop(
    columns=["alive", "alone", "embark_town", "who", "adult_male", "deck", "embarked", "class"]
)

print("\n=== Titanic Dataset ===")
print("Shape after dropping columns:", titanic.shape)
print(titanic.head())


# ── Task 6: Handle Categorical & Missing Values ───────────────

# Label-encode 'sex' column (male → 1, female → 0)
le = LabelEncoder()
titanic["sex"] = le.fit_transform(titanic["sex"])

# Fill missing numerical values with mean
num_cols = titanic.select_dtypes(include=[np.number]).columns
titanic[num_cols] = titanic[num_cols].fillna(titanic[num_cols].mean())

# Fill missing categorical values with mode
cat_cols = titanic.select_dtypes(include=["object", "category"]).columns
for col in cat_cols:
    titanic[col] = titanic[col].fillna(titanic[col].mode()[0])

print("\nMissing values after cleaning:")
print(titanic.isnull().sum())


# ── Task 7: Visualize Cleaned Titanic Data ───────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Histogram of 'fare'
axes[0].hist(titanic["fare"], bins=30, color="steelblue", edgecolor="black")
axes[0].set_title("Fare Distribution")
axes[0].set_xlabel("Fare")
axes[0].set_ylabel("Frequency")

# Pie chart of survival status
survival_counts = titanic["survived"].value_counts()
axes[1].pie(
    survival_counts,
    labels=["Not Survived", "Survived"],
    autopct="%1.1f%%",
    colors=["#ff6b6b", "#6bcb77"],
    startangle=90
)
axes[1].set_title("Passenger Survival Status")

plt.tight_layout()
plt.savefig("part2_titanic_plots.png", dpi=100)
plt.show()


# ── Task 8: Feature Scaling and Train/Test Split ──────────────

# Separate features and target
X = titanic.drop(columns=["survived"])
y = titanic["survived"]

# Scale all numerical features
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"\nTraining samples : {X_train.shape[0]}")
print(f"Testing  samples : {X_test.shape[0]}")
print("\nExperiment 2 complete.")
