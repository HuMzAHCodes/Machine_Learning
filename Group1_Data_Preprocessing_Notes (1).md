# Group 1 — Data Preprocessing & Visualization
### Comprehensive Lecture Notes | ML Lab Manual (CS-471)
#### Covers: Experiment 1 & Experiment 2

---

> **How to use these notes:** Read top to bottom on your first pass. On revision, test yourself — after reading a section heading, close the notes and explain the concept out loud. If you can't, re-read. The storytelling is intentional — follow the *why* before the *how*.

---

## The Story Begins Here — Why Does Any of This Exist?

Imagine you are a doctor in 2005. You want to predict whether a patient will develop diabetes in the next five years. You have a hunch that factors like age, weight, blood pressure, and family history matter. But your hunch is useless until you can express it mathematically, and mathematics only speaks one language — **numbers in a clean, consistent format.**

Now picture your actual data. It comes from three different hospitals. One hospital records weight in kilograms, another in pounds, a third sometimes leaves it blank. One records gender as "M/F", another as "Male/Female", a third as "0/1". Some patients missed their follow-up appointments so half their records are incomplete. One doctor accidentally entered "999" for a patient's age.

You have data. But you do not have *usable* data.

This is the real world. And this is exactly why **data preprocessing exists** — it is the translation layer between messy human reality and the clean mathematical world that machine learning algorithms live in.

Every ML project in history, from Netflix recommendations to COVID-19 models, has spent more time on preprocessing than on the actual algorithm. In industry, the common estimate is that **data work consumes 60–80% of a data scientist's time.** Your two lab experiments are teaching you the single most practically important skill in the field.

---

## What is Data, Really?

Before we touch preprocessing, we need to be precise about what data looks like in an ML context.

Data comes to you as a **table** — rows and columns. Rows are called **samples** or **observations** (one patient, one student, one transaction). Columns are called **features** or **attributes** (age, weight, score, gender).

```
| student_id | gender | lunch    | math_score | reading_score | prep_course |
|------------|--------|----------|------------|---------------|-------------|
| 1          | female | standard | 72         | 68            | none        |
| 2          | male   | free     | NaN        | 55            | completed   |
| 3          | female | standard | 61         | NaN           | none        |
```

In Python, this table is called a **DataFrame** (from the Pandas library). Each column is a **Series**. These are the two fundamental data structures you will work with every single day.

There are two types of features, and this distinction drives almost every preprocessing decision:

**Numerical features** — values that are actual numbers where arithmetic makes sense. Age, score, salary, temperature. You can add them, average them, compare them.

**Categorical features** — values that represent categories or labels. Gender, city, color, yes/no. The word "male" minus the word "female" is meaningless. These need to be converted before any algorithm can use them.

---

## The Four Problems with Raw Data

Every dataset you will ever encounter has some combination of these four problems. Your entire preprocessing job is to solve them.

### Problem 1 — Missing Values

Data goes missing for real reasons. A patient skipped a test. A sensor malfunctioned. A survey respondent left a question blank. A CSV export had a bug.

In Python/Pandas, missing values appear as `NaN` (Not a Number). If you feed `NaN` directly into an ML algorithm, it will crash or produce garbage — most algorithms simply cannot compute with an absence of value.

**You have three choices when you encounter missing values:**

**Choice A — Delete the row.** If very few rows are missing and you have lots of data, this is acceptable. But if 30% of your data has missing values, deleting those rows destroys too much information.

**Choice B — Delete the column.** If an entire feature column is mostly empty (say, 70% missing), that feature is probably useless anyway. Drop it.

**Choice C — Impute (fill in) the missing value.** This is the most common approach. You replace `NaN` with a calculated estimate. The most common strategies:

- **Mean imputation** — replace NaN with the average of that column. Works well when data is roughly symmetric (bell-shaped). Problem: the mean is sensitive to outliers. One patient recorded at 999 kg will pull the mean wildly high.

- **Median imputation** — replace NaN with the middle value of that column. More robust than mean when outliers exist. Your Experiment 1 uses this for math and reading scores — a deliberate choice to show you the alternative.

- **Mode imputation** — replace NaN with the most frequent value. Used for categorical columns.

- **Forward fill / backward fill** — replace NaN with the previous or next valid value in the sequence. Only makes sense when data has a time order (your Experiment 2 uses forward fill on a Date column in an exercise dataset — because the previous date is a reasonable estimate for a missing date).

In sklearn, the tool is called `SimpleImputer`. You tell it the strategy, it learns from the training data, and applies the same fill logic to new data.

```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
imputer.fit(X_train)           # learns the median from training data only
X_train = imputer.transform(X_train)
X_test  = imputer.transform(X_test)   # applies same median — not recalculated
```

**Critical insight:** Always fit your imputer on training data only, then transform both train and test. If you fit on the entire dataset before splitting, you are "leaking" information from the test set into training — a subtle but serious error called **data leakage** that makes your model look better than it really is.

---

### Problem 2 — Categorical Variables

ML algorithms are mathematical engines. They multiply, add, compute distances. They cannot do any of this with the string "female" or the string "completed". These strings must become numbers.

But here is the trap: you cannot just blindly replace "male" with 1 and "female" with 2. If you do that, you have told the algorithm that female is twice male — a completely false relationship. The algorithm will believe this and make wrong predictions.

So how you encode categories depends on the nature of the category:

**Label Encoding** — assigns each unique category an integer (0, 1, 2, 3...).

```
none      → 0
completed → 1
```

This is appropriate **only when the categories have a real order** (small < medium < large) or when there are exactly two categories (binary). Your Experiment 1 uses this for the "test preparation course" column — which has exactly two values: "none" and "completed". Since it's binary, there is no false ordering introduced.

**One-Hot Encoding** — creates a new binary column for each category value.

```
lunch = standard  →  lunch_standard=1, lunch_free=0
lunch = free      →  lunch_standard=0, lunch_free=1
```

This is the correct approach for categories with **no inherent order** and **more than two values**. Your Experiment 1 uses this for the "lunch" column. Now the algorithm sees two independent binary features — no false mathematical relationship imposed.

**Why not always use one-hot?** Because if a column has 500 unique cities, one-hot creates 500 new columns. This is called the **curse of dimensionality** and it makes models slower and sometimes worse. There are advanced encoding techniques for high-cardinality categories, but those come later.

In sklearn, `LabelEncoder` handles label encoding, and `OneHotEncoder` (often used inside a `ColumnTransformer`) handles one-hot encoding.

---

### Problem 3 — Features at Different Scales

Imagine you are predicting house prices and your features are: number of rooms (2–8) and distance to city center in meters (500–50,000).

Distance has numbers that are thousands of times larger than room count. Many ML algorithms — especially those that compute distances between points (like K-Nearest Neighbors) or use gradient descent (like linear regression with regularization) — will be dominated by the large-scale feature. The algorithm will effectively ignore room count because distance's numbers are so much bigger.

This is not because distance matters more — it's just a units problem. We need all features on a comparable scale.

**Feature scaling** solves this. Two main techniques:

**Min-Max Scaling (Normalization)** — squishes every value into the range [0, 1].

```
scaled_value = (value - min) / (max - min)
```

A score of 50 in a range of 0–100 becomes 0.5. The minimum value of any column becomes 0, the maximum becomes 1. This is what `MinMaxScaler` in sklearn does. Your Experiment 1 applies this to the score columns.

**Standardization (Z-score scaling)** — transforms data so that each column has mean=0 and standard deviation=1.

```
scaled_value = (value - mean) / std_deviation
```

`StandardScaler` in sklearn does this. This approach is more robust when your data has outliers, because it does not compress everything to a fixed range — extreme values stretch but do not dominate.

**When to use which?** As a rule of thumb:
- Use **MinMaxScaler** when you know your data has no extreme outliers and you want bounded output (e.g., for neural networks where inputs need to be small).
- Use **StandardScaler** when you are unsure or when your data follows a roughly normal distribution.
- **Tree-based models** (Decision Trees, Random Forest) do not need scaling at all — they split on thresholds, not distances.

Again — fit your scaler on training data only.

---

### Problem 4 — Dirty Data (Duplicates, Wrong Types, Outliers)

This is the most hands-on, least glamorous part of preprocessing. Experiment 2 is entirely about this.

**Duplicate rows** happen because data is merged from multiple sources, or a form was submitted twice. They skew your analysis — if one transaction appears three times, your model thinks that pattern is three times more common than it is. `df.drop_duplicates()` in Pandas handles this.

**Wrong data types** are common in real datasets. Dates stored as plain strings, numbers stored as text because someone accidentally typed a letter. Your Experiment 2 dataset has a Date column stored as a string — you need to convert it with `pd.to_datetime()` so Python understands it as an actual date and can sort it, compute differences, etc.

**Outliers** are values that are so extreme they are almost certainly errors — a Duration of 10,000 minutes for a workout session, an age of 999, a negative price. Your Experiment 2 has Duration values above 120 minutes flagged as invalid. These are replaced with the median. Outliers in training data teach the model wrong patterns.

---

## The Python Libraries — Their Roles, Simply Explained

You will use four libraries constantly. Know what each one is for:

**NumPy** — the foundation of numerical computing in Python. Everything is built on top of it. It provides the array data structure (like a supercharged list) and fast mathematical operations. You rarely call NumPy directly in high-level ML work, but it runs underneath Pandas and sklearn. Think of it as the engine.

**Pandas** — your workbench for data. Loading CSVs, exploring data, cleaning, filtering, merging tables, handling missing values. The DataFrame is Pandas. You will use this more than anything else. `df.head()`, `df.info()`, `df.describe()`, `df.dropna()`, `df.fillna()` — these are daily tools.

**Matplotlib** — the classic Python plotting library. Creates line charts, bar charts, histograms, scatter plots. Not always the prettiest by default, but fully customizable and foundational. `plt.plot()`, `plt.bar()`, `plt.show()`.

**Scikit-learn (sklearn)** — the ML toolkit. Contains implementations of almost every classical ML algorithm, plus all the preprocessing tools we have discussed (SimpleImputer, LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler, train_test_split). It has a consistent, beautiful API: every tool has `.fit()`, `.transform()`, and `.fit_transform()`. This consistency is deliberate and important.

---

## The Train-Test Split — The Most Important Idea in This Group

Here is a thought experiment. You are a teacher preparing students for an exam. You practice with them using 100 sample questions. On exam day, you give them the exact same 100 questions. They score 100%. Does that mean they learned the material?

No. It means they memorized the answers.

This is the exact problem in ML. If you train a model on all your data and then test it on the same data, the model will appear to perform perfectly — it has simply memorized the examples. This is called **overfitting**, and a model that overfits is useless in the real world because it cannot handle new, unseen examples.

The solution is to **split your data before doing anything else:**

- **Training set** (typically 70–80% of data): this is what the model learns from.
- **Test set** (typically 20–30% of data): this is locked away until the very end, used only to evaluate how well the model generalizes.

Your Experiment 1 uses a 75/25 split — 75% for training, 25% for testing.

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
```

`random_state=42` is just a seed for the random number generator — it ensures you get the same split every time you run the code (reproducibility). The number 42 is traditional, but any integer works.

**The golden rule:** fit all preprocessing (imputers, scalers, encoders) on `X_train` only. Transform both `X_train` and `X_test` using what was learned from training. Never let information from the test set influence preprocessing. Data leakage is the most common subtle mistake in ML, and this is its most common source.

---

## The ColumnTransformer — Applying Different Steps to Different Columns

In practice, you often need to apply different preprocessing to different columns simultaneously — scale the numerical columns, one-hot encode the categorical columns. `ColumnTransformer` lets you do this in one clean step.

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), ['math_score', 'reading_score']),
    ('cat', OneHotEncoder(),  ['lunch'])
])
```

This is the professional way to build preprocessing pipelines — it avoids manually looping through columns and makes the code readable and reproducible.

---

## Data Visualization — Why You Must See Your Data Before Modeling

Experiment 2 gives equal weight to visualization, and there is a reason. Numbers alone are deceptive.

In 1973, statistician Francis Anscombe created four datasets — "Anscombe's Quartet" — that have nearly identical means, variances, and correlation values. If you only looked at summary statistics, you would conclude they are the same. But when you plot them, they look completely different — one is linear, one is curved, one has a single outlier destroying the pattern. The statistics lied; the plot told the truth.

Visualization in preprocessing serves three purposes:

**1. Spot problems before modeling.** A histogram of your "age" column that shows a spike at 999 immediately tells you there are bad entries. A bar chart of duration values that shows one bar at 10,000 minutes stands out instantly.

**2. Verify that your cleaning worked.** Experiment 2 asks you to plot before and after cleaning. This is not busywork — it trains the habit of confirming that your code actually did what you intended.

**3. Understand distribution.** Is your target variable (the thing you are predicting) balanced? If you are predicting fraud and 99% of transactions are normal, a model that always predicts "not fraud" will be 99% accurate but completely useless. You cannot know this without looking.

Key plot types you need to know:

- **Line chart** — trends over time. "How do Calories change over different workout dates?"
- **Bar chart** — comparing categories or discrete values. "How does Duration vary across first 20 records?"
- **Histogram** — distribution of a single numerical variable. How spread out are your values? Are there clusters?
- **Scatter plot** — relationship between two numerical variables. Do they move together?
- **Box plot** — shows median, quartiles, and outliers for a distribution. Excellent for spotting outliers quickly.

---

## The Complete Preprocessing Pipeline — Putting It All Together

Here is the sequence of a professional preprocessing workflow. Memorize this order — it matters.

```
1. LOAD           →  Read the CSV into a DataFrame (pd.read_csv)
2. EXPLORE        →  df.head(), df.info(), df.describe(), df.isnull().sum()
3. VISUALIZE      →  Plot key columns before touching anything
4. SPLIT          →  train_test_split FIRST, before any fitting
5. CLEAN          →  Handle duplicates, fix data types, fix outliers
6. IMPUTE         →  Fill missing values (fit on train, transform both)
7. ENCODE         →  Convert categorical to numerical (fit on train, transform both)
8. SCALE          →  Normalize/standardize numericals (fit on train, transform both)
9. VERIFY         →  Check shapes, check for remaining NaNs, visualize again
10. HAND OFF      →  X_train, X_test, y_train, y_test ready for the algorithm
```

Steps 6, 7, and 8 all follow the same rule: fit on train, transform both. This is the heartbeat of correct preprocessing.

---

## Quick Reference — Key Functions and When to Use Them

| Task | Pandas/sklearn Function | When to Use |
|---|---|---|
| Load CSV | `pd.read_csv('file.csv')` | Always first |
| Quick overview | `df.head()`, `df.info()`, `df.describe()` | Always after loading |
| Count missing values | `df.isnull().sum()` | Before imputing |
| Fill missing (numerical) | `SimpleImputer(strategy='mean'/'median')` | When rows with NaN are too many to drop |
| Fill missing (date/time) | `df.fillna(method='ffill')` | When data is time-ordered |
| Drop duplicates | `df.drop_duplicates()` | When rows repeat exactly |
| Fix data types | `pd.to_datetime()`, `astype()` | When types are wrong |
| Label encode (binary / ordered) | `LabelEncoder()` | Binary categories or ordinal data |
| One-hot encode (nominal) | `OneHotEncoder()` / `pd.get_dummies()` | Non-ordered categories with 3+ values |
| Scale to [0,1] | `MinMaxScaler()` | When you need bounded output, no major outliers |
| Standardize (mean=0, std=1) | `StandardScaler()` | General-purpose, more robust to outliers |
| Train/test split | `train_test_split(X, y, test_size=0.25)` | Before any fitting |
| Line chart | `plt.plot(x, y)` | Trends over time |
| Bar chart | `plt.bar(x, y)` | Comparing discrete categories |
| Histogram | `plt.hist(column)` | Distribution of a numerical column |

---

## Common Mistakes to Avoid

**Fitting on the full dataset before splitting.** This is data leakage. The model gets an unfair preview of test data during preprocessing.

**Using mean when outliers are present.** One extreme value can drag the mean far from the true center. Use median when your data is not symmetric.

**Applying label encoding to non-binary categories.** Encoding "red=0, green=1, blue=2" tells the model blue > green > red. Use one-hot instead.

**Forgetting to handle NaN before scaling.** MinMaxScaler and StandardScaler will break or produce NaN outputs if any NaN values remain.

**Dropping too many rows.** If a column has 60% missing, do not fill it with mean — the filled values are mostly fictional. Consider dropping the column entirely.

**Not visualizing before and after.** You will make mistakes in code. The only way to catch them is to look at the results.

---

## Summary — What These Two Experiments Are Really Teaching You

Experiment 1 teaches you the **full preprocessing skeleton** — the sequence of steps every ML project must follow. Missing value handling, encoding, scaling, splitting. All using the sklearn API.

Experiment 2 teaches you **real-world messiness** — the kind of dirty data that does not come with clean lab instructions. Wrong types, outliers, duplicates, time-series fills. Plus it adds visualization as a diagnostic tool, not just a final product.

Together, they are telling you one thing: **no matter how clever your algorithm is, garbage in means garbage out.** The model you will build in Experiments 3 through 9 will be only as good as the preprocessing you do before it.

The best ML engineers are not the ones who know the most exotic algorithms. They are the ones who never let bad data reach the algorithm in the first place.

---

*Notes prepared for CS-471 ML Lab revision | Group 1 of 5*
*Next: Group 2 — Regression (Linear, Polynomial, Gradient Descent)*
