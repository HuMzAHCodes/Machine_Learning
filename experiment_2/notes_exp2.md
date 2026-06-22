# Experiment 2: Data Preprocessing with Pandas & Matplotlib

## Objective
Learn how to clean messy real-world data, handle missing values, remove outliers and duplicates, and visualize data before and after cleaning using two datasets: **rawdata.csv** and **Titanic**.

---

## Key Concepts Learned

### 1. Exploratory Data Analysis (EDA)
Before cleaning, we always explore the data first:
- `df.head(10)` — see first 10 rows
- `df.info()` — column types and non-null counts
- `df.describe()` — statistical summary (mean, min, max, std)

---

### 2. Handling Missing Values

| Method | Code | When to Use |
|--------|------|-------------|
| Fill with mean | `df["col"].fillna(df["col"].mean())` | Numerical columns |
| Fill with mode | `df["col"].fillna(df["col"].mode()[0])` | Categorical columns |
| Forward fill | `df["col"].ffill()` | Time-series / date columns |

---

### 3. Handling Outliers
Values that are unrealistically large/small are replaced with the **median**:
```python
median_val = df["Duration"].median()
df["Duration"] = df["Duration"].apply(
    lambda x: median_val if x > 120 else x
)
```
We use **median** (not mean) because median is not affected by the outlier itself.

---

### 4. Removing Duplicates
```python
df = df.drop_duplicates().reset_index(drop=True)
```
- `drop_duplicates()` removes rows that are identical
- `reset_index(drop=True)` resets row numbers cleanly after removal

---

### 5. Data Type Conversion
```python
df["Date"] = pd.to_datetime(df["Date"])
```
Converting string dates to proper datetime objects allows sorting, plotting, and time-based operations.

---

### 6. Label Encoding
Converts text categories to numbers so ML models can process them:
```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["sex"] = le.fit_transform(df["sex"])
# female → 0, male → 1
```

---

### 7. Feature Scaling (MinMaxScaler)
Normalizes all values to the range **[0, 1]**:
```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
```
**Formula:**
$$X_{scaled} = \frac{X - X_{min}}{X_{max} - X_{min}}$$

This ensures no single feature dominates the model due to a larger scale.

---

### 8. Train/Test Split
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
- `test_size=0.2` → 80% training, 20% testing
- `random_state=42` → ensures same split every run (reproducibility)

---

### 9. Visualization

| Plot Type | Used For |
|-----------|----------|
| Bar chart | Comparing values across records (Duration) |
| Line chart | Trends over time (Calories over Date) |
| Histogram | Distribution of a numerical column (Fare) |
| Pie chart | Proportions of categories (Survival rate) |

---

## Datasets Used

### rawdata.csv
A workout log with 3 columns: `Duration`, `Date`, `Calories`

Issues fixed:
- Missing `Calories` → filled with **mean**
- Missing `Date` → filled with **forward fill**
- Invalid `Duration` (450 min) → replaced with **median**
- Duplicate rows → removed

### Titanic Dataset
Classic ML dataset with passenger survival information.

Issues fixed:
- Irrelevant columns dropped
- `sex` column label-encoded
- Missing numerical values filled with mean
- Missing categorical values filled with mode

---

## Summary
| Step | Tool Used |
|------|-----------|
| Load data | `pd.read_csv()` |
| Explore data | `.head()`, `.info()`, `.describe()` |
| Fix missing values | `.fillna()`, `.ffill()` |
| Fix outliers | `.apply()` with lambda |
| Remove duplicates | `.drop_duplicates()` |
| Encode categories | `LabelEncoder` |
| Scale features | `MinMaxScaler` |
| Split data | `train_test_split` |
| Visualize | `matplotlib.pyplot` |
