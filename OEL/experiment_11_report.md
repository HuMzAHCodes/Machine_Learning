# Experiment 11 — Lab Report
## Open-Ended Machine Learning Project: Diabetes Prediction using Multiple Classifiers

**Course:** CS-471 Machine Learning
**Dataset:** Pima Indians Diabetes Dataset (Kaggle)
**Models Used:** Logistic Regression, Decision Tree, Random Forest, SVM, KNN

---

## 1. Objective

To design and implement a complete end-to-end machine learning pipeline on a real-world medical dataset that:
- Requires significant preprocessing (dirty, incomplete data)
- Applies PCA for dimensionality reduction and evaluates its benefit
- Trains and compares multiple classification models
- Identifies and attempts to fix model limitations
- Justifies every decision made throughout the pipeline

---

## 2. Strategy

Rather than picking one model and running it, the following comprehensive strategy was adopted:

1. Select a dataset that is **not completely clean** — requiring maximum preprocessing
2. Perform thorough **EDA and visualization** before any modeling
3. Apply **PCA** and evaluate whether it actually benefits this dataset
4. Train **all 5 models** on both PCA and original data
5. **Compare results** and justify the winner
6. Identify limitations and **attempt fixes**

This approach demonstrates not just coding ability but **reasoning and decision-making** at every step.

---

## 3. Dataset

**Pima Indians Diabetes Dataset** was selected for the following reasons:

| Requirement | How this dataset fits |
|---|---|
| Not completely clean | Contains impossible zero values in medical columns |
| Maximum preprocessing | Requires outlier detection, imputation, scaling |
| PCA applicable | 8 numerical features |
| Multiple models | Binary classification — all 5 models applicable |
| Medical significance | Low recall has real-world consequences |

**Dataset details:**
- 768 samples, 8 features, 1 binary target (Outcome: 0=No Diabetes, 1=Diabetes)
- Class distribution: 500 No Diabetes (65%), 268 Diabetes (35%) — **imbalanced**
- Features: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age

---

## 4. Cell-by-Cell Walkthrough

---

### Cell 1 — Imports

Imported all required libraries upfront:
- `numpy`, `pandas` — data manipulation
- `matplotlib`, `seaborn` — visualization
- `sklearn` — preprocessing, PCA, all 5 models, metrics

---

### Cell 2 — Load & Explore Dataset

```python
df = pd.read_csv("/content/diabetes.csv")
df.head(), df.info(), df.describe(), df.isnull().sum(), df.duplicated().sum()
```

**What we looked for:**
- `.head()` — first look at data structure
- `.info()` — column types and non-null counts
- `.describe()` — statistical summary (min, max, mean, std)
- `.isnull().sum()` — explicit missing value check
- `.duplicated().sum()` — duplicate row check

**Finding:** No explicit NaN values detected — but `.describe()` revealed minimum values of 0 in medical columns like Glucose, BloodPressure, BMI — which are biologically impossible. These are **disguised missing values** that require special handling.

**Reasoning before this cell:**
> When receiving a new dataset, always explore with all three of `.head()`, `.info()`, `.describe()` together — not just one. Also explicitly check for missing values and duplicates. For this dataset, the disguised zeros in medical columns were the critical finding.

---

### Cell 3 — Boxplot Visualization (Before Cleaning)

```python
# Boxplot for each feature + IQR-based outlier count
```

**What was done:** Plotted boxplots for all 8 features and computed outlier counts using the IQR method (values beyond Q1 - 1.5×IQR or Q3 + 1.5×IQR).

**Results:**
| Feature | Outliers |
|---------|----------|
| Pregnancies | 4 |
| Glucose | 5 |
| BloodPressure | 45 |
| SkinThickness | 1 |
| Insulin | 34 |
| BMI | 19 |
| DiabetesPedigreeFunction | 29 |
| Age | 9 |

**Why this cell was added before preprocessing:**
This visualization was deliberately placed before the cleaning cell to **justify the imputation strategy**. The question was: should we use mean or median to replace missing values?

**Reasoning:**
> Outliers pull the mean upward — replacing missing values with an inflated mean would introduce false data. Median is not affected by extreme values. However, mean is more precise for clean, normally distributed columns. The correct approach is to use **mean for low-outlier columns and median for high-outlier columns** — not blindly applying one strategy to all.

**Decision based on outlier analysis:**
| Column | Outliers | Strategy |
|--------|----------|----------|
| Glucose | 5 (low) | Mean |
| SkinThickness | 1 (very low) | Mean |
| BloodPressure | 45 (high) | Median |
| Insulin | 34 (high) | Median |
| BMI | 19 (high) | Median |

**About outlier removal:** Even though extreme outliers exist (Insulin up to 846), they were **not removed**. In medical datasets, extreme values represent real patients with severe conditions. Removing them would introduce bias — the model would never learn to handle extreme cases. Keeping them and using median imputation is the correct approach.

---

### Cell 4 — Handle Zero Values (Disguised Missing Values)

```python
zero_not_allowed = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[zero_not_allowed] = df[zero_not_allowed].replace(0, np.nan)

# Mean for low-outlier columns
for col in ["Glucose", "SkinThickness"]:
    df[col] = df[col].fillna(df[col].mean())

# Median for high-outlier columns
for col in ["BloodPressure", "Insulin", "BMI"]:
    df[col] = df[col].fillna(df[col].median())
```

**Key decision — Pregnancies column:**
A critical question arose: should Pregnancies=0 also be treated as a disguised missing value?

**Answer: No.** Pregnancies=0 is a perfectly valid value — a patient can genuinely have had zero pregnancies. The rule is:

> "Can this value realistically be zero in real life?"
> - Pregnancies = 0 ✅ → valid, leave it
> - Glucose = 0 ❌ → biologically impossible, replace it
> - BMI = 0 ❌ → biologically impossible, replace it

**Why not Label Encoding here:** Label Encoding applies to categorical text columns (male/female, yes/no). Pregnancies is already a numerical column with actual integer values — no encoding needed.

---

### Cell 5 — Correlation Heatmap

```python
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
```

**Purpose:** This is a **suitability check cell** — it does not perform PCA but answers the question: "Is PCA worth running on this dataset?"

**What correlation values mean:**
| Value | Meaning | PCA action |
|-------|---------|------------|
| Close to +1 | Features increase together — redundant | Merges into fewer PCs |
| Close to -1 | Opposite relationship — redundant | Merges into fewer PCs |
| Close to 0 | Independent features | Keeps in separate PCs |

**Results:**
- `SkinThickness` vs `BMI` → 0.54 (moderate)
- `Pregnancies` vs `Age` → 0.54 (moderate)
- `Glucose` vs `Insulin` → 0.42 (mild)
- Everything else → below 0.4

**Critical observation:**
The maximum correlation was only **0.54** — significantly lower than highly correlated datasets like Iris (0.96 between petal features). This immediately suggested that PCA may not be very beneficial here.

**Reasoning:** The heatmap is a go/no-go check. High correlations → PCA will compress features → worth running. Low correlations → features already independent → PCA has little to compress. However, PCA was still applied to:
1. Fulfill the experiment strategy
2. Verify whether even moderate correlations could benefit any model
3. Let the data prove whether PCA helps or not

**Important limitation of heatmap:** The heatmap shows correlation (normalized, scale-free, -1 to +1). It cannot be used in mathematical calculations — it is purely visual. The covariance matrix is still needed for the actual PCA math because it captures the **magnitude** of spread, not just the direction of relationship.

---

### Cell 6 — Split, Scale & Apply PCA (Combined Cell)

This cell was deliberately designed to do all three steps in the correct order to **avoid data leakage**.

**Why the correct order matters:**

> `fit_transform` must only be applied to training data. Test data must only ever see `transform`. This applies to both scaling and PCA — not just one of them.

The wrong approach (which was initially implemented and then corrected):
```
# WRONG — data leakage
X_scaled = scaler.fit_transform(X)        # fits on full dataset including test
X_pca    = pca.fit_transform(X_scaled)    # PCA sees test data
X_train, X_test = train_test_split(X_pca) # too late
```

The correct approach:
```python
# CORRECT — no leakage
X_train, X_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Scale — fit on train only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# PCA — fit on train only
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca  = pca.transform(X_test_scaled)
```

**Why StandardScaler before PCA:**
PCA finds directions of maximum variance. If features are on different scales, PCA gets dominated by features with large numbers — not because they carry more information, but simply because their values are bigger. `Insulin` ranges 0–846 while `DiabetesPedigreeFunction` ranges 0.08–2.42 — without scaling, PCA would almost entirely ignore the latter.

StandardScaler sets mean=0 and std=1 for all features — giving PCA an equal starting point.

**PCA Results:**
```
Original shape  : (614, 8)
Reduced shape   : (614, 7)
Components kept : 7
Variance retained: 95.13%
```

**Interpretation:** PCA reduced only 1 dimension (8→7) to retain 95% variance. Variance was spread almost evenly:
- PC1: 28.54%, PC2: 18.68%, PC3: 14.27%... (no single dominant component)

Compare to Iris: PC1 alone captured 72%. This confirmed the heatmap's hint — **features are mostly independent, PCA has very little to compress.**

**Training: 614 samples | Testing: 154 samples (80/20 split, stratified)**

---

### Cell 7 — Find Best SVM Kernel

Before training SVM, a critical question arose:

**Reasoning:**
> We should never assume RBF is the best kernel without justification. In Experiment 7, we learned that linear SVM fails on non-linear data and RBF excels. But we need to verify what kind of data we have here — not assume.

All 4 kernels were tested on both PCA and original data:

| Kernel | PCA Accuracy | Original Accuracy |
|--------|-------------|-------------------|
| Linear | 70.13% | 70.13% |
| RBF | 74.68% | 74.03% |
| Polynomial (deg 3) | 72.08% | 71.43% |
| Sigmoid | 70.78% | 70.13% |

**Winner: SVM RBF (74.03% on original, 74.68% on PCA)**

**What this reveals:**
- RBF outperforms Linear → the data has **some non-linear structure**
- For SVM specifically, PCA data (74.68%) beat original (74.03%) — one of the few cases where PCA helped
- This justifies using RBF — not by assumption but by evidence

---

### Cell 8 — Train All 5 Models (PCA vs Original)

```python
models = {
    "Logistic Regression": LogisticRegression(...),
    "Decision Tree"      : DecisionTreeClassifier(...),
    "Random Forest"      : RandomForestClassifier(...),
    "SVM (RBF)"          : SVC(kernel="rbf", probability=True, ...),
    "KNN"                : KNeighborsClassifier(n_neighbors=5)
}
```

**Which models need scaling:**
| Model | Needs Scaling | Reason |
|-------|--------------|--------|
| Logistic Regression | Yes | Gradient-based optimization |
| SVM | Yes | Distance-based (margin calculation) |
| KNN | Yes | Distance-based (nearest neighbors) |
| Decision Tree | No | Threshold comparisons |
| Random Forest | No | Ensemble of threshold-based trees |

Since both `X_train_scaled` and `X_train_pca` are already scaled, all 5 models were safely trained on both.

**Results:**
| Model | PCA Accuracy | Original Accuracy | Winner |
|-------|-------------|-------------------|--------|
| Logistic Regression | 69.48% | 70.78% | Original |
| Decision Tree | 70.78% | 67.53% | PCA |
| Random Forest | 68.83% | 75.32% | Original |
| SVM (RBF) | 74.68% | 74.03% | PCA |
| KNN | 74.68% | 75.32% | Original |

**Best Overall: Random Forest & KNN tied at 75.32% on original data**

**PCA conclusion:**
Original data won for 3 out of 5 models. PCA was **not beneficial** for this dataset — exactly what the heatmap predicted. The low feature correlations (max 0.54) meant PCA had almost nothing to compress, and the slight information loss in the 8th component actually hurt some models.

---

### Cell 9 — Detailed Evaluation of Best Model (Random Forest)

**Why evaluate on original data:**
Random Forest won with 75.32% on `X_train_scaled` — so evaluation must use `X_test_scaled`. The rule is: evaluate on the same data format the model was trained on.

**Results:**
- Accuracy: **75.32%**
- AUC Score: **~0.82**

**Classification Report:**
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| No Diabetes | 0.79 | 0.85 | 0.82 |
| Diabetes | 0.67 | 0.57 | 0.62 |

**Critical finding — Low Recall for Diabetes class (0.57):**

The model only catches **57% of actual diabetes patients** — missing 43%. In a medical context, this is a serious concern.

**Which metric matters more — Precision or Recall?**

In medical diagnosis, **Recall matters more** than Precision:
- **False Negative** (predicting healthy when actually diabetic) → patient goes untreated → serious health consequences
- **False Positive** (predicting diabetic when actually healthy) → patient does more tests → no real harm

A recall of 0.57 means 43% of actual diabetes cases are being missed — unacceptable in a real medical system.

---

### Cell 10 — Hyperparameter Tuning with GridSearchCV

**What GridSearchCV does:**
- **Grid Search** → tries every combination of specified hyperparameter values
- **Cross Validation (cv=5)** → evaluates each combination on 5 different splits — not just one
- **scoring="recall"** → optimizes specifically for catching diabetes patients, not just overall accuracy

```python
param_grid = {
    "n_estimators"     : [100, 200, 300],
    "max_depth"        : [3, 5, 7, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf" : [1, 2, 4]
}
grid_search = GridSearchCV(..., scoring="recall", cv=5)
```

540 combinations tested (108 combinations × 5 folds).

**Best parameters found:** `max_depth=None, min_samples_leaf=2, min_samples_split=2, n_estimators=300`

**Result: No improvement** — accuracy and recall stayed identical (75.32%, recall=0.57).

**Why tuning failed:**
`max_depth=None` means unlimited depth — essentially the same as the default Random Forest. The tuning converged back to the default settings, confirming that **hyperparameters are not the bottleneck** here.

---

### Cell 11 — Class Imbalance Fix

**Root cause identified:**
The real problem is **class imbalance**:
- No Diabetes: 500 samples (65%)
- Diabetes: 268 samples (35%)

The model sees almost twice as many non-diabetes cases during training — naturally becoming better at predicting them. No hyperparameter tuning fixes this structural problem.

**Fix attempted:** `class_weight='balanced'` — tells Random Forest to penalize missing Diabetes cases more heavily during training.

**Result:**
| | Before Fix | After Fix |
|--|------------|-----------|
| Accuracy | 75.32% | 73.38% |
| Recall (Diabetes) | 0.57 | 0.56 |

**Fix did not help** — recall actually dropped slightly. This confirms:

> The dataset itself is the fundamental limitation. `class_weight='balanced'` works better on more severely imbalanced datasets. At 65/35 split, the imbalance is moderate — not severe enough for this fix to make a meaningful difference.

**What would actually fix it in a real project:**
| Fix | Why |
|-----|-----|
| More data (5000+ samples) | Model learns minority class better |
| SMOTE oversampling | Synthetically generates diabetes samples |
| More features (HbA1c, lifestyle) | Better discriminative power |
| Deep Neural Network | Captures complex non-linear patterns |

---

### Cell 12 — Final Comparison Visualization

Side-by-side bar charts comparing:
1. PCA vs Original accuracy for all 5 models
2. Final model ranking on original data

**Final ranking:**
| Model | Original Accuracy |
|-------|-------------------|
| Random Forest | 75.32% |
| KNN | 75.32% |
| SVM (RBF) | 74.03% |
| Logistic Regression | 70.78% |
| Decision Tree | 67.53% |

---

## 5. Results Summary

| Metric | Value |
|--------|-------|
| Best Model | Random Forest (tied with KNN) |
| Best Accuracy | 75.32% |
| Best AUC | ~0.82 |
| PCA benefit | Not beneficial (original won 3/5 models) |
| PCA reduction | 8 → 7 features (minimal) |
| Key limitation | Low recall (0.57) for Diabetes class |
| Root cause | Class imbalance + small dataset size |

---

## 6. Key Decisions & Justifications

| Decision | Justification |
|----------|--------------|
| Dataset choice | Contains disguised zeros, outliers, class imbalance — maximum preprocessing needed |
| Median for high-outlier columns | Outliers inflate mean — median is robust |
| Mean for low-outlier columns | Clean distribution — mean is more precise |
| Pregnancies zeros kept | Biologically valid — a patient can have 0 pregnancies |
| PCA applied despite low correlation | Strategy required it; data proved it wasn't beneficial |
| Split before scale/PCA | Prevents data leakage — fit only on training data |
| RBF kernel for SVM | Tested all 4 kernels — RBF won with evidence, not assumption |
| Recall as optimization metric | Missing a diabetes patient is more dangerous than a false alarm |
| class_weight='balanced' attempted | Responsible fix attempt — data proved dataset size is the real issue |

---

## 7. Conclusion

This experiment demonstrated a complete end-to-end ML pipeline on a real-world medical dataset. The Pima Diabetes dataset required significant preprocessing — handling disguised missing values, outlier-aware imputation, and feature scaling — before any modeling could begin.

PCA was applied and rigorously evaluated — rather than blindly accepting its results, the experiment compared PCA vs original data performance across all 5 models. The heatmap correctly predicted that PCA would not be beneficial (max correlation 0.54), and the model comparison confirmed it — original data outperformed PCA for 3 out of 5 models.

Random Forest emerged as the best model at 75.32% accuracy. However, a critical limitation was identified: low recall (0.57) for the Diabetes class due to class imbalance and small dataset size. Both hyperparameter tuning and class weight balancing were attempted but failed to improve recall — correctly diagnosing that the dataset itself, not the model configuration, is the root constraint.

In a real-world deployment, this model would require a larger, more balanced dataset and additional medical features before being considered for clinical use.

---

*Experiment 11 Lab Report | CS-471 Machine Learning*
