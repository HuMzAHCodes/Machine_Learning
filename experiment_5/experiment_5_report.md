# Experiment 5: Development and Implementation of Logistic Regression for Classification Tasks

**Course:** CS-471 Machine Learning  
**Department:** Computer Software Engineering, MCS NUST  

---

## Objectives
- To design, implement, and evaluate a Logistic Regression model for binary classification
- To preprocess and clean real-world data for classification tasks
- To evaluate model performance using accuracy, confusion matrix, classification report, and ROC curve
- To identify the most influential features using model coefficients

---

## Tools & Libraries Used
- Python 3 (Google Colab)
- NumPy, Pandas, Matplotlib, Seaborn
- Scikit-learn (LogisticRegression, StandardScaler, train_test_split, metrics)

---

## Dataset Used
| Dataset | Source |
|---------|--------|
| Titanic | Loaded automatically via `sns.load_dataset("titanic")` — no download required |

The Titanic dataset contains 891 rows and 15 columns with passenger information. The target variable is `survived` (0 = did not survive, 1 = survived).

---

## Steps Performed

**Cell 1 — Imports:**
Imported all required libraries including LogisticRegression, StandardScaler, and evaluation metrics (accuracy_score, confusion_matrix, classification_report, roc_curve, auc).

**Cell 2 — Load & Explore:**
Loaded the Titanic dataset directly via seaborn. Examined structure using `.head()`, `.info()`, and `.describe()`. Dataset had 891 rows, 15 columns, with missing values in `age`, `fare`, and `embarked` columns.

**Cell 3 — Visualize Target Variable:**
A count plot of the `survived` column showed class imbalance — approximately 342 survivors vs 549 non-survivors. This means the dataset is slightly imbalanced, which is important when interpreting accuracy.

**Cell 4 — Data Cleaning & Preprocessing:**
- Dropped irrelevant columns: `deck`, `embark_town`, `alive`, `who`, `adult_male`, `alone`
- Filled missing `age` and `fare` with median
- Filled missing `embarked` with mode
- Label Encoded `sex` (binary column — male/female → 1/0)
- OneHot Encoded `embarked` using `pd.get_dummies()` with `drop_first=True` (3 unordered categories → k-1 rule)
- Dropped `class` column (already represented numerically by `pclass`)

Result: 891 rows, 9 columns, zero missing values.

**Cell 5 — Feature Selection:**
Selected 8 features as X: `pclass`, `sex`, `age`, `sibsp`, `parch`, `fare`, `embarked_Q`, `embarked_S`. Target variable y set as `survived`.

**Cell 6 — Train/Test Split & Scaling:**
- Split data 80/20 (712 training, 179 testing) before scaling
- Applied `StandardScaler` — `fit_transform` on training data, `transform` on test data only
- StandardScaler normalizes features to mean=0 and std=1, which is important for Logistic Regression as it is sensitive to feature scale

**Cell 7 — Build & Train Model:**
Created LogisticRegression with `random_state=42` and `max_iter=1000`. Trained on scaled training data. First 10 predictions showed only 1 mismatch against actual values, suggesting good model performance.

**Cell 8 — Evaluate Model Performance:**

Accuracy: **81.01%**

Confusion Matrix results:
- True Negatives (correctly predicted Not Survived): 90
- True Positives (correctly predicted Survived): 55
- False Positives (predicted Survived but didn't): 15
- False Negatives (predicted Not Survived but survived): 19

Classification Report:
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Not Survived | 0.83 | 0.86 | 0.84 |
| Survived | 0.79 | 0.74 | 0.76 |

The model performs slightly better at predicting non-survivors than survivors, likely due to class imbalance.

**Cell 9 — ROC Curve & AUC Score:**
ROC curve plotted with AUC = **0.8820**. The blue curve stays well above the red diagonal (random classifier line), confirming strong discriminating power. An AUC above 0.85 is considered very good for a binary classifier.

**Cell 10 — Feature Importance:**
Bar chart of absolute coefficient values revealed:
- **sex** — most influential feature (coefficient ~1.28). Women had significantly higher survival rates
- **pclass** — second most important (~0.79). First class passengers survived more
- **age** — third (~0.40). Younger passengers had higher survival chances
- **embarked_Q** — least important (~0.04). Port of embarkation had minimal effect

---

## Problems Encountered
No technical errors were encountered in this experiment. All cells ran successfully on the first attempt.

---

## Results Summary
| Metric | Value |
|--------|-------|
| Accuracy | 81.01% |
| AUC Score | 0.8820 |
| Best Performing Class | Not Survived (F1 = 0.84) |
| Most Important Feature | sex |
| Least Important Feature | embarked_Q |
