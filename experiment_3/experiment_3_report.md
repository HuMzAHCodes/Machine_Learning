# Experiment 3: Design and Implementation of Linear Regression and Gradient Descent for Supervised Learning Tasks

**Course:** CS-471 Machine Learning  
**Department:** Computer Software Engineering, MCS NUST  

---

## Objectives
- To understand the working of Simple Linear Regression for prediction
- To implement Linear Regression using Scikit-learn
- To visualize model performance and evaluate using regression metrics
- To implement Gradient Descent manually for Linear Regression

---

## Tools & Libraries Used
- Python 3 (Google Colab)
- NumPy, Pandas, Matplotlib, Seaborn
- Scikit-learn (LinearRegression, train_test_split, MinMaxScaler, LabelEncoder, metrics)

---

## Datasets Used
| Task | Dataset | Source |
|------|---------|--------|
| Task 1 | insurance.csv | Kaggle — Medical Cost Personal Dataset |
| Task 2 | Salary_dataset.csv | Kaggle — Salary Dataset Simple Linear Regression |

---

## Task 1: Insurance Charges Prediction

### What We Did
Loaded the insurance dataset containing patient information (age, sex, BMI, children, smoker status, region) and used Linear Regression to predict medical charges.

### Steps Performed

**Cell 1 — Imports:** Imported all required libraries including sklearn's LinearRegression, metrics, and preprocessing tools.

**Cell 2 — Load & Explore:** Loaded `insurance.csv` and examined the structure using `.head()`, `.info()`, and `.describe()`. The dataset had 1338 rows and 7 columns with no missing values.

**Cell 3 — Data Cleaning:** Checked for null values and duplicates. Filled any missing numerical values with mean and categorical with mode. Removed duplicate rows.

**Cell 4 — Encoding (Corrected):**
- `sex` and `smoker` → Label Encoded (binary columns, 2 values only)
- `region` → OneHot Encoded using `pd.get_dummies()` with `drop_first=True`

> **Critical Fix Applied:** Originally all three columns were Label Encoded. This was incorrect because `region` has 4 unordered categories — Label Encoding would imply a false ranking (southwest=3 > southeast=2 etc.). OneHot Encoding was applied instead, following the **k-1 dummy variable rule** to avoid multicollinearity.

**Cell 5 — Split & Scale (Corrected):**
- Data was split **before** scaling (80% train, 20% test)
- `fit_transform()` applied on training data only
- `transform()` applied on test data only
- Linear Regression model trained on scaled training data

> **Critical Fix Applied:** Originally scaling was done on the entire dataset before splitting, causing **data leakage** — the scaler was learning min/max from test data which it should never see. The correct approach is to always split first, then scale.

**Cell 6 — Evaluation:**
- Predicted charges on test set
- Evaluated using MSE, RMSE, and R²
- R² value indicates how much variance in charges the model explains. A value closer to 1.0 means a better fit.

**Cell 7 — Bar Chart (Actual vs Predicted):**
A bar chart comparing actual vs predicted charges for the first 30 test samples. Both bars appear at similar heights where the model predicts well. Larger gaps between bars indicate samples where the model struggled — typically high-charge patients (smokers).

**Cell 8 — Regression Line (BMI vs Charges):**
A scatter plot of BMI vs Charges with a red regression line overlaid. The upward slope confirms that higher BMI is associated with higher medical charges. The spread of points around the line indicates that BMI alone does not fully explain charges — other features like smoker status also play a strong role.

**Cell 9 — Manual Gradient Descent:**
Implemented Gradient Descent from scratch using NumPy on a single feature (BMI). The algorithm iteratively updated slope and intercept over 1000 epochs using learning rate 0.01. The cost plot shows the MSE decreasing steeply at first and then flattening as the model converges — confirming that Gradient Descent successfully minimized the cost function.

---

## Task 2: Salary Prediction with Gradient Descent

### What We Did
Used a simple salary dataset (Years of Experience → Salary) to implement both Scikit-learn Linear Regression and manual Gradient Descent, then compared their results.

### Steps Performed

**Cell 10 — Load Salary Dataset:**
Loaded `Salary_dataset.csv`. The dataset has 2 columns — Years of Experience and Salary.

> **Problem Encountered:** The code used `/content/salary.csv` but the uploaded file was named `Salary_dataset.csv`. Fixed by correcting the filename in `pd.read_csv()`.

**Cell 11 — Scatter Plot & Split:**
A scatter plot of Years of Experience vs Salary shows a strong positive linear relationship — as experience increases, salary increases almost linearly. This confirms Linear Regression is the right model here. Data split into 80% train, 20% test.

**Cell 12 — Sklearn Linear Regression:**
Trained Linear Regression on salary data. Printed slope and intercept. A plot of actual vs predicted salary on the test set shows the red prediction line closely following the actual data points, indicating a good fit. MSE and RMSE values are also printed.

**Cell 13 — Manual Gradient Descent & Comparison:**
Implemented the same Gradient Descent function from Task 1 on salary data. The slope and intercept from Gradient Descent are compared with those from Scikit-learn. Both values are very close to each other, confirming that manual Gradient Descent converges to the same solution as Scikit-learn's optimized implementation. The cost plot again shows smooth convergence.

---

## Problems Encountered & How They Were Fixed

| Problem | Root Cause | Fix Applied |
|---------|-----------|-------------|
| Label Encoding applied to `region` column | `region` has 4 unordered categories — Label Encoding implies false ranking | Switched to `pd.get_dummies()` with `drop_first=True` |
| Data leakage during scaling | `fit_transform()` was called on full dataset before splitting | Moved split before scaling; `fit_transform` on train, `transform` on test only |
| `FileNotFoundError` for salary.csv | Uploaded file was named `Salary_dataset.csv` not `salary.csv` | Corrected the filename in `pd.read_csv()` |

---

## Key Concepts Learned

**Linear Regression:** Models relationship between one independent variable (X) and dependent variable (Y) as Y = b₀ + b₁·X. The model finds the best-fit line by minimizing the difference between actual and predicted values.

**Gradient Descent:** An iterative optimization algorithm that updates slope and intercept step-by-step using partial derivatives of the cost function (MSE). Learning rate controls the step size.

**Evaluation Metrics:**
- **MSE** — Mean of squared errors; penalizes large errors heavily
- **RMSE** — Square root of MSE; same units as target variable
- **R²** — Proportion of variance explained; ranges from 0 to 1

**Data Leakage:** When information from test data leaks into the training process (e.g., scaling on full dataset). Always split first, then scale.

**k-1 Dummy Variable Rule:** For a categorical column with k categories, create only k-1 dummy columns. The dropped column is implied when all others are 0. This avoids multicollinearity.
