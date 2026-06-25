# Experiment 4: Design and Implementation of Polynomial Regression

**Course:** CS-471 Machine Learning  
**Department:** Computer Software Engineering, MCS NUST  

---

## Objectives
- To understand the limitations of Simple Linear Regression on non-linear data
- To implement Polynomial Regression using Scikit-learn
- To compare Linear vs Polynomial Regression performance across multiple degrees
- To visualize overfitting as polynomial degree increases

---

## Tools & Libraries Used
- Python 3 (Google Colab)
- NumPy, Pandas, Matplotlib
- Scikit-learn (LinearRegression, PolynomialFeatures, mean_squared_error, r2_score)

---

## Dataset Used
| Dataset | Source |
|---------|--------|
| Position_Salaries.csv | Kaggle — Position Salaries Dataset |

The dataset contains 10 rows with 3 columns: Position (job title), Level (numeric position level 1–10), and Salary. Only Level and Salary are used as features.

---

## Steps Performed

**Cell 1 — Imports & Load:**
Imported all required libraries and loaded `Position_Salaries.csv`. The dataset is very small (10 rows) which makes it a perfect candidate for demonstrating polynomial regression — a small dataset with non-linear salary growth across position levels.

**Cell 2 — Prepare Data & Scatter Plot:**
Extracted Level as X (feature) and Salary as y (target). The scatter plot reveals a clear non-linear pattern — salary grows slowly at lower levels but increases exponentially at higher levels. This immediately tells us that a straight line (Linear Regression) will not fit this data well.

**Cell 3 — Simple Linear Regression:**
Fitted a Linear Regression model on the full dataset and plotted the regression line against the data points. The straight red line clearly fails to capture the curve in the data — it underestimates salaries at both very low and very high levels while overestimating in the middle. The R² value is noticeably low, confirming poor fit.

**Cell 4 — Polynomial Regression (Degree 2, 3, 4):**
Applied PolynomialFeatures to transform X into higher-degree features before fitting LinearRegression. For each degree, a separate plot is generated:

- **Degree 2** — The curve bends once and fits the data better than linear but still misses the sharp salary jump at higher levels
- **Degree 3** — The curve bends twice and captures the general trend more accurately. R² improves noticeably
- **Degree 4** — The curve fits almost all data points very closely. R² approaches 1.0, indicating near-perfect fit on this small dataset

> **Note on PolynomialFeatures:** This transformer does not create a new model — it expands the input features mathematically. For degree 2, it transforms [X] into [1, X, X²], then LinearRegression fits a line through this expanded feature space. The result appears as a curve when plotted against original X.

**Cell 5 — Model Evaluation Comparison Table:**
A summary table comparing all models side by side on R² and RMSE:

| Model | R² | RMSE |
|-------|----|------|
| Linear | Low (~0.6) | High |
| Poly Degree 2 | Medium | Medium |
| Poly Degree 3 | High | Low |
| Poly Degree 4 | Very High (~0.99) | Very Low |

As degree increases, R² improves and RMSE drops — but this does not always mean the model generalizes better to unseen data (see Cell 7).

**Cell 6 — Smooth Polynomial Curve (Degree 4):**
Instead of connecting only the 10 data points, we generate 300 evenly spaced X values between min and max level and predict salary for each. This produces a smooth, continuous curve that clearly shows the polynomial shape. The curve passes through or very near all data points, confirming degree 4 fits this dataset very well.

**Cell 7 — Overfitting Demo (Degrees 1 to 6):**
All 6 degrees plotted in a 2x3 grid. This is the most important visualization in this experiment:

- **Degree 1** — Straight line, clearly underfits
- **Degree 2, 3** — Reasonable curves, good generalization
- **Degree 4** — Excellent fit on this data
- **Degree 5, 6** — The curve starts to wiggle aggressively to pass through every single point. R² may be perfect (1.0) but the model has memorized the training data rather than learning the pattern — this is **overfitting**

This demonstrates the bias-variance tradeoff: as model complexity increases, training error drops but the risk of poor generalization to new data increases.

---

## Problems Encountered
No technical errors were encountered in this experiment. The dataset loaded cleanly and all cells ran successfully.

---

## Key Concepts Learned

**Polynomial Regression:** An extension of Linear Regression where the relationship between X and y is modeled as an nth-degree polynomial. Internally it is still a Linear Regression — just on transformed (expanded) features.

**PolynomialFeatures:** A Scikit-learn transformer that expands input features to include all polynomial combinations up to the specified degree. For degree d and feature X: output = [1, X, X², ..., Xᵈ]

**Overfitting:** When a model learns the training data too well — including its noise — and fails to generalize to new data. Signs: very high R² on training data, poor performance on test data, wildly oscillating prediction curves.

**Underfitting:** When a model is too simple to capture the underlying pattern. Signs: low R², high RMSE, predictions far from actual values even on training data.

**Bias-Variance Tradeoff:**
- High bias (underfitting) → model too simple, misses the pattern
- High variance (overfitting) → model too complex, memorizes noise
- Goal → find the sweet spot degree where the model generalizes well

**R² (Coefficient of Determination):** Measures how much of the variance in y is explained by the model. Ranges from 0 to 1. Closer to 1 = better fit. However, on small datasets, a very high R² at high degrees is a warning sign of overfitting, not necessarily a good thing.

**Why fit_transform on X, not split first?**
This dataset has only 10 rows. Splitting into train/test would leave too few samples to fit meaningful polynomial curves. In such cases, the full dataset is used for demonstration purposes. In real-world problems with sufficient data, always split first then apply PolynomialFeatures.
