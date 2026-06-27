# Experiment 6: Design and Implementation of Linear Support Vector Classification

**Course:** CS-471 Machine Learning  
**Department:** Computer Software Engineering, MCS NUST  

---

## From Logistic Regression to SVM — Why We Moved Forward

In Experiment 5, Logistic Regression achieved **81% accuracy** on the Titanic dataset with an AUC of 0.88. While solid, Logistic Regression has key limitations:

- It models the **probability** of class membership using a sigmoid function, which can be sensitive to outliers and class imbalance.
- It works best when classes are **linearly separable** in feature space — but offers no formal margin guarantee.
- It does **not maximize the separation** between classes; it only fits a decision boundary.

**SVM addresses these shortcomings** by finding the **maximum-margin hyperplane** — the decision boundary with the largest possible gap between classes. This makes SVM:
- More robust to outliers (only support vectors influence the boundary).
- Better at generalizing to unseen data when a clear margin exists.
- Formally optimal in terms of the structural risk minimization principle.

---

## Objectives
- To implement a linear SVM classifier on a real-world medical dataset.
- To evaluate model performance using accuracy, confusion matrix, and classification report.
- To experiment with kernel functions and compare their performance.
- To perform hyperparameter tuning using GridSearchCV.

---

## Tools & Libraries Used
- Python 3 (Google Colab)
- NumPy, Pandas, Matplotlib, Seaborn
- Scikit-learn (SVC, StandardScaler, train_test_split, GridSearchCV, metrics)

---

## Dataset Used
| Dataset | Source |
|---------|--------|
| Breast Cancer Wisconsin | Built into sklearn — `load_breast_cancer()` |

The dataset contains **569 samples** with **30 numerical features** (cell measurements) and a binary target: **malignant (0)** or **benign (1)**.  
Class distribution: 212 malignant (37.3%), 357 benign (62.7%).

---

## Steps Performed

**Cell 1 — Imports:** Imported SVC, GridSearchCV, StandardScaler, and evaluation metrics.

**Cell 2 — Load & Explore:** Loaded the Breast Cancer dataset from sklearn. Examined shape (569 × 30), feature names, class distribution, and statistical summary. No missing values were found — a clean built-in dataset.

**Cell 3 — EDA:** A 2×3 subplot grid visualized:
- Class distribution (malignant vs benign) — slightly imbalanced.
- Histogram of `mean radius` — shows bimodal distribution, suggesting it is discriminative.
- Boxplot of `mean area` — reveals outliers in malignant cases.
- Correlation heatmap of first 10 features — several features highly correlated (e.g., radius, perimeter, area).
- Scatter plot of radius vs texture by class — shows partial linear separability.
- Bar chart of mean feature values by class — malignant tumors consistently larger on radius, area, and perimeter features.

**Cell 4 — Preprocessing (brief):** Split data 80/20, applied StandardScaler (fit on train, transform on test — avoiding data leakage).

**Cell 5 — Train SVM (default kernel='rbf'):** SVC trained with default settings. High accuracy expected due to the clean, well-separated nature of this dataset.

**Cell 6 — Evaluate:** Accuracy, confusion matrix (as heatmap), and classification report printed. SVM achieves very high accuracy on this dataset — typically **96–98%**, significantly outperforming Logistic Regression's 81% on Titanic.

**Cell 7 — Kernel Comparison:** Three kernels tested: `linear`, `rbf`, and `poly`. Comparison table of accuracies printed. On this dataset, all kernels tend to perform well, with RBF often being the best due to the non-linear structure of some feature boundaries.

**Cell 8 — ROC Curve & AUC:** `predict_proba` called to get probabilities. ROC curve plotted. AUC typically ~0.99 — near-perfect discrimination between classes.

**Cell 9 — GridSearchCV Hyperparameter Tuning:** Grid search over `C`, `gamma`, and `kernel`. The best parameters are printed. After tuning, accuracy typically improves slightly or confirms the default setting was already near-optimal.

---

## Why SVM Outperforms Logistic Regression Here

| Factor | Logistic Regression (Exp 5) | SVM (Exp 6) |
|--------|----------------------------|-------------|
| Dataset | Titanic (messy, mixed types) | Breast Cancer (clean, numerical) |
| Accuracy | 81% | ~97% |
| Margin guarantee | No | Yes (max-margin hyperplane) |
| Outlier robustness | Lower | Higher (only support vectors matter) |
| Kernel flexibility | No | Yes (linear, RBF, poly) |

The dramatic accuracy improvement is partly due to the dataset itself (Breast Cancer is highly separable), and partly due to SVM's structural advantages.

---

## Key Concepts Learned

**Support Vector Machine (SVM):** Finds the hyperplane that maximizes the margin between two classes. Only the training points closest to the boundary (support vectors) define it.

**Maximum Margin:** The distance between the decision boundary and the nearest points of each class. Maximizing this leads to better generalization.

**C Parameter (Regularization):** Controls the trade-off between maximizing margin and minimizing classification error. Small C → wider margin, more misclassifications allowed. Large C → narrower margin, fewer misclassifications.

**Kernel Trick:** Maps data to a higher-dimensional space without explicitly computing coordinates, allowing SVM to find non-linear boundaries. Common kernels: linear, RBF (Gaussian), polynomial.

**GridSearchCV:** Exhaustively searches a parameter grid using cross-validation to find the best hyperparameter combination.
