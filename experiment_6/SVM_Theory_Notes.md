# Support Vector Machines — Complete Theory Notes
**Course:** CS-471 Machine Learning  
**Prerequisites:** Linear Algebra, Calculus, Basic ML Concepts

---

## 1. What is SVM?

Support Vector Machine is a supervised ML algorithm used primarily for classification. It finds the optimal hyperplane that best separates data points from different classes.

**Historical Context:** Developed at AT&T Bell Labs by Vladimir Vapnik and Corinna Cortes (1990s). Gained early popularity for handwritten digit recognition.

**Core Principle:** Find a decision boundary that correctly classifies all training examples AND maximizes the distance (margin) between the boundary and the nearest points from each class.

---

## 2. Fundamental Concepts

### 2.1 Hyperplane

A hyperplane is the decision boundary that separates classes. Its dimension depends on the feature space:

| Dimensions | Hyperplane |
|---|---|
| 1D | A point |
| 2D | A line |
| 3D | A plane |
| nD | An (n-1)-dimensional subspace |

**Mathematical form:**
```
w·x + b = 0
```
- `w` = weight vector (perpendicular to the hyperplane)
- `x` = feature vector
- `b` = bias term

**Decision rule:**
```
w·x + b ≥ 0  →  Class +1
w·x + b < 0  →  Class -1
```

### 2.2 Margin

The margin is the distance between the hyperplane and the nearest data points from each class.

```
Margin = 2 / ||w||
```

Maximizing margin = minimizing `||w||`. This is the entire optimization goal of SVM.

### 2.3 Support Vectors

Support vectors are the data points that lie exactly on the margin boundaries. They are the only points that determine and "support" the decision boundary.

**Key properties:**
- Only support vectors influence the model
- Removing any non-support vector leaves the boundary unchanged
- Makes SVM memory efficient — only these points are stored after training
- Makes SVM robust — outliers far from the boundary have zero influence

---

## 3. Mathematical Formulation

### 3.1 Hard Margin SVM (Perfectly Separable Data)

```
Minimize:   ½||w||²
Subject to: yᵢ(w·xᵢ + b) ≥ 1,  for all i
```

Why `½||w||²`? Maximizing margin = minimizing `||w||`. The ½ and squaring make the derivative clean. This is a convex quadratic problem — it has one global optimum, always.

### 3.2 Soft Margin SVM (Real-World Data)

Real data has noise and overlap. Forcing perfect separation causes overfitting. **Slack variables (ξᵢ)** allow controlled violations:

| Slack Value | Meaning |
|---|---|
| ξᵢ = 0 | Correctly classified, outside margin |
| 0 < ξᵢ < 1 | Correctly classified, inside margin |
| ξᵢ ≥ 1 | Misclassified |

```
Minimize:   ½||w||² + C·Σξᵢ
Subject to: yᵢ(w·xᵢ + b) ≥ 1 - ξᵢ
            ξᵢ ≥ 0 for all i
```

### 3.3 The C Parameter (Regularization)

| C Value | Effect | Result |
|---|---|---|
| Very Small | Wide margin, many violations | Underfitting (high bias) |
| Moderate | Balanced margin | Good generalization |
| Large | Narrow margin, few violations | Overfitting (high variance) |
| Very Large | Almost hard margin | Severe overfitting |

**Rule of thumb:** Start with C=1. Increase if underfitting. Decrease if overfitting.

---

## 4. The Kernel Trick

### 4.1 The Problem

Many real datasets are not linearly separable. A straight hyperplane cannot separate them correctly.

### 4.2 The Solution

Map data to a higher-dimensional space where it becomes linearly separable — then apply a linear SVM there.

**The trick:** A kernel function `K(x, y)` computes the dot product between two points *as if* they were in the higher-dimensional space — without explicitly computing the transformation. This saves enormous computation.

```
K(x, y) = φ(x)·φ(y)
```

`φ` is the mapping function. We never compute `φ(x)` directly — the kernel does it implicitly.

### 4.3 The Four Kernels

**Linear Kernel**
```
K(x, y) = x·y + c
```
No transformation. Fastest. Most interpretable. Use when data is already linearly separable or when dimensionality is already high (text classification).

**RBF (Radial Basis Function) Kernel** — most commonly used
```
K(x, y) = exp(-γ||x - y||²)
```
γ (gamma) controls the influence radius of each training point:

| Gamma | Effect | Boundary |
|---|---|---|
| Small | Distant points influence each other | Smooth, simple |
| Large | Only nearby points matter | Complex, detailed |
| Too Large | Each point only influences itself | Overfitting |

Use when data is complex and non-linear, or when you have no prior knowledge about structure.

**Polynomial Kernel**
```
K(x, y) = (x·y + c)^d
```
`d` = degree. Higher degree → more complex boundary → higher overfitting risk. Use when data has known polynomial or interaction-based relationships.

**Sigmoid Kernel**
```
K(x, y) = tanh(γ·x·y + c)
```
Behaves like a 2-layer neural network. Rarely the best choice in practice.

---

## 5. The Dual Problem

### 5.1 Why the Dual Form Matters

The original (primal) optimization is in terms of `w` and `b`. The dual reformulation introduces **Lagrange multipliers (αᵢ)** — one per training point.

**Dual optimization:**
```
Maximize:   Σαᵢ - ½ΣᵢΣⱼ αᵢαⱼ yᵢyⱼ K(xᵢ, xⱼ)
Subject to: Σαᵢyᵢ = 0
            0 ≤ αᵢ ≤ C
```

### 5.2 Why This Is Important

- The kernel `K(xᵢ, xⱼ)` replaces the dot product — this is exactly where the kernel trick plugs in
- For non-support vectors, αᵢ = 0 — they contribute nothing
- Only support vectors have αᵢ > 0 — sparse, memory-efficient solution
- The problem is still convex — global optimum guaranteed

---

## 6. Why Maximum Margin Works

**Statistical Learning Theory perspective:** Maximum margin lowers the VC dimension of the model. Lower VC dimension = better generalization = less overfitting. The model is theoretically guaranteed to generalize well.

**Geometric perspective:** A wider margin means new points are more likely to land on the correct side. The boundary is farther from both classes, so it is more robust to noise and small variations in new data.

---

## 7. Multiclass Classification

SVM is natively binary. Two strategies extend it to multiple classes:

**One-vs-Rest (OvR):** Train K classifiers — each one distinguishes one class from all others. Predict the class whose classifier gives the highest confidence. Simpler, K classifiers only.

**One-vs-One (OvO):** Train K(K-1)/2 classifiers — one per pair of classes. Predict by majority vote across all classifiers. More accurate but slower. This is sklearn's default for SVM.

---

## 8. Advantages and Limitations

**Advantages:**

| Aspect | Why |
|---|---|
| Effective in high dimensions | Works well when features > samples |
| Memory efficient | Only support vectors are stored |
| Versatile | Kernel choice adapts to any data shape |
| Robust to outliers | Only support vectors matter; distant outliers have zero influence |
| Theoretically sound | Convex optimization = global optimum guaranteed |

**Limitations:**

| Aspect | Why |
|---|---|
| Poor scalability | Training is O(n²) to O(n³) — slow on large datasets |
| Black box (RBF) | Hard to explain the decision to stakeholders |
| Parameter sensitivity | C, gamma, kernel all need tuning |
| No probability output | Needs Platt scaling (extra step) for probabilities |
| Feature scaling mandatory | Unscaled features break the distance calculation |

---

## 9. SVM vs Other Algorithms

### SVM vs Logistic Regression

| Aspect | SVM | Logistic Regression |
|---|---|---|
| Goal | Maximize margin | Maximize likelihood |
| Boundary depends on | Support vectors only | All data points |
| Probability output | No (needs extra step) | Yes, directly |
| Kernel flexibility | Yes | No |
| Outlier sensitivity | Low | High |
| Training speed | Medium | Fast |

Use SVM for complex boundaries and high-dimensional data. Use Logistic Regression when you need probabilities, interpretability, or are working with large datasets.

### SVM vs Neural Networks

| Aspect | SVM | Neural Networks |
|---|---|---|
| Optimization | Convex — global optimum | Non-convex — local minima |
| Large datasets | Challenging | Handles well |
| Feature learning | Manual (kernel choice) | Automatic |
| Training time | Faster (medium data) | Slower |

### SVM vs Decision Trees

| Aspect | SVM | Decision Trees |
|---|---|---|
| Decision boundary | Smooth | Axis-aligned rectangles |
| Interpretability | Lower | Higher (visualizable) |
| Missing values | Cannot handle | Can handle |
| Outlier sensitivity | Low | High |

---

## 10. Implementation Guidelines

### Feature Scaling — Non-Negotiable

SVM computes Euclidean distances. Features with larger ranges dominate. Always standardize:

```
StandardScaler  →  mean=0, std=1     (preferred for SVM)
MinMaxScaler    →  range [0,1]       (also works)
RobustScaler    →  uses median/IQR   (when outliers are present)
```

**Golden rule:** Fit scaler on training data only. Transform both train and test.

### Kernel Selection Flowchart

```
Is data linearly separable (or very high-dimensional)?
├── YES → Linear Kernel
└── NO  → Do you know the data structure?
          ├── Polynomial relationships → Polynomial Kernel
          ├── Smooth/continuous patterns → RBF Kernel
          └── No prior knowledge → Test Linear AND RBF
```

**Default recommendation:** If unsure, always test Linear and RBF first.

### Hyperparameter Tuning Grids

```python
# RBF Kernel
{'C': [0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1]}

# Polynomial Kernel
{'C': [0.1, 1, 10, 100], 'degree': [2, 3, 4]}

# Linear Kernel
{'C': [0.1, 1, 10, 100]}
```

**Process:** Start with a coarse grid (powers of 10). Use 5-fold cross-validation. Refine around the best values. Always tune on training data only.

---

## 11. Common Pitfalls

| Pitfall | Cause | Fix |
|---|---|---|
| Not scaling | Unequal feature ranges | Always use StandardScaler |
| Wrong kernel | Poor performance | Start linear, then try RBF |
| Overfitting with RBF | γ too high or C too large | Decrease both; use GridSearchCV |
| Underfitting with linear | Data is non-linear | Switch to RBF or polynomial |
| Data leakage | Scaler fit on full data | Fit scaler on train only |
| Class imbalance ignored | Model biased to majority | Use `class_weight='balanced'` |

---

## 12. Advanced Variants (Know for Interviews)

**ν-SVM:** Uses parameter ν ∈ (0,1] instead of C. ν directly controls the fraction of support vectors and the fraction of errors — more intuitive than C.

**One-Class SVM:** Used for anomaly detection. Trained on only one class. Points outside the learned boundary are flagged as anomalies.

**SVR (Support Vector Regression):** Extends SVM to regression. Finds a tube (width ε) that contains most training points. Points outside the tube are penalized.

**SGD-SVM:** Stochastic Gradient Descent approximation for very large datasets. Loses global optimum guarantee but scales to millions of samples.

---

## 13. Quick Pre-Submission Checklist

- [ ] Features scaled?
- [ ] Train/test split correct with `stratify=y`?
- [ ] No data leakage (scaler fit on train only)?
- [ ] Multiple kernels compared?
- [ ] Hyperparameters tuned with GridSearchCV?
- [ ] Cross-validation performed?
- [ ] Evaluation metrics appropriate for the problem (not just accuracy)?
- [ ] Results make domain sense?

---

## 14. Key Takeaways

1. **Goal:** Find the hyperplane with maximum margin between classes
2. **Support Vectors:** Only the boundary points matter — everything else is irrelevant after training
3. **Kernel Trick:** Enables non-linear boundaries without explicit high-dimensional transformation
4. **C Parameter:** Controls regularization — the overfitting/underfitting dial
5. **Gamma Parameter:** Controls influence radius of each support vector (RBF)
6. **Feature Scaling:** Mandatory — not optional
7. **Dual Problem:** Enables the kernel trick and produces a sparse, memory-efficient solution

**When to use SVM:**
- Small to medium datasets (< 100,000 samples)
- Complex, non-linear patterns
- High-dimensional data
- When robustness to outliers matters
- When interpretability is not the primary concern

**When to avoid SVM:**
- Very large datasets (> 100,000 samples) — use gradient boosting or neural networks
- When calibrated probabilities are needed directly
- Very noisy data with heavy class overlap
- When you need to explain every decision to a non-technical stakeholder

---

*SVM Theory Notes | CS-471 Machine Learning*  
*References: Cortes & Vapnik (1995), Vapnik (1998), Schölkopf & Smola (2002), Bishop (2006), sklearn docs*
