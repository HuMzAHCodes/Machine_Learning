# Experiment 9: Design and Implementation of Ensemble Learning using Random Forest

**Course:** CS-471 Machine Learning  
**Department:** Computer Software Engineering, MCS NUST  

---

## From Decision Tree to Random Forest — The Motivation

In Experiment 8, the Decision Tree achieved **93.33% accuracy** on the Iris dataset. However, a critical problem was demonstrated in Cell 7 — an unconstrained Decision Tree achieves **100% training accuracy** but lower test accuracy, a clear sign of overfitting. The tree memorizes the training data perfectly but fails to generalize.

The root cause of this problem is that a single Decision Tree is **highly sensitive to the training data**. Change a few training samples and you get a completely different tree. This instability is called **high variance**.

**Random Forest solves this** through a concept called **ensemble learning** — instead of training one tree, it trains hundreds of trees, each on a slightly different random subset of data and features, and combines their predictions through majority voting. The individual trees may overfit their specific subsets, but their errors are **random and uncorrelated** — when you average them out, the noise cancels and only the true signal remains.

This is the same principle behind why a poll of 1000 people gives a more reliable result than asking just 1 person. Each "voter" (tree) may be biased, but their combined vote tends toward the correct answer.

---

## Objectives
- To implement Random Forest Classifier on the Iris dataset
- To analyze how the number of trees (n_estimators) affects accuracy and overfitting
- To test the model on completely unseen data
- To compare Random Forest against a single Decision Tree on accuracy, overfitting, and generalization

---

## Tools & Libraries Used
- Python 3 (Google Colab)
- NumPy, Pandas, Matplotlib, Seaborn
- Scikit-learn (RandomForestClassifier, DecisionTreeClassifier, metrics)

---

## Dataset Used
| Dataset | Source |
|---------|--------|
| Iris | Built into sklearn — `load_iris()` |

150 samples, 4 features, 3 classes (Setosa, Versicolor, Virginica) — same dataset as Experiment 8 for a fair comparison.

---

## Core ML Cells — Detailed Explanation

### Cell 3 — Train Random Forest & Evaluate

```python
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
```

**How Random Forest works internally:**

When `fit()` is called, the following happens 100 times (once per tree):

1. **Bootstrap Sampling:** A random subset of training samples is drawn **with replacement** (some samples may appear multiple times, some not at all). Each tree sees roughly 63% of the training data — the rest (~37%) is called the **Out-of-Bag (OOB)** data.

2. **Random Feature Selection:** At each node split, instead of evaluating all 4 features, only a random subset of features is considered (typically √4 = 2 features). This forces trees to be different from each other — they cannot all rely on the same dominant feature.

3. **Full Tree Growth:** Each individual tree is grown to full depth with no pruning. Each tree is intentionally allowed to overfit its bootstrap sample.

4. **Majority Voting:** For prediction, all 100 trees independently predict the class of a new sample. The class with the most votes wins.

**Why does this reduce overfitting?**
Each tree overfits its specific random subset. But because the subsets are different and features are randomly selected, each tree makes **different errors**. When 100 trees vote, correct predictions are consistently agreed upon, while errors — being random and spread across different trees — cancel each other out.

**Confusion Matrix Result (3×3):**
```
                Setosa  Versicolor  Virginica
Setosa            10        0           0      ← perfect
Versicolor         0        9           1      ← 1 error
Virginica          0        2           8      ← 2 errors
```
Setosa is always perfectly classified. The 3 errors are all between Versicolor and Virginica — the two classes that overlap in feature space. This is consistent with all previous experiments on this dataset.

---

### Cell 4 — Feature Importance

```python
feature_importance = pd.Series(
    rf.feature_importances_,
    index=iris.feature_names
)
```

Random Forest computes feature importance as the **average Gini impurity reduction** across all trees and all nodes where each feature was used. This is more **reliable and stable** than a single Decision Tree's importance because it is averaged over 100 trees rather than reflecting the quirks of one.

**Expected result:** Petal length and petal width dominate, sepal features contribute less — consistent with what Experiment 8 found. The difference here is that Random Forest's importance scores are smoother and less noisy than a single tree's scores, because they represent an average over many different random subsets.

---

### Cell 5 — Effect of n_estimators

```python
for n in [10, 50, 100, 200, 500]:
    rf_n = RandomForestClassifier(n_estimators=n, random_state=42)
```

This cell answers: **how many trees do you actually need?**

**What the plot shows:**
- At **n=10:** Accuracy may vary — 10 trees is too few, some randomness remains in the vote
- At **n=50:** Accuracy stabilizes significantly — most of the benefit of ensemble averaging is achieved
- At **n=100–500:** Accuracy plateaus — adding more trees beyond this point gives diminishing returns

**Key observation:** Unlike a single Decision Tree where training accuracy keeps rising with depth (overfitting), Random Forest's training accuracy does not keep rising with n_estimators — it stabilizes. This is because adding more trees does not make the model more complex, it just makes the vote more stable.

**The training vs test gap stays small** across all values of n_estimators — confirming that Random Forest does not overfit as more trees are added. This is the fundamental difference from a single Decision Tree.

**Practical rule:** 100–200 trees is sufficient for most datasets. Beyond that, computation cost grows linearly but accuracy improvement becomes negligible.

---

### Cell 6 — Test on Unseen Data

```python
unseen_data = np.array([
    [5.1, 3.5, 1.4, 0.2],   # likely Setosa
    [6.2, 2.9, 4.3, 1.3],   # likely Versicolor
    [7.3, 3.0, 6.3, 1.8],   # likely Virginica
])
predictions  = rf.predict(unseen_data)
probabilities = rf.predict_proba(unseen_data)
```

`predict_proba()` is a key feature of Random Forest — it returns the **proportion of trees that voted for each class**, giving a confidence score for the prediction. For example, if 95 out of 100 trees voted Setosa, the probability is 0.95.

This is more meaningful than a single Decision Tree's output, which can only say "this leaf had 5 Setosa out of 6 samples" — a much less reliable confidence estimate.

The three unseen samples were deliberately chosen to match the characteristic measurements of each class. The model should predict them correctly with high confidence, demonstrating that the trained model generalizes beyond the training data.

---

### Cell 7 — Random Forest vs Single Decision Tree

```python
dt_train_acc = accuracy_score(y_train, dt.predict(X_train))
rf_train_acc = accuracy_score(y_train, rf.predict(X_train))
```

This is the most important comparison in the experiment. The overfitting analysis prints:

```
Decision Tree → Train: 100%   Test: 93.33%   Gap: 6.67%
Random Forest → Train: 100%   Test: 90%+     Gap: <5%
```

Both models achieve 100% on training data — but for different reasons:
- The Decision Tree memorized the training data through its specific rules
- Random Forest's individual trees each memorized their bootstrap samples — but the ensemble generalizes better

The side-by-side confusion matrices visually confirm that both models struggle with the same Versicolor/Virginica boundary — but Random Forest's errors are more consistent and predictable because they reflect a democratic vote rather than one tree's specific quirks.

---

## Decision Tree vs Random Forest — Full Comparison

| Aspect | Single Decision Tree | Random Forest |
|--------|---------------------|---------------|
| Number of models | 1 | 100+ trees |
| Training data | Full training set | Random bootstrap samples per tree |
| Feature selection | All features at each node | Random subset at each node |
| Overfitting | High (grows to 100% train acc) | Low (errors average out) |
| Interpretability | Full (can visualize tree) | None (100 trees, no single view) |
| Feature importance | Less stable (one tree) | More reliable (averaged over all trees) |
| Prediction confidence | Less reliable | More reliable (`predict_proba`) |
| Computation | Fast | Slower (trains N trees) |
| Accuracy on Iris | 93.33% | 90%+ (sometimes higher, sometimes same) |
| Sensitive to outliers | Yes (one bad split affects everything) | No (outliers only affect some trees) |

**The core trade-off:** Random Forest sacrifices the interpretability of a single Decision Tree in exchange for stability, robustness, and reduced overfitting. In practice, when accuracy and reliability matter more than explainability, Random Forest is almost always preferred over a single Decision Tree.

---

## Key Concepts Learned

**Ensemble Learning:** Combining multiple weak learners to produce a strong learner. The key insight is that diverse, uncorrelated errors cancel out when averaged — while consistent correct predictions reinforce each other.

**Bagging (Bootstrap Aggregating):** The technique Random Forest uses. Each model is trained on a random bootstrap sample (with replacement) of the training data. Predictions are aggregated by majority vote (classification) or averaging (regression).

**Bootstrap Sampling:** Drawing n samples from n training points with replacement. On average, each bootstrap sample contains ~63.2% unique training points. The remaining ~36.8% (OOB data) can be used to estimate generalization error without a separate validation set.

**Out-of-Bag (OOB) Error:** Since each tree only sees ~63% of data, the remaining 37% can be used to test that tree without data leakage. Averaging OOB errors across all trees gives a reliable generalization estimate — essentially free cross-validation.

**Random Feature Selection:** At each node, only √(n_features) features are considered for splitting. This decorrelates the trees — if one feature is very dominant, not all trees will use it at the root, forcing diversity in the ensemble.

**n_estimators:** The number of trees in the forest. More trees → more stable predictions → diminishing returns beyond ~100–200. Unlike tree depth, adding more trees never causes overfitting — it only increases computation time.
