# Experiment 8: Design and Implementation of Decision Tree Classification

**Course:** CS-471 Machine Learning  
**Department:** Computer Software Engineering, MCS NUST  

---

## From SVM to Decision Trees — The Motivation

In Experiments 6 and 7, SVM achieved excellent accuracy (~97%) on medical and synthetic datasets. However, SVM has a significant limitation — it is a **black box**. You cannot easily explain *why* it made a particular decision. The model finds a hyperplane in a high-dimensional transformed space, and there is no intuitive way to explain that to a non-technical person.

**Decision Trees solve this** by being completely **interpretable and transparent**. Every prediction can be traced back through a series of simple if-then-else rules on actual feature values. A doctor, a manager, or any non-technical stakeholder can look at the tree and understand exactly why a sample was classified the way it was.

Additionally, SVM requires feature scaling — it is sensitive to the magnitude of features because it relies on distance calculations. **Decision Trees require no scaling at all** — they split data based on threshold values, so the absolute scale of a feature is irrelevant.

However, Decision Trees come with their own problem — **overfitting**. A tree allowed to grow without restriction will memorize the training data perfectly but generalize poorly. This experiment explores that problem directly through the depth analysis in Cell 7.

---

## Objectives
- To implement a Decision Tree Classifier on the Iris dataset
- To visualize the trained tree and understand its decision logic
- To analyze how tree depth affects overfitting and underfitting
- To compare Decision Tree performance against SVM and Logistic Regression

---

## Tools & Libraries Used
- Python 3 (Google Colab)
- NumPy, Pandas, Matplotlib, Seaborn
- Scikit-learn (DecisionTreeClassifier, plot_tree, SVC, LogisticRegression, metrics)

---

## Dataset Used
| Dataset | Source |
|---------|--------|
| Iris | Built into sklearn — `load_iris()` |

150 samples, 4 features (sepal length, sepal width, petal length, petal width), 3 classes (Setosa, Versicolor, Virginica) — 50 samples per class. A perfectly balanced dataset, making accuracy a reliable metric.

---

## Core ML Cells — Detailed Explanation

### Cell 4 — Train Decision Tree (Default Depth)

```python
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
```

**Why no scaling?** Unlike SVM and Logistic Regression which use distance-based or gradient-based optimization, a Decision Tree splits data by finding the best threshold on a single feature at each node. For example: "Is petal length ≤ 2.45 cm?" — the answer is the same regardless of whether the feature is scaled or not. Scaling would not change any threshold comparison, so it is unnecessary.

**How the tree builds itself:**
The algorithm starts at the root node with all training data. It evaluates every possible split on every feature and picks the one that results in the **highest information gain** (or equivalently, the **lowest Gini impurity**). The selected feature and threshold become the root node's decision rule. The data is then split into two subsets and the process repeats recursively on each subset until:
- All samples in a node belong to the same class (pure node), or
- The tree reaches its maximum depth, or
- Further splitting provides no benefit

**Gini Impurity:** Measures how often a randomly chosen element from a node would be incorrectly classified. A pure node (all one class) has Gini = 0. The algorithm always tries to minimize this.

$$Gini = 1 - \sum_{i=1}^{k} p_i^2$$

where $p_i$ is the proportion of class $i$ in the node.

**Result:** Default Decision Tree achieves **~93.33% accuracy** on the test set — a strong result on a clean, well-structured dataset.

---

### Cell 6 — Visualize the Decision Tree

```python
plot_tree(dt, feature_names=iris.feature_names,
          class_names=iris.target_names, filled=True, rounded=True)
```

This is one of the most powerful features of Decision Trees — you can literally print the model's entire decision logic as a diagram. Each node shows:
- The **feature and threshold** used for splitting (e.g., "petal length ≤ 2.45")
- The **Gini impurity** of that node
- The **number of samples** that reached that node
- The **class distribution** at that node
- The **predicted class** (majority class) if this were a leaf

**Color coding:** Nodes are colored by their dominant class. A deeply colored node means it is nearly pure (mostly one class). A lightly colored node means it is still mixed.

**Reading the tree:** Start at the root. If petal length ≤ 2.45, go left → pure Setosa leaf (Gini=0). Otherwise go right and continue down through more splits on petal width and petal length to separate Versicolor from Virginica.

**Key insight from the tree:** Setosa is perfectly separable from the other two classes with a single rule (petal length ≤ 2.45). Versicolor and Virginica require more splits because they overlap in feature space — this is also visible in the scatter plots from Cell 3.

The printed depth and node count confirm how complex the default (unconstrained) tree is — it likely has more nodes than necessary, which is a sign of overfitting to the training data.

---

### Cell 7 — Effect of Tree Depth (Overfitting Analysis)

```python
for depth in [2, 3, 4, 5, 6, 7, 8]:
    dt_depth = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt_depth.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, dt_depth.predict(X_train))
    test_acc  = accuracy_score(y_test,  dt_depth.predict(X_test))
```

This is the most important cell in the experiment. It directly demonstrates the **bias-variance tradeoff** in Decision Trees.

**What the plot shows:**
- At **low depth (e.g., depth=2):** Both training and test accuracy are moderate. The tree is too simple to capture all patterns — this is **underfitting** (high bias).
- At **medium depth (e.g., depth=3 or 4):** Training and test accuracy are both high and close to each other. This is the **sweet spot** — the tree generalizes well.
- At **high depth (e.g., depth=7 or 8):** Training accuracy reaches 100% but test accuracy drops or plateaus. The tree has memorized the training data including its noise — this is **overfitting** (high variance).

**The characteristic signature of overfitting:** The training accuracy line keeps rising toward 100%, while the test accuracy line flattens or dips. The growing gap between the two lines is the overfitting gap.

**Why does this happen?** As depth increases, the tree creates increasingly specific rules for smaller and smaller subsets of training data. At full depth, it may create a leaf for just 1 or 2 training samples — rules that will not generalize to new data.

**Practical implication:** Always use `max_depth` to constrain the tree. The optimal depth can be found by looking at where the test accuracy peaks before it starts dropping.

---

### Cell 8 — Compare Decision Tree with SVM and Logistic Regression

```python
models = {
    "Decision Tree"      : DecisionTreeClassifier(random_state=42),
    "SVM (RBF)"          : SVC(kernel="rbf", random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000)
}
```

Three models are trained and compared on the same train/test split. Note that Decision Tree uses **unscaled** data while SVM and Logistic Regression use **StandardScaler-scaled** data — each model gets the input format it needs.

**Results:**
| Model | Accuracy |
|-------|----------|
| Decision Tree | 93.33% |
| SVM (RBF) | 96.67% |
| Logistic Regression | 93.33% |

**Why SVM scores highest (96.67%):** The Iris dataset has well-defined feature boundaries. SVM's maximum-margin hyperplane finds the most robust separator between Versicolor and Virginica (the two harder-to-separate classes), giving it an edge.

**Why Decision Tree and Logistic Regression tie (93.33%):** Both correctly classify Setosa (trivially separable) but make 2 errors each on the Versicolor/Virginica boundary. Decision Tree makes rule-based splits that may not find the optimal boundary between these two overlapping classes. Logistic Regression fits a linear boundary which also struggles with the overlap.

**The key comparison point:** Decision Tree's 93.33% comes with full interpretability — you can explain every single prediction. SVM's 96.67% is a black box. In domains like medicine or law where explainability is legally or ethically required, a Decision Tree at 93.33% may be the preferred choice over SVM at 96.67%.

---

### Cell 9 — Feature Importance

```python
feature_importance = pd.Series(
    dt.feature_importances_,
    index=iris.feature_names
).sort_values(ascending=True)
```

Decision Trees provide **built-in feature importance scores** — a major advantage over SVM (which requires a separate linear model for importance, as seen in Experiment 6). The importance of each feature is computed as the **total Gini impurity reduction** it contributes across all nodes where it is used as a split criterion.

**Expected result:**
- `petal length` and `petal width` → highest importance (used at root and early splits)
- `sepal length` and `sepal width` → lower importance (used at deeper, less impactful nodes)

This aligns with what we saw in Cell 3's scatter plots — petal features showed much cleaner class separation than sepal features. The Decision Tree independently discovered the same thing through its splitting process.

**Why this matters:** Feature importance from Decision Trees is a fast, built-in way to perform **feature selection** — identifying which features actually drive the model's decisions and potentially removing irrelevant ones to simplify future models.

---

## Results Summary
| Model | Accuracy | Needs Scaling | Interpretable |
|-------|----------|---------------|---------------|
| Decision Tree | 93.33% | No | Yes (fully) |
| SVM (RBF) | 96.67% | Yes | No |
| Logistic Regression | 93.33% | Yes | Partially |

---

## Key Concepts Learned

**Decision Tree Structure:**
- **Root Node** — first split, uses the most informative feature
- **Internal Nodes** — subsequent splits on features
- **Leaf Nodes** — final prediction (majority class of samples that reach it)
- **Branches** — outcomes of each split condition

**Gini Impurity vs Information Gain:** Both measure the quality of a split. Gini measures class mixing at a node. Information Gain measures how much a split reduces entropy (disorder). Sklearn's DecisionTreeClassifier uses Gini by default.

**Overfitting in Decision Trees:** An unconstrained tree will grow until it perfectly memorizes training data — creating leaf nodes for single samples. `max_depth`, `min_samples_split`, and `min_samples_leaf` are the key parameters to prevent this.

**Why No Scaling for Decision Trees:** Splits are threshold comparisons on raw feature values. Multiplying all values by a constant (scaling) does not change which side of the threshold a value falls on — so scaling has zero effect on the tree's decisions.

**Feature Importance:** Measures each feature's contribution to reducing Gini impurity across all nodes. A feature used at the root (early splits, affecting many samples) will have much higher importance than one used deep in the tree (affecting few samples).

**Interpretability vs Accuracy Trade-off:** Decision Trees sacrifice some accuracy for full transparency. In regulated industries (healthcare, finance, legal), interpretability is not optional — it is a requirement. This is where Decision Trees shine despite lower accuracy than SVM.
