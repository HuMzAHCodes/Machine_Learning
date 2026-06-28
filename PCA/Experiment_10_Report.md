# Experiment 10 — Lab Report
## Dimensionality Reduction using Principal Component Analysis (PCA)
**Course:** CS-471 Machine Learning
**Dataset:** Fisher's Iris Dataset (4 features → 2 principal components)

---

## 1. Objective

To implement Principal Component Analysis (PCA) from scratch using NumPy, verify it against sklearn's implementation, and visually confirm that the 4-dimensional Iris dataset can be faithfully represented in 2 dimensions with minimal information loss.

---

## 2. Steps Taken (Overview)

1. Loaded and explored the Iris dataset — shape, class distribution, feature ranges
2. Visualized feature distributions by class (histograms) and feature correlations (heatmap)
3. **Manual PCA — Step 1:** Mean-centered the data
4. **Manual PCA — Step 2:** Computed the covariance matrix
5. **Manual PCA — Step 3:** Computed eigenvalues and eigenvectors via `np.linalg.eigh()`
6. **Manual PCA — Step 4:** Sorted eigenvalues in descending order
7. **Manual PCA — Step 5:** Selected top 2 eigenvectors (feature vector)
8. **Manual PCA — Step 6:** Projected data from 4D to 2D
9. Visualized the 2D projection colored by class
10. Reproduced PCA using sklearn (`StandardScaler` + `PCA`) as a cross-check
11. Compared manual vs sklearn projections side by side
12. Ran full 4-component PCA to justify the choice of k=2

---

## 3. Link to Previous Experiments — Why This Experiment Was Needed

Every classifier in Groups 2 and 3 has a hidden enemy: **too many features**.

In **Experiment 6 (SVM)**, the Breast Cancer dataset had 30 features. SVM computes distances between every pair of points across all 30 dimensions. As dimensions increase, distances between points become increasingly similar — all points start appearing equidistant. The algorithm loses its ability to distinguish close neighbors from far ones. This is the **curse of dimensionality**.

In **Experiments 8 and 9 (Decision Tree and Random Forest)**, more features mean more possible splits at every node. The tree searches a larger space, training slows down, and the risk of fitting to noise increases.

In **Experiment 5 (Logistic Regression)**, highly correlated features cause **multicollinearity** — the model's coefficient estimates become unstable because two features are telling it almost the same thing.

**The shared shortcoming across all Group 3 classifiers:** none of them has a built-in mechanism to handle redundant, correlated, high-dimensional input. They accept whatever features you give them and try to work with it.

**PCA solves this upstream** — before the data ever reaches a classifier. It identifies which directions in feature space carry the most information, compresses the data into those directions, and discards the rest. What enters the classifier is leaner, uncorrelated, and faster to process. Experiment 10 is the preprocessing upgrade that makes every previous classifier better.

---

## 4. Cell-by-Cell Walkthrough

---

### Cell 4 — Feature Distributions by Class (Histograms)

**What was done:** Plotted a histogram for each of the 4 features (sepal length, sepal width, petal length, petal width), with all three classes overlaid in different colors.

**What the visuals tell:**
Petal length and petal width show very clean separation — Setosa occupies a completely different range from Versicolor and Virginica. Sepal width is nearly useless: all three classes overlap almost entirely. Sepal length is partially useful but still shows heavy overlap between Versicolor and Virginica.

**Why this matters for PCA:** The features are not equally informative. PCA will naturally weight petal features more heavily in the first principal component because they carry more variance. The histogram confirmed this before the math did.

---

### Cell 5 — Correlation Heatmap

**What was done:** Computed the correlation matrix of all 4 features and displayed it as an annotated heatmap.

**What the visual tells:**
Petal length and petal width have a correlation of approximately 0.96 — nearly perfectly correlated. This means they are carrying almost identical information. Keeping both in a classifier is redundant — it doubles the noise without doubling the signal. Sepal length also correlates strongly with both petal features (r ≈ 0.87).

**Why this matters for PCA:** High correlation between features is exactly the condition PCA is designed to exploit. It identifies that these correlated features collectively point in one dominant direction — and compresses them into a single principal component. This is the visual proof that PCA will work well on this dataset.

---

### Cell 6 — Mean-Centering the Data

**What was done:** Subtracted the mean of each feature column from every row. Verified that after centering, all feature means equal approximately zero.

**The point:** PCA finds directions of maximum variance. Variance is measured relative to the mean. If the data is not centered, PCA finds the direction toward the mean of the dataset rather than the direction of actual spread — corrupting every subsequent step. Mean-centering is not optional; it is the mathematical requirement for PCA to work correctly.

---

### Cell 7 — Covariance Matrix

**What was done:** Computed the 4×4 covariance matrix from the mean-centered data using `np.cov()`, then visualized it as a heatmap.

**What the visual tells:**
The diagonal shows the variance of each individual feature. The off-diagonal values show covariance between pairs. The heatmap visually confirms what the correlation heatmap showed in Cell 5 — the petal features have very large off-diagonal values, meaning they co-vary strongly. The covariance matrix is the mathematical compression of all redundancy in the dataset. PCA then decomposes this matrix to find the redundancy's structure.

---

### Cell 8 — Eigenvalues and Eigenvectors

**What was done:** Applied `np.linalg.eigh()` to the covariance matrix to extract eigenvalues and eigenvectors.

**What this produces:**
Each eigenvector defines a direction in the original 4D feature space — a candidate axis for a principal component. Each corresponding eigenvalue tells how much variance the data has along that direction. A large eigenvalue means the data is spread out along that eigenvector — that direction carries a lot of information. A small eigenvalue means the data is tightly clustered along that direction — little information, safe to discard.

The eigenvectors are mutually orthogonal — they are perpendicular to each other. This is what makes principal components uncorrelated: they point in completely independent directions.

---

### Cell 9 — Sorting and Explained Variance

**What was done:** Sorted eigenvalues from largest to smallest. Computed the explained variance ratio for each component (eigenvalue divided by sum of all eigenvalues) and the cumulative variance.

**Key result:**
PC1 alone explains approximately 72% of total variance. PC2 explains approximately 23%. Together they explain ~95% — meaning only 5% of total information is lost by dropping PC3 and PC4. This is the quantitative justification for choosing k=2.

---

### Cell 10 — Scree Plot

**What was done:** Plotted individual explained variance per component (bar chart) and cumulative explained variance vs number of components (line chart with 90% and 95% threshold lines).

**What the visual tells:**
The bar chart makes the dominance of PC1 immediately obvious — it towers over PC3 and PC4. The cumulative line crosses the 95% threshold at k=2 and barely moves after that. This is the standard visual used in any real project to justify the choice of k. The "elbow" in the curve — where adding more components gives diminishing returns — sits clearly at k=2 for this dataset.

---

### Cell 11 — Feature Vector (PC Loadings)

**What was done:** Selected the top 2 eigenvectors as the projection matrix and printed the weight (loading) of each original feature in each principal component.

**What this reveals:**
PC1 has large weights on all petal features and sepal length — it is essentially a "size" axis. Larger values on PC1 correspond to larger flowers overall. PC2 is dominated by sepal width and acts as a "shape contrast" axis. This is the only point in PCA where you can peek inside and understand what the components represent in terms of original features — something you cannot do with the projected data itself.

---

### Cell 12 — Projecting to 2D

**What was done:** Multiplied the mean-centered data by the transpose of the feature vector to produce the final 2D representation. Original shape (150, 4) reduced to (150, 2).

**The mechanism:** Each original 150-dimensional sample (in 4D) is re-expressed as a point on the 2D coordinate system defined by PC1 and PC2. All the information that lived across 4 axes is now compressed into 2 axes, with the maximum possible variance retained.

---

### Cell 13 — 2D Scatter Plot (Manual PCA Result)

**What was done:** Scatter plotted all 150 samples in the PC1-PC2 plane, colored by species.

**What the visual tells:**
Setosa forms a tight, completely isolated cluster on the left side of the plot — no overlap whatsoever with the other two classes. Versicolor and Virginica are well-separated along PC1 but show a small region of overlap near their boundary. This overlap mirrors reality — these two species are biologically more similar than Setosa.

**The critical insight:** The original 4D data had these same 3 cluster structures, but you could not see them because human perception stops at 3D. PCA collapsed 4 dimensions into 2 while preserving the cluster structure. Any classifier trained on this 2D data will be able to separate the classes — the information needed for classification survived the compression.

---

### Cell 14–15 — sklearn PCA Cross-Check

**What was done:** Applied `StandardScaler` followed by `PCA(n_components=2)` from sklearn. Compared the explained variance ratios and scatter plot to the manual result.

**What was confirmed:**
sklearn's result shows slightly different explained variance ratios because `StandardScaler` divides by standard deviation in addition to mean-centering — giving all features equal initial scale before PCA. The manual implementation only mean-centered. Despite this difference, both methods produce scatter plots with identical cluster structure. The sign/direction of axes may differ (eigenvectors can point in opposite directions and still define the same subspace) but the relative positions of points — and therefore the cluster separation — are preserved.

---

### Cell 16 — Side-by-Side Comparison

**What was done:** Plotted manual PCA and sklearn PCA results side by side in one figure.

**What the visual tells:**
Both plots show Setosa cleanly isolated and Versicolor/Virginica partially overlapping. The axes are scaled differently and may be mirrored, but the geometry of the three clusters is identical. This confirms the manual implementation is mathematically correct and that the choice of preprocessing (mean-center vs full standardization) affects the axis scale but not the fundamental structure.

---

### Cell 17 — Full Variance Breakdown (Justifying k=2)

**What was done:** Ran `PCA(n_components=4)` to compute the complete variance breakdown across all components.

**What this confirms:**

| Component | Variance Explained | Cumulative |
|---|---|---|
| PC1 | ~72% | ~72% |
| PC2 | ~23% | ~95% |
| PC3 | ~4% | ~99% |
| PC4 | ~1% | 100% |

PC3 and PC4 together add only ~5% more information. Retaining them would add 2 dimensions to train on while contributing almost no discriminative signal — pure noise trade-off. k=2 is the correct choice for this dataset.

---

## 5. Results Summary

| Metric | Value |
|---|---|
| Original dimensions | 4 |
| Reduced dimensions | 2 |
| Variance retained (manual) | ~95% |
| Variance retained (sklearn) | ~95% |
| Setosa separability | Perfect — zero overlap |
| Versicolor / Virginica | Minor boundary overlap |

---

## 6. Conclusion

PCA successfully reduced the Iris dataset from 4 dimensions to 2 while retaining approximately 95% of total variance. The 2D scatter plot confirms that the three-class structure of the dataset is fully preserved in the reduced space — Setosa is perfectly separable and the Versicolor/Virginica boundary is clearly visible.

More broadly, this experiment establishes PCA as the solution to the shared shortcoming of every classifier in Group 3: the inability to handle correlated, high-dimensional input on its own. PCA applied upstream of any Group 3 classifier would reduce training time, remove multicollinearity, and in many cases improve generalization — particularly for SVM, which is most sensitive to both scale and dimensionality.

The manual implementation confirmed that PCA is not a black-box transformation — it is a structured eigendecomposition of the covariance matrix, where every step has a precise geometric and statistical meaning.

---

*Experiment 10 Lab Report | CS-471 Machine Learning*
