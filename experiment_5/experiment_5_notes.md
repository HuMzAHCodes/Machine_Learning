# Experiment 5: Notes — Logistic Regression for Classification

---

## What is Logistic Regression?
Unlike Linear Regression which predicts continuous values, Logistic Regression predicts the **probability** that an input belongs to a particular class. It outputs a value between 0 and 1 using the **sigmoid function**.

**Sigmoid Function:**
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

- Output close to 1 → likely positive class (survived)
- Output close to 0 → likely negative class (did not survive)
- Threshold = 0.5 by default → if probability > 0.5, predict class 1

---

## Classification vs Regression

| | Regression | Classification |
|--|------------|----------------|
| Output | Continuous number | Category/Class |
| Example | Predict salary | Predict survived/not |
| Algorithm | Linear Regression | Logistic Regression |
| Metric | MSE, RMSE, R² | Accuracy, F1, AUC |

---

## StandardScaler vs MinMaxScaler

| | StandardScaler | MinMaxScaler |
|--|----------------|--------------|
| Formula | (X - mean) / std | (X - min) / (max - min) |
| Output range | mean=0, std=1 | [0, 1] |
| Use when | Logistic Regression, SVM, Neural Networks | When bounded range needed |
| Sensitive to outliers | Less sensitive | More sensitive |

Logistic Regression uses **StandardScaler** because it relies on gradient-based optimization which works better when features are centered around 0.

---

## Evaluation Metrics for Classification

### Accuracy
```
Accuracy = (TP + TN) / Total
```
Percentage of correctly classified samples. Can be misleading on imbalanced datasets.

### Confusion Matrix
```
                  Predicted
                  0        1
Actual  0    TN (90)   FP (15)
        1    FN (19)   TP (55)
```
- **TP** — True Positive: predicted survived, actually survived
- **TN** — True Negative: predicted not survived, actually not survived
- **FP** — False Positive: predicted survived, actually did not (Type I error)
- **FN** — False Negative: predicted not survived, actually survived (Type II error)

### Precision
```
Precision = TP / (TP + FP)
```
Of all passengers predicted to survive, how many actually did?

### Recall (Sensitivity)
```
Recall = TP / (TP + FN)
```
Of all passengers who actually survived, how many did the model catch?

### F1-Score
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
Harmonic mean of Precision and Recall. Best metric when classes are imbalanced.

---

## ROC Curve & AUC

**ROC Curve** (Receiver Operating Characteristic) plots True Positive Rate vs False Positive Rate at various thresholds.

**AUC** (Area Under Curve):
- AUC = 1.0 → perfect classifier
- AUC = 0.5 → random classifier (diagonal line)
- AUC = 0.88 → strong classifier (our result)

The further the ROC curve bows toward the top-left corner, the better the model.

---

## Feature Importance in Logistic Regression
Logistic Regression assigns a **coefficient** to each feature. The larger the absolute value, the more that feature influences the prediction.

In our Titanic model:
- `sex` had the highest coefficient → most important
- `embarked_Q` had the lowest → least important

This aligns with historical knowledge — gender was the primary factor in Titanic survival ("women and children first").

---

## Key Rules Applied
- Label Encoding for binary columns (`sex`)
- OneHot Encoding with `drop_first=True` for multi-category columns (`embarked`)
- Split before scaling to avoid data leakage
- `fit_transform` on train, `transform` on test only
- Used `StandardScaler` (not MinMaxScaler) for Logistic Regression
