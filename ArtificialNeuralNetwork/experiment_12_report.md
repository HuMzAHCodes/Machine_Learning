# Experiment 12 — Lab Report
## Design and Implementation of Artificial Neural Network (ANN) for Classification

**Course:** CS-471 Machine Learning
**Dataset:** Churn Modelling Dataset (Kaggle)
**Framework:** TensorFlow 2.x / Keras

---

## 1. Objective

To design, implement, train, and evaluate an Artificial Neural Network for binary classification — predicting whether a bank customer will exit (churn) or stay, based on their demographic and financial profile.

---

## 2. From Classical ML to Neural Networks — The Motivation

In Experiments 5 through 11, we used classical machine learning models — Logistic Regression, Decision Trees, Random Forest, SVM, and KNN. These models performed well on structured tabular data but share a fundamental architectural limitation:

**They cannot learn complex, hierarchical feature interactions automatically.**

For example, in Experiment 11 (Diabetes Prediction), Random Forest achieved 75.32% — a solid result, but it required manual feature engineering and preprocessing decisions. The model treats each feature independently at each split — it cannot discover that the combination of Age + Balance + IsActiveMember creates a complex non-linear pattern that predicts churn.

**ANN solves this** through its layered architecture:
- Each neuron learns a weighted combination of inputs
- Hidden layers learn increasingly abstract representations
- The network discovers complex feature interactions **automatically** during training
- No manual feature engineering required beyond basic preprocessing

Additionally, all classical models from previous experiments are **shallow** — they have at most one level of decision-making. ANN introduces **depth** — multiple transformation layers stacked together, each refining the representation learned by the previous one.

**Why ANN over Random Forest for this task:**
- Churn prediction involves subtle interactions between financial behavior, demographics, and engagement — patterns too complex for threshold-based or linear models
- ANN's backpropagation finds these non-linear patterns across all features simultaneously
- ANN scales better to larger datasets (10,000 records here vs 768 in Experiment 11)

---

## 3. Dataset

**Churn Modelling Dataset** — Finance domain
- **10,000 records, 14 columns**
- **Target:** `Exited` (0 = stayed, 1 = exited the bank)
- **Class distribution:** ~7,963 Not Exited (80%), ~2,037 Exited (20%) — imbalanced

| Feature | Type | Description |
|---------|------|-------------|
| RowNumber | Irrelevant | Dropped |
| CustomerId | Irrelevant | Dropped |
| Surname | Irrelevant | Dropped |
| CreditScore | Numerical | Customer credit score |
| Geography | Categorical (3) | France, Spain, Germany |
| Gender | Categorical (2) | Male, Female |
| Age | Numerical | Customer age |
| Tenure | Numerical | Years with bank |
| Balance | Numerical | Account balance |
| NumOfProducts | Numerical | Number of bank products |
| HasCrCard | Binary | Has credit card |
| IsActiveMember | Binary | Active member status |
| EstimatedSalary | Numerical | Estimated annual salary |
| Exited | Target | 0=Stay, 1=Exit |

---

## 4. Cell-by-Cell Walkthrough

---

### Cell 1 — Imports

Imported TensorFlow/Keras alongside standard sklearn preprocessing tools. TensorFlow version printed to confirm correct installation.

---

### Cell 2 — Load & Explore

Loaded dataset and examined shape (10000×14), class distribution, and missing values. No missing values found. Class imbalance noted: ~80% Not Exited vs ~20% Exited — flagged as a potential recall issue for the minority class.

---

### Cell 3 — Feature Selection & Encoding

**Dropped irrelevant columns:**
`RowNumber`, `CustomerId`, `Surname` — these carry zero predictive information about customer churn. Row numbers are sequential IDs. Customer IDs are arbitrary. Surnames have no causal relationship with banking behavior.

**Encoding decisions:**
- `Gender` → **Label Encoding** (binary column — Male/Female → 1/0)
- `Geography` → **OneHot Encoding** with `drop_first=True` (3 unordered categories → k-1 rule to avoid multicollinearity)

These encoding decisions follow the same rules established in Experiments 1, 5, and 11:
> Binary columns → Label Encoding. Multi-category unordered columns → OneHot Encoding with k-1 dummy variables.

Final feature count after encoding: **11 features**

---

### Cell 4 — Split & Scale

**Split first, then scale** — maintaining the no-data-leakage rule established in Experiment 6:
- `fit_transform` on `X_train` only
- `transform` on `X_test` only

**Why StandardScaler for ANN:**
ANN uses gradient-based optimization (backpropagation + Adam optimizer). Gradients are computed as partial derivatives of the loss with respect to weights. If features are on vastly different scales, gradients for large-scale features dominate the weight updates — causing unstable, slow, or divergent training. StandardScaler normalizes all features to mean=0, std=1 — ensuring balanced gradient updates across all neurons.

**Split:** 80% training (8,000), 20% testing (2,000)

---

### Cell 5 — ANN Architecture

```
Input Layer  →  11 features
Hidden Layer 1  →  6 neurons, ReLU activation
Hidden Layer 2  →  6 neurons, ReLU activation
Output Layer    →  1 neuron, Sigmoid activation
```

**Why Sequential model:**
The Sequential API stacks layers linearly — each layer's output feeds directly into the next. This is the simplest and most appropriate architecture for tabular binary classification where data flows in one direction.

**Why ReLU (Rectified Linear Unit) for hidden layers:**
$$ReLU(x) = \max(0, x)$$
ReLU outputs the input directly if positive, otherwise 0. It solves the **vanishing gradient problem** that plagued older activation functions (sigmoid, tanh) in deep networks — gradients don't shrink to near-zero during backpropagation, allowing effective learning in deeper layers.

**Why Sigmoid for output layer:**
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$
Sigmoid outputs a value between 0 and 1 — interpretable as the probability of the customer exiting. The threshold of 0.5 converts this to a binary prediction.

**Rule:**
- Binary classification → **1 output neuron + Sigmoid**
- Multi-class classification → **N output neurons + Softmax** (one per class)

**Why 6 neurons per hidden layer:**
A common heuristic is to use a number between the input size (11) and output size (1). 6 is a reasonable middle ground — enough capacity to learn patterns without being too large and overfitting.

---

### Cell 6 — Compile ANN

```python
ann.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
```

**Adam Optimizer:**
Adam (Adaptive Moment Estimation) is an advanced variant of Stochastic Gradient Descent. It adapts the learning rate for each parameter individually based on first and second moment estimates of gradients. It converges faster and more reliably than vanilla SGD — making it the default choice for most ANN tasks.

**Binary Crossentropy Loss:**
$$L = -[y \log(\hat{y}) + (1-y) \log(1-\hat{y})]$$
The standard loss function for binary classification. It measures the difference between the predicted probability and the actual label. For multi-class problems, categorical crossentropy would be used instead.

**Rule:**
- Binary classification → **binary_crossentropy**
- Multi-class classification → **categorical_crossentropy**

---

### Cell 7 — Train ANN

```python
history = ann.fit(X_train, y_train, batch_size=32, epochs=100, validation_split=0.2)
```

**Batch size = 32:**
Instead of computing gradients on all 8,000 training samples at once (too slow) or 1 sample at a time (too noisy), mini-batches of 32 samples balance speed and stability. This is called **mini-batch gradient descent**.

**Epochs = 100:**
One epoch = one full pass through the training data. 100 epochs gives the network enough iterations to converge without over-training.

**Validation split = 0.2:**
20% of training data (1,600 samples) is held out during training to monitor generalization. If validation accuracy diverges from training accuracy, it signals overfitting.

**Results:**
```
Final Training Accuracy  : 86.37%
Final Validation Accuracy: 85.69%
```

The 0.68% gap between training and validation accuracy confirms **no overfitting** — the model generalizes well to unseen data within the training process.

---

### Cell 8 — Training History Plots

**Accuracy plot:**
Both training and validation accuracy rise steeply in the first 20 epochs, then plateau around 86% and 85.5% respectively. The two lines remain close throughout — a healthy sign. No divergence observed.

**Loss plot:**
Both training and validation loss drop steeply from ~0.60 to ~0.33 in the first 20 epochs, then converge smoothly. The validation loss never rises above the training loss — confirming no overfitting.

**Key insight from the plots:**
The model converged around epoch 20 — the remaining 80 epochs provided marginal improvement. In a production setting, **EarlyStopping** callback would terminate training at epoch ~25, saving computation while achieving the same result.

---

### Cell 9 — Evaluate on Test Set

**Test Accuracy: 85%**

**Classification Report:**
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Not Exited | 0.88 | 0.95 | 0.91 | 1607 |
| Exited | 0.71 | 0.45 | 0.55 | 393 |

**Recall for Exited class = 0.45** — the model only catches 45% of customers who actually exit.

This is the same class imbalance problem observed in Experiment 11. With 1607 Not Exited vs 393 Exited in the test set, the model is biased toward predicting the majority class.

**Business implication:**
Missing a churning customer (False Negative) means the bank loses that customer without any retention attempt — a direct revenue loss. In this business context, **Recall matters more than Precision** — the bank would prefer to flag more customers for retention outreach (some false alarms acceptable) rather than miss actual churners.

---

### Cell 10 — Save Model & Single Prediction

Model saved in `.keras` format for future use without retraining.

**Single customer prediction:**
Input: CreditScore=600, Female, Age=40, Tenure=3, Balance=60000, 2 products, HasCrCard=1, IsActiveMember=1, Salary=50000, Spain

- **Raw probability: 0.0249** → only 2.49% chance of exiting
- **Prediction: Will STAY in the bank** ✅

This demonstrates the model's real-world usability — given a new customer's profile, it outputs both a probability and a binary decision.

---

## 5. Results Summary

| Metric | Value |
|--------|-------|
| Test Accuracy | 85% |
| Training Accuracy | 86.37% |
| Validation Accuracy | 85.69% |
| Overfitting Gap | 0.68% (negligible) |
| Recall (Exited) | 0.45 (limitation) |
| Convergence Epoch | ~20 |

---

## 6. ANN vs Classical ML — Comparison

| Aspect | Classical ML (Exp 5–11) | ANN (Exp 12) |
|--------|------------------------|--------------|
| Feature interactions | Manual or limited | Learned automatically |
| Architecture depth | Shallow (1 level) | Deep (multiple layers) |
| Interpretability | High (DT, RF) to Medium (LR) | Low (black box) |
| Scaling with data | Limited | Excellent |
| Training time | Fast | Slower (100 epochs) |
| Hyperparameters | Few | Many (layers, neurons, lr) |
| Performance here | 75.32% (RF, Exp 11) | 85% |

ANN outperforms Random Forest by ~10% on this larger, more complex dataset — justifying the added complexity of neural network training.

---

## 7. Key Concepts Learned

**Neuron:** The basic unit of an ANN. Computes a weighted sum of inputs, adds a bias, and applies an activation function: $output = activation(\sum w_i x_i + b)$

**Backpropagation:** The training algorithm. Computes the gradient of the loss with respect to each weight using the chain rule, then updates weights in the direction that reduces loss.

**ReLU:** Activation function for hidden layers. Solves vanishing gradient problem. Computationally efficient.

**Sigmoid:** Activation function for binary output. Outputs probability between 0 and 1.

**Adam Optimizer:** Adaptive learning rate optimizer. Combines momentum and RMSProp. Default choice for most neural network tasks.

**Binary Crossentropy:** Loss function for binary classification. Penalizes confident wrong predictions heavily.

**Batch Size:** Number of samples processed before weight update. Mini-batch (32) balances speed and stability.

**Epoch:** One complete pass through the training dataset. Multiple epochs allow the network to refine its weights iteratively.

**Validation Split:** Portion of training data held out to monitor generalization during training — early warning system for overfitting.

---

*Experiment 12 Lab Report | CS-471 Machine Learning*
