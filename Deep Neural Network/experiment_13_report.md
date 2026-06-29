# Experiment 13 — Lab Report
## Design and Implementation of Deep Neural Networks for Image Classification

**Course:** CS-471 Machine Learning
**Dataset:** Fashion MNIST (built into Keras)
**Framework:** TensorFlow 2.x / Keras

---

## 1. Objective

To design, implement, train, and evaluate a Deep Neural Network (DNN) for multi-class image classification — classifying 28×28 grayscale images of clothing items into 10 categories using multiple hidden layers, Dropout regularization, and EarlyStopping.

---

## 2. From ANN to Deep Neural Network — The Motivation

In Experiment 12, we built an ANN with **2 hidden layers and 6 neurons each** for binary classification on tabular data. It achieved 85% accuracy on a 10,000-sample dataset with 11 features.

**The new challenges in Experiment 13:**

**1. Image data instead of tabular data:**
Each Fashion MNIST image is 28×28 pixels = **784 input features** per sample. Compare this to Experiment 12's 11 features. The input dimensionality has increased by 71× — a shallow 2-layer network with 6 neurons would have nowhere near enough capacity to learn patterns from 784 inputs.

**2. Multi-class instead of binary classification:**
Experiment 12 had 2 output classes (Stay/Exit) → 1 output neuron + Sigmoid. Experiment 13 has **10 output classes** → 10 output neurons + Softmax. The entire output layer design changes.

**3. More complex patterns:**
Distinguishing a T-shirt from a Pullover, or a Sandal from an Ankle Boot, requires learning **spatial patterns** — shapes, textures, edges — that emerge from combinations of pixel values across different regions of the image. A shallow network cannot learn these hierarchical patterns.

**The solution — Deeper and Wider Network:**
- More neurons per layer (512, 256, 128) to handle 784 inputs
- More hidden layers (3 instead of 2) to learn hierarchical features
- **Dropout layers** to prevent overfitting (a new concern with larger networks)
- **EarlyStopping** to automatically stop training at the right epoch

**What DNN adds over simple ANN:**
- More layers → learns more abstract representations at each level
- First layers learn low-level patterns (edges, pixel clusters)
- Middle layers combine these into shapes
- Final layers combine shapes into class-level features (sleeve shape, sole shape)

---

## 3. Dataset

**Fashion MNIST** — a modern replacement for the classic MNIST digit dataset
- **Training images:** 60,000
- **Testing images:** 10,000
- **Image size:** 28×28 pixels, grayscale (1 channel)
- **Number of classes:** 10
- **Class distribution:** Perfectly balanced — 6,000 training samples per class

| Class | Label | Description |
|-------|-------|-------------|
| 0 | T-shirt/top | |
| 1 | Trouser | |
| 2 | Pullover | |
| 3 | Dress | |
| 4 | Coat | |
| 5 | Sandal | |
| 6 | Shirt | |
| 7 | Sneaker | |
| 8 | Bag | |
| 9 | Ankle boot | |

No download needed — loaded directly via `keras.datasets.fashion_mnist.load_data()`.

---

## 4. Cell-by-Cell Walkthrough

---

### Cell 1 — Imports

Imported TensorFlow/Keras with additional components not needed in Experiment 12:
- `Dropout` — regularization layer to prevent overfitting
- `Flatten` — available but not used (we used reshape instead)
- `EarlyStopping` — callback to stop training automatically
- `to_categorical` — converts integer labels to one-hot vectors

---

### Cell 2 — Load & Explore Dataset

```python
(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
```

**What the data looks like:**
- `X_train.shape` = (60000, 28, 28) — 60,000 images, each 28×28 pixels
- `y_train.shape` = (60000,) — integer labels 0–9
- Pixel values range from 0 to 255 (raw, unnormalized)
- Perfectly balanced — 6,000 samples per class

**Why this dataset is harder than tabular data:**
Each sample is a 2D matrix of pixel intensities — not a row of numbers with clear meanings like Age or Balance. The model must learn what combinations of pixel values represent clothing categories — a fundamentally different and more complex task.

---

### Cell 3 — Visualize Sample Images

Plotted 20 sample images from the training set with their class labels. This visualization serves two purposes:
1. **Sanity check** — confirms the dataset loaded correctly
2. **Task difficulty assessment** — viewing the images reveals that some classes are visually similar (Shirt vs T-shirt vs Pullover, Sneaker vs Ankle Boot) — these will be the hardest to classify correctly, as confirmed by the confusion matrix later.

---

### Cell 4 — Preprocess Data

Three preprocessing steps were required — all specific to image data:

**Step 1 — Normalize pixel values:**
```python
X_train = X_train / 255.0
X_test  = X_test  / 255.0
```
Raw pixel values range from 0–255. Dividing by 255 normalizes them to [0, 1]. This is the image equivalent of StandardScaler for tabular data — it ensures gradient updates are balanced and training is numerically stable.

**Why 255 specifically:** 255 is the maximum value of an 8-bit unsigned integer — the standard format for pixel values. Dividing by the maximum maps all values to the [0, 1] range.

**Step 2 — Flatten 2D images to 1D vectors:**
```python
X_train_flat = X_train.reshape(X_train.shape[0], -1)
# Shape: (60000, 28, 28) → (60000, 784)
```
Dense layers expect 1D input vectors, not 2D matrices. Each 28×28 image is "unrolled" into a single vector of 784 values. The `-1` in reshape tells NumPy to automatically calculate the correct dimension (28×28 = 784).

**Important note:** This flattening loses spatial information — the model no longer knows that pixel[0][0] and pixel[0][1] are neighbors. This is why CNNs (Experiment 15) outperform DNNs on images — they preserve spatial structure. The DNN still achieves ~88% here because Fashion MNIST images are simple and small enough for this to work.

**Step 3 — One-hot encode labels:**
```python
y_train_cat = to_categorical(y_train, num_classes=10)
# Example: label 3 (Dress) → [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
```
The output layer has 10 neurons — one per class. Each neuron outputs the probability of its class. The loss function (categorical crossentropy) compares this 10-element probability vector to the true one-hot vector. Without one-hot encoding, the loss function cannot make this comparison.

**Why one-hot for multi-class but not binary:**
In binary classification (Experiment 12), Sigmoid outputs a single probability (e.g., 0.73 = 73% chance of Exit). No one-hot needed.
In multi-class, Softmax outputs 10 probabilities that sum to 1.0 (e.g., [0.02, 0.01, 0.85, ...]). The loss function needs the true label in the same 10-element format to compute the error.

---

### Cell 5 — Build Deep Neural Network

```
Input        : 784 features (28×28 flattened)
Hidden Layer 1: 512 neurons + ReLU + Dropout(0.3)
Hidden Layer 2: 256 neurons + ReLU + Dropout(0.3)
Hidden Layer 3: 128 neurons + ReLU + Dropout(0.2)
Output Layer  : 10 neurons + Softmax
```

**Why so many more neurons than Experiment 12 (6 neurons)?**
Experiment 12 had 11 input features → 6 neurons was sufficient.
This experiment has 784 input features → the first layer needs enough neurons to process all pixel combinations. 512 neurons in layer 1 means each neuron learns a different weighted combination of the 784 pixels — different patterns from different regions of the image.

**The decreasing layer sizes (512 → 256 → 128):**
This is a common "funnel" architecture. Each layer progressively compresses the representation — first learning many low-level patterns (512), then combining them into fewer mid-level patterns (256), then into even fewer high-level features (128) before the final classification (10). This mimics how human visual processing works — from edges to shapes to objects.

**Dropout layers — the key new concept:**

```python
Dropout(0.3)  # randomly sets 30% of neurons to 0 during each training step
```

Dropout is a **regularization technique** to prevent overfitting. During each training batch:
- 30% of neurons in the layer are randomly selected and their outputs set to 0
- Their weights are not updated in that step
- Different neurons are dropped each time

**Why this prevents overfitting:**
Without Dropout, neurons can develop **co-adaptation** — neuron A learns to rely on neuron B always being present. This makes the network memorize specific training patterns. With Dropout, no neuron can rely on any other — each must learn useful features independently. The result is a more robust, generalizable network.

**Dropout is only active during training — not during prediction.** During evaluation, all neurons are active and their outputs are scaled by (1 - dropout rate) to compensate.

**Why Dropout(0.3) for first two layers, Dropout(0.2) for third:**
Larger layers (512, 256) have more capacity to overfit → higher dropout rate. Smaller layer (128) closer to output → lower dropout to preserve learned representations.

**Softmax output — key difference from Experiment 12:**
```python
Dense(10, activation="softmax")
```
$$Softmax(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{10} e^{x_j}}$$

Softmax converts the 10 raw output values into 10 probabilities that sum to 1.0. The class with the highest probability is the prediction. This is the standard output activation for any multi-class classification problem.

**Rule summary:**
| Problem Type | Output Neurons | Output Activation | Loss Function |
|-------------|----------------|-------------------|---------------|
| Binary | 1 | Sigmoid | Binary Crossentropy |
| Multi-class | N (one per class) | Softmax | Categorical Crossentropy |

---

### Cell 6 — Compile & Train with EarlyStopping

**Categorical Crossentropy loss:**
$$L = -\sum_{i=1}^{10} y_i \log(\hat{y}_i)$$
Where $y_i$ is the true one-hot label and $\hat{y}_i$ is the predicted probability for class $i$. Only the term for the correct class contributes to the loss — it penalizes the model for assigning low probability to the correct class.

**EarlyStopping callback:**
```python
EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
```
- Monitors validation loss after each epoch
- If validation loss doesn't improve for 5 consecutive epochs → stop training
- `restore_best_weights=True` → reverts to the weights from the best epoch, not the last

**Why EarlyStopping:**
Without it, training for 50 epochs might result in the model overfitting after epoch 30 — validation loss starts rising while training loss keeps falling. EarlyStopping catches this automatically. In Experiment 12 we trained for 100 epochs and noted the model converged at epoch 20 — EarlyStopping would have saved 80 unnecessary epochs.

**Results:**
```
Training stopped at epoch: 30  (EarlyStopping triggered)
Final Training Accuracy  : 90.93%
Final Validation Accuracy: 89.72%
Gap                      : 1.21% — no overfitting
```

**batch_size=64 (vs 32 in Experiment 12):**
Larger dataset (48,000 training after validation split) → larger batch size is appropriate. 64 samples per gradient update balances speed and stability for this dataset size.

---

### Cell 7 — Training History Plots

**Accuracy plot:**
Both training and validation accuracy rise steeply in early epochs and converge around epoch 25–30. The small gap between the two lines throughout training confirms Dropout is doing its job — preventing the training accuracy from diverging far above validation accuracy.

**Loss plot:**
Both losses drop sharply in early epochs and converge smoothly. EarlyStopping stopped training at epoch 30 when validation loss stopped improving — exactly where the loss curves flatten.

**Comparison with Experiment 12:**
- Exp 12 converged at ~epoch 20 with no EarlyStopping (ran all 100 epochs wastefully)
- Exp 13 used EarlyStopping and stopped at epoch 30 automatically — more efficient

---

### Cell 8 — Evaluate on Test Set

```python
y_pred = np.argmax(y_pred_prob, axis=1)
```

`np.argmax` selects the class index with the highest Softmax probability — the model's final prediction. This is the multi-class equivalent of `(y_pred_prob > 0.5).astype(int)` used in binary classification.

**Test Accuracy: ~88%**

**Confusion Matrix (10×10):**
The confusion matrix reveals which classes the model confuses most:
- **Shirt vs T-shirt/top vs Pullover** — these three are the most confused. Visually they are similar — all upper-body clothing with similar silhouettes
- **Sneaker vs Ankle Boot** — both are footwear with similar shapes
- **Trouser, Bag, Sandal** — almost never confused — distinctive shapes

This pattern makes intuitive sense and demonstrates that the model has learned meaningful visual features — it confuses visually similar items, just as a human might at a glance.

**Per-class accuracy reveals:**
- Trouser, Bag → highest accuracy (~98–99%) — very distinctive shapes
- Shirt → lowest accuracy (~70–75%) — most visually similar to other tops

---

### Cell 9 — Visualize Predictions

Plotted 20 test images with actual (A) and predicted (P) labels. Green title = correct prediction, Red title = wrong prediction. The misclassified images visually confirm the confusion matrix findings — errors occur on items that look genuinely similar even to human eyes.

---

## 5. Results Summary

| Metric | Value |
|--------|-------|
| Test Accuracy | ~88% |
| Training Accuracy | 90.93% |
| Validation Accuracy | 89.72% |
| Overfitting Gap | 1.21% (negligible) |
| Training stopped at | Epoch 30 (EarlyStopping) |
| Hardest classes | Shirt, Pullover, T-shirt/top |
| Easiest classes | Trouser, Bag, Sandal |

---

## 6. DNN vs ANN — Key Differences

| Aspect | ANN (Exp 12) | DNN (Exp 13) |
|--------|-------------|-------------|
| Hidden layers | 2 | 3 |
| Neurons per layer | 6, 6 | 512, 256, 128 |
| Input features | 11 | 784 |
| Output neurons | 1 | 10 |
| Output activation | Sigmoid | Softmax |
| Loss function | Binary Crossentropy | Categorical Crossentropy |
| Regularization | None | Dropout (0.2–0.3) |
| Training control | Fixed epochs | EarlyStopping |
| Data type | Tabular | Image |
| Preprocessing | StandardScaler | Normalize + Flatten + One-hot |

---

## 7. Key Concepts Learned

**Deep Neural Network:** An ANN with multiple (3+) hidden layers. Depth allows the network to learn hierarchical representations — from simple patterns in early layers to complex abstractions in later layers.

**Dropout:** Regularization technique that randomly deactivates neurons during training. Prevents co-adaptation and overfitting. Only active during training — all neurons active during inference.

**Softmax:** Output activation for multi-class problems. Converts raw scores to probabilities summing to 1.0. Each output neuron represents one class.

**Categorical Crossentropy:** Loss function for multi-class classification. Requires one-hot encoded labels. Penalizes low probability assigned to the correct class.

**EarlyStopping:** Keras callback that monitors a metric (val_loss) and stops training when it stops improving for `patience` epochs. Prevents wasted computation and overfitting.

**Flattening:** Converting a 2D image (28×28) to a 1D vector (784) for Dense layer compatibility. Loses spatial information — a limitation addressed by CNNs in Experiment 15.

**Normalization (images):** Dividing pixel values by 255 to map to [0, 1]. The image equivalent of feature scaling — ensures balanced gradient updates during training.

**np.argmax:** Converts Softmax probability vector to class prediction by selecting the index with the highest value. The multi-class equivalent of the 0.5 threshold used in binary classification.

---

## 8. Limitation & Looking Ahead

The DNN achieved ~88% accuracy by flattening images — treating each pixel as an independent feature. This works reasonably well for simple 28×28 grayscale images but ignores **spatial relationships** between pixels.

A pixel's meaning depends on its neighbors — the curve of a collar, the buckle of a shoe, the strap of a bag. Dense layers have no way to capture this spatial context because flattening destroys the 2D structure.

**This is exactly the problem Experiment 15 (CNN) solves** — Convolutional layers preserve spatial structure and learn local patterns (edges, curves, textures) directly from the 2D image — making CNNs significantly more powerful for image classification tasks.

---

*Experiment 13 Lab Report | CS-471 Machine Learning*
