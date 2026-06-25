"""
================================================================================
SUPPORT VECTOR MACHINES - COMPLETE THEORETICAL LECTURE
================================================================================
Course: CS-471 Machine Learning
Topic: Support Vector Machines (SVM)
Prerequisites: Linear Algebra, Calculus, Basic Machine Learning Concepts
================================================================================
"""


# ================================================================================
# 1. INTRODUCTION TO SVM
# ================================================================================

"""
1.1 What is SVM?
----------------
Support Vector Machine (SVM) is a supervised machine learning algorithm used 
primarily for classification tasks. It works by finding the optimal hyperplane 
that best separates data points from different classes.

1.2 Historical Context
----------------------
- Developed at AT&T Bell Labs by Vladimir Vapnik and Corinna Cortes (1990s)
- Gained popularity for handwritten digit recognition
- One of the most robust classification algorithms

1.3 Core Idea
-------------
The fundamental principle of SVM is to find a decision boundary that:
1. Correctly classifies all training examples
2. Maximizes the distance (margin) between the boundary and the nearest 
   points from each class
"""


# ================================================================================
# 2. FUNDAMENTAL CONCEPTS
# ================================================================================

"""
2.1 Hyperplane
--------------
A hyperplane is a decision boundary that separates data points into different 
classes.

In different dimensions:
- 1D: A point
- 2D: A line  
- 3D: A plane
- nD: An (n-1)-dimensional subspace

Mathematical Representation:
    w·x + b = 0
    
Where:
    w = weight vector (normal/perpendicular to hyperplane)
    x = feature vector
    b = bias term

2.2 Decision Rule
-----------------
For a new point x, the classification is determined by:
    If w·x + b ≥ 0  → Class 1 (+1)
    If w·x + b < 0  → Class 2 (-1)

2.3 Margin
----------
The margin is the distance between the hyperplane and the nearest data points 
from each class.

Formula:
    Margin = 2/||w||

2.4 Support Vectors
-------------------
Support vectors are the data points that lie on the margin boundaries. These 
points "support" the decision boundary.

Key Properties:
- They define the optimal hyperplane
- Only support vectors influence the model
- Removing non-support vectors doesn't change the boundary
- Makes SVM memory efficient and robust
"""


# ================================================================================
# 3. MATHEMATICAL FORMULATION
# ================================================================================

"""
3.1 Hard Margin SVM (Perfectly Separable Data)
----------------------------------------------
Optimization Problem:
    Minimize: ½||w||²
    Subject to: yᵢ(w·xᵢ + b) ≥ 1, for all i

Why ½||w||²?
- Maximizing margin = Minimizing ||w||
- ½||w||² is mathematically convenient for optimization
- Leads to a convex quadratic programming problem (guarantees global optimum)

3.2 Soft Margin SVM (Not Perfectly Separable)
---------------------------------------------
Why Soft Margin is Needed:
- Real-world data often has overlap/noise
- Perfect separation may lead to overfitting
- We need to allow some misclassifications

Slack Variables (ξᵢ):
- Allow points to violate the margin
- ξᵢ = 0: Point is correctly classified and outside margin
- 0 < ξᵢ < 1: Point is correctly classified but inside margin
- ξᵢ ≥ 1: Point is misclassified

Optimization Problem:
    Minimize: ½||w||² + C·Σξᵢ
    Subject to: yᵢ(w·xᵢ + b) ≥ 1 - ξᵢ
                ξᵢ ≥ 0 for all i

3.3 Understanding the C Parameter
---------------------------------
C = Regularization Parameter

C Value         Effect                              Result
---------       --------                            ------
Very Small      Wide margin, many violations        Underfitting (high bias)
Small           Smooth boundary, some violations    Good generalization
Large           Narrow margin, few violations       Complex boundary
Very Large      Almost hard margin                  Overfitting (high variance)

Choosing C:
- Start with C=1
- Increase if underfitting (training accuracy low)
- Decrease if overfitting (gap between train/test accuracy)
"""


# ================================================================================
# 4. THE KERNEL TRICK
# ================================================================================

"""
4.1 The Problem
---------------
Many real-world datasets are not linearly separable in their original feature 
space. Finding a linear hyperplane would result in poor classification.

4.2 The Solution: Kernel Trick
-------------------------------
Transform data to a higher-dimensional space where it becomes linearly separable.

4.3 What is a Kernel?
---------------------
A kernel is a function that computes the similarity (dot product) between two 
data points in a transformed feature space, without explicitly computing the 
transformation.

Formula:
    K(x, y) = φ(x)·φ(y)
    
Where φ is the mapping function to higher dimension.

4.4 Why It's Called a "Trick"
-----------------------------
We don't compute φ(x) explicitly. Instead, we define K(x,y) directly, which is 
computationally efficient. The kernel function implicitly performs the transformation.

4.5 Common Kernel Functions
---------------------------
"""

# A. Linear Kernel
"""
K(x,y) = x·y + c

Characteristics:
- Simplest kernel
- No transformation to higher dimension
- Fastest to compute
- Most interpretable

When to Use:
- Data is already linearly separable
- High-dimensional data (text classification)
- Need for interpretability
"""

# B. RBF (Radial Basis Function) / Gaussian Kernel
"""
K(x,y) = exp(-γ||x-y||²)

Parameters:
- γ (gamma): Controls the influence radius

Gamma Effect:
Gamma           Influence                   Boundary
-----           ---------                   --------
Small           Far points influence each    Smooth, simple
                other
Large           Only nearby points matter    Complex, detailed
Too Large       Each point only influences   Overfitting
                itself

Characteristics:
- Most versatile kernel
- Handles complex, non-linear patterns
- One of the most commonly used kernels
- Can approximate any continuous function

When to Use:
- Complex, non-linear data
- No prior knowledge about data structure
- Most real-world problems
"""

# C. Polynomial Kernel
"""
K(x,y) = (x·y + c)^d

Parameters:
- d (degree): Degree of polynomial
- c (constant): Usually set to 1

Degree Effect:
Degree      Boundary Type      Complexity
------      -------------      ----------
1           Linear             Simple
2           Quadratic          Medium
3           Cubic              Complex
>3          Very complex       Overfitting risk

When to Use:
- Data with interaction effects
- Known polynomial relationships
- Image processing tasks
"""

# D. Sigmoid Kernel
"""
K(x,y) = tanh(x·y + c)

Characteristics:
- Similar to neural network activation function
- Less commonly used
- Can behave like a 2-layer neural network

When to Use:
- Neural network-like problems
- Specialized applications
- Rarely used in practice
"""


# ================================================================================
# 5. THE DUAL PROBLEM
# ================================================================================

"""
5.1 Primal Form
---------------
The original optimization problem we saw earlier.

Hard Margin:
    Minimize: ½||w||²
    Subject to: yᵢ(w·xᵢ + b) ≥ 1

Soft Margin:
    Minimize: ½||w||² + C·Σξᵢ
    Subject to: yᵢ(w·xᵢ + b) ≥ 1 - ξᵢ

5.2 Lagrangian Formulation
--------------------------
We use Lagrange multipliers (αᵢ) to combine the objective and constraints.

Lagrangian (Hard Margin):
    L(w,b,α) = ½||w||² - Σαᵢ[yᵢ(w·xᵢ + b) - 1]

5.3 Dual Form
-------------
After setting derivatives to zero and substituting back:

Dual Optimization Problem:
    Maximize: Σαᵢ - ½ΣᵢΣⱼ αᵢαⱼ yᵢyⱼ K(xᵢ, xⱼ)
    Subject to: Σαᵢyᵢ = 0
                0 ≤ αᵢ ≤ C for all i

5.4 Advantages of Dual Form
---------------------------
1. Kernel trick becomes possible (K replaces dot product)
2. Only support vectors have αᵢ > 0
3. Provides sparse solution
4. Optimization is quadratic programming

5.5 The Sparse Solution
-----------------------
- For most points, αᵢ = 0 (not support vectors)
- Only support vectors have αᵢ > 0
- This makes SVM memory efficient
- Classification depends only on support vectors
"""


# ================================================================================
# 6. SVM DECISION BOUNDARY
# ================================================================================

"""
6.1 How SVM Finds the Optimal Boundary
---------------------------------------
1. Start with all possible hyperplanes
2. For each hyperplane, calculate the margin
3. Select the hyperplane with the maximum margin
4. This is the optimal decision boundary

6.2 Why Maximum Margin Works
----------------------------
Statistical Learning Theory:
- Maximum margin = Lower VC dimension
- Lower VC dimension = Better generalization
- Less prone to overfitting

Geometric Intuition:
- Wider margin = More room for error
- New points are more likely to be correctly classified
- Boundary is more robust to noise

6.3 Role of Support Vectors
---------------------------
- Define the margin boundaries
- If a point is not a support vector, removing it doesn't change the boundary
- Only the difficult/interesting points matter
"""


# ================================================================================
# 7. MULTICLASS CLASSIFICATION WITH SVM
# ================================================================================

"""
7.1 One-vs-Rest (OvR)
---------------------
Train K binary classifiers, one for each class.

For class k:
    - Positive examples: All points from class k
    - Negative examples: All points from other classes

Prediction:
    - Run all K classifiers
    - Choose class with highest confidence (largest distance to hyperplane)

Advantages:
    - Simple to implement
    - K classifiers only
    - Works well in practice

Disadvantages:
    - Class imbalance issues
    - Hard to calibrate confidence

7.2 One-vs-One (OvO)
--------------------
Train K(K-1)/2 binary classifiers, one for each pair of classes.

For each pair (i,j):
    - Train classifier on examples from class i and class j only

Prediction:
    - Run all K(K-1)/2 classifiers
    - Majority voting determines final class

Advantages:
    - Each classifier is trained on balanced data
    - More accurate than OvR

Disadvantages:
    - More classifiers to train (K(K-1)/2)
    - Slower prediction

7.3 Implementation in Scikit-learn
----------------------------------
# Automatically uses OvR
svm = SVC(decision_function_shape='ovr')

# Explicitly use OvO
from sklearn.multiclass import OneVsOneClassifier
svm = OneVsOneClassifier(SVC())
"""


# ================================================================================
# 8. ADVANTAGES AND LIMITATIONS
# ================================================================================

"""
8.1 Advantages
--------------
Aspect                  Explanation
---------------------   --------------------------------------------------
Effective in High       Works well when features > samples
Dimensions
Memory Efficient        Only stores support vectors
Versatile               Different kernels for different data
Robust                  Only support vectors matter, outliers have limited
                        influence
Theoretically Sound     Strong mathematical foundations
Global Optimum          Convex optimization guarantees global optimum

8.2 Limitations
---------------
Aspect                  Explanation
---------------------   --------------------------------------------------
Scalability             Training time O(n³) for full SVM
Interpretability        Hard to explain (especially with RBF)
Parameter Tuning        Many parameters to optimize (C, gamma, degree)
Memory for Large Data   All support vectors stored
No Probabilistic        Doesn't provide direct probabilities
Output
Sensitive to Scaling    Features must be normalized
"""


# ================================================================================
# 9. IMPLEMENTATION GUIDELINES
# ================================================================================

"""
9.1 Step-by-Step Process
------------------------
1. Preprocess Data
   ├── Handle missing values
   ├── Scale features (MANDATORY for SVM)
   └── Encode categorical variables

2. Split Data
   ├── Training set (usually 80%)
   └── Test set (20%)

3. Choose Kernel
   ├── Linear: Simple, interpretable
   ├── RBF: Most versatile
   └── Poly: For interaction effects

4. Tune Hyperparameters
   ├── Use GridSearchCV
   ├── Test various C values
   ├── Test various gamma values (for RBF)
   └── Use cross-validation

5. Train Model
   └── Fit on training data

6. Evaluate Model
   ├── Accuracy
   ├── Confusion Matrix
   ├── Classification Report
   └── ROC-AUC

7. Interpret Results
   ├── For linear: Check coefficients
   ├── For RBF: Compare to baseline
   └── Validate with domain knowledge

9.2 Feature Scaling
-------------------
Why Scaling is Mandatory:
    - SVM uses Euclidean distance
    - Features with larger ranges dominate
    - Leads to biased decision boundaries

Scaling Methods:
    - StandardScaler: Mean=0, Std=1 (preferred for SVM)
    - MinMaxScaler: Range [0,1] (also works)
    - RobustScaler: Uses median and IQR (if outliers present)

Rule: Always fit scaler on training data only, then transform both train and test.

9.3 Kernel Selection
--------------------
Decision Flowchart:
    Is data linearly separable?
    ├── YES → Linear Kernel
    └── NO → Continue

    Do we know the data structure?
    ├── YES → Choose based on knowledge
    │   ├── Polynomial relationships → Polynomial Kernel
    │   └── Smooth/continuous patterns → RBF Kernel
    └── NO → Continue

    Dataset size?
    ├── Small/Medium → Test all kernels
    └── Large → Use RBF (default) or Linear (high-dim)

    Final Recommendation: If unsure, ALWAYS test Linear and RBF.

9.4 Hyperparameter Tuning
-------------------------
Common Parameter Grids:

RBF Kernel:
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [0.001, 0.01, 0.1, 1]
    }

Polynomial Kernel:
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'degree': [2, 3, 4]
    }

Linear Kernel:
    param_grid = {
        'C': [0.1, 1, 10, 100]
    }

Tuning Tips:
1. Start with coarse grid (e.g., powers of 10)
2. Use cross-validation (5-fold is standard)
3. Refine around best values
4. Watch for overfitting (train vs validation accuracy)
"""


# ================================================================================
# 10. SVM VS OTHER ALGORITHMS
# ================================================================================

"""
10.1 SVM vs Logistic Regression
-------------------------------
Aspect                  SVM                     Logistic Regression
---------------------   --------------------    --------------------
Primary Goal            Maximize margin         Maximize likelihood
Decision Boundary       Depends on support      Depends on all data
                        vectors
Interpretability        Lower                   Higher
Probability Output      No (needs Platt         Yes
                        scaling)
Kernel Flexibility      Yes                     No
Outlier Sensitivity     Low                     High
Training Speed          Medium                  Fast

When to Use SVM:
    - Complex decision boundaries
    - High-dimensional data
    - Need for robust classifier
    - Interpretability not primary concern

When to Use Logistic Regression:
    - Need for probabilities
    - Interpretability required
    - Linear decision boundaries
    - Large datasets

10.2 SVM vs Neural Networks
---------------------------
Aspect                  SVM                     Neural Networks
---------------------   --------------------    --------------------
Architecture            Fixed (kernel)          Flexible (many layers)
Training                Convex optimization     Non-convex
Global Optimum          Guaranteed              Not guaranteed
Large Datasets          Challenging             Good
Interpretability        More (linear)           Less
Feature Learning        Manual (kernel)         Automatic
Training Time           Faster (for medium      Slower
                        data)

10.3 SVM vs Decision Trees
--------------------------
Aspect                  SVM                     Decision Trees
---------------------   --------------------    --------------------
Decision Boundary       Smooth                  Axis-aligned
Interpretability        Lower                   Higher
Non-linearity           Kernel trick            Automatic
Missing Values          Not handled             Can handle
Outlier Sensitivity     Low                     High
Training Time           Medium                  Fast
"""


# ================================================================================
# 11. ADVANCED TOPICS
# ================================================================================

"""
11.1 ν-SVM
----------
Alternative formulation that uses parameter ν instead of C.

Properties:
- ν ∈ (0,1]
- ν controls fraction of support vectors
- ν controls fraction of errors
- More intuitive than C

11.2 One-Class SVM
------------------
Used for anomaly detection/outlier detection.

Idea:
- Find boundary that contains most data
- Points outside boundary are anomalies
- Uses only one class for training

11.3 SVM Regression (SVR)
-------------------------
Extension to regression problems.

Idea:
- Find tube that contains most data points
- Points outside tube are penalized
- Margin is in vertical direction (errors allowed within ε)

11.4 Stochastic Gradient Descent for SVM
----------------------------------------
Approximation method for large datasets.

Advantages:
- Scales to millions of samples
- Online learning possible
- Faster than full SVM

Disadvantages:
- Loss of global optimum guarantee
- More parameters to tune
"""


# ================================================================================
# 12. COMMON PITFALLS AND SOLUTIONS
# ================================================================================

"""
12.1 Not Scaling Features
-------------------------
Problem: Features with larger ranges dominate
Solution: Always use StandardScaler or MinMaxScaler

12.2 Wrong Kernel Choice
------------------------
Problem: Poor performance or overfitting
Solution: Start with linear, then try RBF

12.3 Overfitting with RBF
-------------------------
Problem: Too complex boundary
Solution: 
- Decrease C (allow more violations)
- Decrease gamma (less influence)
- Increase cross-validation folds

12.4 Underfitting with Linear
-----------------------------
Problem: Boundary too simple
Solution:
- Try RBF or polynomial kernel
- Increase C
- Add more features

12.5 Data Leakage
-----------------
Problem: Scaler fit on test data
Solution: Fit scaler ONLY on training data

12.6 Ignoring Class Imbalance
-----------------------------
Problem: Model biased toward majority class
Solution:
- Use class_weight='balanced'
- Apply SMOTE for oversampling
- Adjust decision threshold
"""


# ================================================================================
# 13. PRACTICAL TIPS
# ================================================================================

"""
13.1 Starting Point
-------------------
1. Always scale features first
2. Start with linear kernel (baseline)
3. Then try RBF (if needed)
4. Use GridSearchCV for tuning
5. Evaluate thoroughly

13.2 When to Choose Which Kernel
--------------------------------
Scenario                                    Recommended Kernel
---------------------                       ------------------
High-dimensional data (p >> n)              Linear
Text classification                         Linear
Image classification                        RBF
Medical diagnosis                           RBF
Known polynomial relationships              Polynomial
No prior knowledge                          RBF (test linear too)

13.3 Quick Debug Checklist
--------------------------
1. ✓ Features scaled?
2. ✓ Train/test split correct?
3. ✓ No data leakage?
4. ✓ Multiple kernels tested?
5. ✓ Hyperparameters tuned?
6. ✓ Cross-validation performed?
7. ✓ Evaluation metrics appropriate?
8. ✓ Results consistent with domain knowledge?
"""


# ================================================================================
# 14. SUMMARY
# ================================================================================

"""
14.1 Key Concepts to Remember
-----------------------------
1. Goal: Find optimal hyperplane with maximum margin
2. Support Vectors: Only boundary points matter
3. Kernel Trick: Automatic non-linearity
4. C Parameter: Controls regularization (overfitting vs underfitting)
5. Gamma Parameter: Controls influence radius of support vectors
6. Feature Scaling: Absolutely mandatory for SVM
7. Dual Problem: Enables kernel trick and sparse solution

14.2 Why SVM Works Well
-----------------------
- Theoretically Sound: Convex optimization with global optimum
- Generalization: Maximum margin bounds VC dimension
- Flexibility: Kernels handle various data distributions
- Sparsity: Only support vectors influence the model

14.3 When to Use SVM
--------------------
Use SVM when:
    ✓ Small to medium datasets (<100,000 samples)
    ✓ Complex, non-linear patterns
    ✓ High-dimensional data
    ✓ Need for robustness to outliers
    ✓ Interpretability is not primary concern

Avoid SVM when:
    ✗ Very large datasets (>100,000 samples)
    ✗ Need for probabilistic outputs
    ✗ High interpretability requirements
    ✗ Very noisy data with class overlap

14.4 Final Takeaway
-------------------
SVM is one of the most powerful and robust classification algorithms in 
machine learning. Its strength lies in:
1. Finding truly optimal decision boundaries
2. Handling non-linearity automatically through kernels
3. Being robust to outliers (only support vectors matter)
4. Having strong theoretical foundations

However, it requires careful preprocessing (feature scaling is mandatory), 
kernel selection, and hyperparameter tuning. When used correctly, SVM achieves 
state-of-the-art performance on many real-world problems.
"""


# ================================================================================
# 15. REFERENCES
# ================================================================================

"""
1. Cortes, C., & Vapnik, V. (1995). Support-vector networks. 
   Machine Learning, 20(3), 273-297.

2. Vapnik, V. (1998). Statistical Learning Theory. Wiley.

3. Schölkopf, B., & Smola, A. J. (2002). Learning with Kernels. MIT Press.

4. Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer.

5. Hastie, T., Tibshirani, R., & Friedman, J. (2009). 
   The Elements of Statistical Learning. Springer.

6. Scikit-learn Documentation: 
   https://scikit-learn.org/stable/modules/svm.html
"""

# ================================================================================
# END OF THEORETICAL LECTURE
# ================================================================================