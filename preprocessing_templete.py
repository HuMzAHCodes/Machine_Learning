"""
================================================================================
DATA PREPROCESSING PIPELINE - COMPLETE TEMPLATE
================================================================================
A systematic approach to preparing data for machine learning models
================================================================================
"""


# ================================================================================
# PHASE 1: INITIAL DATA EXPLORATION
# ================================================================================

"""
1.1 Load and Inspect Dataset
----------------------------
Actions:
    - Load dataset using pandas (read_csv, read_excel, etc.)
    - Display dataset shape (rows, columns)
    - List all column names
    - Show first 5 rows (head)
    - Show last 5 rows (tail)
    - Check data types of each column

Purpose:
    - Understand data structure
    - Identify potential issues early
    - Plan preprocessing strategy
"""

"""
1.2 Basic Statistics
--------------------
Actions:
    - Generate statistical summary (describe)
    - Display data information (info)
    - Count missing values per column
    - Calculate percentage of missing values
    - Count duplicate rows

Purpose:
    - Understand data distribution
    - Identify data quality issues
    - Plan missing value handling
"""


# ================================================================================
# PHASE 2: DATA QUALITY ASSESSMENT
# ================================================================================

"""
2.1 Visual Data Exploration
---------------------------
Visualizations:
    - Histograms: Distribution of numerical features
    - Box plots: Outlier detection
    - Count plots: Categorical variable distribution
    - Scatter plots: Relationships between features
    - Pair plots: Multi-feature relationships

Purpose:
    - Visual pattern identification
    - Outlier detection
    - Feature relationship understanding
"""

"""
2.2 Outlier Detection
---------------------
Methods:

    1. IQR (Interquartile Range) Method:
       - Calculate Q1 (25th percentile) and Q3 (75th percentile)
       - IQR = Q3 - Q1
       - Lower bound = Q1 - 1.5 * IQR
       - Upper bound = Q3 + 1.5 * IQR
       - Points outside bounds are outliers
       
       When to Use:
           - Most common method
           - Distribution independent
           - Good for skewed data

    2. Z-Score Method:
       - Calculate z-score = (x - mean) / std
       - Points with |z-score| > 3 are outliers
       
       When to Use:
           - When data is normally distributed
           - Symmetric data
           - Mathematical rigor needed

    3. Visualization:
       - Box plots for visual detection
       - Scatter plots for multi-dimensional outliers
       
       When to Use:
           - Initial exploration
           - Presentation purposes
           - Quick assessment

Outlier Handling Strategies:
    1. Cap/Winsorize: Replace outliers with boundary values
       When to Use: When outliers are valid but extreme

    2. Remove: Delete outlier rows
       When to Use: When outliers are errors or very few

    3. Transform: Apply log/box-cox transformation
       When to Use: When outliers are due to skewness

    4. Keep: Leave outliers as is
       When to Use: When outliers are valid and important
"""

"""
2.3 Correlation Analysis
------------------------
Methods:
    - Pearson Correlation: Linear relationships
      When to Use: Continuous, normally distributed variables

    - Spearman Correlation: Monotonic relationships
      When to Use: Ordinal data or non-normal distributions

    - Kendall Correlation: Rank-based
      When to Use: Small datasets or many ties

Actions:
    - Generate correlation matrix
    - Visualize with heatmap
    - Identify highly correlated pairs (|correlation| > 0.8)
    - Check correlation with target variable

Purpose:
    - Identify redundant features
    - Understand feature relationships
    - Guide feature selection
"""

"""
2.4 Data Distribution Analysis
------------------------------
Actions:
    - Calculate skewness for each feature
    - Interpret skewness values
    - Identify features needing transformation

Skewness Interpretation:
    - Skewness > 1: Highly skewed right (positive skew)
    - Skewness < -1: Highly skewed left (negative skew)
    - 0.5 < Skewness < 1: Moderately skewed right
    - -1 < Skewness < -0.5: Moderately skewed left
    - -0.5 < Skewness < 0.5: Approximately symmetric

Transformation Recommendations:
    - Right Skewed: Log, Square Root, Box-Cox
    - Left Skewed: Square, Exponential, Box-Cox
    - Symmetric: No transformation needed
"""


# ================================================================================
# PHASE 3: DATA CLEANING
# ================================================================================

"""
3.1 Handle Missing Values
-------------------------
For Numerical Columns:
    1. Mean Imputation:
       - Replace missing with mean value
       When to Use: Symmetric distribution, no outliers

    2. Median Imputation:
       - Replace missing with median value
       When to Use: Skewed distribution, outliers present

    3. Mode Imputation:
       - Replace missing with most frequent value
       When to Use: Discrete numerical values

    4. Forward Fill:
       - Use previous value
       When to Use: Time series data

    5. Backward Fill:
       - Use next value
       When to Use: Time series data

    6. Interpolation:
       - Linear or polynomial interpolation
       When to Use: Time series with smooth trends

    7. Remove Row:
       - Delete rows with missing values
       When to Use: Few missing values (<5%)

    8. Remove Column:
       - Delete entire column
       When to Use: >50% missing values

For Categorical Columns:
    1. Mode Imputation:
       - Replace with most frequent category
       When to Use: Most common method

    2. New Category:
       - Create "Missing" category
       When to Use: Missing has meaning

    3. Forward/Backward Fill:
       - Use neighboring values
       When to Use: Ordered categorical data

    4. Remove Row:
       - Delete rows with missing values
       When to Use: Few missing values

    5. Remove Column:
       - Delete entire column
       When to Use: >50% missing values
"""

"""
3.2 Handle Duplicates
---------------------
Actions:
    - Check for duplicate rows
    - Display sample duplicates
    - Remove duplicate rows (drop_duplicates)
    - Keep first or last occurrence

When to Remove Duplicates:
    - Duplicates are errors
    - Duplicates skew analysis
    - Duplicates represent data entry issues

When to Keep Duplicates:
    - Duplicates are valid (e.g., repeated measurements)
    - Duplicates represent multiple events
    - Duplicates are intentional
"""

"""
3.3 Handle Outliers
-------------------
Strategies:

    1. Remove Outliers:
       When to Use:
           - Outliers are data entry errors
           - Outliers are measurement errors
           - Few outliers (<1% of data)
           - Data collection errors confirmed

    2. Cap/Winsorize:
       When to Use:
           - Outliers are valid but extreme
           - Want to preserve data points
           - Domain knowledge suggests bounds

    3. Transform:
       When to Use:
           - Outliers due to skewness
           - Want to reduce impact
           - Log/box-cox transformations

    4. Separate Treatment:
       When to Use:
           - Outliers represent a separate class
           - Different behavior is meaningful
           - Want to model outliers separately

    5. Keep:
       When to Use:
           - Outliers are valid
           - Domain knowledge supports them
           - Want to maintain data integrity
"""

"""
3.4 Data Type Correction
------------------------
Actions:
    - Identify incorrect data types
    - Convert to appropriate types

Common Conversions:
    1. Categorical:
       - Convert object to category
       - Use when: Few unique values (<20)

    2. DateTime:
       - Convert string to datetime
       - Use when: Date/time information

    3. Numerical:
       - Convert object to numeric
       - Use when: Numbers stored as text

    4. Boolean:
       - Convert to bool
       - Use when: Binary values
"""


# ================================================================================
# PHASE 4: FEATURE ENGINEERING
# ================================================================================

"""
4.1 Handle Categorical Variables
--------------------------------
Encoding Methods:

    1. Label Encoding:
       - Convert categories to numbers (0,1,2,...)
       When to Use:
           - Ordinal categories (Low, Medium, High)
           - Binary categories (Yes/No)
           - Tree-based algorithms
       
       Caution: Don't use for nominal with linear models

    2. One-Hot Encoding:
       - Create binary column for each category
       When to Use:
           - Nominal categories (colors, countries)
           - No order between categories
           - Linear models
       
       Caution: Increases dimensionality

    3. Ordinal Encoding:
       - Assign ordered numbers (1,2,3,...)
       When to Use:
           - Ordinal categories with known order
           - Domain knowledge available

    4. Frequency Encoding:
       - Replace with frequency/count
       When to Use:
           - High cardinality
           - Categorical importance

    5. Target Encoding:
       - Replace with target mean
       When to Use:
           - High cardinality
           - Classification problems
       
       Caution: Risk of overfitting

    6. Binary Encoding:
       - Convert to binary representation
       When to Use:
           - High cardinality
           - Want to reduce dimensions

    7. BaseN Encoding:
       - Base-n representation
       When to Use:
           - High cardinality
           - Custom encoding needed

    8. Hash Encoding:
       - Hash to fixed dimensions
       When to Use:
           - Very high cardinality
           - Memory constraints
"""

"""
4.2 Feature Scaling
-------------------
Methods:

    1. StandardScaler:
       - Transform to mean=0, std=1
       Formula: (x - mean) / std
       
       When to Use:
           - SVM (MANDATORY)
           - Linear models
           - PCA
           - Distance-based algorithms
           - Neural networks
           - Gaussian assumptions
       
       Characteristics:
           - Most common
           - Preserves distribution shape
           - Sensitive to outliers

    2. MinMaxScaler:
       - Scale to [0,1] range
       Formula: (x - min) / (max - min)
       
       When to Use:
           - No distribution assumption
           - Need bounded values
           - Neural networks (recommended)
           - Image data (pixels)
       
       Characteristics:
           - Bounded range
           - Sensitive to outliers
           - Preserves zeros

    3. RobustScaler:
       - Use median and IQR
       Formula: (x - median) / IQR
       
       When to Use:
           - Outliers present
           - Robust to outliers
           - Don't want outlier influence
       
       Characteristics:
           - Not sensitive to outliers
           - Uses robust statistics
           - Good for skewed data

    4. Normalizer:
       - Scale to unit norm
       Formula: x / ||x||
       
       When to Use:
           - Magnitude-based models
           - Text data
           - Sparse data
       
       Characteristics:
           - L1 or L2 norm
           - Changes distribution

    5. PowerTransformer:
       - Make data Gaussian
       - Box-Cox or Yeo-Johnson
       
       When to Use:
           - Highly skewed data
           - Need normal distribution
       
       Characteristics:
           - Handles skewness
           - Improves normality

    6. QuantileTransformer:
       - Mapping to normal/uniform
       
       When to Use:
           - Non-normal distributions
           - Want specific distribution
       
       Characteristics:
           - Non-linear transformation
           - Robust to outliers
"""

"""
4.3 Feature Transformation
--------------------------
Methods:

    1. Log Transformation:
       Formula: log(x) or log(1+x)
       When to Use:
           - Right skewness
           - Exponential relationships
           - Positive values only
       
       Advantages: Compresses large values

    2. Square Root:
       Formula: sqrt(x)
       When to Use:
           - Right skewness
           - Count data
           - Positive values
       
       Advantages: Stronger than log

    3. Box-Cox:
       Formula: (x^λ - 1)/λ
       When to Use:
           - Need normal distribution
           - Positive values only
           - Fine-tuned transformation
       
       Advantages: Finds optimal λ

    4. Yeo-Johnson:
       Formula: Similar to Box-Cox
       When to Use:
           - Any values (including negative)
           - Need normal distribution
       
       Advantages: Handles negative values

    5. Square:
       Formula: x²
       When to Use:
           - Left skewness
           - Symmetric around zero
       
       Advantages: Simple

    6. Exponential:
       Formula: exp(x)
       When to Use:
           - Left skewness
           - Small values
       
       Advantages: Strong transformation

    7. Reciprocal:
       Formula: 1/x
       When to Use:
           - Heavy right skewness
           - Positive values
       
       Advantages: Very strong

    8. Power:
       Formula: x^p
       When to Use:
           - Known relationships
           - Domain knowledge
       
       Advantages: Flexible
"""


# ================================================================================
# PHASE 5: DATA SPLITTING
# ================================================================================

"""
5.1 Separate Features and Target
--------------------------------
Actions:
    - Identify target column
    - Split into X (features) and y (target)
    - Verify shapes

When to Use Which:
    - Target is known: Use specific column
    - Last column: Often target in datasets
    - First column: Sometimes ID/Index
"""

"""
5.2 Train-Test Split
--------------------
Methods:

    1. Regular Split:
       - Random split into train and test
       When to Use:
           - Regression problems
           - Large datasets
       
       Parameters:
           - test_size: 0.2 (common)
           - random_state: Set for reproducibility

    2. Stratified Split:
       - Maintains class proportions
       When to Use:
           - Classification problems
           - Imbalanced data
       
       Parameters:
           - stratify: target variable

    3. Time Series Split:
       - Maintains temporal order
       When to Use:
           - Time series data
           - Sequential data
       
       Parameters:
           - Use TimeSeriesSplit

Split Ratios:
    - 80/20: Most common
    - 70/30: When data is abundant
    - 90/10: When data is scarce
    - 60/20/20: When validation set needed

Important: Always use stratify for classification to maintain class balance.
"""

"""
5.3 Validation Split
--------------------
Purpose:
    - Hyperparameter tuning
    - Model selection
    - Avoid overfitting

Methods:
    1. Hold-out Validation:
       - Single validation set
       When to Use: Large datasets

    2. K-Fold Cross-Validation:
       - K separate train/validation splits
       When to Use: Standard approach
       Common K Values: 5 or 10

    3. Stratified K-Fold:
       - K-Fold with class balance
       When to Use: Classification problems

    4. Leave-One-Out:
       - One sample for validation
       When to Use: Very small datasets

    5. Time Series Cross-Validation:
       - Temporal order maintained
       When to Use: Time series data

K-Fold Cross-Validation Rules:
    - K=5: Standard, good balance
    - K=10: More stable, more computation
    - K=3: Faster, less stable
"""


# ================================================================================
# PHASE 6: QUICK REFERENCE - WHEN TO USE WHICH TECHNIQUE
# ================================================================================

"""
SCENARIO                                    TECHNIQUE TO USE
---------------------                       --------------------------------
Missing Numerical Data (symmetric)          Mean Imputation
Missing Numerical Data (skewed)             Median Imputation
Missing Numerical Data (time series)        Forward/Backward Fill
Missing Categorical Data                    Mode Imputation
Missing Categorical Data (meaningful)       "Missing" Category

Outliers Present                            RobustScaler or Cap/Winsorize
Outliers (data errors)                      Remove Outliers
Outliers (valid but extreme)                Keep or Cap

Normal Distribution                         StandardScaler
No Distribution Assumption                  MinMaxScaler
Skewed Data                                 Log/Box-Cox Transformation
Binary Categories                           Label Encoding
Nominal Categories (few)                    One-Hot Encoding
Ordinal Categories                          Ordinal Encoding
High Cardinality Categories                 Frequency or Target Encoding

Classification Problem                      Stratified Split
Regression Problem                          Regular Split
Time Series Data                            Time Series Split
Hyperparameter Tuning                       K-Fold Cross-Validation
Small Dataset                               Leave-One-Out

SVM Algorithm                               StandardScaler (MANDATORY)
Neural Networks                             MinMaxScaler or StandardScaler
Distance-based Algorithms                   StandardScaler
Tree-based Algorithms                       No scaling needed
Linear Models                               StandardScaler

Highly Correlated Features                  Remove one or PCA
Redundant Features                          Feature Selection
High Dimensionality                         PCA or Feature Selection
Memory Constraints                          Dimensionality Reduction

Imbalanced Classification                   Stratified Split + Class Weights
Text Data                                   Normalizer (L2)
Image Data                                  MinMaxScaler (pixels [0,1])
Medical Data                                StandardScaler or RobustScaler
"""


# ================================================================================
# PHASE 7: COMMON PITFALLS TO AVOID
# ================================================================================

"""
1. Data Leakage
   - Fitting scaler on test data
   - Fitting encoders on test data
   - Feature selection before split
   
   Solution: Fit ONLY on training data

2. Improper Validation
   - No cross-validation for tuning
   - Using test for validation
   
   Solution: Use validation set for tuning

3. Ignoring Missing Values
   - Training with missing values
   - Not handling systematically
   
   Solution: Always handle missing values

4. Not Scaling Features
   - Using unscaled features for distance-based models
   
   Solution: Scale features for SVM, KNN, etc.

5. Wrong Encoding Choice
   - One-Hot for ordinal categories
   - Label for nominal categories
   
   Solution: Match encoding to category type

6. Class Imbalance
   - Not stratifying split
   - Using accuracy on imbalanced data
   
   Solution: Use stratified split and balanced metrics

7. Overfitting
   - No validation
   - Too complex models
   - Not regularizing
   
   Solution: Cross-validation, regularization

8. Underfitting
   - Too simple models
   - Not enough features
   - Wrong algorithm choice
   
   Solution: Increase model complexity
"""


# ================================================================================
# PHASE 8: PREPROCESSING CHECKLIST
# ================================================================================

"""
PHASE 1: INITIAL EXPLORATION
    [ ] Dataset loaded successfully
    [ ] Shape identified
    [ ] Column names and types known
    [ ] Basic statistics reviewed

PHASE 2: DATA QUALITY ASSESSMENT
    [ ] Missing values quantified
    [ ] Outliers detected and analyzed
    [ ] Correlation checked
    [ ] Skewness evaluated
    [ ] Visualizations created

PHASE 3: DATA CLEANING
    [ ] Missing values handled
    [ ] Duplicates removed
    [ ] Outliers processed
    [ ] Data types corrected

PHASE 4: FEATURE ENGINEERING
    [ ] Categorical variables encoded
    [ ] Features scaled appropriately
    [ ] Transformations applied if needed
    [ ] Features selected (if applicable)

PHASE 5: DATA SPLITTING
    [ ] Features and target separated
    [ ] Train/test split completed
    [ ] Validation split done (if needed)
    [ ] No data leakage confirmed
    [ ] Stratification applied (classification)

DATA QUALITY VERIFICATION
    [ ] No missing values in train/test
    [ ] All features scaled
    [ ] All categorical encoded
    [ ] Target variable clean
    [ ] Train/test sets representative
"""


# ================================================================================
# END OF PREPROCESSING TEMPLATE
# ================================================================================