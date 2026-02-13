# Machine Learning Practical Exercise: Logistic Regression vs Decision Trees

## Overview
This document summarizes the complete practical exercise implementing and comparing Logistic Regression and Decision Tree models using Scikit-learn.

---

## 1. ENVIRONMENT SETUP ✓

### Libraries Installed & Used:
- **numpy**: Numerical computing
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning algorithms and metrics
- **matplotlib**: Data visualization

All libraries imported successfully!

---

## 2. DATA PREPARATION ✓

### Dataset Summary:
The dataset contains 10 samples with 2 features and a binary classification target.

```
   StudyHours  PrevExamScore  Pass
0           1             30     0
1           2             40     0
2           3             45     0
3           4             50     0
4           5             60     0
5           6             65     1
6           7             70     1
7           8             75     1
8           9             80     1
9          10             85     1
```

**Features:**
- `StudyHours`: Number of hours spent studying (1-10)
- `PrevExamScore`: Previous exam score (30-85)

**Target:**
- `Pass`: Binary classification (0 = Fail, 1 = Pass)

### Data Split:
- **Training Data**: 8 samples (80%)
- **Testing Data**: 2 samples (20%)
- **Random State**: 42 (for reproducibility)

---

## 3. LOGISTIC REGRESSION MODEL ✓

### Model Configuration:
```python
LogisticRegression(random_state=42)
```

### Performance Metrics:

**Accuracy: 1.0000 (100%)**

**Confusion Matrix:**
```
[[1 0]
 [0 1]]
```
- True Negatives: 1
- False Positives: 0
- False Negatives: 0
- True Positives: 1

**Classification Report:**
```
              precision    recall  f1-score   support

           0       1.00      1.00      1.00         1
           1       1.00      1.00      1.00         1

    accuracy                           1.00         2
   macro avg       1.00      1.00      1.00         2
weighted avg       1.00      1.00      1.00         2
```

**Interpretation:**
- Perfect predictions on the test set
- No false positives or false negatives
- Both classes have perfect precision, recall, and F1-scores

---

## 4. DECISION TREE MODEL ✓

### Model Configuration:
```python
DecisionTreeClassifier(random_state=42)
```

### Tree Properties:
- **Tree Depth**: 1
- **Number of Leaves**: 2

### Performance Metrics:

**Accuracy: 1.0000 (100%)**

**Confusion Matrix:**
```
[[1 0]
 [0 1]]
```
- True Negatives: 1
- False Positives: 0
- False Negatives: 0
- True Positives: 1

**Classification Report:**
```
              precision    recall  f1-score   support

           0       1.00      1.00      1.00         1
           1       1.00      1.00      1.00         1

    accuracy                           1.00         2
   macro avg       1.00      1.00      1.00         2
weighted avg       1.00      1.00      1.00         2
```

**Interpretation:**
- Perfect predictions matching Logistic Regression
- Very simple tree structure (depth of 1) indicates clear linear separability
- Two leaf nodes corresponding to the two classes

---

## 5. MODEL PERFORMANCE COMPARISON ✓

### Summary:
```
Logistic Regression Accuracy: 1.0000 (100%)
Decision Tree Accuracy:       1.0000 (100%)

✓ Both models have equal accuracy
```

### Key Findings:

1. **Both models achieve perfect accuracy** on the test set
2. **No misclassifications** for either model
3. **Decision Tree is simpler** (depth 1) - indicates data is linearly separable
4. **Logistic Regression** confirms linear decision boundary

---

## 6. DECISION TREE TUNING FOR OVERFITTING PREVENTION ✓

### Hyperparameter Tuning Results:
Testing different max_depth values to find optimal configuration:

```
Depth 1: Train Acc = 1.0000, Test Acc = 1.0000
Depth 2: Train Acc = 1.0000, Test Acc = 1.00 00
Depth 3: Train Acc = 1.0000, Test Acc = 1.0000
Depth 4: Train Acc = 1.0000, Test Acc = 1.0000
Depth 5: Train Acc = 1.0000, Test Acc = 1.0000
```

### Analysis:
- **No overfitting detected** - training and test accuracies are identical
- **Optimal max_depth**: 1 (simplest model with perfect performance)
- **Recommendation**: Use max_depth=1 for this dataset (Occam's Razor principle)
- All deeper trees achieve same accuracy but add unnecessary complexity

---

## 7. REFLECTION AND ANALYSIS ✓

### 7.1 Model Performance Analysis

Both models achieved perfect classification on the test set, correctly identifying 1 fail and 1 pass case.

### 7.2 Characteristics of Each Model

#### LOGISTIC REGRESSION

**✓ Advantages:**
- Simple, fast, and efficient algorithm
- Works exceptionally well with linearly separable data
- Requires minimal computational resources
- Less prone to overfitting on small datasets
- Provides probability estimates for predictions
- Smaller model file size

**✗ Disadvantages:**
- Cannot capture non-linear relationships
- May underfit on complex, non-linear datasets
- Assumes linear decision boundaries
- Requires feature scaling for optimal performance
- Limited interpretability (harder to explain why)

#### DECISION TREE

**✓ Advantages:**
- Excellent for capturing non-linear relationships
- Highly interpretable - visual representation of decision logic
- No feature scaling required
- Automatic feature interaction discovery
- Handles both numerical and categorical features
- Easy to understand decision paths

**✗ Disadvantages:**
- Prone to overfitting if not properly tuned
- Can become unstable with small changes in data
- Greedy algorithm may miss truly optimal splits
- Can be biased toward high-cardinality features
- Requires careful pruning/depth limiting

### 7.3 Data Complexity Impact on Models

**Observations:**
1. **Clear Linear Separability**: The dataset shows distinct separation along the feature space
2. **Monotonic Relationships**: Both features show monotonic relationship with the target
3. **Small Dataset Size**: Limits tree complexity; prevents overfitting even without pruning
4. **Feature Independence**: Features don't have complex interactions requiring deep trees

**Model Impact:**
- Database separability favors Logistic Regression
- Small sample size limits decision tree depth benefits
- Both models achieve maximum performance easily

### 7.4 When to Use Each Model

#### Use Logistic Regression when:
✓ Your data is linearly separable
✓ You need fast, efficient predictions
✓ Interpretability as feature coefficients is important
✓ You have limited computing resources
✓ Probability estimates are needed
✓ Simplicity and robustness are priorities

#### Use Decision Trees when:
✓ Your data has non-linear relationships
✓ Visual interpretability of decision paths is crucial
✓ You have sufficient data to prevent overfitting
✓ Features have complex interactions
✓ You want automatic feature selection
✓ Mixed numerical and categorical features exist

### 7.5 Recommendations for This Problem

**Based on the analysis, Logistic Regression is more suitable because:**

1. **Decision Boundary is Clearly Linear**
   - The data shows perfect linear separation
   - No complex non-linear boundaries needed
   - Logistic Regression captures this efficiently

2. **Simplicity Wins**
   - Both models achieve 100% accuracy
   - Logistic Regression is simpler (fewer parameters)
   - Follows Occam's Razor principle
   - Better generalization expected on new data

3. **Training Data is Sufficient**
   - Dataset size is adequate for both models
   - No indication of underfitting
   - Simple model less likely to overfit

4. **Model Simplicity vs. Accuracy Trade-off**
   - Logistic Regression provides optimal trade-off
   - Simpler model with same accuracy
   - Better performance on unseen data expected

### 7.6 Best Practices Demonstrated

✓ **Proper Data Split**: 80/20 train-test split with random_state for reproducibility
✓ **Multiple Evaluation Metrics**: Accuracy, confusion matrix, precision, recall, F1-score
✓ **Consistent Comparison**: Both models evaluated on same test set
✓ **Hyperparameter Tuning**: Systematic testing of max_depth parameter
✓ **Overfitting Prevention**: Monitoring train/test metrics during tuning
✓ **Visual Representation**: Tree structure and performance comparison

---

## 8. CONCLUSION

### Key Takeaways:

1. **Both Logistic Regression and Decision Trees are powerful tools**, each with distinct advantages

2. **Model Selection is Problem-Dependent**:
   - For this linear dataset: **Logistic Regression is recommended**
   - For complex non-linear datasets: **Decision Trees excel**
   - For production systems: **Consider ensemble methods** (Random Forest, Gradient Boosting)

3. **Logistic Regression Strengths**:
   - Computational efficiency and speed
   - Excellent with linear relationships
   - Reduced overfitting risk
   - Better generalization on new data

4. **Decision Tree Strengths**:
   - Superior interpretability through visual trees
   - Automatic non-linear pattern discovery
   - No feature engineering required
   - Mixed data type support

5. **Best Practices for Model Selection**:
   - Always start with a simple baseline model (Logistic Regression)
   - Increase complexity only if needed
   - Use cross-validation for robust evaluation
   - Consider ensemble methods for improved performance
   - Monitor both training and test metrics to detect overfitting

### Next Steps for Enhancement:

1. **Feature Engineering**: Create interaction terms for more complex relationships
2. **Cross-Validation**: Use k-fold CV for more robust performance estimates
3. **Ensemble Methods**: Try Random Forest, Gradient Boosting for better accuracy
4. **Hyperparameter Grid Search**: Use GridSearchCV for systematic tuning
5. **Larger Dataset**: Test on larger datasets to see where each model excels
6. **Class Imbalance**: Evaluate with imbalanced classes (if applicable)

---

## 9. PRACTICAL EXERCISE STATUS

✅ **COMPLETED SUCCESSFULLY!**

All required steps have been implemented and executed:
- ✅ Environment setup and library installation
- ✅ Data loading and preparation
- ✅ Data splitting (train/test)
- ✅ Logistic Regression implementation and evaluation
- ✅ Decision Tree implementation and evaluation
- ✅ Model performance comparison
- ✅ Decision tree visualization
- ✅ Hyperparameter tuning for overfitting prevention
- ✅ Comprehensive reflection and analysis

---

**Date**: February 13, 2026  
**Notebook Location**: `/home/azureuser/cloudfiles/code/Users/2419100457/smu.ipynb`
