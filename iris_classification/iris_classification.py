"""
Iris Flower Classification Project
==================================
Multi-class classification using multiple ML algorithms to predict iris flower species.

Libraries: scikit-learn, pandas, numpy, matplotlib, seaborn
Dataset: Iris dataset (150 samples, 4 features, 3 classes)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             f1_score, precision_score, recall_score)
from sklearn.tree import plot_tree

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 70)
print("IRIS FLOWER CLASSIFICATION - MULTI-CLASS CLASSIFICATION PROJECT")
print("=" * 70)

# ==================== STEP 1: LOAD AND EXPLORE DATA ====================
print("\n" + "=" * 70)
print("STEP 1: LOADING AND EXPLORING DATA")
print("=" * 70)

iris = load_iris()
X = iris.data
y = iris.target

# Create DataFrame
df = pd.DataFrame(X, columns=iris.feature_names)
df['Species'] = iris.target_names[y]

print(f"\nDataset Shape: {df.shape}")
print(f"Features: {iris.feature_names}")
print(f"Target Classes: {iris.target_names}")
print(f"\nFirst 10 samples:")
print(df.head(10))

print(f"\nDataset Statistics:")
print(df.describe())

print(f"\nClass Distribution:")
print(df['Species'].value_counts())

# ==================== STEP 2: DATA SPLITTING ====================
print("\n" + "=" * 70)
print("STEP 2: SPLITTING DATA INTO TRAIN AND TEST SETS")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                      random_state=42, stratify=y)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")
print(f"Feature dimension: {X_train.shape[1]}")

# Feature scaling for better performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeature scaling applied (StandardScaler)")
print(f"Training data mean: {X_train_scaled.mean(axis=0).round(4)}")
print(f"Training data std: {X_train_scaled.std(axis=0).round(4)}")

# ==================== STEP 3: MODEL 1 - LOGISTIC REGRESSION ====================
print("\n" + "=" * 70)
print("STEP 3: MODEL 1 - LOGISTIC REGRESSION (Multi-class)")
print("=" * 70)

lr_model = LogisticRegression(max_iter=200, random_state=42, multi_class='multinomial')
lr_model.fit(X_train_scaled, y_train)

y_pred_lr = lr_model.predict(X_test_scaled)
y_pred_proba_lr = lr_model.predict_proba(X_test_scaled)

accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f"\nAccuracy: {accuracy_lr:.4f}")
print(f"Precision (weighted): {precision_score(y_test, y_pred_lr, average='weighted'):.4f}")
print(f"Recall (weighted): {recall_score(y_test, y_pred_lr, average='weighted'):.4f}")
print(f"F1-Score (weighted): {f1_score(y_test, y_pred_lr, average='weighted'):.4f}")

print("\nConfusion Matrix:")
cm_lr = confusion_matrix(y_test, y_pred_lr)
print(cm_lr)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr, target_names=iris.target_names))

# ==================== STEP 4: MODEL 2 - RANDOM FOREST ====================
print("\n" + "=" * 70)
print("STEP 4: MODEL 2 - RANDOM FOREST CLASSIFIER")
print("=" * 70)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
y_pred_proba_rf = rf_model.predict_proba(X_test)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"\nAccuracy: {accuracy_rf:.4f}")
print(f"Precision (weighted): {precision_score(y_test, y_pred_rf, average='weighted'):.4f}")
print(f"Recall (weighted): {recall_score(y_test, y_pred_rf, average='weighted'):.4f}")
print(f"F1-Score (weighted): {f1_score(y_test, y_pred_rf, average='weighted'):.4f}")

print("\nConfusion Matrix:")
cm_rf = confusion_matrix(y_test, y_pred_rf)
print(cm_rf)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf, target_names=iris.target_names))

print("\nFeature Importance (Random Forest):")
for name, importance in zip(iris.feature_names, rf_model.feature_importances_):
    print(f"  {name}: {importance:.4f}")

# ==================== STEP 5: MODEL 3 - SUPPORT VECTOR MACHINE ====================
print("\n" + "=" * 70)
print("STEP 5: MODEL 3 - SUPPORT VECTOR MACHINE (SVM)")
print("=" * 70)

svm_model = SVC(kernel='rbf', random_state=42, probability=True)
svm_model.fit(X_train_scaled, y_train)

y_pred_svm = svm_model.predict(X_test_scaled)
y_pred_proba_svm = svm_model.predict_proba(X_test_scaled)

accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"\nAccuracy: {accuracy_svm:.4f}")
print(f"Precision (weighted): {precision_score(y_test, y_pred_svm, average='weighted'):.4f}")
print(f"Recall (weighted): {recall_score(y_test, y_pred_svm, average='weighted'):.4f}")
print(f"F1-Score (weighted): {f1_score(y_test, y_pred_svm, average='weighted'):.4f}")

print("\nConfusion Matrix:")
cm_svm = confusion_matrix(y_test, y_pred_svm)
print(cm_svm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_svm, target_names=iris.target_names))

# ==================== STEP 6: MODEL COMPARISON ====================
print("\n" + "=" * 70)
print("STEP 6: MODEL PERFORMANCE COMPARISON")
print("=" * 70)

models_summary = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'Support Vector Machine'],
    'Accuracy': [accuracy_lr, accuracy_rf, accuracy_svm],
    'Precision': [
        precision_score(y_test, y_pred_lr, average='weighted'),
        precision_score(y_test, y_pred_rf, average='weighted'),
        precision_score(y_test, y_pred_svm, average='weighted')
    ],
    'Recall': [
        recall_score(y_test, y_pred_lr, average='weighted'),
        recall_score(y_test, y_pred_rf, average='weighted'),
        recall_score(y_test, y_pred_svm, average='weighted')
    ],
    'F1-Score': [
        f1_score(y_test, y_pred_lr, average='weighted'),
        f1_score(y_test, y_pred_rf, average='weighted'),
        f1_score(y_test, y_pred_svm, average='weighted')
    ]
})

print("\n" + models_summary.to_string())

best_model_idx = models_summary['Accuracy'].idxmax()
best_model_name = models_summary.loc[best_model_idx, 'Model']
best_accuracy = models_summary.loc[best_model_idx, 'Accuracy']

print(f"\n✓ Best Performing Model: {best_model_name}")
print(f"  Accuracy: {best_accuracy:.4f}")

# ==================== STEP 7: CROSS-VALIDATION ====================
print("\n" + "=" * 70)
print("STEP 7: CROSS-VALIDATION ANALYSIS")
print("=" * 70)

cv_scores_lr = cross_val_score(LogisticRegression(max_iter=200, random_state=42), 
                               X_train_scaled, y_train, cv=5, scoring='accuracy')
cv_scores_rf = cross_val_score(RandomForestClassifier(n_estimators=100, random_state=42), 
                               X_train, y_train, cv=5, scoring='accuracy')
cv_scores_svm = cross_val_score(SVC(kernel='rbf', random_state=42), 
                                X_train_scaled, y_train, cv=5, scoring='accuracy')

print(f"\nLogistic Regression CV Scores: {cv_scores_lr.round(4)}")
print(f"  Mean: {cv_scores_lr.mean():.4f} (+/- {cv_scores_lr.std():.4f})")

print(f"\nRandom Forest CV Scores: {cv_scores_rf.round(4)}")
print(f"  Mean: {cv_scores_rf.mean():.4f} (+/- {cv_scores_rf.std():.4f})")

print(f"\nSVM CV Scores: {cv_scores_svm.round(4)}")
print(f"  Mean: {cv_scores_svm.mean():.4f} (+/- {cv_scores_svm.std():.4f})")

# ==================== STEP 8: VISUALIZATION ====================
print("\n" + "=" * 70)
print("STEP 8: GENERATING VISUALIZATIONS")
print("=" * 70)

# 1. Model Accuracy Comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Accuracy Comparison
ax1 = axes[0, 0]
models = models_summary['Model'].tolist()
accuracies = models_summary['Accuracy'].tolist()
colors = ['#3498db', '#e74c3c', '#2ecc71']
bars = ax1.bar(models, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.set_title('Model Accuracy Comparison', fontsize=13, fontweight='bold')
ax1.set_ylim([0.90, 1.0])
ax1.grid(axis='y', alpha=0.3)
for i, (bar, acc) in enumerate(zip(bars, accuracies)):
    ax1.text(i, acc + 0.003, f'{acc:.4f}', ha='center', fontweight='bold')

# Plot 2: Confusion Matrix - Random Forest (Best Model)
ax2 = axes[0, 1]
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax2,
            xticklabels=iris.target_names, yticklabels=iris.target_names)
ax2.set_title('Confusion Matrix - Random Forest', fontsize=13, fontweight='bold')
ax2.set_ylabel('Actual', fontsize=11)
ax2.set_xlabel('Predicted', fontsize=11)

# Plot 3: All Metrics Comparison
ax3 = axes[1, 0]
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x_pos = np.arange(len(metrics))
width = 0.25

ax3.bar(x_pos - width, models_summary.iloc[0][metrics], width, label='Logistic Regression', 
        color='#3498db', alpha=0.7, edgecolor='black')
ax3.bar(x_pos, models_summary.iloc[1][metrics], width, label='Random Forest', 
        color='#e74c3c', alpha=0.7, edgecolor='black')
ax3.bar(x_pos + width, models_summary.iloc[2][metrics], width, label='SVM', 
        color='#2ecc71', alpha=0.7, edgecolor='black')

ax3.set_ylabel('Score', fontsize=12)
ax3.set_title('Performance Metrics Comparison', fontsize=13, fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(metrics)
ax3.legend(fontsize=10)
ax3.grid(axis='y', alpha=0.3)
ax3.set_ylim([0.90, 1.0])

# Plot 4: Feature Importance - Random Forest
ax4 = axes[1, 1]
feature_importance = rf_model.feature_importances_
feature_names_short = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
colors_feat = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
bars = ax4.barh(feature_names_short, feature_importance, color=colors_feat, edgecolor='black', linewidth=1.5)
ax4.set_xlabel('Importance', fontsize=12)
ax4.set_title('Feature Importance - Random Forest', fontsize=13, fontweight='bold')
ax4.grid(axis='x', alpha=0.3)
for i, (bar, imp) in enumerate(zip(bars, feature_importance)):
    ax4.text(imp + 0.01, i, f'{imp:.4f}', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('/home/azureuser/cloudfiles/code/iris_classification/iris_model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved: iris_model_comparison.png")

# 2. First Decision Tree from Random Forest
fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(rf_model.estimators_[0], 
          feature_names=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'],
          class_names=iris.target_names, 
          filled=True, 
          rounded=True,
          ax=ax)
plt.title('First Decision Tree from Random Forest Ensemble', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('/home/azureuser/cloudfiles/code/iris_classification/iris_tree_visualization.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved: iris_tree_visualization.png")

# 3. Cross-validation scores comparison
fig, ax = plt.subplots(figsize=(12, 6))
cv_data = [cv_scores_lr, cv_scores_rf, cv_scores_svm]
cv_labels = ['Logistic Regression', 'Random Forest', 'SVM']
cv_means = [cv_scores_lr.mean(), cv_scores_rf.mean(), cv_scores_svm.mean()]
cv_stds = [cv_scores_lr.std(), cv_scores_rf.std(), cv_scores_svm.std()]

x_pos = np.arange(len(cv_labels))
ax.bar(x_pos, cv_means, yerr=cv_stds, capsize=10, color=['#3498db', '#e74c3c', '#2ecc71'], 
       alpha=0.7, edgecolor='black', linewidth=2)
ax.set_ylabel('Cross-Validation Accuracy', fontsize=12)
ax.set_title('5-Fold Cross-Validation Results', fontsize=13, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(cv_labels)
ax.set_ylim([0.90, 1.0])
ax.grid(axis='y', alpha=0.3)

for i, (mean, std) in enumerate(zip(cv_means, cv_stds)):
    ax.text(i, mean + std + 0.005, f'{mean:.4f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('/home/azureuser/cloudfiles/code/iris_classification/iris_cv_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved: iris_cv_comparison.png")

# ==================== STEP 9: ANALYSIS AND INSIGHTS ====================
print("\n" + "=" * 70)
print("STEP 9: ANALYSIS AND INSIGHTS")
print("=" * 70)

print("""
KEY FINDINGS:
=============

1. DATASET CHARACTERISTICS:
   - Multi-class Classification Problem (3 iris species)
   - 150 samples with 4 features
   - Well-balanced dataset (50 samples per class)
   - Features: Sepal Length, Sepal Width, Petal Length, Petal Width

2. MODEL PERFORMANCE:
""")

print(f"   ✓ {best_model_name}: {best_accuracy:.4f}")
print(f"   - Logistic Regression: {accuracy_lr:.4f}")
print(f"   - Random Forest: {accuracy_rf:.4f}")
print(f"   - SVM: {accuracy_svm:.4f}")

print("""
3. FEATURE IMPORTANCE (Random Forest):
   The Petal features (Length and Width) are more important for
   distinguishing iris species compared to Sepal features. This makes
   biological sense as petal characteristics vary more among species.

4. MODEL CHARACTERISTICS:

   LOGISTIC REGRESSION (Multi-class):
   ✓ Pros: Simple, interpretable, fast training
   ✗ Cons: Assumes linear boundaries, may underfit on complex patterns
   
   RANDOM FOREST:
   ✓ Pros: Handles non-linear patterns, robust, provides feature importance
   ✗ Cons: More complex, requires more memory, longer training time
   
   SUPPORT VECTOR MACHINE (SVM):
   ✓ Pros: Effective in high-dimensional spaces, versatile kernels
   ✗ Cons: Requires feature scaling, slower on large datasets

5. CROSS-VALIDATION INSIGHTS:
   - All models show consistent performance across different folds
   - Low standard deviations indicate stable, reliable models
   - Limited overfitting risk with this balanced dataset

6. RECOMMENDATIONS:
   - For production: Use Random Forest (best accuracy + interpretability)
   - For speed: Use Logistic Regression (fast inference)
   - For precision: Cross-validate and ensemble predictions
   - Feature scaling is important for LR and SVM but not for RF

7. NEXT STEPS TO IMPROVE:
   - Hyperparameter tuning (GridSearchCV, RandomizedSearchCV)
   - Ensemble methods combining all three models
   - Feature engineering and selection
   - Larger dataset evaluation
""")

# ==================== STEP 10: SUMMARY ====================
print("\n" + "=" * 70)
print("STEP 10: PROJECT SUMMARY")
print("=" * 70)

summary_dict = {
    'Total Samples': len(X),
    'Training Samples': len(X_train),
    'Testing Samples': len(X_test),
    'Number of Features': X.shape[1],
    'Number of Classes': len(np.unique(y)),
    'Best Model': best_model_name,
    'Best Accuracy': f'{best_accuracy:.4f}',
    'Feature Scaling Applied': 'Yes (for LR and SVM)',
    'Cross-Validation Folds': 5
}

for key, value in summary_dict.items():
    print(f"{key:.<40} {value}")

print("\n" + "=" * 70)
print("IRIS CLASSIFICATION PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 70)
print("\nGenerated Files:")
print("  ✓ iris_model_comparison.png - Model performance visualization")
print("  ✓ iris_tree_visualization.png - Decision tree structure")
print("  ✓ iris_cv_comparison.png - Cross-validation results")
print("\n")
