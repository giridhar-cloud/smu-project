"""
Machine Learning Practical Exercise: Logistic Regression vs Decision Trees
==================================================================================
This script implements and compares two popular machine learning models:
1. Logistic Regression - Simple, linear classification
2. Decision Tree - Flexible, non-linear classification

Author: ML Practical Exercise
Date: February 13, 2026
Dataset: Student Pass/Fail Prediction based on Study Hours and Previous Exam Score
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn import tree


def load_data():
    """Load and prepare the dataset."""
    print("=" * 60)
    print("STEP 1: DATA LOADING AND PREPARATION")
    print("=" * 60)
    
    # Sample dataset: Study hours, previous exam scores, and pass/fail labels
    data = {
        'StudyHours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'PrevExamScore': [30, 40, 45, 50, 60, 65, 70, 75, 80, 85],
        'Pass': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # 0 = Fail, 1 = Pass
    }
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    print("\nDataset Summary:")
    print(df)
    print(f"\nDataset shape: {df.shape}")
    
    return df


def split_data(df):
    """Split data into training and testing sets."""
    print("\n" + "=" * 60)
    print("STEP 2: DATA SPLITTING")
    print("=" * 60)
    
    # Features (X) and Target (y)
    X = df[['StudyHours', 'PrevExamScore']]  # Features
    y = df['Pass']  # Target variable (0 = Fail, 1 = Pass)
    
    # Split data into 80% training and 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTraining data: {X_train.shape}, {y_train.shape}")
    print(f"Testing data: {X_test.shape}, {y_test.shape}")
    print("\nTraining set:")
    print(X_train)
    print("\nTraining target:")
    print(y_train.values)
    
    return X_train, X_test, y_train, y_test


def train_logistic_regression(X_train, y_train, X_test, y_test):
    """Train and evaluate Logistic Regression model."""
    print("\n" + "=" * 60)
    print("STEP 3: LOGISTIC REGRESSION")
    print("=" * 60)
    
    # Initialize and train the Logistic Regression model
    logreg_model = LogisticRegression(random_state=42)
    logreg_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_logreg = logreg_model.predict(X_test)
    
    # Evaluate
    accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
    
    print(f"\nAccuracy: {accuracy_logreg:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_logreg))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_logreg))
    
    return logreg_model, y_pred_logreg, accuracy_logreg


def train_decision_tree(X_train, y_train, X_test, y_test):
    """Train and evaluate Decision Tree model."""
    print("\n" + "=" * 60)
    print("STEP 4: DECISION TREE")
    print("=" * 60)
    
    # Initialize and train the Decision Tree Classifier
    tree_model = DecisionTreeClassifier(random_state=42)
    tree_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_tree = tree_model.predict(X_test)
    
    # Evaluate
    accuracy_tree = accuracy_score(y_test, y_pred_tree)
    
    print(f"\nAccuracy: {accuracy_tree:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_tree))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_tree))
    
    print(f"\nDecision Tree Properties:")
    print(f"  - Tree Depth: {tree_model.get_depth()}")
    print(f"  - Number of Leaves: {tree_model.get_n_leaves()}")
    
    return tree_model, y_pred_tree, accuracy_tree


def compare_models(accuracy_logreg, accuracy_tree, y_test, y_pred_logreg, y_pred_tree):
    """Compare model performance."""
    print("\n" + "=" * 60)
    print("STEP 5: MODEL PERFORMANCE COMPARISON")
    print("=" * 60)
    
    print(f"\nLogistic Regression Accuracy: {accuracy_logreg:.4f}")
    print(f"Decision Tree Accuracy:       {accuracy_tree:.4f}")
    
    if accuracy_logreg > accuracy_tree:
        print(f"\n✓ Logistic Regression performs better by {(accuracy_logreg - accuracy_tree)*100:.2f}%")
    elif accuracy_tree > accuracy_logreg:
        print(f"\n✓ Decision Tree performs better by {(accuracy_tree - accuracy_logreg)*100:.2f}%")
    else:
        print("\n✓ Both models have equal accuracy")
    
    # Create comparison visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    models = ['Logistic\nRegression', 'Decision\nTree']
    accuracies = [accuracy_logreg, accuracy_tree]
    colors = ['#3498db', '#e74c3c']
    
    axes[0].bar(models, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Model Accuracy Comparison', fontsize=13, fontweight='bold')
    axes[0].set_ylim([0, 1])
    axes[0].grid(axis='y', alpha=0.3)
    
    for i, acc in enumerate(accuracies):
        axes[0].text(i, acc + 0.02, f'{acc:.4f}', ha='center', fontweight='bold')
    
    # Plot prediction results
    x_range = np.arange(len(y_test))
    axes[1].plot(x_range, y_test.values, 'ko-', linewidth=2, markersize=8, label='Actual', alpha=0.7)
    axes[1].plot(x_range, y_pred_logreg, 's--', linewidth=2, markersize=6, label='LogReg Pred', alpha=0.7)
    axes[1].plot(x_range, y_pred_tree, '^--', linewidth=2, markersize=6, label='Tree Pred', alpha=0.7)
    axes[1].set_xlabel('Test Sample Index', fontsize=12)
    axes[1].set_ylabel('Prediction (0=Fail, 1=Pass)', fontsize=12)
    axes[1].set_title('Predictions on Test Set', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim([-0.1, 1.1])
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✓ Comparison visualization saved as 'model_comparison.png'")
    plt.show()


def visualize_tree(tree_model):
    """Visualize the decision tree structure."""
    print("\n" + "=" * 60)
    print("STEP 6: DECISION TREE VISUALIZATION")
    print("=" * 60)
    
    plt.figure(figsize=(14, 8))
    tree.plot_tree(tree_model, 
                   feature_names=['StudyHours', 'PrevExamScore'], 
                   class_names=['Fail', 'Pass'], 
                   filled=True, 
                   rounded=True,
                   fontsize=10)
    plt.title('Decision Tree Structure for Pass/Fail Classification', 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('decision_tree_visualization.png', dpi=150, bbox_inches='tight')
    print("\n✓ Tree visualization saved as 'decision_tree_visualization.png'")
    plt.show()


def tune_decision_tree(X_train, y_train, X_test, y_test):
    """Tune decision tree to prevent overfitting."""
    print("\n" + "=" * 60)
    print("STEP 7: DECISION TREE TUNING FOR OVERFITTING PREVENTION")
    print("=" * 60)
    
    depths = range(1, 6)
    train_accuracies = []
    test_accuracies = []
    
    print("\nTesting different max_depth values:")
    for depth in depths:
        tuned_tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
        tuned_tree.fit(X_train, y_train)
        
        train_acc = accuracy_score(y_train, tuned_tree.predict(X_train))
        test_acc = accuracy_score(y_test, tuned_tree.predict(X_test))
        
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        print(f"Depth {depth}: Train Acc = {train_acc:.4f}, Test Acc = {test_acc:.4f}")
    
    # Plot tuning results
    plt.figure(figsize=(11, 5))
    plt.plot(depths, train_accuracies, 'o-', linewidth=2, markersize=8, 
             label='Training Accuracy', color='#2ecc71')
    plt.plot(depths, test_accuracies, 's-', linewidth=2, markersize=8, 
             label='Testing Accuracy', color='#e74c3c')
    plt.xlabel('Tree Depth', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Decision Tree Tuning: Impact of Max Depth on Model Performance', 
              fontsize=13, fontweight='bold')
    plt.xticks(depths)
    plt.ylim([0, 1.05])
    plt.grid(alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('tree_tuning.png', dpi=150, bbox_inches='tight')
    print("\n✓ Tuning visualization saved as 'tree_tuning.png'")
    plt.show()
    
    optimal_depth = depths[test_accuracies.index(max(test_accuracies))]
    print(f"\n✓ Optimal max_depth: {optimal_depth} (Test Accuracy: {max(test_accuracies):.4f})")


def print_analysis(accuracy_logreg, accuracy_tree):
    """Print detailed reflection and analysis."""
    print("\n" + "=" * 60)
    print("STEP 8: REFLECTION AND ANALYSIS")
    print("=" * 60)
    
    best_model = "Logistic Regression" if accuracy_logreg >= accuracy_tree else "Decision Tree"
    
    analysis = """
FINDINGS FROM THE PRACTICAL EXERCISE:
=====================================

1. MODEL PERFORMANCE ANALYSIS:
   - Logistic Regression Accuracy: {:.4f}
   - Decision Tree Accuracy:       {:.4f}
   - Best Performer:               {}

2. CHARACTERISTICS OF EACH MODEL:

   LOGISTIC REGRESSION:
   ✓ Advantages:
     • Simple, fast, and efficient
     • Works well with linearly separable data
     • Requires less computational resources
     • Less prone to overfitting
     • Provides probability estimates
   
   ✗ Disadvantages:
     • Cannot capture non-linear relationships
     • May underfit on complex datasets
     • Assumes linear decision boundaries
     • Requires feature scaling for optimal results

   DECISION TREE:
   ✓ Advantages:
     • Can capture non-linear relationships
     • Highly interpretable (visual tree structure)
     • No feature scaling required
     • Automatic feature interaction discovery
     • Handles both numerical and categorical data
   
   ✗ Disadvantages:
     • Prone to overfitting if not properly tuned
     • Training complexity O(n*log n)
     • Small data changes lead to different trees
     • Greedy algorithm may miss optimal splits

3. WHEN TO USE EACH MODEL:

   Use Logistic Regression when:
   - You need a fast, interpretable model
   - Your data has linear decision boundaries
   - You have limited training data
   - You need to minimize overfitting risk
   - Features have clear linear relationships

   Use Decision Trees when:
   - You need to capture complex patterns
   - Interpretability of decision paths is important
   - You have sufficient data to prevent overfitting
   - Features have complex interactions
   - You want automatic feature selection

4. RECOMMENDATIONS:
   For this particular problem (pass/fail prediction):
   {} is recommended because:
   - The data shows clear linear separability
   - Model simplicity outweighs complexity
   - Less prone to overfitting with limited data
   - Better expected generalization

CONCLUSION:
Both Logistic Regression and Decision Trees are valuable tools. The choice
depends on your problem characteristics, data complexity, and requirements.
Always start simple and increase complexity only when needed.
"""
    
    print(analysis.format(accuracy_logreg, accuracy_tree, best_model, best_model))


def main():
    """Main execution function."""
    print("\n" + "=" * 60)
    print("MACHINE LEARNING PRACTICAL EXERCISE")
    print("Logistic Regression vs Decision Trees")
    print("=" * 60)
    
    # Load data
    df = load_data()
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Train models
    logreg_model, y_pred_logreg, accuracy_logreg = train_logistic_regression(
        X_train, y_train, X_test, y_test
    )
    tree_model, y_pred_tree, accuracy_tree = train_decision_tree(
        X_train, y_train, X_test, y_test
    )
    
    # Compare models
    compare_models(accuracy_logreg, accuracy_tree, y_test, y_pred_logreg, y_pred_tree)
    
    # Visualize tree
    visualize_tree(tree_model)
    
    # Tune tree
    tune_decision_tree(X_train, y_train, X_test, y_test)
    
    # Analysis
    print_analysis(accuracy_logreg, accuracy_tree)
    
    print("\n" + "=" * 60)
    print("PRACTICAL EXERCISE COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
