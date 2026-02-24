# Program 5: Decision Tree Classification

## Objective
Implement a **Decision Tree Classifier** to classify iris flowers based on their features using the scikit-learn library.

## Overview
Decision Trees are supervised learning algorithms used for both classification and regression tasks. They work by recursively splitting the dataset based on feature values to create a tree-like structure of decisions.

## Algorithm Explained

### What is a Decision Tree?
A Decision Tree is a flowchart-like structure where:
- **Internal nodes** represent features/attributes
- **Branches** represent decision rules
- **Leaf nodes** represent outcomes/predictions

### How Decision Trees Work
1. **Select Best Feature**: Choose the feature that best splits the data based on a criterion
2. **Split Dataset**: Divide the data based on the selected feature's value
3. **Repeat Recursively**: Continue splitting until a stopping condition is met
4. **Make Predictions**: Classify new instances by following the path from root to leaf

### Key Concepts

#### 1. Splitting Criteria
- **Entropy**: Measures the impurity or randomness in the data
  - Formula: `Entropy = -Σ(p_i * log2(p_i))`
  - Lower entropy = more pure node
  - 0 = perfectly pure, 1 = maximum impurity

- **Information Gain**: Reduction in entropy after splitting
  - Higher information gain = better split

- **Gini Index**: Alternative to entropy
  - Measures probability of incorrect classification
  - Formula: `Gini = 1 - Σ(p_i)²`

#### 2. Hyperparameters
- **criterion**: Splitting criterion ('gini' or 'entropy')
- **max_depth**: Maximum depth of the tree (prevents overfitting)
- **min_samples_split**: Minimum samples required to split a node
- **min_samples_leaf**: Minimum samples required at a leaf node

