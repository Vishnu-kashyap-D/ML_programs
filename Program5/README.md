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

