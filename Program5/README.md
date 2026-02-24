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

## Dataset: Iris Dataset
The Iris dataset contains 150 samples of iris flowers with 4 features:
- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)

**Target Classes** (3 species):
1. Setosa (0)
2. Versicolor (1)
3. Virginica (2)

**In this program**, we use only 2 features:
- Sepal width (cm)
- Petal width (cm)

## Implementation Steps

### Step 1: Import Libraries
```python
from sklearn import tree
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sb
```

### Step 2: Load and Prepare Data
```python
# Load iris dataset
d = load_iris()
df = pd.DataFrame(d.data, columns=d.feature_names)

# Select only 2 features for simplicity
df = df[["sepal width (cm)", "petal width (cm)"]]
y = d.target
```

### Step 3: Split Data
```python
# 80% training, 20% testing
x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.2)
```

### Step 4: Create and Train Model
```python
# Create Decision Tree with entropy criterion
model = DecisionTreeClassifier(criterion='entropy', max_depth=4)
model.fit(x_train, y_train)
```

**Parameters Used**:
- `criterion='entropy'`: Uses information gain for splitting
- `max_depth=4`: Limits tree depth to prevent overfitting

### Step 5: Evaluate Model
```python
# Calculate accuracy
score = model.score(x_test, y_test)
print(f"Accuracy: {score * 100:.2f}%")
```

### Step 6: Visualize Results

#### Confusion Matrix
```python
y_pred = model.predict(x_test)
sb.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Blues')
```

The confusion matrix shows:
- **Rows**: Actual classes
- **Columns**: Predicted classes
- **Diagonal**: Correct predictions
- **Off-diagonal**: Misclassifications

#### Decision Tree Visualization
```python
tree.plot_tree(model, feature_names=df.columns)
```

This displays the complete tree structure showing:
- Decision rules at each node
- Entropy values
- Sample counts
- Class distributions

## Advantages of Decision Trees
1. **Easy to Understand**: Tree structure is intuitive and visual
2. **No Feature Scaling**: Works with features on different scales
3. **Handles Non-linear Relationships**: Can capture complex patterns
4. **Feature Importance**: Shows which features are most important
5. **Handles Mixed Data**: Works with both numerical and categorical data

## Disadvantages of Decision Trees
1. **Overfitting**: Can create overly complex trees
2. **Instability**: Small data changes can create different trees
3. **Bias**: Biased toward features with more levels
4. **Not Optimal**: Greedy algorithm may not find globally optimal tree

## Preventing Overfitting
1. **Pruning**: Remove branches that don't improve accuracy
2. **Max Depth**: Limit tree depth
3. **Min Samples Split**: Require minimum samples to split
4. **Min Samples Leaf**: Require minimum samples in leaf nodes

## Files in this Program
- **5DT.ipynb**: Main Decision Tree implementation with detailed steps
- **p5.ipynb**: Alternate implementation of Decision Tree classifier
- **README.md**: This documentation file

## Expected Output
- **Accuracy**: Typically 70-95% depending on train-test split
- **Confusion Matrix**: Heatmap showing classification performance
- **Decision Tree Plot**: Visual representation of the tree structure

## How to Run
1. Open either `5DT.ipynb` or `p5.ipynb` in Jupyter Notebook
2. Execute cells sequentially from top to bottom
3. Observe the output including accuracy, confusion matrix, and tree visualization

## Learning Outcomes
After completing this program, you will understand:
- How Decision Trees make classification decisions
- The concept of entropy and information gain
- How to implement Decision Trees using scikit-learn
- How to evaluate classifier performance
- How to visualize decision boundaries and tree structure
- How to prevent overfitting using hyperparameters

## References
- [Scikit-learn Decision Trees Documentation](https://scikit-learn.org/stable/modules/tree.html)
- [Iris Dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html)
- Decision Tree Theory and Applications

---
**Author**: ML Lab Program  
**Date**: February 2026  
**Language**: Python 3.x  
**Libraries**: scikit-learn, pandas, seaborn, matplotlib
