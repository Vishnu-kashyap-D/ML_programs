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

