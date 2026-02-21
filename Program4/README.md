# Program 4: K-Nearest Neighbors (KNN) Classification

## Overview
This program demonstrates the implementation of the K-Nearest Neighbors (KNN) algorithm for multi-class classification using the Iris dataset from scikit-learn.

## Files

### 1. 4Knn.ipynb (Short Implementation)
A concise implementation of KNN classification:
- Loads the Iris dataset using `load_iris()`
- Splits data into 80% training and 20% testing sets
- Trains a KNN classifier with **k=5** neighbors
- Evaluates the model using accuracy score
- Visualizes predictions with a scatter plot (sepal width vs petal width)
- Displays confusion matrix as a heatmap

### 2. p4.ipynb (Detailed Implementation)
A step-by-step implementation of KNN:
- Loads the Iris dataset and separates features and target
- Performs train-test split (80-20 ratio)
- Trains KNN classifier with default parameters (k=5)
- Evaluates model accuracy on the test set
- Generates scatter plot visualization of predicted classes
- Computes and visualizes confusion matrix using seaborn heatmap

## Dataset
**Iris Dataset**
- One of the most well-known datasets in machine learning
- **150 samples** with **4 features** each:
  - Sepal Length (cm)
  - Sepal Width (cm)
  - Petal Length (cm)
  - Petal Width (cm)
- **3 classes** (species):
  - Setosa (0)
  - Versicolor (1)
  - Virginica (2)
- 50 samples per class

## Key Concepts Demonstrated

### 1. K-Nearest Neighbors Algorithm
- A **non-parametric**, **instance-based** learning algorithm
- Classifies a new data point based on the majority vote of its **k nearest neighbors**
- Uses **Euclidean distance** (default) to find the nearest neighbors
- The value of **k** determines how many neighbors influence the classification
- No explicit training phase — the algorithm memorizes the entire training set

### 2. How KNN Works
1. Choose the number of neighbors (k)
2. Calculate the distance between the new point and all training points
3. Select the k closest training points
4. Assign the class that appears most frequently among the k neighbors

### 3. Choosing the Right k Value
- **Small k** (e.g., k=1): More sensitive to noise, complex decision boundary
- **Large k** (e.g., k=20): Smoother decision boundary, but may miss local patterns
- **Odd k** is preferred for binary classification to avoid ties
- Common practice: Start with k = √n (where n = number of training samples)

### 4. Evaluation Metrics Used
- **Accuracy Score**: Proportion of correctly classified samples out of total samples
- **Confusion Matrix**: A table showing:
  - True Positives, True Negatives
  - False Positives, False Negatives
  - Helps identify which classes are being confused with each other

### 5. Visualization
- **Scatter Plot**: Plots test samples using two features (sepal width vs petal width), colored by predicted class
- **Heatmap**: Visual representation of the confusion matrix using seaborn's heatmap with annotations

## Libraries Used
| Library | Purpose |
|---------|---------|
| `sklearn.datasets` | Loading the Iris dataset |
| `sklearn.neighbors` | KNeighborsClassifier |
| `sklearn.model_selection` | train_test_split |
| `sklearn.metrics` | confusion_matrix |
| `pandas` | Data manipulation |
| `matplotlib` | Scatter plot visualization |
| `seaborn` | Confusion matrix heatmap |

## How to Run
1. Install the required libraries:
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn
   ```

2. Open the notebook files in Jupyter Notebook or VS Code

3. Run all cells sequentially

## Expected Output
- Model accuracy typically ranges between **93-100%** (Iris is a relatively easy dataset)
- Scatter plot showing clusters of predicted classes
- Confusion matrix heatmap showing classification performance across all 3 classes

## Author
Vishnu Kashyap D
vishnukashyapd18@gmail.com

## Repository
https://github.com/Vishnu-kashyap-D/ML_programs.git
