# Program 3: Logistic Regression for Binary Classification

## Overview
This program demonstrates the implementation of Logistic Regression algorithm for binary classification using the Breast Cancer dataset from scikit-learn.

## Files

### 1. 3logistic_short.ipynb
A concise implementation of Logistic Regression that includes:
- Loading the Breast Cancer dataset
- Splitting data into training and testing sets
- Feature scaling using StandardScaler
- Training a Logistic Regression model
- Model evaluation using multiple metrics:
  - Accuracy Score
  - Precision Score
  - Recall Score
  - Confusion Matrix
- Visualization using scatter plot and confusion matrix heatmap

### 2. p3.ipynb
A detailed implementation with step-by-step approach:
- Loading the Breast Cancer dataset
- Feature selection (using only 'mean radius' and 'mean texture')
- Data preprocessing with StandardScaler
- Train-test split (80-20 ratio)
- Logistic Regression model training
- Model evaluation:
  - Accuracy Score
  - Confusion Matrix
  - Classification Report
- Visualizations including confusion matrix heatmap

## Dataset
**Breast Cancer Wisconsin Dataset**
- Binary classification problem (Malignant vs Benign)
- 30 numerical features derived from cell nuclei characteristics
- 569 samples total

## Key Concepts Demonstrated

### 1. Logistic Regression
A linear model for binary classification that:
- Uses the logistic (sigmoid) function to map predictions to probabilities
- Outputs values between 0 and 1
- Uses a decision boundary (typically 0.5) to classify samples

### 2. Feature Scaling
- StandardScaler normalizes features to have mean=0 and std=1
- Important for logistic regression as it uses gradient descent
- Ensures all features contribute equally to the model

### 3. Evaluation Metrics
- **Accuracy**: Overall correctness of predictions
- **Precision**: How many predicted positives are actually positive
- **Recall**: How many actual positives were correctly predicted
- **Confusion Matrix**: Shows true positives, true negatives, false positives, and false negatives

## How to Run
1. Ensure you have the required libraries installed:
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn
   ```

2. Open the notebook files in Jupyter Notebook or VS Code

3. Run all cells sequentially

## Expected Output
- Model accuracy typically ranges between 90-95%
- Confusion matrix showing classification results
- Visualizations of model performance

## Author
Vishnu Kashyap D
vishnukashyapd18@gmail.com

## Repository
https://github.com/Vishnu-kashyap-D/ML_programs.git
