# Program 1: Iris Dataset Exploration and Visualization

## Overview
This program demonstrates exploratory data analysis (EDA) and visualization techniques using the classic Iris dataset from scikit-learn.

## Dataset
**Iris Dataset**: A multivariate dataset introduced by Ronald Fisher containing 150 samples of iris flowers from three species (Setosa, Versicolor, and Virginica), with 4 features each:
- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)

## Files
1. **p1.ipynb** - Main program for data exploration
2. **1graph.ipynb** - Focused visualization script

## Features Implemented

### Data Exploration
- Load Iris dataset using `sklearn.datasets.load_iris()`
- Convert to pandas DataFrame for easy manipulation
- Display first 5 rows using `head()`
- Generate statistical summary using `describe()`
- Check for missing values using `isnull().sum()`
- Display dataset information using `info()`

### Visualizations
1. **Scatter Plot**: Visualizes relationship between petal length and sepal width, color-coded by species
2. **Histograms**: Distribution plots for all features
3. **Box Plot**: Shows the distribution and outliers for each feature using Seaborn

## Libraries Used
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical operations
- **matplotlib**: Basic plotting and visualization
- **seaborn**: Statistical data visualization
- **sklearn**: Dataset loading

## How to Run
```python
# Install required libraries
pip install pandas numpy matplotlib seaborn scikit-learn

# Run the notebook
jupyter notebook p1.ipynb
```

## Learning Outcomes
- Understanding basic data exploration techniques
- Identifying data distribution patterns
- Detecting outliers using box plots
- Visualizing multi-class data using scatter plots
- Statistical analysis of datasets

## Author
Vishnu Kashyap D
