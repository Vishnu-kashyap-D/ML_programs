# Program 2: Linear Regression on California Housing Dataset

## Overview
This program implements Linear Regression to predict housing prices using the California Housing dataset. It demonstrates supervised learning for regression tasks.

## Dataset
**California Housing Dataset**: Contains information from the 1990 California census with 20,640 samples and 8 features:
- MedInc - Median income in block group
- HouseAge - Median house age in block group
- AveRooms - Average number of rooms per household
- AveBedrms - Average number of bedrooms per household
- Population - Block group population
- AveOccup - Average number of household members
- Latitude - Block group latitude
- Longitude - Block group longitude

**Target**: Median house value for California districts (in $100,000s)

## Files
1. **p2.ipynb** - Complete implementation with model coefficients
2. **2Linear.ipynb** - Simplified Linear Regression implementation

## Features Implemented

### Data Preprocessing
- Load California Housing dataset using `fetch_california_housing()`
- Convert to pandas DataFrame for manipulation
- Feature selection: Using only 'MedInc' (Median Income) as predictor
- Train-test split (80-20 ratio)
- Feature scaling using StandardScaler

### Model Training
- **Algorithm**: Linear Regression
- **Training**: Fit model on training data
- **Prediction**: Generate predictions on test data
- **Model Parameters**: Extract intercept and coefficients

### Model Evaluation
1. **R² Score**: Model accuracy using `score()` method
2. **Mean Absolute Error (MAE)**: Average absolute difference between predicted and actual values
3. **Mean Squared Error (MSE)**: Average squared difference between predicted and actual values

### Visualization
- **Scatter Plot**: Actual values (first 20 test samples)
- **Line Plot**: Predicted values overlaid in red
- Visual comparison of model predictions vs actual values

## Libraries Used
- **scikit-learn**: Machine learning algorithms and utilities
  - `LinearRegression`: Linear regression model
  - `train_test_split`: Data splitting
  - `StandardScaler`: Feature normalization
  - `fetch_california_housing`: Dataset loading
  - `mean_absolute_error`, `mean_squared_error`: Evaluation metrics
- **pandas**: Data manipulation
- **matplotlib**: Visualization

## How to Run
```python
# Install required libraries
pip install pandas matplotlib scikit-learn

# Run the notebook
jupyter notebook p2.ipynb
```

## Key Concepts
- **Linear Regression**: Finding the best-fit line to predict continuous values
- **Feature Scaling**: Normalizing features for better model performance
- **Train-Test Split**: Separating data for training and validation
- **Model Evaluation**: Using multiple metrics to assess performance

## Expected Output
- R² score indicating model accuracy
- MAE and MSE values for error quantification
- Visualization showing relationship between median income and house prices
- Model coefficients showing the linear relationship

## Learning Outcomes
- Understanding simple linear regression
- Implementing supervised learning for regression
- Feature scaling and preprocessing techniques
- Model evaluation using multiple metrics
- Interpreting regression results

## Author
Vishnu Kashyap D
