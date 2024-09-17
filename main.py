# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load the Boston Housing dataset from OpenML
boston = fetch_openml(name="house_prices", as_frame=True, version=1)
data = boston.frame

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Step 1: Data Preprocessing

# Select relevant features (e.g., number of rooms, lot size, year built, etc.)
features = ['GrLivArea', 'YearBuilt', 'TotalBsmtSF', 'GarageArea', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd']

# Target variable (SalePrice)
target = 'SalePrice'

# Handle missing values by filling with the median (or you can drop them)
data.fillna(data.median(numeric_only=True), inplace=True)

# Split the data into features (X) and target (y)
X = data[features]
y = data[target]

# Normalize the features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Step 3: Build Linear Regression Model
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Step 4: Make Predictions on Test Data
y_pred = model.predict(X_test)

# Step 5: Evaluate the Model

# Calculate Mean Squared Error (MSE) and R-squared (R2) score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2) Score: {r2}")

# Step 6: Visualize Predictions vs Actual Values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='b')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)
plt.xlabel('Actual House Prices')
plt.ylabel('Predicted House Prices')
plt.title('Actual vs Predicted House Prices')
plt.show()
