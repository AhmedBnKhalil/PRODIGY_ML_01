import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the datasets
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv')

# Identifying columns with high missing values
missing_train = train_data.isnull().mean() * 100
columns_to_drop = missing_train[missing_train > 50].index.tolist()

# Dropping columns with high missing values from both train and test datasets
train_data_clean = train_data.drop(columns=columns_to_drop)
test_data_clean = test_data.drop(columns=columns_to_drop)

# Filling missing values for remaining columns
for column in train_data_clean.columns:
    if train_data_clean[column].dtype == 'object':  # Categorical data
        mode_value = train_data_clean[column].mode()[0]
        train_data_clean[column] = train_data_clean[column].fillna(mode_value)
        if column in test_data_clean.columns:
            test_data_clean[column] = test_data_clean[column].fillna(mode_value)
    else:  # Numerical data
        median_value = train_data_clean[column].median()
        train_data_clean[column] = train_data_clean[column].fillna(median_value)
        if column in test_data_clean.columns:
            test_data_clean[column] = test_data_clean[column].fillna(median_value)

# Selecting features for the regression model
selected_features = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars', '1stFlrSF', 'FullBath', 'LotArea',
                     'YearBuilt']

# Prepare the final datasets with the selected features
X_train = train_data_clean[selected_features]
y_train = train_data_clean['SalePrice']
X_test = test_data_clean[selected_features]

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Compute the root mean squared error by comparing with the sample_submission
rmse = np.sqrt(mean_squared_error(sample_submission['SalePrice'], predictions))

# Output the RMSE and sample of predictions
print("Root Mean Squared Error:", rmse, '\n')

print("Sample Predictions:", predictions[:10])
