import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.tree import DecisionTreeRegressor


np.random.seed(42)


url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
data = pd.read_csv(url, delim_whitespace=True, header=None,
                       names=["mpg", "cylinders", "displacement", "horsepower", "weight",
                              "acceleration", "model_year", "origin", "car name"])

# Clean the data by eliminating unnecessary columns and rows with invalid values
# Compare the performance of your model with scikit-learn's decision tree

data = data[data['horsepower'] != '?'].reset_index(drop=True)

# Dropping the 'car name' column as it is unique for each car and might introduce high variance
data = data.drop('car name', axis=1)

# Setting the target attribute
target = data['mpg']
features = data.drop('mpg', axis=1)

features.rename(columns={'cylinders': 0}, inplace=True)
features.rename(columns={'displacement': 1}, inplace=True)
features.rename(columns={'horsepower': 2}, inplace=True)
features.rename(columns={'weight': 3}, inplace=True)
features.rename(columns={'acceleration': 4}, inplace=True)
features.rename(columns={'model_year': 5}, inplace=True)
features.rename(columns={'origin': 6}, inplace=True)

# Splitting the data into train and test sets, converting all features and target to real numbers
X_train_df = pd.DataFrame(features[:275].reset_index(drop=True), dtype=np.float64)
y_train_series = pd.Series(target[:275].reset_index(drop=True), dtype=np.float64, name=None)
X_test_df = pd.DataFrame(features[275:].reset_index(drop=True), dtype=np.float64)
y_test_series = pd.Series(target[275:].reset_index(drop=True), dtype=np.float64, name=None)

# Hyperparameters
num_folds = 8  # number of folds for cross-validation
fold_size = X_train_df.shape[0] // num_folds
possible_depths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
best_depth = 0
lowest_error = np.inf

# Finding the optimal depth for the decision tree using cross-validation for best performance
for depth in possible_depths:
    accumulated_error = 0
    for fold in range(num_folds):
        X_train_fold = pd.concat((X_train_df[:fold * fold_size], X_train_df[(fold + 1) * fold_size:]), axis=0).reset_index(drop=True)
        X_val_fold = X_train_df[fold * fold_size: (fold + 1) * fold_size].reset_index(drop=True)
        y_train_fold = pd.concat((y_train_series[:fold * fold_size], y_train_series[(fold + 1) * fold_size:]), axis=0).reset_index(drop=True)
        y_val_fold = y_train_series[fold * fold_size: (fold + 1) * fold_size].reset_index(drop=True)

        model = DecisionTreeRegressor(max_depth=depth)
        model.fit(X_train_fold, y_train_fold)
        y_pred_fold = model.predict(X_val_fold)
        accumulated_error += rmse(y_pred_fold, y_val_fold)

    if accumulated_error / num_folds < lowest_error:  # if error for this depth is minimum, update best_depth
        best_depth = depth
        lowest_error = accumulated_error / num_folds

# Applying our decision tree
custom_model = DecisionTree(criterion="information_gain", max_depth=best_depth)
custom_model.fit(X_train_df, y_train_series)
custom_y_pred = custom_model.predict(X_test_df)
custom_rmse = rmse(custom_y_pred, y_test_series)

# Applying scikit-learn's decision tree
sklearn_tree = DecisionTreeRegressor(max_depth=best_depth)
sklearn_tree.fit(X_train_df, y_train_series)
sklearn_y_pred = sklearn_tree.predict(X_test_df)
sklearn_rmse = rmse(sklearn_y_pred, y_test_series)

print("Optimal Depth:", best_depth)
print("RMSE for custom decision tree:", custom_rmse)
print("RMSE for scikit-learn decision tree:", sklearn_rmse)

# Optimal Depth: 4
# RMSE for our decision tree: 11.072494355044617
# RMSE for sklearn decision tree: 6.913069937096257