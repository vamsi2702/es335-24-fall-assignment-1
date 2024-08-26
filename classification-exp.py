import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

np.random.seed(42)

# Code given in the question
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# For plotting
plt.scatter(X[:, 0], X[:, 1], c=y)

# Write the code for Q2 a) and b) below. Show your results.

features_df = pd.DataFrame(X)
labels_df = pd.DataFrame(y)

X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X, y, train_size=0.7, test_size=0.3)
train_data = pd.DataFrame(X_train_split)

classifier = DecisionTree(criterion="information_gain")
classifier.fit(train_data, pd.Series(y_train_split))
predicted_labels = classifier.predict(pd.DataFrame(X_test_split))

print("Accuracy:", accuracy(y_test_split, predicted_labels))
for feature in features_df.columns:
    print("Precision:", precision(y_test_split, predicted_labels, feature))
    print("Recall:", recall(y_test_split, predicted_labels, feature))

X_train_df = pd.DataFrame(X_train_split)
y_train_series = pd.Series(y_train_split)

num_folds = 5  # number of folds for cross-validation
fold_size = X_train_df.shape[0] // num_folds
tree_depths = tuple(range(1, 11))
best_depth = 0
lowest_error = np.inf

# Determining the best depth for the decision tree using cross-validation for optimal performance
for depth in tree_depths:
    accumulated_error = 0
    for fold in range(num_folds):
        X_train_fold = pd.concat((X_train_df[:fold * fold_size], X_train_df[(fold + 1) * fold_size:]), axis=0).reset_index(drop=True)
        X_validation_fold = X_train_df[fold * fold_size: (fold + 1) * fold_size].reset_index(drop=True)
        y_train_fold = pd.concat((y_train_series[:fold * fold_size], y_train_series[(fold + 1) * fold_size:]), axis=0).reset_index(drop=True)
        y_validation_fold = y_train_series[fold * fold_size: (fold + 1) * fold_size].reset_index(drop=True)

        model = DecisionTree(criterion="information_gain", max_depth=depth)
        model.fit(X_train_fold, y_train_fold)
        y_pred_fold = model.predict(X_validation_fold)
        accumulated_error += rmse(y_pred_fold, y_validation_fold)

    if accumulated_error / num_folds < lowest_error: 
        best_depth = depth
        lowest_error = accumulated_error / num_folds

print("Optimal Depth:", best_depth)


# Accuracy: 0.9666666666666667
# Precision: 1.0
# Recall: 0.9375
# Precision: 0.9333333333333333
# Recall: 1.0
# Optimal Depth: 1