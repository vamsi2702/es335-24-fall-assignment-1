"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these functions are here to simply help you.
"""

from typing import Literal
import pandas as pd
import numpy as np
def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Convert categorical variables into one-hot encoded vectors.
    
    Parameters:
    - X: pd.DataFrame, input data
    
    Returns:
    - pd.DataFrame, one-hot encoded data
    """
    categorical_columns = X.select_dtypes(include=['object', 'category']).columns
    return pd.get_dummies(X, columns=categorical_columns, drop_first=True)
def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """
    try:
        return any(y % 1 != 0)  # True if any value has a non-zero decimal part
    except TypeError:
        return False 

def entropy(Y: pd.Series) -> float:
    """
    Calculate entropy for a categorical variable.

    Parameters:
    - Y: pd.Series of categorical data

    Returns:
    - Entropy value
    """
    # Count the occurrences of each unique value in the series
    value_counts = Y.value_counts()

    # Calculate the probabilities of each unique value
    probabilities = value_counts / len(Y)

    nonzero_probabilities = probabilities[probabilities > 0]

    # Calculate entropy using the formula: H(S) = -p1*log2(p1) - p2*log2(p2) - ...
    entropy_value = -np.sum(nonzero_probabilities * np.log2(nonzero_probabilities))

    return entropy_value

def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    gini = 1
    for value in np.unique(Y) :
        p = (np.sum(Y == value))/(np.size(Y)) # prob of value in series
        if p>0:
          gini -= p**2
    return gini

def information_gain(Y: pd.Series, attr: pd.Series, criterion: Literal["information_gain", "gini_index"]) -> float:
    """
    Function to calculate the information gain
    Y = target attr, attr = attr i am splitting on
    """
    
    def compute_weighted_variance(Y, attr):
        parent_var = np.var(Y)
        mean_value = attr.mean()
        group_var = [np.var(Y[attr <= mean_value]), np.var(Y[attr > mean_value])]
        group_weight = [sum(attr <= mean_value) / len(Y), sum(attr > mean_value) / len(Y)]
        return parent_var - sum(w * v for w, v in zip(group_weight, group_var))
    
    parent_impurity = entropy(Y) if criterion == "information_gain" else gini_index(Y)
    weighted_impurity = sum(
        len(Y[attr == value]) / len(Y) * (entropy(Y[attr == value]) if criterion == "information_gain" else gini_index(Y[attr == value]))
        for value in attr.unique()
    )
    return parent_impurity - weighted_impurity

def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion: Literal["information_gain", "gini_index"], features: pd.Series):
    max_gain = -float('inf') 
    best_feature = None

    for feature in features:
        attr = X[feature]
        gain = information_gain(y, attr, criterion)  
        if gain > max_gain:
            max_gain = gain
            best_feature = feature

    return best_feature, max_gain

def split_data(X: pd.DataFrame, y: pd.Series, attribute: str, value: any) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    mask = X[attribute] <= value
    X_left = X[mask]
    y_left = y[mask]
    X_right = X[~mask]
    y_right = y[~mask]
    return X_left, y_left, X_right, y_right

