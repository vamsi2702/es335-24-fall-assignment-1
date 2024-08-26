"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""

from typing import Literal
import pandas as pd
import numpy as np

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
    
    def compute_weighted_impurity(Y, attr, unique_attr, impurity_func):
        parent_impurity = impurity_func(Y)
        weighted_impurity = sum(
            len(Y[attr == u_attr]) / len(Y) * impurity_func(Y[attr == u_attr]) 
            for u_attr in unique_attr
        )
        return parent_impurity - weighted_impurity

    is_real_attr = check_ifreal(attr)
    is_real_output = check_ifreal(Y)
    
    if is_real_attr and is_real_output:
        return compute_weighted_variance(Y, attr)

    elif not is_real_attr and is_real_output:
        parent_variance = np.var(Y)
        unique_attr = np.unique(attr)
        variance_diff = sum(
            (np.var(Y[attr == u_attr]) * len(Y[attr == u_attr])) / len(Y)
            for u_attr in unique_attr
        )
        return parent_variance - variance_diff

    elif is_real_attr and not is_real_output:
        impurity_func = entropy if criterion == "information_gain" else gini_index
        parent_impurity = impurity_func(Y)
        threshold = attr.mean()
        subsets = [Y[attr <= threshold], Y[attr > threshold]]
        weighted_impurity = sum(
            (len(subset) / len(Y)) * impurity_func(subset) 
            for subset in subsets
        )
        return parent_impurity - weighted_impurity

    else:
        impurity_func = entropy if criterion == "information_gain" else gini_index
        unique_attr = np.unique(attr)
        return compute_weighted_impurity(Y, attr, unique_attr, impurity_func)

        
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
    
    if check_ifreal(X[attribute]): # Real Input
        mask = X[attribute] <= value
    else: # Discrete Input
        mask = X[attribute] == value    

    X_left = X[mask]
    y_left = y[mask]
    X_right = X[~mask]
    y_right = y[~mask]

    return X_left, y_left, X_right, y_right
