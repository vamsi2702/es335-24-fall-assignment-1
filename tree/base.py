"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import *

np.random.seed(42)

class TreeNode():
    def __init__(self, feature=None, threshold=None, info_gain=None, value = None):

      self.feature = feature
      self.threshold = threshold
      self.info_gain = info_gain
      self.children = {}
      self.value = value
      self.split_pt= None 

@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    max_depth: int  # The maximum depth the tree can grow to

    def __init__(self, criterion, max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.root = None
        self.input_type = 'D' # Discrete is considerd default for both input and output
        self.output_type = 'D'


    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        if check_ifreal(X.iloc[:, 0]):
            self.input_type = 'R'

        if check_ifreal(y):
            self.output_type = 'R'

        self.root = self.build_tree(X, y)

    
    def build_tree(self, X: pd.DataFrame, y: pd.Series, curr_depth=0):
        """
        Function to train and construct the decision tree
        """
        if len(y.unique()) == 1 or curr_depth >= self.max_depth:
            return TreeNode(value=y.iloc[0] if self.output_type == "D" else y.mean())

        best_feature, _ = opt_split_attribute(X, y, self.criterion, X.columns)
        if best_feature is None:
            return TreeNode(value=y.mode().iloc[0] if self.output_type == "D" else y.mean())

        new_node = TreeNode(feature=best_feature)
        
        if self.input_type == "R":
            val = X[best_feature].mean()
            X_left, y_left, X_right, y_right = split_data(X, y, best_feature, val)
            if len(y_left) > 0 and len(y_right) > 0:
                new_node.split_pt = val
                new_node.children[f"<={val}"] = self.build_tree(X_left.drop(columns=best_feature), y_left, curr_depth+1)
                new_node.children[f">{val}"] = self.build_tree(X_right.drop(columns=best_feature), y_right, curr_depth+1)
        else:
            for val in X[best_feature].unique():
                X_subset, y_subset = X[X[best_feature] == val], y[X[best_feature] == val]
                new_node.children[val] = self.build_tree(X_subset.drop(columns=best_feature), y_subset, curr_depth+1)

        new_node.value = y.mode().iloc[0] if self.output_type == "D" else y.mean()
        return new_node
        
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """
        predictions = []
        for index, row in X.iterrows():
            node = self.root
            while node.children:
                feature_value = row[node.feature]
                if self.input_type == "D":
                    try:
                        node = node.children[feature_value]
                    except KeyError:
                        break
                else:
                    if feature_value <= node.split_pt:
                        node = node.children[f"<={node.split_pt}"]
                    elif feature_value > node.split_pt:
                        node = node.children[f">{node.split_pt}"]

            predictions.append(node.value)
        
        return pd.Series(predictions)
    

    def plot(self, node=None, indent="  "):
        
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """

        if node is None:
            node = self.root

        if node.children:
            if self.input_type == 'R':
                print(f"{indent}?({node.feature} <= {node.split_pt})")
            else:
                print(f"{indent}?({node.feature})")
            
            for decision, child in node.children.items():
                print(f"{indent}{decision}: ", end="")
                self.plot(child, indent + "  ")
        else:
            print(f"Class: {node.value}")