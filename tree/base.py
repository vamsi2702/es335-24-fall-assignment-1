from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import *

np.random.seed(42)

class TreeNode():
    def __init__(self, feature=None, threshold=None, info_gain=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.info_gain = info_gain
        self.children = {}
        self.value = value
        self.split_pt = None 

@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]
    max_depth: int

    def __init__(self, criterion, max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.root = None
        self.output_type = 'D'  # Default output type is discrete

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        # Convert categorical inputs to one-hot encoded vectors
        X = one_hot_encoding(X)
        
        if check_ifreal(y):
            self.output_type = 'R'

        self.root = self.build_tree(X, y)

    def build_tree(self, X: pd.DataFrame, y: pd.Series, curr_depth=0):
        if len(y.unique()) == 1 or curr_depth >= self.max_depth:
            return TreeNode(value=y.iloc[0] if self.output_type == "D" else y.mean())

        best_feature, _ = opt_split_attribute(X, y, self.criterion, X.columns)
        if best_feature is None:
            return TreeNode(value=y.mode().iloc[0] if self.output_type == "D" else y.mean())

        new_node = TreeNode(feature=best_feature)
        
        val = X[best_feature].mean()
        X_left, y_left, X_right, y_right = split_data(X, y, best_feature, val)
        if len(y_left) > 0 and len(y_right) > 0:
            new_node.split_pt = val
            new_node.children[f"<={val}"] = self.build_tree(X_left.drop(columns=best_feature), y_left, curr_depth+1)
            new_node.children[f">{val}"] = self.build_tree(X_right.drop(columns=best_feature), y_right, curr_depth+1)

        new_node.value = y.mode().iloc[0] if self.output_type == "D" else y.mean()
        return new_node
        
    def predict(self, X: pd.DataFrame) -> pd.Series:
        # Convert categorical inputs to one-hot encoded vectors
        X = one_hot_encoding(X)
        
        predictions = []
        for index, row in X.iterrows():
            node = self.root
            while node.children:
                feature_value = row[node.feature]
                if feature_value <= node.split_pt:
                    node = node.children[f"<={node.split_pt}"]
                else:
                    node = node.children[f">{node.split_pt}"]

            predictions.append(node.value)
        
        return pd.Series(predictions)

    def plot(self) -> None:
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
        self._print_tree_recursive(self.root, depth=0)

    def _print_tree_recursive(self, node, depth):
        if node is None:
            return

        indent = '    ' * depth
        if not node.children:
            print(f"{indent}Value: {node.value}")
        else:
            print(f"{indent}?(feature '{node.feature}' > {node.threshold})")
            for decision, child in node.children.items():
                print(f"{indent}   {decision}: ", end="")
                self._print_tree_recursive(child, depth + 1)