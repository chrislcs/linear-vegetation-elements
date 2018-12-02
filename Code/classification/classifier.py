# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier


class BalancedRandomForest(object):
    """
    A balanced random forest classifier. Takes a bootstrap sample of the
    minority class and takes a random sample from the majority class.

    Parameters
    ----------
    n_estimators : integer
        The number of trees in the forest.
    criterion : string
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "entropy" for the information gain.
    max_features : int, float, string or None
        The number of features to consider when looking for the best split:
        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a percentage and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.
    max_depth : integer or None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.
    min_samples_split : int, float
        The minimum number of samples required to split an internal node:
        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a percentage and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.
    min_samples_leaf : int, float
        The minimum number of samples required to be at a leaf node:
        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a percentage and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.
    min_weight_fraction_leaf : float
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.
    max_leaf_nodes : int or None
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.
    min_impurity_split : float
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.
    ratio : string or float
        Determines the size of the random sample from the majority class.
        'auto' takes a sample equal to the bootstrap sample of the minority
        class. If the ratio is set with a float the majority sample will be
        1/ratio times as large as the bootstrap sample of the minority class.

    Attributes
    ----------
    forest : list of DecisionTreeClassifier
        The collection of fitted sub-estimators.
    n_estimators : int
        The number of decision trees.
    tree_params : dictionary
        The parameters used for each decision tree.
    ratio : float
        The ratio between majority and minority class samples.
    n_features : int
        The number of features.
    n_classes : int
        The number of classes (currently only 2 supported).
    n_samples : int
        The number of samples.
    feature_importances_ : array
        The feature importances (the higher, the more important the feature).

    Methods
    -------
    fit(X, y)
        Build a forest of trees from the training set (X, y).
    predict(X)
        Predict class for X.
    predict_proba(X)
        Predict class probabilities for X.
    """

    def __init__(self, n_estimators=10, criterion='gini', max_depth=None,
                 min_samples_split=2, min_samples_leaf=5,
                 min_weight_fraction_leaf=0.0, max_features='sqrt',
                 max_leaf_nodes=None, min_impurity_decrease=0,
                 ratio='auto'):
        self.n_estimators = n_estimators
        self.tree_params = {'criterion': criterion,
                            'max_depth': max_depth,
                            'min_samples_split': min_samples_split,
                            'min_samples_leaf': min_samples_leaf,
                            'min_weight_fraction_leaf': min_weight_fraction_leaf,
                            'max_features': max_features,
                            'max_leaf_nodes': max_leaf_nodes,
                            'min_impurity_decrease': min_impurity_decrease}
        self.forest = []
        self.ratio = ratio

    def fit(self, X, y):
        """
        Build a forest of trees from the training set (X, y).

        Parameters
        ----------
        X : array-like
            The feature values.
        y : (Mx1) array-like
            The class labels as integers from 0 to n.
        """
        self.n_features = np.shape(X)[1]
        self.n_classes = max(y) + 1
        if self.n_classes != 2:
            raise ValueError('Currently only works for binary classifications')
        self.n_samples = len(X)
        importances = np.zeros(self.n_features)

        # Check which class is in majority and which in minority
        class_idx = (np.where(y == 0)[0], np.where(y == 1)[0])
        min_class = 0 if len(class_idx[0]) < len(class_idx[1]) else 1
        n_min_class = len(class_idx[min_class])

        for _ in range(self.n_estimators):
            # Take a bootstrap sample of the minority class
            samples_min = np.random.choice(class_idx[min_class],
                                           n_min_class,
                                           replace=True)
            # Take a random sample for the majority class
            n_maj_samples = n_min_class
            if self.ratio != 'auto':
                n_maj_samples = int(n_maj_samples / self.ratio)
            samples_maj = np.random.choice(class_idx[1-min_class],
                                           n_maj_samples,
                                           replace=True)
            # Merge the two samples
            samples = np.concatenate((samples_min, samples_maj))
            # Retrieve the sample values
            X_sample = X.iloc[samples]
            y_sample = y.iloc[samples]
            # Fit a decision tree classifier using the sample data
            tree = DecisionTreeClassifier(**self.tree_params)
            tree.fit(X_sample, y_sample)
            # Add the feature importances
            importances += tree.feature_importances_
            # Add the tree to the forest
            self.forest.append(tree)
        # Average the feature importances over all trees
        self.feature_importances_ = importances / self.n_estimators

    def predict(self, X):
        """
        Predict class for X.

        Parameters
        ----------
        X : array-like
            The feature values.

        Returns
        -------
        preds : (Mx1) array-like
            The predicted classes.
        """
        votes = np.zeros((len(X), self.n_estimators), dtype=np.int32)
        for i, tree in enumerate(self.forest):
            preds = tree.predict(X)
            votes[:, i] = preds
        preds = np.argmax(np.apply_along_axis(np.bincount, 1,
                                              votes, minlength=2), axis=1)
        return preds

    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        Parameters
        ----------
        X : array-like
            The feature values.

        Returns
        -------
        probas : (Mx2) array-like
            The probabilities the samples belong to each class.
        """
        probas = np.zeros((len(X), self.n_classes))
        for tree in self.forest:
            proba = tree.predict_proba(X)
            probas += proba
        probas /= self.n_estimators
        return probas
