# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from imblearn.metrics import geometric_mean_score
from .classifier import BalancedRandomForest


def grid_search(df, features, class_column, param_dict, n_folds=3):
    """
    Perform a cross validated grid search to look for the best parameter
    settings.

    Parameters
    ----------
    df : DataFrame
        The feature values and the corrisponding classes.
    features : list of strings
        The names of the features (column names in the dataframe).
    class_column : string
        The name of the column with the class labels (the labels should be
        integers from 0 to n for n classes)
    param_dict : dictionary
        The parameters and the values that need to be checked.
    n_folds : int
        The amount of folds to perform the cross validation.

    Returns
    -------
    gs_scores : DataFrane
        A table with the Matthews Correlation Coefficient, the Area under
        ROC-curve, and geometric mean values of all the different combinations
        of parameter settings.
    param_grid : sklearn ParameterGrid
        All the different combinations of the parameters.
    """
    skf = StratifiedKFold(n_splits=n_folds)

    param_grid = ParameterGrid(param_dict)
    n_params = len(param_grid)
    gs_scores = pd.DataFrame(np.zeros((n_params, 3)),
                             columns=['mcc', 'roc_auc', 'gmean'])

    n = 0
    n_total = n_params * n_folds
    for train_index, test_index in skf.split(df[features], df[class_column]):
        for i in range(n_params):
            parameters = param_grid[i]
            train_data = df.iloc[train_index]
            test_data = df.iloc[test_index]

            clf = BalancedRandomForest(n_estimators=100, **parameters)
            clf.fit(train_data[features], train_data[class_column])

            preds = clf.predict(test_data[features])
            probas = clf.predict_proba(test_data[features])

            mcc = matthews_corrcoef(test_data[class_column], preds)
            roc_auc = roc_auc_score(test_data[class_column], probas[:, 1])
            gmean = geometric_mean_score(test_data[class_column], preds)
            gs_scores.loc[i, 'mcc'] += mcc
            gs_scores.loc[i, 'roc_auc'] += roc_auc
            gs_scores.loc[i, 'gmean'] += gmean

            n += 1
            print("Done %d of %d.." % (n, n_total))

    gs_scores['mcc'] /= n_folds
    gs_scores['roc_auc'] /= n_folds
    gs_scores['gmean'] /= n_folds

    return gs_scores, param_grid
