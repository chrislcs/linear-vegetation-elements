# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from imblearn.metrics import geometric_mean_score
from .classifier import BalancedRandomForest


def cross_validation(df, features, class_column, n_folds=10,
                     n_estimators=1000, max_features='log2',
                     min_samples_leaf=1, min_samples_split=5,
                     ratio=0.4):
    """
    Perform a cross validation to evaluate the performance of a classification
    method.

    Parameters
    ----------
    df : DataFrame
        The feature values and the corrisponding classes.
    features : list of strings
        The names of the features (column names in the dataframe).
    class_column : string
        The name of the column with the class labels (the labels should be
        integers from 0 to n for n classes)
    n_folds : int
        The amount of folds to perform the cross validation.

    Returns
    -------
    cv_scores : DataFrane
        A table with the Matthews Correlation Coefficient, the Area under
        ROC-curve, and geometric mean values of all the folds and the
        average of those in the last row.
    confusion_matrices : list
        The confusion matrices of each fold.
    """
    skf = StratifiedKFold(n_splits=n_folds)

    cv_scores = pd.DataFrame(np.zeros((n_folds + 1, 3)),
                             columns=['mcc', 'roc_auc', 'gmean'])
    confusion_matrices = []

    i = 0
    for train_index, test_index in skf.split(df[features], df[class_column]):
        train_data = df.iloc[train_index]
        test_data = df.iloc[test_index]

        clf = BalancedRandomForest(n_estimators=n_estimators,
                                   max_features=max_features,
                                   min_samples_leaf=min_samples_leaf,
                                   min_samples_split=min_samples_split,
                                   ratio=ratio)
        clf.fit(train_data[features], train_data[class_column])

        preds = clf.predict(test_data[features])
        probas = clf.predict_proba(test_data[features])

        mcc = matthews_corrcoef(test_data[class_column], preds)
        roc_auc = roc_auc_score(test_data[class_column], probas[:, 1])
        gmean = geometric_mean_score(test_data[class_column], preds)
        cv_scores.loc[i, 'mcc'] = mcc
        cv_scores.loc[i, 'roc_auc'] = roc_auc
        cv_scores.loc[i, 'gmean'] = gmean

        df_confusion = pd.crosstab(test_data[class_column], preds,
                                   rownames=['Actual'],
                                   colnames=['Predicted'],
                                   margins=True)
        confusion_matrices.append(df_confusion)

        i += 1
        print("Done %d of %d.." % (i, n_folds))

    cv_scores.loc[i, 'mcc'] = np.average(cv_scores['mcc'][:i])
    cv_scores.loc[i, 'roc_auc'] = np.average(cv_scores['roc_auc'][:i])
    cv_scores.loc[i, 'gmean'] = np.average(cv_scores['gmean'][:i])

    return cv_scores, confusion_matrices
