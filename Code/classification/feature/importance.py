# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""


def mean_decrease_impurity(clf, features, plot=False):
    """
    Assess the importance of the features by analysing the mean decrease in
    gini impurity for each feature.

    Parameters
    ----------
    clf : Classifier
        A classifier with a feature_importance_ attribute
    features : list of strings
        The names of the features used when fitting the classifier.
    plot : bool
        Plot the feature importances using a bar diagram.

    Returns
    -------
    scores : list of tuples
        The features and the corrisponding mean decrease in gini impurity.
    """
    scores = sorted(zip(map(lambda x: round(x, 4), clf.feature_importances_),
                        features), reverse=True)

    if plot:
        import matplotlib.pyplot as plt

        widths, names = zip(*scores)
        xs = range(len(names))
        plt.figure()
        plt.barh(xs, widths[::-1], height=0.4, tick_label=names[::-1])
        plt.tight_layout()
        plt.show()

    return scores
