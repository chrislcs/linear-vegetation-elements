# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

from sklearn.cluster import DBSCAN


def dbscan(points, eps, min_samples):
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean',
                algorithm='auto', leaf_size=30, p=None, n_jobs=1)
    db.fit(points)
    labels = db.labels_
    return labels
