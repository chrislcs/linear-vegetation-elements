# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import numpy as np


def structure_tensor(points):
    """
    Computes the structure tensor of points by computing the eigenvalues
    and eigenvectors of the covariance matrix of a point cloud.

    Parameters
    ----------
    points : (Mx3) array
        X, Y and Z coordinates of points.

    Returns
    -------
    eigenvalues : (1x3) array
        The eigenvalues corrisponding to the eigenvectors of the covariance
        matrix.
    eigenvectors : (3,3) array
        The eigenvectors of the covariance matrix.
    """
    if len(points) > 3:
        cov_mat = np.cov(points, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eig(cov_mat)
        order = np.argsort(-eigenvalues)
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        return eigenvalues, eigenvectors
    else:
        raise ValueError('Not enough points to compute eigenvalues/vectors.')
