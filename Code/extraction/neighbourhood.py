# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import numpy as np
import math
from .tensor import structure_tensor


def compute_features(points, point_neighbours, features,
                     distances=None, neighborhood_radius=None):
    """
    Computes neighbourhood features of a point.

    Height difference (delta z)
    Height standart deviation
    Radius
    Density

    Normal vector (X, Y, Z)

    Linearity:
        L_λ = (λ_1 - λ_2) / λ_1
    Planarity:
        P_λ = (λ_2 − λ_3) / λ_1
    Sphericity:
        S_λ = λ_3 / λ_1
    Omnivariance:
        O_λ = ³√(λ_1 * λ_2 * λ_3)
    Anisotropy:
        A_λ = (λ_1 − λ_3) / λ_1
    Eigenentropy:
        E_λ = −1 * sum[i=1 to 3](λ_i * ln(λ_i))
    Sum of λs:
        Σλ = λ_1 + λ_2 + λ_3
    Change of curvature:
        C_λ = λ_3 / (λ_1 + λ_2 + λ_3)
    (Blomley, Weimann, Leitloff and Jutzi, 2014)

    Roughness:
        Distance between point and best fit plane

    Parameters
    ----------
    points : (Mx3) array
        X, Y and Z coordinates of points.
    point_neighbours : array
        The indices of the neighbouring points of each point.
    features : list of strings
        The features to be calculated. Possible features: 'delta_z', 'std_z',
        'radius', 'density', 'linearity', 'planarity', 'sphericity',
        'omnivariance', 'anisotropy', 'eigenentropy', 'sum_eigenvalues',
        'curvature', 'roughness', 'norm_x', 'norm_y', 'norm_z'
    distances : array
        The distances between the point and the neighbouring points. Needed
        for the computation of 'radius' and 'density'.

    Returns
    -------
    feature_values : array
        The computed values for each of the features specified. For each
        feature a column.
    """
    pca_features = ['linearity', 'planarity', 'sphericity', 'omnivariance',
                    'anisotropy', 'eigenentropy', 'sum_eigenvalues',
                    'curvature', 'verticality', 'norm_x', 'norm_y', 'norm_z']
    n_points = len(points)

    if any(i in pca_features for i in features):
        eigenvalues = np.zeros((n_points, 3))
        normal_vectors = np.zeros((n_points, 3))

    if 'delta_z' in features:
        delta_z = np.zeros(n_points)
    if 'std_z' in features:
        std_z = np.zeros(n_points)
    if 'radius'in features:
        if distances is None:
            raise ValueError('No distances given.')
        radius = np.zeros(n_points)
    if 'density' in features:
        if distances is None and neighborhood_radius is None:
            raise ValueError('No distances or radius given.')
        density = np.zeros(n_points)
    if 'roughness' in features:
        roughness = np.zeros(n_points)

    print(" Computing structure tensors..")
    for i in range(n_points):
        local_points = points[point_neighbours[i]]

        if len(local_points) > 3:
            if any(i in pca_features for i in features):
                evalues, evectors = structure_tensor(local_points)
                eigenvalues[i, :] = evalues
                normal_vector = evectors[:, 2]
                if normal_vector[2] < 0:
                    normal_vector *= -1
                normal_vectors[i, :] = normal_vector

            if 'delta_z' in features:
                delta_z[i] = max(local_points[:, 2]) - min(local_points[:, 2])
            if 'std_z' in features:
                std_z[i] = np.std(local_points[:, 2])
            if 'radius'in features:
                radius[i] = max(distances[i])
            if 'density' in features:
                if distances is not None:
                    density[i] = len(local_points) / ((4/3) * math.pi *
                                                      max(distances[i])**3)
                elif neighborhood_radius is not None:
                    density[i] = len(local_points) / ((4/3) * math.pi *
                                                      neighborhood_radius)
                else:
                    raise ValueError('No distances or radius given.')
            if 'roughness' in features:
                mean_point = np.mean(local_points, axis=0)
                a, b, c = normal_vector
                d = -(np.dot(normal_vector, mean_point))
                dist = abs((a * points[i][0] + b * points[i][1] +
                            c * points[i][2] + d) /
                           (math.sqrt(a**2 + b**2 + c**2)))
                roughness[i] = dist
        else:
            if any(i in pca_features for i in features):
                eigenvalues[i, :] = [float('nan'), float('nan'), float('nan')]
                normal_vectors[i, :] = [
                    float('nan'), float('nan'), float('nan')]
            if 'roughness' in features:
                roughness[i] = float('nan')
    print(" Done!")

    print(" Computing features..")
    if any(i in pca_features for i in features):
        sum_eigenvalues = np.sum(eigenvalues, axis=1)
        eigenvalues = (eigenvalues /
                       np.tile(np.array([sum_eigenvalues]).transpose(), (1, 3)))

    feature_values = np.zeros((n_points, len(features)))
    for i, f in enumerate(features):
        if f == 'linearity':
            feature_values[:, i] = ((eigenvalues[:, 0] - eigenvalues[:, 1]) /
                                    eigenvalues[:, 0])
        elif f == 'planarity':
            feature_values[:, i] = ((eigenvalues[:, 1] - eigenvalues[:, 2]) /
                                    eigenvalues[:, 0])
        elif f == 'sphericity':
            feature_values[:, i] = eigenvalues[:, 2] / eigenvalues[:, 0]
        elif f == 'omnivariance':
            feature_values[:, i] = (eigenvalues[:, 0] * eigenvalues[:, 1] *
                                    eigenvalues[:, 2]) ** (1./3.)
        elif f == 'anisotropy':
            feature_values[:, i] = ((eigenvalues[:, 0] - eigenvalues[:, 2]) /
                                    eigenvalues[:, 0])
        elif f == 'eigenentropy':
            feature_values[:, i] = (-1 *
                                    ((eigenvalues[:, 0] * np.log(eigenvalues[:, 0])) +
                                     (eigenvalues[:, 1] * np.log(eigenvalues[:, 1])) +
                                     (eigenvalues[:, 2] * np.log(eigenvalues[:, 2]))))
        elif f == 'sum_eigenvalues':
            feature_values[:, i] = sum_eigenvalues
        elif f == 'curvature':
            feature_values[:, i] = eigenvalues[:, 2] / sum_eigenvalues
        elif f == 'roughness':
            feature_values[:, i] = roughness
        elif f == 'norm_x':
            feature_values[:, i] = normal_vectors[:, 0]
        elif f == 'norm_y':
            feature_values[:, i] = normal_vectors[:, 1]
        elif f == 'norm_z':
            feature_values[:, i] = normal_vectors[:, 2]
        elif f == 'delta_z':
            feature_values[:, i] = delta_z
        elif f == 'std_z':
            feature_values[:, i] = std_z
        elif f == 'radius':
            feature_values[:, i] = radius
        elif f == 'density':
            feature_values[:, i] = density
        else:
            raise ValueError('Inavailable features.')
    print(" Done!")

    return feature_values
