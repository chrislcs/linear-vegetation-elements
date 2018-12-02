# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import numpy as np
from scipy.spatial import cKDTree
from .alphashape import AlphaShape
from .boundingbox import BoundingBox


def segment_object(points, min_size, threshold,
                   alpha=0.4,
                   k_init=30, max_dist_init=float('inf'),
                   knn=8, max_dist=float('inf')):
    """
    Segment an object into smaller rectangular objects.

    Parameters
    ----------
    points : (Mx2) array
        The coordinates of the points.
    min_size : int
        The minimum size (number of points) an object needs to be.
    threshold : float
        The threshold value the rectangularity needs to be for the points to
        be considered of the same object.
    alpha : float
        The alpha value used for the computation of the alpha shapes.
    k_init : int
        The size of the initial object when segmenting.
    max_dist_init : float or int
        The max distance points can be from the starting point for them to be
        added to the initial object
    k : int
        The number of neighbours considered when growing a segment.
    max_dist : float or int
        The max distance a point can be from another point and still be
        considered from the same neighbourhood.

    Returns
    -------
    segments : list of arrays
        The points belonging to each rectangular segment.
    """
    segments = []
    kdtree = cKDTree(points)

    points_remain = np.array(range(len(points)))
    while len(points_remain) > 3:
        seg = grow_rectangular(points, points_remain, kdtree, threshold,
                               alpha, k_init=k_init,
                               max_dist_init=max_dist_init,
                               knn=knn, max_dist=max_dist)

        if len(seg) > min_size:
            points_segment = points[seg]
            segments.append(points_segment)

        seg_i = [np.argwhere(points_remain == i)[0][0] for i in seg]
        points_remain = np.delete(points_remain, seg_i)

    return segments


def grow_rectangular(points, points_remain, kdtree, threshold, alpha=0.4,
                     k_init=30, max_dist_init=float('inf'),
                     knn=8, max_dist=float('inf')):
    """
    Grow a region based on rectangularity

    Parameters
    ----------
    points : (Mx2) array
        The coordinates of all the points.
    points_remain : (Mx2) array
        The coordinates of the remaining points which need to be segmented.
    kdtree : SciPy KDtree object
        A SciPy KDtree data structure objects of the points.
    threshold : float
        The threshold value the rectangularity needs to be for the points to
        be considered of the same object.
    alpha : float
        The alpha value used for the computation of the alpha shapes.
    k_init : int
        The size of the initial object when segmenting.
    max_dist_init : float or int
        The max distance points can be from the starting point for them to be
        added to the initial object
    k : int
        The number of neighbours considered when growing a segment.
    max_dist : float or int
        The max distance a point can be from another point and still be
        considered from the same neighbourhood.

    Returns
    -------
    cluster : list
        The indices of the points belonging to the same region according to
        the rectangularity constraint.
    """
    # Initiate starting region
    # Locate point with the smallest x value
    min_x = np.argmin(points[points_remain, 0])
    p0_i = np.where(np.logical_and(points[:, 0] == points[points_remain][min_x][0],
                                   points[:, 1] == points[points_remain][min_x][1]))[0][0]
    p0 = points[p0_i]
    # Determine start cluster
    p0_nn_dist, p0_nn_i = kdtree.query(p0, k=k_init,
                                       distance_upper_bound=max_dist_init)
    p0_nn_i = np.delete(p0_nn_i, np.argwhere(p0_nn_dist == float('inf')))
    cluster = list(set(p0_nn_i).intersection(points_remain))
    # Check if there are enough points nearby
    if len(cluster) < 4:
        return [p0_i]

    # Compute initial rectangularity and check if it is above the treshold
    alpha_shape = AlphaShape(points[cluster], alpha)
    bbox = BoundingBox(points[cluster])
    rectangularity = alpha_shape.area / bbox.area
    if rectangularity < threshold:
        return [p0_i]

    # Start region growing
    point_queue = cluster[:]

    while point_queue != []:
        # Get next point in queue and determine its neighbourhood
        p = point_queue.pop(0)

        p_nn_dist, p_nn_i = kdtree.query(points[p], k=knn,
                                         distance_upper_bound=max_dist)

        p_nn_i = np.delete(p_nn_i, np.argwhere(p_nn_dist == float('inf')))
        p_nn_i = list(set(p_nn_i).intersection(points_remain))

        # for each point in the neighbourhood check if the region would still
        # rectangular if the point was added. Add point to region and point
        # queue if that is the case.
        for pi in p_nn_i:
            if pi not in cluster:
                alpha_shape.add_point(points[pi])
                bbox = BoundingBox(points[cluster + [pi]])
                rectangularity = alpha_shape.area / bbox.area
                if rectangularity > threshold:
                    cluster.append(pi)
                    point_queue.append(pi)
                else:
                    alpha_shape.remove_last_added()

    return cluster
