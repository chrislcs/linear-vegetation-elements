# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import math
import numpy as np
from .vegetationobject import VegetationObject


def angle_difference(a1, a2):
    """
    Returns the minimal angle difference between two orientations.

    Parameters
    ----------
    a1 : float
        An angle in radians
    a2 : float
        Another angle in radians

    Returns
    -------
    angle : float
        The minimal angle difference in radians
    """
    pos1 = abs(math.pi - abs(abs(a1 - a2) - math.pi))
    if a1 < math.pi:
        pos2 = abs(math.pi - abs(abs((a1 + math.pi) - a2) - math.pi))
    elif a2 < math.pi:
        pos2 = abs(math.pi - abs(abs(a1 - (a2 + math.pi)) - math.pi))
    else:
        return pos1
    return pos1 if pos1 < pos2 else pos2


def merge_objects(objects, max_dist, max_dir_dif,  max_c_dir_dif,
                  min_elong=0, max_width=float('inf'), alpha=0.4):
    """
    Merges objects if they are in proximity, have the same direction and are
    aligned with each other.

    Parameters
    ----------
    objects : list of VegetationObject objects
        The linear elements to be merged.
    max_dist : float or int
        The maximum distance two objects can be from each other to be
        considered for merging
    max_dir_dif : float
        The maximum difference of the directions of the objects in radians.
        Ensures the objects point in the same direction.
    max_c_dir_dif : float
        The maximum difference of the angle between the center points and
        the directions of the objects in radians. Ensures the objects are
        aligned.
    min_elong : float or int
        Minimum elongatedness of the objects to be considered for merging.
        To ensure only elongated objects will be merged.
    max_width : float or int
        The maximum width of the objects to be considered for merging. To
        ensure only narrow objects will be merged.
    alpha : float
        The alpha used to compute the alpha shape of the newly created merged
        objects.

    Returns
    -------
    objects : list of LinearElement objects
        The merged linear objects.
    """
    objects = objects[:]
    n_objects = len(objects)
    shift = objects[0].shift

    # Calculate distances between all polygons
    distances = np.full((n_objects, n_objects), np.inf)
    for i in range(n_objects):
        poly1 = objects[i].shape
        for j in range(i+1, n_objects):
            poly2 = objects[j].shape
            dist = poly1.distance(poly2)
            distances[i, j] = dist

    # Select the polygons which are close to each other as candidates
    candidates = distances < max_dist
    candidates = np.transpose(np.where(candidates == True))

    # Determine if the candidates meet the criteria for merging
    to_merge = {}
    for c in candidates:
        if (objects[c[0]].bbox.width < max_width and
                objects[c[1]].bbox.width < max_width):
            # Check if the objects point in the same direction
            dir_dif = angle_difference(objects[c[0]].direction,
                                       objects[c[1]].direction)

            # Check if the objects are aligned by comparing the angle between
            # the two center points and the directions the objects face
            x = objects[c[0]].shape.centroid.x-objects[c[1]].shape.centroid.x
            y = objects[c[0]].shape.centroid.y-objects[c[1]].shape.centroid.y
            centroid_dir = math.atan2(y, x)
            c_dir_dif1 = angle_difference(objects[c[0]].direction,
                                          centroid_dir)
            c_dir_dif2 = angle_difference(objects[c[1]].direction,
                                          centroid_dir)

            # Set to merge if all criteria are met
            if (dir_dif < max_dir_dif and
                    objects[c[0]].elongatedness > min_elong and
                    objects[c[1]].elongatedness > min_elong and
                    c_dir_dif1 < max_c_dir_dif and c_dir_dif2 < max_c_dir_dif):
                for key, value in to_merge.items():
                    if c[0] in value:
                        to_merge.setdefault(key, []).append(c[1])
                        break
                    if c[1] in value:
                        to_merge.setdefault(key, []).append(c[0])
                        break
                else:
                    to_merge.setdefault(c[0], []).append(c[1])

    # Create the new objects from the objects to be merged
    to_remove = []
    for key, value in to_merge.items():
        merge_objects_idx = [key]
        merge_objects_idx.extend(value)
        merge_objects = [objects[i] for i in merge_objects_idx]
        points = merge_objects[0].points
        for x in merge_objects[1:]:
            points = np.append(points, x.points, axis=0)
        new_object = VegetationObject(points, alpha, shift)
        new_object.width = max([x.bbox.width for x in merge_objects])
        new_object.length = sum([x.bbox.length for x in merge_objects])
        new_object.elongatedness = (new_object.bbox.length /
                                    new_object.bbox.width)
#        new_object.direction = (sum([x.direction for x in merge_objects]) /
#                                len(merge_objects))
        objects.append(new_object)
        to_remove.extend(merge_objects_idx)

    # Remove the objects which are now part of a merged object
    objects = [x for i, x in enumerate(objects) if i not in to_remove]

    return objects
