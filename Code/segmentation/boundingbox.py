# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import math
import numpy as np
from scipy.spatial import ConvexHull


class BoundingBox(object):
    """
    Compute the minimum oriented bounding box around a set of points
    by rotating caliphers.

    Parameters
    ----------
    points : (Mx2) array
        The coordinates of the points.

    Attributes
    ----------
    hull : (Mx2x2) array
        The coordinates of the points of the edges which form the
        convex hull of the points.
    hull_points : (Mx2) array
        The points of the convex hull.
    corner_points : (4x1) array
        The coordinates of the corner points of the bounding box.
    length : float
        The length of the long side of the bounding box.
    width : float
        The length of the short side of the bounding box.
    area : area
        The area of the bounding box.
    """

    def __init__(self, points):
        self.points = points
        self.hull = ConvexHull(points).simplices
        self.hull_points = self._to_unique_points()
        angles = self._compute_edge_angles()
        self._compute_bbox(angles)

    @staticmethod
    def _rotate(points, angle):
        """
        Rotate points in a coordinate system using a rotation matrix based on
        an angel.

        Parameters
        ----------
        points : (Mx2) array
            The coordinates of the points.
        angle : float
            The angle by which the points will be rotated (in radians).

        Returns
        -------
        points_rotated : (Mx2) array
            The coordinates of the rotated points.
        """
        # Compute rotation matrix
        rot_matrix = np.array(((math.cos(angle), -math.sin(angle)),
                               (math.sin(angle), math.cos(angle))))
        # Apply rotation matrix to the points
        points_rotated = np.dot(points, rot_matrix)

        return np.array(points_rotated)

    def _to_unique_points(self):
        """
        Extracts the unique points present in the convex hull edges.
        """
        hull_unique_i = np.array(list(set([p for s in self.hull for p in s])))

        return self.points[hull_unique_i]

    def _compute_edge_angles(self):
        """
        Compute the angles between the edges of the convex hull and the x-axis.

        Returns
        -------
        edge_angles : (Mx1) array
            The angles between the edges and the x-axis.
        """
        edges = self.points[self.hull]
        edges_count = len(edges)
        edge_angles = np.zeros(edges_count)
        for i in range(edges_count):
            edge_x = edges[i][1][0] - edges[i][0][0]
            edge_y = edges[i][1][1] - edges[i][0][1]
            edge_angles[i] = math.atan2(edge_y, edge_x)

        return np.unique(edge_angles)

    def _compute_bbox(self, angles):
        """
        Compute the oriented minimum bounding box.

        Parameters
        ----------
        angles : (Mx1) array-like
            The angles the edges of the convex hull and the x-axis.
        """
        # Start with basic rectangle around the points
        min_bbox = {'angle': 0, 'minmax': (0, 0, 0, 0),
                    'width': 0, 'height': 0, 'area': float('inf')}

        for a in angles:
            # Rotate the points and compute the new bounding box
            rotated_points = self._rotate(self.hull_points, a)
            min_x = min(rotated_points[:, 0])
            max_x = max(rotated_points[:, 0])
            min_y = min(rotated_points[:, 1])
            max_y = max(rotated_points[:, 1])
            width = max_x - min_x
            height = max_y - min_y
            area = width * height

            # Save if the new bounding box is smaller than the current smallest
            if area < min_bbox['area']:
                min_bbox = {'angle': a, 'minmax': (min_x, max_x, min_y, max_y),
                            'width': width, 'height': height, 'area': area}

        # Extract the rotated corner points of the minimum bounding box
        c1 = (min_bbox['minmax'][0], min_bbox['minmax'][2])
        c2 = (min_bbox['minmax'][0], min_bbox['minmax'][3])
        c3 = (min_bbox['minmax'][1], min_bbox['minmax'][3])
        c4 = (min_bbox['minmax'][1], min_bbox['minmax'][2])
        rotated_corner_points = [c1, c2, c3, c4]

        # Set the length, width and area of the minimum bounding box
        self.length = max(min_bbox['width'], min_bbox['height'])
        self.width = min(min_bbox['width'], min_bbox['height'])
        self.area = min_bbox['area']
        # Save the angle of the longest edge
        if (min_bbox['minmax'][1] - min_bbox['minmax'][0] >
                min_bbox['minmax'][3] - min_bbox['minmax'][2]):
            self.angle = a
        else:
            self.angle = a+math.pi if a-math.pi < 2*math.pi else a-math.pi

        # Rotate the corner points back to the original system
        self.corner_points = np.array(self._rotate(rotated_corner_points,
                                                   2*np.pi-min_bbox['angle']))
        [c1, c2, c3, c4] = self.corner_points
        self.edges = np.array([[c1, c2], [c2, c3], [c3, c4], [c4, c1]])
