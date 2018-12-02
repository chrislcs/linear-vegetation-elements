# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import math
import numpy as np
from .alphashape import AlphaShape
from .boundingbox import BoundingBox


class VegetationObject(object):
    """
    A vegetation object.

    Parameters
    ----------
    points : (Mx2) array
        The coordinates of the points of the vegetation object.
    alpha : float
        Influences the shape of the alpha shape. Higher values lead to more
        edges being deleted.

    Attributes
    ----------
    area : float
        The area of the vegetation object.
    bbox_area : float
        The area of the minimum bounding box around the vegetation object.
    elongatedness : float
        The elongatedness of the vegetation object, defined as the length
        divided by the width.
    direction : float
        The angle between the long side of the object and the x-axis in
        radians.
    """

    def __init__(self, points, alpha, shift=None):
        self.points = points
        if shift is None:
            self.shift = np.min(points, axis=0)
            self.alpha_shape = AlphaShape(points - self.shift, alpha)
        else:
            self.shift = shift
            self.alpha_shape = AlphaShape(points, alpha)
        self.shape = self.alpha_shape.to_shape(shift)
        self.bbox = BoundingBox(points)
        self.width = self.bbox.width
        self.length = self.bbox.length
        self.elongatedness = self.length / self.width
        self.direction = self._compute_direction()

    def _compute_direction(self):
        """
        Compute the angle between the long side of the object and the x-axis.
        """
        idx = np.argmin(self.bbox.corner_points[:, 0])
        [c1, c2, c3, _] = np.roll(self.bbox.corner_points, 4-idx, axis=0)
        dist12 = math.hypot(*c1-c2)
        dist23 = math.hypot((c2[0]-c3[0]), (c2[1]-c3[1]))
        if dist12 > dist23:
            x = c1[0] - c2[0]
            y = c1[1] - c2[1]
        else:
            x = c2[0] - c3[0]
            y = c2[1] - c3[1]
        return math.atan2(y, x)
