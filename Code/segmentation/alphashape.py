# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import numpy as np
from numba import njit
from CGAL.CGAL_Triangulation_2 import Delaunay_triangulation_2
from CGAL.CGAL_Kernel import Point_2
from shapely.geometry import Polygon
from shapely.ops import cascaded_union


@njit
def triangle_geometry(triangle):
    """
    Compute the area and circumradius of a triangle.

    Parameters
    ----------
    triangle : (1x3) array-like
        The indices of the points which form the triangle.

    Returns
    -------
    area : float
        The area of the triangle
    circum_r : float
        The circumradius of the triangle
    """
    point_a = triangle[0]
    point_b = triangle[1]
    point_c = triangle[2]
    # Lengths of sides of triangle
    x_diff_ab = point_a[0]-point_b[0]
    y_diff_ab = point_a[1]-point_b[1]
    x_diff_bc = point_b[0]-point_c[0]
    y_diff_bc = point_b[1]-point_c[1]
    x_diff_ca = point_c[0]-point_a[0]
    y_diff_ca = point_c[1]-point_a[1]

    length_a = ((x_diff_ab * x_diff_ab) + (y_diff_ab * y_diff_ab))**0.5
    length_b = ((x_diff_bc * x_diff_bc) + (y_diff_bc * y_diff_bc))**0.5
    length_c = ((x_diff_ca * x_diff_ca) + (y_diff_ca * y_diff_ca))**0.5
    # Semiperimeter of triangle
    semiperimeter = (length_a + length_b + length_c) / 2.0
    # Area of triangle by Heron's formula
    area = (semiperimeter * (semiperimeter - length_a) *
            (semiperimeter - length_b) * (semiperimeter - length_c))**0.5
    if area != 0:
        circumradius = (length_a * length_b * length_c) / (4.0 * area)
    else:
        circumradius = 0

    return area, circumradius


class AlphaShape(object):
    """
    Compute the alpha shape (a concave hull) of points.

    Parameters
    ----------
    points : (Mx2) array
        The coordinates of the points.
    alpha : float
        Influences the shape of the alpha shape. Higher values lead to more
        edges being deleted.

    Attributes
    ----------
    points : (Mx2) array
        The coordinates of the points used to compute the alpha shape.
    alpha : float
        The alpha used to compute the alpha shape.
    tri : SciPy Delaunay object
        The Delaunay triangulation of the points.
    triangles : array of simplices of triangles
        The Delaunay triangles in indices

    Methods
    -------
    to_shape :
    """

    def __init__(self, points, alpha):
        if len(points) < 4:
            raise ValueError('Not enough points to compute an alpha shape.')

        self.area = 0
        self.alpha = alpha
        self.points = [Point_2(*p) for p in points]
        self.tri = Delaunay_triangulation_2()
        self.tri.insert(self.points)
        self.triangles = []
        self._add_triangles(self.tri.finite_faces())
        self.saved = None
        self.points_added = False

    @staticmethod
    def _compute_area(faces, alpha):
        for f in faces:
            p1 = f.vertex(0).point()
            p2 = f.vertex(1).point()
            p3 = f.vertex(2).point()
            triangle = ((p1.x(), p1.y()),
                        (p2.x(), p2.y()),
                        (p3.x(), p3.y()))
            area, circumradius = triangle_geometry(triangle)
            if area != 0:
                if circumradius < (1.0 / alpha):
                    yield area

    def _add_triangles(self, faces):
        """
        Add the edges between the given vertices if the circumradius of the
        triangle is bigger than 1/alpha.

        Parameters
        ----------
        simplices : (Mx3) array
            Indices of the points forming the vertices of a triangulation.
        """
        for f in faces:
            p1 = f.vertex(0).point()
            p2 = f.vertex(1).point()
            p3 = f.vertex(2).point()
            triangle = ((p1.x(), p1.y()),
                        (p2.x(), p2.y()),
                        (p3.x(), p3.y()))
            area, circumradius = triangle_geometry(triangle)
            if area != 0:
                if circumradius < (1.0 / self.alpha):
                    self.triangles.append(triangle)
                    self.area += area

    def add_point(self, point):
        """
        Adds points to the alpha shape.

        Parameters
        ----------
        point : (1x2) array-like
            The coordinates of the points.
        """
        tri_old = self.tri.deepcopy()

        vertex = self.tri.insert(Point_2(*point))

        self.saved = {'area': self.area,
                      'last_added_vertex': vertex}

        new_faces = []
        faces_circ = self.tri.incident_faces(vertex)
        first_face = faces_circ.next()
        if not self.tri.is_infinite(first_face):
            new_faces.append(first_face)
        while True:
            f = faces_circ.next()
            if f == first_face:
                break
            if not self.tri.is_infinite(f):
                new_faces.append(f)

        affected_vertices = []
        for f in new_faces:
            for i in range(3):
                affected_vertex = f.vertex(i)
                if (affected_vertex not in affected_vertices and
                        affected_vertex != vertex):
                    affected_vertices.append(affected_vertex)

        old_faces = []
        for affected_vertex in affected_vertices:
            faces_circ = self.tri.incident_faces(affected_vertex)
            first_face = faces_circ.next()
            if not self.tri.is_infinite(first_face):
                if first_face not in new_faces:
                    new_faces.append(first_face)
            while True:
                f = faces_circ.next()
                if f == first_face:
                    break
                if not self.tri.is_infinite(f):
                    if f not in new_faces:
                        new_faces.append(f)

            faces_circ = tri_old.incident_faces(
                tri_old.nearest_vertex(affected_vertex.point()))
            first_face = faces_circ.next()
            if not tri_old.is_infinite(first_face):
                if first_face not in old_faces:
                    old_faces.append(first_face)
            while True:
                f = faces_circ.next()
                if f == first_face:
                    break
                if not tri_old.is_infinite(f):
                    if f not in old_faces:
                        old_faces.append(f)

        old_area = sum(self._compute_area(old_faces, self.alpha))
        new_area = sum(self._compute_area(new_faces, self.alpha))

        d_area = new_area - old_area

        self.area += d_area

        self.points_added = True

    def remove_last_added(self):
        """
        Removes the last added points.
        """
        if self.saved is None:
            raise RuntimeError('No points were added.')

        self.area = self.saved['area']
        self.tri.remove(self.saved['last_added_vertex'])
        self.saved = None

    def recompute_triangles(self):
        self.triangles = []
        self._add_triangles(self.tri.finite_faces())

    def to_shape(self, shift):
        """
        Convert the alpha shape to a Shapely polygon object.

        Parameters
        ----------
        shift :
        """
        if self.points_added:
            self.recompute_triangles()

        polygons = []
        for t in self.triangles:
            polygons.append(Polygon(np.array(t) + shift))

        return cascaded_union(polygons)
