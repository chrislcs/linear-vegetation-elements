# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import fiona
import fiona.crs
from shapely.geometry import mapping, shape, Polygon, MultiPolygon


def read_manual_shp(path):
    manual_shapes_file = fiona.open(path)
    manual_shapes_linear = []
    manual_shapes_nonlinear = []
    for polygon in manual_shapes_file:
        if polygon['properties']['linear'] == 1:
            manual_shapes_linear.append(shape(polygon['geometry']))
        else:
            manual_shapes_nonlinear.append(shape(polygon['geometry']))

    return manual_shapes_linear, manual_shapes_nonlinear


def polygons_to_shapefile(filename, polygons, epsg):
    """
    Export linear elements to a shapefile.

    Parameters
    ----------
    filename : string
        The path and filename where the shapefile will be saved.
    polygons : list of Polygons
        The polygons to be saved in the shapefile.
    epsg : int
        The epsg number of the coordinate system to use for the shapefile.

    Output
    ------
    Shapefile
    """
    crs = fiona.crs.from_epsg(epsg)
    schema = {'geometry': 'Polygon',
              'properties': {'id': 'int'}}

    with fiona.open(filename, 'w', 'ESRI Shapefile',
                    schema=schema, crs=crs) as c:
        for i, s in enumerate(polygons):
            if type(s) == Polygon or type(s) == MultiPolygon:
                c.write({
                    'geometry': mapping(s),
                    'properties': {'id': i}, })
