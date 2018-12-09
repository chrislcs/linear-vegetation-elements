# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import math
import pandas as pd
import numpy as np
import time
import pickle

from segmentation.vegetationobject import VegetationObject
from segmentation.merge import merge_objects
from segmentation.linearity import extract_linear
from segmentation.accuracy import spatial_assessment, numerical_assessment
from utils.io import polygons_to_shapefile, read_manual_shp


INPUT = '../Data/segments.pkl'
alpha = 0.4
point_cloud = pd.read_csv('../Data/vegetation_dbscan.csv')
points = point_cloud[['X', 'Y']].values.copy()
shift = np.min(points, axis=0)

# %%
with open(INPUT, 'rb') as f:
    segments = pickle.load(f)

# %%
vegetation_objects = []
for s in segments:
    vegetation_objects.append(VegetationObject(s, alpha, shift))

# %% Merge neighbouring elongated objects if pointing in the same direction
print('Merging objects..')
t = time.time()
max_dist = 5
max_dir_dif = math.radians(30)
min_elong = 1.3
max_c_dir_dif = math.radians(15)
max_width = 60
vegetation_objects = merge_objects(vegetation_objects, max_dist, max_dir_dif,
                                    max_c_dir_dif, min_elong, max_width, alpha)
print('Done! Time elapsed: %.2f' % (time.time() - t))

# %% Extract the linear elements
linear_objects = extract_linear(vegetation_objects,
                                min_elongatedness=1.5,
                                max_width=60)

# %% Assess the accuracy using a manually annotated dataset

manual_shapes_path = "../Data/shp/alpha_shape_parts.shp"
manual_linear, manual_nonlinear = read_manual_shp(manual_shapes_path)

tp, fp, tn, fn = spatial_assessment(manual_linear,
                                    manual_nonlinear,
                                    linear_objects)
accuracy_assessment = numerical_assessment(tp, fp, tn, fn)

# %% Export to shapefile
print('Exporting to shapefile..')
filename = '../Data/shp/linear_objects_test.shp'
epsg = 28992
polygons_to_shapefile(filename, linear_objects, epsg)
