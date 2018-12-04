# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import math
import pandas as pd
import numpy as np
import time
import os

from segmentation.clustering import dbscan
from segmentation.regiongrowing import segment_object
from segmentation.vegetationobject import VegetationObject
from segmentation.merge import merge_objects
from segmentation.linearity import extract_linear
from segmentation.accuracy import spatial_assessment, numerical_assessment
from utils.io import polygons_to_shapefile, read_manual_shp
from utils.data import sample


# %%
veg_path = '../Data/veg_classification.csv'
point_cloud = pd.read_csv(veg_path)

# %% Downsample vegetation points
veg_ds_path = sample(veg_path, 1.0, vegetation_class=0, overwrite=True)

# %% Load point cloud data
point_cloud = pd.read_csv(veg_ds_path)
points = point_cloud.as_matrix(columns=['X', 'Y']).copy()
shift = np.min(points, axis=0)
points -= shift

# %% Cluster using DBScan
eps = 3
min_samples = 4

clusters = dbscan(points, eps, min_samples)
point_cloud['cluster'] = clusters

point_cloud.to_csv('../Data/vegetation_dbscan.csv', index=False)

# %% Segment the points into rectangular objects
min_size = 5
rectangularity_limit = 0.55
alpha = 0.4
k_init = 10
max_dist_init = 15.0
knn = 8
max_dist = 5

print('Growing rectangular regions..')
t = time.time()
segments = []
num_clusters = max(point_cloud['cluster'])
# TODO add parallel processing to increase performance
# (each cluster can be segmented completely seperately, thus allowing parallelism)
for i in range(num_clusters):
    print("{} of {}..".format(i, num_clusters))
    cluster = point_cloud.loc[point_cloud['cluster'] == i]
    cluster_points = cluster.as_matrix(columns=['X', 'Y']).copy()
    cluster_points -= shift
    cluster_segments = segment_object(cluster_points,
                                      min_size, rectangularity_limit,
                                      alpha, k_init,
                                      max_dist_init, knn, max_dist)
    segments.extend(cluster_segments)
print('Done! Time elapsed: %.2f' % (time.time() - t))

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
vegetation_objects_ = merge_objects(vegetation_objects, max_dist, max_dir_dif,
                                    max_c_dir_dif, min_elong, max_width, alpha)
print('Done! Time elapsed: %.2f' % (time.time() - t))

# %% Extract the linear elements
linear_objects = extract_linear(vegetation_objects_,
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
filename = '../Data/shp/linear_objects.shp'
epsg = 28992
polygons_to_shapefile(filename, linear_objects, epsg)

 # %%
#import matplotlib.pyplot as plt
#import random
#plt.figure()
#colors = 'rgbcyk'
#for p in segments:
#     plt.scatter(p[:, 0], p[:, 1], c=colors[random.randint(0, 5)])

# plt.figure()
#num_clusters = max(clusters)
# for i in range(num_clusters):
#    cluster = point_cloud.loc[point_cloud['cluster'] == i]
#    cluster_points = cluster.as_matrix(columns=['X', 'Y'])
#    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[random.randint(0,5)])
