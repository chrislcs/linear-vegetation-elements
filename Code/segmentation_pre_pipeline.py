# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import pandas as pd
import numpy as np

from segmentation.clustering import dbscan
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
