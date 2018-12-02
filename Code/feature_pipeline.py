# -*- coding: utf-8 -*-
"""
Computes point neighbourshood parameters and removes irrelevant points.

@author: Chris Lucas
"""

import pandas as pd
import time
import os
from scipy.spatial import cKDTree

from extraction.neighbourhood import compute_features
from utils.data import las_to_csv, sample

# %% file paths
las_path = "../Data/ResearchArea.las"

# %% Prepare data and load into python
# downsample point cloud and convert to csv
las = sample(las_path, 0.3)
csv_path = las_to_csv(las)

# Load the csv point cloud file
print("Loading point cloud csv file using pandas..")
point_cloud = pd.read_csv(csv_path)
point_cloud.drop(columns=['ScanDirectionFlag', 'EdgeOfFlightLine', 'Classification',
                          'ScanAngleRank', 'UserData', 'PointSourceId', 'GpsTime',
                          'Red', 'Green', 'Blue'], inplace=True)
point_cloud.rename(columns={'Intensity': 'intensity', 'ReturnNumber': 'return_number',
                            'NumberOfReturns': 'number_of_returns'}, inplace=True)

points = point_cloud.as_matrix(columns=['X', 'Y', 'Z'])

# %% Compute nearest neighbours
print("Computing nearest neighbours..")
neighbours = 10
kdtree = cKDTree(points)
distances, point_neighbours = kdtree.query(points, neighbours)
print("Done!")

# %% Compute point features
features = ['delta_z', 'std_z', 'radius', 'density', 'norm_z',
            'linearity', 'planarity', 'sphericity', 'omnivariance',
            'anisotropy', 'eigenentropy', 'sum_eigenvalues',
            'curvature']
print("Computing covariance features..")
t = time.time()
feature_values = compute_features(points, point_neighbours,
                                  features, distances)
print("Done! Runtime: %s" % str(time.time() - t))

for i, f in enumerate(features):
    point_cloud[f] = pd.Series(feature_values[:, i])

# %% Trim the data by deleting all non scatter points from the point cloud
print("Trimming data..")
point_cloud.query('sphericity > 0.03', inplace=True)
point_cloud.reset_index(drop=True, inplace=True)
print("Done!")

# %% Compute normalized return number
point_cloud['norm_returns'] = (point_cloud['return_number'] /
                               point_cloud['number_of_returns'])

# %% Output data
las_path_root = os.path.splitext(las_path)[0]
out_filename = '{}_params.csv'.format(las_path_root)
point_cloud.to_csv(out_filename, index=False)
