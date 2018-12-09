# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import time
import multiprocessing
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm

from segmentation.regiongrowing import segment_object


PROCESSES = multiprocessing.cpu_count() - 1
INPUT = '../Data/vegetation_dbscan.csv'
OUTPUT = '../Data/segments.pkl'

min_size = 5
rectangularity_limit = 0.55
alpha = 0.4
k_init = 10
max_dist_init = 15.0
knn = 8
max_dist = 5

point_cloud = pd.read_csv(INPUT)
points = point_cloud[['X', 'Y']].values.copy()
shift = np.min(points, axis=0)
num_clusters = max(point_cloud['cluster'])
inputs = point_cloud['cluster'].value_counts().index

segments = []

pbar = tqdm(total=point_cloud['cluster'].count())


def run_segmentation(i):
    cluster = point_cloud.loc[point_cloud['cluster'] == i]
    cluster_points = cluster[['X', 'Y']].values.copy()
    cluster_points -= shift
    cluster_segments = segment_object(cluster_points,
                                      min_size, rectangularity_limit,
                                      alpha, k_init,
                                      max_dist_init, knn, max_dist)
    return cluster_segments


def add_result(result):
    segments.extend(result)
    pbar.update(sum([len(r) for r in result]))


def run_multi():
    with multiprocessing.Pool(processes=PROCESSES) as p:

        for i in inputs:
            p.apply_async(run_segmentation, args=(i,), callback=add_result)

        p.close()
        p.join()

    pbar.close()


if __name__ == '__main__':
    t = time.time()
    run_multi()
    print('Done! Time elapsed: %.2f' % (time.time() - t))

    with open(OUTPUT, 'wb') as f:
        pickle.dump(segments, f)
