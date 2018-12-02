# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

from io import BytesIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib import rc
from owslib.wms import WebMapService
from scipy.spatial import cKDTree

from ..segmentation.boundingbox import BoundingBox
from ..segmentation.alphashape import AlphaShape


# %% Font
rc('text', usetex=True)
rc('font', family='serif')
rc('font', size=14)

# %% Data
veg_ss = pd.read_csv('../Data/vegpart_small_ss.csv')
veg_ss.rename(columns={'//X': 'X'}, inplace=True)

points = veg_ss.as_matrix(columns=['X', 'Y'])

shift = np.min(points, axis=0)

# %% Background image
offset = 50
min_x = min(points[:, 0]) - offset
max_x = max(points[:, 0]) + offset
min_y = min(points[:, 1]) - offset
max_y = max(points[:, 1]) + offset

dif_x = max_x - min_x
dif_y = max_y - min_y
aspect_ratio = dif_x / dif_y
resolution = int(dif_x * 4)
img_size = (resolution, int(resolution / aspect_ratio))

wms_url = 'https://geodata.nationaalgeoregister.nl/luchtfoto/rgb/wms?'
wms_version = '1.3.0'
wms_layer = 'Actueel_ortho25'
wms_srs = 'EPSG:28992'
wms_format = 'image/png'

wms = WebMapService(wms_url, version=wms_version)
wms_img = wms.getmap(layers=[wms_layer],
                     srs=wms_srs,
                     bbox=(min_x, min_y, max_x, max_y),
                     size=img_size,
                     format=wms_format,
                     transparent=True)

img = mpimg.imread(BytesIO(wms_img.read()))

# %%

kdtree = cKDTree(points)

start_point = points[np.argmin(points[:, 1])]
_, indices = kdtree.query(start_point, k=20)
start_region = points[indices]

grow_point = start_region[np.argmax(start_region[:, 1])]
_, indices = kdtree.query(grow_point, k=range(2, 10))
neighbors = points[indices]

added_point = neighbors[np.argmax(neighbors[:, 1])]

considered_region = np.vstack([start_region, added_point])
bbox = BoundingBox(considered_region)
bbox_points1 = np.zeros((5, 2))
bbox_points1[:4] = bbox.corner_points
bbox_points1[-1] = bbox_points1[0]
alpha_shape = AlphaShape(considered_region-shift, 4).to_shape(shift)
alpha_shape_points1 = np.array(alpha_shape.exterior.coords)

considered_points = neighbors[neighbors[:, 1] != added_point[1]]
added_point2 = considered_points[np.argmax(considered_points[:, 1])]
considered_region = np.vstack([considered_region, added_point2])
bbox = BoundingBox(considered_region)
bbox_points2 = np.zeros((5, 2))
bbox_points2[:4] = bbox.corner_points
bbox_points2[-1] = bbox_points2[0]
alpha_shape = AlphaShape(considered_region-shift, 4).to_shape(shift)
alpha_shape_points2 = np.array(alpha_shape.exterior.coords)

considered_points = considered_points[considered_points[:, 1]
                                      != added_point2[1]]
added_point3 = considered_points[np.argmax(considered_points[:, 1])]
considered_region2 = np.vstack([considered_region, added_point3])
bbox = BoundingBox(considered_region2)
bbox_points3 = np.zeros((5, 2))
bbox_points3[:4] = bbox.corner_points
bbox_points3[-1] = bbox_points3[0]
alpha_shape = AlphaShape(considered_region-shift, 4).to_shape(shift)
alpha_shape_points3 = np.array(alpha_shape.exterior.coords)

# %%
path_data = [(Path.MOVETO, (143549.0, 434338.7)),
             (Path.CURVE4, (143548.5, 434338.7)),
             (Path.CURVE4, (143548.0, 434338.7)),
             (Path.CURVE4, (143548.5, 434339.8)),
             (Path.CURVE4, (143549.0, 434340.6)),
             (Path.CURVE4, (143550.0, 434340.8)),
             (Path.CURVE4, (143550.5, 434340.7)),
             (Path.CURVE4, (143551.3, 434340.6)),
             (Path.CURVE4, (143551.3, 434339.9)),
             (Path.CURVE4, (143550.5, 434339.8)),
             (Path.CURVE4, (143549.5, 434339.6)),
             (Path.CURVE4, (143549.5, 434339.2)),
             (Path.LINETO, (143549.0, 434338.7))]
codes, verts = zip(*path_data)
path = Path(verts, codes)

# %% Plot
rc('font', size=17)

fig, axes = plt.subplots(ncols=4, sharex=True, sharey=True)
fig.add_subplot(111, frameon=False)

ax = axes[0]
ax.imshow(img, extent=(min_x, max_x, min_y, max_y), alpha=0.6)
veg_plot = ax.scatter(veg_ss['X'], veg_ss['Y'], c='k', marker='o', s=1)

neighbor_plot = ax.scatter(neighbors[:, 0], neighbors[:, 1],
                           c='b', marker='*', s=80)
grow_plot = ax.scatter(grow_point[0], grow_point[1], c='r', marker='X', s=80)
start_plot = ax.scatter(start_region[:, 0], start_region[:, 1],
                        c='g', marker='o', s=10)
patch = patches.PathPatch(path, facecolor='none', lw=2)
ax.add_patch(patch)
ax.axis('equal')
# ax.set_autoscale_on(False)
#ax.set_xlim([143540, 143560])
#ax.set_ylim([434325, 434355])
# ax.set_title('Title')
ax.set_xlabel('(a)', labelpad=30)
ax.get_xaxis().get_major_formatter().set_useOffset(False)
ax.get_yaxis().get_major_formatter().set_useOffset(False)
ax.legend(handles=[veg_plot, start_plot, grow_plot, neighbor_plot, patch],
          labels=['Vegetation Points', 'Start Region',
                  'Grow Point', 'Neighboring Points', 'Points to Consider'],
          prop={'size': 11},
          loc=4)

ax = axes[1]
ax.imshow(img, extent=(min_x, max_x, min_y, max_y), alpha=0.6)
veg_plot = ax.scatter(veg_ss['X'], veg_ss['Y'],
                      c='k', marker='o', s=1, zorder=1)
bbox_plot, = ax.plot(bbox_points1[:, 0], bbox_points1[:, 1], 'b-', zorder=2)
alpha_shape_plot, = ax.plot(alpha_shape_points1.hull[:, 0], alpha_shape_points1.hull[:, 1],
                            c='m', zorder=3)
current_plot = ax.scatter(start_region[:, 0], start_region[:, 1],
                          c='g', marker='o', s=10, zorder=4)
added_plot = ax.scatter(added_point[0], added_point[1],
                        c='r', marker='X', s=80, zorder=5)

ax.axis('equal')
# ax.set_title('Title')
ax.set_xlabel('(b)', labelpad=30)
ax.legend(handles=[veg_plot, current_plot, added_plot,
                   bbox_plot, alpha_shape_plot],
          labels=['Vegetation Points', 'Current Region',
                  'Considered Point', 'Bounding Box', 'Concave Hull'],
          prop={'size': 11},
          loc=4)

ax = axes[2]
ax.imshow(img, extent=(min_x, max_x, min_y, max_y), alpha=0.6)
veg_plot = ax.scatter(veg_ss['X'], veg_ss['Y'],
                      c='k', marker='o', s=1, zorder=1)
bbox_plot, = ax.plot(bbox_points2[:, 0], bbox_points2[:, 1], 'b-', zorder=2)
alpha_shape_plot, = ax.plot(alpha_shape_points2.hull[:, 0], alpha_shape_points2.hull[:, 1],
                            c='m', zorder=3)
current_plot = ax.scatter(considered_region[:, 0], considered_region[:, 1],
                          c='g', marker='o', s=10, zorder=4)
added_plot = ax.scatter(added_point2[0], added_point2[1],
                        c='r', marker='X', s=80, zorder=5)

ax.axis('equal')
# ax.set_title('Title')
ax.set_xlabel('(c)', labelpad=30)
ax.legend(handles=[veg_plot, start_plot, added_plot,
                   bbox_plot, alpha_shape_plot],
          labels=['Vegetation Points', 'Current Region',
                  'Considered Point', 'Bounding Box', 'Concave Hull'],
          prop={'size': 11},
          loc=4)

ax = axes[3]
ax.imshow(img, extent=(min_x, max_x, min_y, max_y), alpha=0.6)
veg_plot = ax.scatter(veg_ss['X'], veg_ss['Y'],
                      c='k', marker='o', s=1, zorder=1)
bbox_plot, = ax.plot(bbox_points3[:, 0], bbox_points3[:, 1], 'b-', zorder=2)
alpha_shape_plot, = ax.plot(alpha_shape_points3.hull[:, 0], alpha_shape_points3.hull[:, 1],
                            c='m', zorder=3)
current_plot = ax.scatter(considered_region2[:, 0], considered_region2[:, 1],
                          c='g', marker='o', s=10, zorder=4)
added_plot = ax.scatter(added_point3[0], added_point3[1],
                        c='r', marker='X', s=80, zorder=5)

ax.axis('equal')
# ax.set_title('Title')
ax.set_xlabel('(d)', labelpad=30)
ax.legend(handles=[veg_plot, start_plot, added_plot,
                   bbox_plot, alpha_shape_plot],
          labels=['Vegetation Points', 'Current Region',
                  'Considered Point', 'Bounding Box', 'Concave Hull'],
          prop={'size': 11},
          loc=4)

#plt.title('The region growing process')
plt.xlabel('X-coordinate (m)', labelpad=10)
plt.ylabel('Y-coordinate (m)', labelpad=35)
plt.tick_params(labelcolor='none',
                top='off', bottom='off',
                left='off', right='off')
plt.tight_layout()
plt.show()
