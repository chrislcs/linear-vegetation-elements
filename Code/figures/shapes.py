# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

from io import BytesIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import rc
from owslib.wms import WebMapService
from scipy.spatial import ConvexHull

from ..segmentation.boundingbox import BoundingBox
from ..segmentation.alphashape import AlphaShape

# %%
rc('text', usetex=True)
rc('font', family='serif')
rc('font', size=14)

# %% Data
veg_ss = pd.read_csv('../Data/vegpart_small_ss.csv')
veg_ss.rename(columns={'//X': 'X'}, inplace=True)

points = veg_ss.as_matrix(columns=['X', 'Y'])

# %% Shapes
convex_hull = ConvexHull(points)

shift = np.min(points, axis=0)
points_shifted = points - shift
alpha_shape = AlphaShape(points_shifted, 0.4).to_shape(shift)
alpha_shape_points = np.array(alpha_shape.exterior.coords)

bbox = BoundingBox(points)
bbox_points = np.zeros((5, 2))
bbox_points[:4] = bbox.corner_points
bbox_points[-1] = bbox_points[0]

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

fig, axes = plt.subplots(ncols=3, sharex=True, sharey=True)
fig.add_subplot(111, frameon=False)

ax = axes[0]
ax.imshow(img, extent=(min_x, max_x, min_y, max_y), alpha=0.6)
veg_plot = ax.scatter(veg_ss['X'], veg_ss['Y'], c='k', marker='o', s=1.2)
for simplex in convex_hull.simplices:
    convex_hull_plot, = ax.plot(points[simplex, 0], points[simplex, 1], 'r-')
ax.axis('equal')
ax.set_title('Convex Hull')
ax.set_xlabel('(a)', labelpad=30)
ax.get_xaxis().get_major_formatter().set_useOffset(False)

ax = axes[1]
ax.imshow(img, extent=(min_x, max_x, min_y, max_y), alpha=0.6)
veg_plot = ax.scatter(veg_ss['X'], veg_ss['Y'], c='k', marker='o', s=1.2)
bbox_plot, = ax.plot(bbox_points[:, 0], bbox_points[:, 1], 'b-')
ax.axis('equal')
ax.set_title('Bounding Box')
ax.set_xlabel('(b)', labelpad=30)

ax = axes[2]
ax.imshow(img, extent=(min_x, max_x, min_y, max_y), alpha=0.6)
veg_plot = ax.scatter(veg_ss['X'], veg_ss['Y'], c='k', marker='o', s=1.2)
alpha_shape_plot, = ax.plot(alpha_shape_points[:, 0], alpha_shape_points[:, 1],
                            c='m')
ax.axis('equal')
ax.set_title('Alpha Shape')
ax.set_xlabel('(c)', labelpad=30)

plt.tick_params(labelcolor='none',
                top='off', bottom='off',
                left='off', right='off')
plt.xlabel('X-coordinate (m)', labelpad=10)
plt.ylabel('Y-coordinate (m)', labelpad=30)
# plt.legend(handles=[veg_plot, convex_hull_plot, bbox_plot, concave_hull_plot],
#           labels=['Vegetation Points', 'Convex Hull', 'Bounding Box', 'Concave Hull'],
#           bbox_to_anchor=(1.01, 1), loc='upper left')
plt.tight_layout()
plt.show()
