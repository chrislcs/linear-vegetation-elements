# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

from io import BytesIO
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import rc
from owslib.wms import WebMapService

# %%
rc('text', usetex=True)
rc('font', family='serif')
rc('font', size=14)

# %%
veg = pd.read_csv('../Data/vegpart_small.csv')
veg.rename(columns={'//X': 'X'}, inplace=True)
veg_ss = pd.read_csv('../Data/vegpart_small_ss.csv')
veg_ss.rename(columns={'//X': 'X'}, inplace=True)

points = veg_ss.as_matrix(columns=['X', 'Y'])

# %%
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
plt.figure()

plt.imshow(img, extent=(min_x, max_x, min_y, max_y), alpha=0.6)
veg_plot = plt.scatter(veg['X'], veg['Y'], marker='o', s=1)
veg_ss_plot = plt.scatter(veg_ss['X'], veg_ss['Y'], c='r', marker='o', s=1.2)
plt.axis('equal')
plt.legend(handles=[veg_plot, veg_ss_plot],
           labels=['Before', 'After'],
           prop={'size': 12})
plt.xlabel('X-coordinate (m)')
plt.ylabel('Y-coordinate (m)')
plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
plt.tight_layout()
