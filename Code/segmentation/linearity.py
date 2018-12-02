# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""


def extract_linear(vegetation_objects, min_elongatedness=1.5, max_width=60):
    linear_objects = []

    for obj in vegetation_objects:
        if (obj.elongatedness > min_elongatedness and
                obj.width < max_width):
            linear_objects.append(obj.shape)

    return linear_objects
