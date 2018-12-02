# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import os
import pdal


def las_to_csv(input_path, output_path=None, overwrite=False):
    """
    Convert a LAS to a CSV file.

    Parameters
    ----------
    input_path : string
        Path to the input file.
    output_path : string
        Path to the output file.
    overwrite : bool
        Force recalculation if output file already exist.

    Returns
    -------
    out_filename : string
        Returns the output file path based on the input files.

    Output
    ------
     : CSV file
         The resulting CSV file.
    """
    input_path = os.path.abspath(input_path).replace('\\', '/')

    if output_path is None:
        path_root, _ = os.path.splitext(input_path)
        output_path = '{}{}'.format(path_root, ".csv")
    else:
        output_path = os.path.abspath(output_path).replace('\\', '/')

    if os.path.isfile(output_path) and overwrite is False:
        print('CSV output file already exists.')
    else:
        json = """{{
        "pipeline":[
            {{
                "type":"readers.las",
                "filename":"{}"
            }},
            {{
                "type":"writers.text",
                "filename":"{}"
            }}
        ]}}
        """

        pipeline = pdal.Pipeline(json.format(input_path, output_path))

        pipeline.validate()
        pipeline.execute()

    return output_path


def sample(input_path, distance, output_path=None,
           vegetation_class=0, overwrite=False):
    """
    Convert to 2D and downsample a point cloud
    by specifying a minimum distance between points.

    Parameters
    ----------
    input_path : string
        Path to the input csv file.
    distance : float or int
        Minimum distance between points
    output_path : string
        Path to the output file.
    overwrite : bool
        Force recalculation if output file already exist.

    Returns
    -------
    output_path : string
        Returns the output file path based on the input files.

    Output
    ------
     : file
         The downsampled point cloud
    """

    json = """{{
    "pipeline":[
        {{
            "type":"readers.text",
            "filename":"{input_path}"
        }},
        {{
            "type":"filters.range",
            "limits":"Classification[{veg_class}:{veg_class}]"
        }},
        {{
            "type":"filters.assign",
            "assignment":"Z[-99999:]=0"
        }},
        {{
            "type":"filters.sample",
            "radius":{radius}
        }},
        {{
            "type":"writers.text",
            "filename":"{output_path}",
            "order":"X,Y",
            "keep_unspecified":"false"
        }}
    ]}}
    """

    input_path = os.path.abspath(input_path).replace('\\', '/')

    if output_path is None:
        path_root, ext = os.path.splitext(input_path)
        output_path = '{}_sub_{}{}'.format(path_root,
                                           str(distance).replace('.', '_'),
                                           ext)
    else:
        output_path = os.path.abspath(output_path).replace('\\', '/')

    if os.path.isfile(output_path) and overwrite is False:
        print('Subsampled output file already exists.')
    else:
        pipeline = pdal.Pipeline(json.format(input_path=input_path,
                                             veg_class=vegetation_class,
                                             radius=distance,
                                             output_path=output_path))
        pipeline.validate()
        pipeline.execute()

    return output_path
