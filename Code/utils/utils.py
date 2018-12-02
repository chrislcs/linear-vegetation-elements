# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import pandas as pd


def merge_dataframes(dfs, key_field_name):
    """
    Merges dataframes containing data of one class into one dataframe with
    the class in a column.

    Parameters
    ----------
    dfs : dict of DataFrames
        Dictionary with the class as key and the value as the dataframes
        to be merged.

    Returns
    -------
    df : DataFrame
        The merged dataframe.
    """
    df = pd.DataFrame()
    for k, v in dfs.items():
        v[key_field_name] = k
        df = df.append(v)
    return df
