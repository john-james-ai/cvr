#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ======================================================================================================================== #
# Project  : Explainable Recommendation (XRec)                                                                             #
# Version  : 0.1.0                                                                                                         #
# File     : \file.py                                                                                                      #
# Language : Python 3.8                                                                                                    #
# ------------------------------------------------------------------------------------------------------------------------ #
# Author   : John James                                                                                                    #
# Email    : john.james.ai.studio@gmail.com                                                                                #
# URL      : https://github.com/john-james-ai/xrec                                                                         #
# ------------------------------------------------------------------------------------------------------------------------ #
# Created  : Monday, December 27th 2021, 2:03:27 am                                                                        #
# Modified : Sunday, January 16th 2022, 4:21:22 pm                                                                         #
# Modifier : John James (john.james.ai.studio@gmail.com)                                                                   #
# ------------------------------------------------------------------------------------------------------------------------ #
# License  : BSD 3-clause "New" or "Revised" License                                                                       #
# Copyright: (c) 2021 Bryant St. Labs                                                                                      #
# ======================================================================================================================== #
import pandas as pd
import random

# ------------------------------------------------------------------------------------------------------------------------ #
def sample_pickle(filepath: str, frac: float, stratify: str = None, random_state: int = None) -> pd.DataFrame:
    """Sampling on pickled dataframes

    Args:
        filepath (str): Path to the file
        frac (float): The fraction of observations to draw as fraction
        stratify (str): None or the column to stratify
        random_state (int): Pseudo random generator seed

    Returns:
        DataFrame containing the requested sample.
    """

    # Read the dataframe from pickle
    df = pd.read_pickle(filepath)

    # Sample if stratify
    if stratify is not None:
        df = df.groupby(by=[stratify]).sample(frac=frac, random_state=random_state)
    else:

        df = df.sample(frac=frac, random_state=random_state)

    return df
