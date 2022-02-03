#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ============================================================================ #
# Project  : Deep Learning for Conversion Rate Prediction (CVR)                #
# Version  : 0.1.0                                                             #
# File     : \data.py                                                          #
# Language : Python 3.7.12                                                     #
# ---------------------------------------------------------------------------- #
# Author   : John James                                                        #
# Email    : john.james.ai.studio@gmail.com                                    #
# URL      : https://github.com/john-james-ai/cvr                              #
# ---------------------------------------------------------------------------- #
# Created  : Saturday, December 25th 2021, 12:22:02 pm                         #
# Modified : Wednesday, February 2nd 2022, 10:17:54 pm                         #
# Modifier : John James (john.james.ai.studio@gmail.com)                       #
# ---------------------------------------------------------------------------- #
# License  : BSD 3-clause "New" or "Revised" License                           #
# Copyright: (c) 2022 Bryant St. Labs                                          #
# ============================================================================ #
import numpy
import pandas as pd

# ---------------------------------------------------------------------------- #
def to_list(x, ignore: str = None) -> list:
    """Converts a non iterable to a list if it isn't already a list."""
    if x is ignore:
        result = x
    else:
        result = x if isinstance(x, list) else [x]
    return result


def sample(
    data: pd.DataFrame, frac: float, stratify: str = None, random_state: int = None
) -> pd.DataFrame:
    """Sample from the current DataFrame.

    Args:
        data (pd.DataFrame): The data to sample
        frac (float): The fraction of observations to draw as fraction
        stratify (str): None or the column to stratify
        random_state (int): Pseudo random generator seed

    Returns:
        DataFrame containing the requested sample.
    """

    # Sample if stratify
    if stratify is not None:
        df = data.groupby(by=[stratify]).sample(frac=frac, random_state=random_state)
    else:

        df = data.sample(frac=frac, random_state=random_state)

    return df
