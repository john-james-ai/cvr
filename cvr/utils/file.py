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
# Modified : Wednesday, January 19th 2022, 2:46:08 am                                                                      #
# Modifier : John James (john.james.ai.studio@gmail.com)                                                                   #
# ------------------------------------------------------------------------------------------------------------------------ #
# License  : BSD 3-clause "New" or "Revised" License                                                                       #
# Copyright: (c) 2021 Bryant St. Labs                                                                                      #
# ======================================================================================================================== #
#%%
import pandas as pd
import numpy as np


# ------------------------------------------------------------------------------------------------------------------------ #
def sample_csv(source: str, destination: str, frac: float = 0.01, random_state: int = None) -> None:
    """Samples from source and saves to destination

    Args:
        source (str): Path to input file
        destination (str): Path to output file
        frac (float): The fraction of observations to draw as fraction
        random_state (int): Pseudo random generator seed

    """
    df = pd.read_csv(source, low_memory=False, index_col=False, sep="\t")
    n = int(len(df) * frac)
    df2 = df.sample(n, replace=False, random_state=random_state)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    df2.to_csv(destination, sep="\t")


# ------------------------------------------------------------------------------------------------------------------------ #
def sample_pkl(source: str, destination: str, frac: float = 0.01, random_state: int = None) -> None:
    """Samples from source and saves to destination

    Args:
        source (str): Path to input file
        destination (str): Path to output file
        frac (float): The fraction of observations to draw as fraction
        random_state (int): Pseudo random generator seed

    """
    df = pd.read_pickle(source)
    n = int(len(df) * frac)
    df2 = df.sample(n, replace=False, random_state=random_state)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    df2.to_pickle(destination)


source = "data\criteo\staged\criteo.pkl"
destination = "tests\\test_data\staged\criteo.pkl"
frac = 0.01
random_state = 55
sample_pkl(source, destination, frac, random_state)
