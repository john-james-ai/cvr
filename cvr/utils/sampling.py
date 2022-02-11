#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project  : Deep Learning for Conversion Rate Prediction (CVR)                                    #
# Version  : 0.1.0                                                                                 #
# File     : \sampling.py                                                                          #
# Language : Python 3.7.12                                                                         #
# ------------------------------------------------------------------------------------------------ #
# Author   : John James                                                                            #
# Email    : john.james.ai.studio@gmail.com                                                        #
# URL      : https://github.com/john-james-ai/cvr                                                  #
# ------------------------------------------------------------------------------------------------ #
# Created  : Monday, December 27th 2021, 2:03:27 am                                                #
# Modified : Thursday, February 10th 2022, 8:27:42 pm                                              #
# Modifier : John James (john.james.ai.studio@gmail.com)                                           #
# ------------------------------------------------------------------------------------------------ #
# License  : BSD 3-clause "New" or "Revised" License                                               #
# Copyright: (c) 2022 Bryant St. Labs                                                              #
# ================================================================================================ #
#%%
import os
import pandas as pd

# ------------------------------------------------------------------------------------------------ #


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


# ------------------------------------------------------------------------------------------------ #
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


# ------------------------------------------------------------------------------------------------ #
def sample_df(
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


# ------------------------------------------------------------------------------------------------ #
def sample_file(
    source: str, destination: str, nrows: int, sep: str = "\t", random_state: int = None
) -> pd.DataFrame:
    """Sample from file

    Args:
        source (str): Path to input file
        destination (str): Path to output file
        frac (float): The fraction of observations to draw
        random_state (int): Pseudo random generator seed

    Returns:
        DataFrame containing the requested sample.
    """
    df = pd.read_csv(
        source,
        nrows=nrows,
        low_memory=False,
        index_col=None,
        sep="\t",
        header=None,
        encoding="utf-8",
    )
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    df.to_csv(destination, sep="\t", header=False, index=False)
