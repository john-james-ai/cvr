#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ======================================================================================================================== #
# Project  : Conversion Rate Prediction (CVR)                                                                              #
# Version  : 0.1.0                                                                                                         #
# File     : \transformer.py                                                                                               #
# Language : Python 3.7                                                                                                    #
# ------------------------------------------------------------------------------------------------------------------------ #
# Author   : John James                                                                                                    #
# Email    : john.james.ai.studio@gmail.com                                                                                #
# URL      : https://github.com/john-james-ai/cvr                                                                          #
# ------------------------------------------------------------------------------------------------------------------------ #
# Created  : Tuesday, January 11th 2022, 7:54:07 am                                                                        #
# Modified : Tuesday, January 18th 2022, 11:39:17 am                                                                       #
# Modifier : John James (john.james.ai.studio@gmail.com)                                                                   #
# ------------------------------------------------------------------------------------------------------------------------ #
# License  : BSD 3-clause "New" or "Revised" License                                                                       #
# Copyright: (c) 2021 Bryant St. Labs                                                                                      #
# ======================================================================================================================== #
import os
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
from scipy import stats
import logging

from cvr.visuals.visualize import Visual

# ------------------------------------------------------------------------------------------------------------------------ #


class Transformer:
    """Applies and evaluates transformations and their effects on data distributions."""

    def __init__(self, visual: Visual) -> None:

        self._visual = visual

        self._tests = {}

        self._xformers = {
            "Original Data": self.identity,
            "Square Root": self.sqrt,
            "Cube Root": self.cuberoot,
            "Log": self.log,
            "Log2": self.log2,
            "Log10": self.log10,
        }

    def explore(self, df: pd.DataFrame, columns: list) -> None:
        """Applies transformations to the columns designated in the DataFrame and plots the data."""
        results = {}
        for col in columns:
            title = "Distribution and Transformation Analysis\n{}".format(col)
            xforms = {}
            for k, v in self._xformers.items():
                xforms[k] = v(df[col])
            x4m = pd.DataFrame.from_dict(xforms, orient="columns")
            self._visual.multihist(x4m, columns=list(self._xformers.keys()), title=title, rows=3, cols=2)
        return

    def apply(self, df: pd.DataFrame, transformation: str, columns: list) -> None:
        """Applies the transformation to the dataframe

        Args:
            df (pd.DataFrame): The data
            transformation (str): The transformation key
            columns (list): List of columns to which the transformation should be applied.
        """
        for col in columns:
            df[col] = self._xformers[transformation](df[col])
        return df

    @property
    def best_performers(self):
        return self._tests

    def identity(self, x) -> np.array:
        return x

    def square(self, x) -> np.array:
        return np.square(x)

    def cube(self, x) -> np.array:
        return x ** 3

    def sqrt(self, x) -> np.array:
        return np.sqrt(x)

    def cuberoot(self, x):
        return x ** (1 / 3)

    def log(self, x):
        return np.log(x)

    def log2(self, x):
        return np.log2(x)

    def log10(self, x):
        return np.log10(x)

    def recip(self, x):
        return 1 / x
