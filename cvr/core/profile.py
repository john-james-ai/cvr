#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ============================================================================ #
# Project  : Deep Learning for Conversion Rate Prediction (CVR)                #
# Version  : 0.1.0                                                             #
# File     : \profile.py                                                       #
# Language : Python 3.7.12                                                     #
# ---------------------------------------------------------------------------- #
# Author   : John James                                                        #
# Email    : john.james.ai.studio@gmail.com                                    #
# URL      : https://github.com/john-james-ai/cvr                              #
# ---------------------------------------------------------------------------- #
# Created  : Sunday, January 16th 2022, 1:33:06 pm                             #
# Modified : Thursday, February 3rd 2022, 4:15:18 pm                           #
# Modifier : John James (john.james.ai.studio@gmail.com)                       #
# ---------------------------------------------------------------------------- #
# License  : BSD 3-clause "New" or "Revised" License                           #
# Copyright: (c) 2022 Bryant St. Labs                                          #
# ============================================================================ #
import pandas as pd
from pandas.api.types import is_categorical_dtype
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import numpy as np
from typing import Union

# plotting modules
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("whitegrid")
sns.set_palette("mako")

from cvr.data import categorical_columns, numeric_columns, criteo_columns
from cvr.utils.printing import Printer
from cvr.utils.format import titlelize, titlelize_df
from cvr.visualize.features import (
    CategoricalFeatureVisualizer,
    NumericFeatureVisualizer,
)

# ---------------------------------------------------------------------------- #


class DataProfiler:
    """Data profiler for Dataset objects."""

    def __init__(self, data: pd.DataFrame) -> None:
        self._data = data
        self._title = "Criteo Sponsored Search Conversion Log Dataset"

        self._printer = Printer()
        self._categorical_visualizer = CategoricalFeatureVisualizer()
        self._numeric_visualizer = NumericFeatureVisualizer()

    # ------------------------------------------------------------------------ #
    @property
    def summary(self, verbose: bool = True) -> None:
        subtitle = "Dataset Summary"
        summary = self._data.summary

        if verbose:
            self._printer.print_title(self._title, subtitle)
            self._printer.print_dictionary(content=summary)
        else:
            return summary

    # ------------------------------------------------------------------------ #
    @property
    def info(self, verbose: bool = False) -> None:
        subtitle = "Overview of Columns"
        info = self._data.info

        if verbose:
            self._printer.print_title(self._title, subtitle)
            self._printer.print_dictionary(content=info)
        else:
            return info

    # ------------------------------------------------------------------------ #
    #                      DESCRIPTIVE STATISTICS                              #
    # ------------------------------------------------------------------------ #
    def describe(self, column: str, verbose=False) -> pd.DataFrame:
        self._data.describe(column)

    # ------------------------------------------------------------------------ #
    def topn_plot(self, column: str) -> None:
        data = self._data.rank_frequencies(column, 10)

        self._categorical_visualizer.fit(self._data[column])
        self._categorical_visualizer.topn()

    def cfd_plot(self, column: str) -> None:

        self._categorical_visualizer.fit(self._data[column])
        self._categorical_visualizer.cfd()

    def zipf_plot(self, column: str) -> None:

        self._categorical_visualizer.fit(self._data[column])
        self._categorical_visualizer.zipf()
