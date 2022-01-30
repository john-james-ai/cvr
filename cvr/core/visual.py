#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ======================================================================================================================== #
# Project  : Conversion Rate Prediction (CVR)                                                                              #
# Version  : 0.1.0                                                                                                         #
# File     : \visual.py                                                                                                    #
# Language : Python 3.7.12                                                                                                 #
# ------------------------------------------------------------------------------------------------------------------------ #
# Author   : John James                                                                                                    #
# Email    : john.james.ai.studio@gmail.com                                                                                #
# URL      : https://github.com/john-james-ai/cvr                                                                          #
# ------------------------------------------------------------------------------------------------------------------------ #
# Created  : Sunday, January 30th 2022, 5:25:46 am                                                                         #
# Modified : Sunday, January 30th 2022, 6:22:32 am                                                                         #
# Modifier : John James (john.james.ai.studio@gmail.com)                                                                   #
# ------------------------------------------------------------------------------------------------------------------------ #
# License  : BSD 3-clause "New" or "Revised" License                                                                       #
# Copyright: (c) 2022 Bryant St. Labs                                                                                      #
# ======================================================================================================================== #
from abc import ABC, abstractmethod
from bokeh.palettes import RdBu9
import pandas as pd

# ------------------------------------------------------------------------------------------------------------------------ #
class Visualizer(ABC):
    def __init__(
        self,
        cmap: list = list(RdBu9),
        width: int = 12,
        height: int = 6,
        label_fontsize: int = 16,
        heading_fontsize: int = 20,
        palette: str = "mako",
    ) -> None:
        self._cmap = cmap
        self._width = width
        self._height = height
        self._label_fontsize = label_fontsize
        self._heading_fontsize = heading_fontsize
        self._palette = palette

    @abstractmethod
    def fit(self, X: pd.Series):
        pass

    @abstractmethod
    def plot(self, X: pd.Series) -> None:
        pass
