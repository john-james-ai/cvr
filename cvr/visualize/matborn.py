#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ============================================================================ #
# Project  : Deep Learning for Conversion Rate Prediction (CVR)                #
# Version  : 0.1.0                                                             #
# File     : \frequency.py                                                     #
# Language : Python 3.7.12                                                     #
# ---------------------------------------------------------------------------- #
# Author   : John James                                                        #
# Email    : john.james.ai.studio@gmail.com                                    #
# URL      : https://github.com/john-james-ai/cvr                              #
# ---------------------------------------------------------------------------- #
# Created  : Thursday, January 27th 2022, 6:45:26 am                           #
# Modified : Thursday, February 3rd 2022, 6:06:49 pm                           #
# Modifier : John James (john.james.ai.studio@gmail.com)                       #
# ---------------------------------------------------------------------------- #
# License  : BSD 3-clause "New" or "Revised" License                           #
# Copyright: (c) 2022 Bryant St. Labs                                          #
# ============================================================================ #
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import logging
import seaborn as sns

sns.set_style("whitegrid")
import pandas as pd
from typing import Union

# ---------------------------------------------------------------------------- #
from cvr.utils.format import titlelize
from cvr.visualize import PALETTES, FONTSIZE, SIZE, FONTDICT

# ---------------------------------------------------------------------------- #


class Matborn:
    def __init__(
        self,
        width: int = SIZE["width"],
        height: int = SIZE["height"],
        palette=PALETTES["blue"],
        rows: int = 1,
        cols: int = 3,
        sharex="none",
        sharey="none",
        title=None,
        label_fontsize=16,
        heading_fontsize=20,
    ) -> None:
        self._width = width
        self._height = height
        self._palette = palette
        self._rows = rows
        self._cols = cols
        self._sharex = sharex
        self._sharey = sharey
        self._title = title
        self._label_fontsize = label_fontsize
        self._heading_fontsize = heading_fontsize

    @abstractmethod
    def fit(
        self, x: [str, int, float], y: Union[str, int, float], df: pd.DataFrame = None
    ) -> None:
        pass

    @abstractmethod
    def plot(self) -> None:
        pass

class BarPlot(Matborn):

    def __init__(self)


        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(hsize, vsize))
        fig.suptitle(title, fontsize=heading_fontsize)

        df_barplot = df.sort_values(by="Count", ascending=False, ignore_index=True)
        order = df_barplot[col_category][:n_categories]

        sns.barplot(
            x=col_freq,
            y=col_category,
            palette=self._palette,
            order=order,
            data=df_barplot.loc[:n_categories],
            ax=axes[0],
        )
        axes[0].set_title("Top {} Categories".format(str(n_categories)))

        sns.lineplot(
            x=col_rank, y=col_pct_cum, palette=self._palette, data=df, ax=axes[1]
        )
        axes[1].set_title("Cumulative Frequency Distribution")
        axes[1].set(xlabel="Rank", ylabel="Percent of Data")

        title = title or "Frequency Distribution Analysis\n{}".format(column)
        fig.suptitle(title, fontsize=heading_fontsize)

        plt.tight_layout()
        plt.show()

class