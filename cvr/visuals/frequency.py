#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ======================================================================================================================== #
# Project  : Explainable Recommendation (XRec)                                                                             #
# Version  : 0.1.0                                                                                                         #
# File     : \visualize.py                                                                                                 #
# Language : Python 3.8                                                                                                    #
# ------------------------------------------------------------------------------------------------------------------------ #
# Author   : John James                                                                                                    #
# Email    : john.james.ai.studio@gmail.com                                                                                #
# URL      : https://github.com/john-james-ai/xrec                                                                         #
# ------------------------------------------------------------------------------------------------------------------------ #
# Created  : Tuesday, December 21st 2021, 7:45:33 pm                                                                       #
# Modified : Thursday, January 27th 2022, 8:10:19 am                                                                       #
# Modifier : John James (john.james.ai.studio@gmail.com)                                                                   #
# ------------------------------------------------------------------------------------------------------------------------ #
# License  : BSD 3-clause "New" or "Revised" License                                                                       #
# Copyright: (c) 2021 Bryant St. Labs                                                                                      #
# ======================================================================================================================== #
import matplotlib.pyplot as plt
import logging
import seaborn as sns

sns.set_style("whitegrid")
import pandas as pd
import datashader as ds
import colorcet as cc
from bokeh.palettes import RdBu9

# ------------------------------------------------------------------------------------------------------------------------ #
from cvr.utils.format import titlelize

# ------------------------------------------------------------------------------------------------------------------------ #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------------------------------ #


class Frequency:
    def __init__(
        self,
        cmap: list = list(RdBu9),
        width: int = 12,
        height: int = 6,
        palette="mako",
    ) -> None:
        self._cmap = cmap
        self._width = width
        self._height = height
        self._palette = palette

    def analysis(
        self,
        df: pd.DataFrame,
        column: str,
        col_category: str,
        col_freq: str,
        col_cum: str,
        col_pct_cum: str,
        col_rank: str,
        n_categories: int = 10,
        title: str = None,
        hsize: int = 16,
        vsize: int = 8,
        label_fontsize=16,
        heading_fontsize=20,
    ) -> None:

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(hsize, vsize))
        fig.suptitle(title, fontsize=heading_fontsize)

        df_barplot = df.sort_values(by="Count", ascending=False, ignore_index=True)
        order = df_barplot[col_category][:n_categories]

        sns.barplot(
            x=col_freq, y=col_category, palette=self._palette, order=order, data=df_barplot.loc[:n_categories], ax=axes[0]
        )
        axes[0].set_title("Top {} Categories".format(str(n_categories)))

        sns.lineplot(x=col_rank, y=col_pct_cum, palette=self._palette, data=df, ax=axes[1])
        axes[1].set_title("Cumulative Frequency Distribution")

        title = title or "Frequency Distribution Analysis\n{}".format(column)
        fig.suptitle(title, fontsize=heading_fontsize)

        plt.tight_layout()
        plt.show()
