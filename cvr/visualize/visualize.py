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
# Modified : Sunday, January 30th 2022, 5:52:50 am                                                                         #
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


class Visual(ABC):
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

    def multihist(
        self,
        df: pd.DataFrame,
        columns: list,
        rows: int = 1,
        cols: int = 3,
        bins: int = 50,
        bandwidth: int = 3,
        hue: str = None,
        kde: bool = False,
        dodge: bool = False,
        discrete: bool = False,
        sharex=False,
        sharey=False,
        stat: str = "count",
        title: str = None,
        hsize: int = 16,
        vsize: int = 8,
        label_fontsize=16,
        heading_fontsize=20,
    ) -> None:

        if rows * cols < len(columns):
            raise Exception("Not enough rows and columns to fit the data")

        df = df[columns]

        fig, axes = plt.subplots(nrows=rows, ncols=cols, sharex=sharex, sharey=sharey, figsize=(hsize, vsize))
        fig.suptitle(title, fontsize=heading_fontsize)
        for i, ax in enumerate(fig.axes):
            sns.histplot(
                data=df,
                x=columns[i],
                bins=bins,
                hue=hue,
                kde=kde,
                discrete=False,
                stat=stat,
                palette=self._palette,
                ax=ax,
            )
            ax.set_title(columns[i])
            ax.set(xlabel=None)

        plt.tight_layout()
        plt.show()

    def multicount(
        self,
        df: pd.DataFrame,
        columns: list,
        rows: int = 1,
        cols: int = 3,
        orient: str = "h",
        stat: str = "count",
        max_horizontal=20,
        max_vertical=10,
        sharex="none",
        sharey="none",
        hsize: int = 20,
        vsize: int = 10,
        title=None,
        label_fontsize=16,
        heading_fontsize=20,
    ) -> None:

        if rows * cols < len(columns):
            raise Exception("Not enough rows and columns to fit the data")

        df = df[columns]

        fig, axes = plt.subplots(nrows=rows, ncols=cols, sharex=sharex, sharey=sharey, figsize=(hsize, vsize))
        fig.suptitle(title, fontsize=heading_fontsize)
        for i, ax in enumerate(fig.axes):
            counts = df[columns[i]].value_counts(sort=True).to_frame().reset_index()
            counts.columns = ["Value", "Count"]

            # Get the number of categories and orient the plot and subset the data accordingly
            n_categories = len(counts)
            if n_categories > max_vertical:
                orient = "h"
                if n_categories > max_horizontal:
                    counts = counts.iloc[0:max_horizontal]
                    # Subset the dataframe by the top remaining values in counts
                    df = df.loc[df[columns[i]].isin(counts["Value"])]
                    logger.info("Output for {} truncated at {} values".format(columns[i], max_horizontal))

            # Truncate product title
            if "title" in columns[i]:
                counts["Value"] = counts["Value"].values.astype(str).split()[0]

            sns.countplot(
                data=df,
                y=columns[i],
                order=counts["Value"],
                orient=orient,
                palette=self._palette,
                ax=ax,
            )

        plt.tight_layout()
        plt.show()

    def hist(
        self,
        df: pd.DataFrame,
        column: str,
        orient: str = "v",
        bins: int = 50,
        binwidth: int = 3,
        hue: str = None,
        kde: bool = False,
        dodge: bool = False,
        discrete: bool = False,
        stat: str = "count",
        title: str = None,
        label_fontsize=16,
        heading_fontsize=20,
    ) -> None:
        """Plot univariate or bivariate histograms to show distributions of datasets."""
        fig, ax = plt.subplots(figsize=(self._width, self._height))

        # Check if data are discrete
        discrete = isinstance(df[column], int)

        multiple = "dodge" if dodge else None
        if "v" in orient:
            if dodge:
                sns.histplot(
                    data=df,
                    x=column,
                    bins=bins,
                    hue=hue,
                    kde=kde,
                    multiple="dodge",
                    discrete=discrete,
                    stat=stat,
                    palette=self._palette,
                    ax=ax,
                )
            else:
                sns.histplot(
                    data=df,
                    x=column,
                    bins=bins,
                    hue=hue,
                    kde=kde,
                    discrete=discrete,
                    stat=stat,
                    palette=self._palette,
                    ax=ax,
                )
            plt.xlabel(titlelize(column), fontsize=label_fontsize)
            plt.ylabel("Count", fontsize=label_fontsize)
        else:
            if dodge:
                sns.histplot(
                    data=df,
                    y=column,
                    bins=bins,
                    hue=hue,
                    kde=kde,
                    multiple="dodge",
                    discrete=discrete,
                    stat=stat,
                    palette=self._palette,
                    ax=ax,
                )
            else:
                sns.histplot(
                    data=df,
                    y=column,
                    bins=bins,
                    hue=hue,
                    kde=kde,
                    discrete=discrete,
                    stat=stat,
                    palette=self._palette,
                    ax=ax,
                )
            plt.ylabel(titlelize(column), fontsize=label_fontsize)
            plt.xlabel("Count", fontsize=label_fontsize)
        if title:
            plt.title(titlelize(title), fontdict={"fontsize": heading_fontsize})
        plt.tight_layout()
        plt.show()

    def countplot(
        self,
        df: pd.DataFrame,
        column: str,
        hue: str = None,
        orient: str = "v",
        threshold=20,
        title=None,
        label_fontsize=16,
        heading_fontsize=20,
    ) -> None:
        """Show the counts of observations in each categorical bin using bars or lines.

        Args:
            df (pd.DataFrame): Data
            column (str): column name
            hue (str): column name by which to subset 'column'
            orient (str): The orientation for the bars
            threshold (n): Maximum number of categories for bars. Plots points connected by lines

        """
        # Prep Data
        counts = df[column].value_counts(sort=True, ascending=False).to_frame().reset_index()
        cumcounts = df[column].value_counts(sort=True, ascending=False).cumsum().to_frame().reset_index()
        counts.columns = ["Value", "Count"]
        cumcounts.columns = ["Value", "Cumulative"]

        # Truncate product title
        if "title" in column:
            counts["Value"] = counts["Value"].astype(str).apply(lambda x: x.split()[0])
            logger.info("Truncated product_title to first segment.")

        # Get the number of categories and plot the data.
        n_categories = len(counts)
        if n_categories <= threshold:
            fig, ax = plt.subplots(figsize=(self._width, self._height))
            sns.countplot(
                x=column,
                data=df,
                hue=hue,
                order=counts["Value"],
                orient=orient,
                palette=self._palette,
                ax=ax,
            )
            plt.xticks(rotation=45)
            if title:
                plt.title(titlelize(title), fontdict={"fontsize": heading_fontsize})

        else:
            fig, axes = plt.subplots(figsize=(self._width, self._height), nrows=2, ncols=1)
            sns.lineplot(
                x=counts.index,
                y=counts["Count"],
                markers=True,
                palette=self._palette,
                ax=axes[0],
            )
            sns.lineplot(
                x=cumcounts.index,
                y=cumcounts["Cumulative"],
                markers=True,
                palette=self._palette,
                ax=axes[1],
            )
            axes[0].axes.xaxis.set_ticklabels([])
            axes[0].title.set_text("Frequency Distribution")
            axes[1].title.set_text("Cumulative Frequency Distribution")
            axes[1].axes.xaxis.set_ticklabels([])

            if title:
                plt.suptitle(titlelize(title), fontsize=heading_fontsize)
        plt.tight_layout()
        plt.show()

    def boxplot(
        self,
        df: pd.DataFrame,
        column: str,
        hue: str = None,
        orient: str = "h",
        title=None,
        label_fontsize=16,
        heading_fontsize=20,
    ) -> None:
        """Show the counts of observations in each categorical bin using bars.

        Args:
            df (pd.DataFrame): Data
            column (str): column name
            hue (str): column name by which to subset 'column'

        """

        fig, ax = plt.subplots(figsize=(self._width, self._height))

        sns.boxplot(
            x=column,
            data=df,
            hue=hue,
            orient=orient,
            palette=self._palette,
            ax=ax,
        )

        if title:
            plt.title(titlelize(title), fontdict={"fontsize": heading_fontsize})
        plt.tight_layout()
        plt.show()

    def lineplot(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        title=None,
        label_fontsize=16,
        heading_fontsize=20,
    ) -> None:
        """Show the counts of observations in each categorical bin using bars.

        Args:
            df (pd.DataFrame): Data
            column (str): column name
            hue (str): column name by which to subset 'column'

        """

        fig, ax = plt.subplots(figsize=(self._width, self._height))

        sns.lineplot(
            x=x,
            y=y,
            data=df,
            palette=self._palette,
            ax=ax,
        )

        if title:
            plt.title(titlelize(title), fontdict={"fontsize": heading_fontsize})
        plt.tight_layout()
        plt.show()
