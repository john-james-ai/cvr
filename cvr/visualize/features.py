#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ======================================================================================================================== #
# Project  : Conversion Rate Prediction (CVR)                                                                              #
# Version  : 0.1.0                                                                                                         #
# File     : \features.py                                                                                                  #
# Language : Python 3.7.12                                                                                                 #
# ------------------------------------------------------------------------------------------------------------------------ #
# Author   : John James                                                                                                    #
# Email    : john.james.ai.studio@gmail.com                                                                                #
# URL      : https://github.com/john-james-ai/cvr                                                                          #
# ------------------------------------------------------------------------------------------------------------------------ #
# Created  : Sunday, January 30th 2022, 5:52:19 am                                                                         #
# Modified : Sunday, January 30th 2022, 11:11:30 am                                                                        #
# Modifier : John James (john.james.ai.studio@gmail.com)                                                                   #
# ------------------------------------------------------------------------------------------------------------------------ #
# License  : BSD 3-clause "New" or "Revised" License                                                                       #
# Copyright: (c) 2022 Bryant St. Labs                                                                                      #
# ======================================================================================================================== #
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import datashader as ds
import colorcet as cc
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import PowerTransformer
import logging
from cvr.core.visual import Visualizer

sns.set_style("whitegrid")
# ------------------------------------------------------------------------------------------------------------------------ #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------------------------------ #
class CategoricalFeatureVisualizer(Visualizer):
    def __init__(self, n_categories: int = 10) -> None:
        super(CategoricalFeatureVisualizer, self).__init__()
        self._counts = None
        self._column = None
        self._n_categories = n_categories
        self._title = "Categorical Variable Analysis"

    def fit(self, X: pd.Series):
        """Fits the data to the visualizer.

        Args:
            X (pd.Series): Pandas series containing the data.
        """
        self._column = X.to_frame().columns.values[0]
        self._counts = X.value_counts().to_frame().reset_index()
        self._counts.columns = ["Category", "Count"]
        self._counts["Cumulative"] = self._counts["Count"].cumsum()
        self._counts["Percent of Data"] = self._counts["Cumulative"] / len(X.dropna()) * 100
        self._counts["Rank"] = np.arange(1, len(self._counts) + 1)
        self._counts["Category Rank"] = self._counts["Rank"].astype("category")
        return self

    def plot(self) -> None:
        """Renders plot from fitted data."""

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(self._width, self._height))
        fig.suptitle(self._title, fontsize=self._heading_fontsize)

        df_barplot = self._counts.sort_values(by="Count", ascending=False, ignore_index=True)
        order = df_barplot["Category Rank"][: self._n_categories]

        sns.barplot(
            x="Count",
            y="Category Rank",
            palette=self._palette,
            order=order,
            data=df_barplot.loc[: self._n_categories],
            ax=axes[0],
        )
        axes[0].set_title("Top {} Categories".format(str(self._n_categories)))

        sns.lineplot(x="Rank", y="Percent of Data", palette=self._palette, data=self._counts, ax=axes[1])
        axes[1].set_title("Cumulative Frequency Distribution")
        axes[1].set(xlabel="Categories by Rank", ylabel="Percent of Data")

        title = "{}\n{}".format(self._title, self._column)
        fig.suptitle(title, fontsize=self._heading_fontsize)

        plt.tight_layout()
        plt.show()


# ------------------------------------------------------------------------------------------------------------------------ #
class NumericFeatureVisualizer(Visualizer):
    def __init__(self, width: int = 16, height: int = 12):
        super(NumericFeatureVisualizer, self).__init__(width=width, height=height)
        self._xformers = {
            "Original Data": self._identity,
            "Square Root": self._sqrt,
            "Cube Root": self._cuberoot,
            "Log": self._log,
            "Reciprocal": self._recip,
            "Yeo-Johnson": self._yeo_johnson,
        }
        self._data = {}
        self._tests = {}
        self._column = None
        self._title = "Numeric Feature Analysis"
        self._power_transformer = PowerTransformer(method="yeo-johnson", standardize=True)

    def fit(self, X: pd.Series):
        """Fits the data to various transformations.

        Args:
            X (pd.Series): A pandas series containing the data
        """
        self._column = X.to_frame().columns.values[0]
        X = X.dropna().to_numpy()

        for name, transformer in self._xformers.items():
            start = datetime.now()
            logger.debug("Fitting {} transformer".format(name))
            data = transformer(X)
            self._data[name] = pd.DataFrame(data, columns=[self._column])
            self._tests[name] = self._data[name].skew(axis=0, skipna=True).values[0]

            duration = datetime.now() - start
            logger.debug("Fitted {} transformer. Duration = {}".format(name, duration))
        return self

    def plot(self) -> None:

        title = "{}\n{}".format(self._title, self._column)

        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(self._width, self._height))
        fig.suptitle(title, fontsize=self._heading_fontsize)

        for transformation, ax in zip(self._data.keys(), axes.flatten()):
            start = datetime.now()
            logger.debug("Printing histogram for {}".format(transformation))
            sns.histplot(
                data=self._data[transformation], x=self._column, stat="density", kde=True, palette=self._palette, ax=ax
            )

            title = "{}\nSkew = {}".format(transformation, str(self._tests[transformation]))
            ax.set_title(label=title)
            duration = datetime.now() - start
            logger.debug("Printing histogram for {}. Duration = {}".format(transformation, duration))

        plt.tight_layout()
        plt.show()

    def _identity(self, x) -> np.array:
        return x

    def _square(self, x) -> np.array:
        return np.square(x)

    def _cube(self, x) -> np.array:
        return x ** 3

    def _sqrt(self, x) -> np.array:
        return np.sqrt(x)

    def _cuberoot(self, x):
        return x ** (1 / 3)

    def _log(self, x):
        return np.log(x + 1)

    def _recip(self, x):
        return 1 / (x + 1)

    def _yeo_johnson(self, x):
        x = x.reshape(-1, 1)
        return self._power_transformer.fit_transform(x)
