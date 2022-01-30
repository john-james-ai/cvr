#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ======================================================================================================================== #
# Project  : Explainable Recommendation (XRec)                                                                             #
# Version  : 0.1.0                                                                                                         #
# File     : \test_visualization.py                                                                                        #
# Language : Python 3.8                                                                                                    #
# ------------------------------------------------------------------------------------------------------------------------ #
# Author   : John James                                                                                                    #
# Email    : john.james.ai.studio@gmail.com                                                                                #
# URL      : https://github.com/john-james-ai/xrec                                                                         #
# ------------------------------------------------------------------------------------------------------------------------ #
# Created  : Friday, December 31st 2021, 1:29:25 pm                                                                        #
# Modified : Sunday, January 30th 2022, 10:31:13 am                                                                        #
# Modifier : John James (john.james.ai.studio@gmail.com)                                                                   #
# ------------------------------------------------------------------------------------------------------------------------ #
# License  : BSD 3-clause "New" or "Revised" License                                                                       #
# Copyright: (c) 2021 Bryant St. Labs                                                                                      #
# ======================================================================================================================== #
#%%
import os
import pytest
import logging
import pandas as pd
import inspect
import cProfile

from cvr.visualize.features import CategoricalFeatureVisualizer, NumericFeatureVisualizer


# ------------------------------------------------------------------------------------------------------------------------ #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------------------------------ #


class CategoricalFeatureVisualizerTests:
    def __init__(self):
        filepath = "tests\data\criteo\staged\criteo_sample.pkl"
        self._df = pd.read_pickle(filepath)
        self._viz = CategoricalFeatureVisualizer()
        self._column = "product_brand"

    def test_plot(self):

        logger.info("\t\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        self._viz.fit(self._df[self._column])
        self._viz.plot()

        logger.info("\t\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))


class NumericFeatureVisualizerTests:
    def __init__(self):
        filepath = "tests\data\criteo\staged\criteo_sample.pkl"
        self._df = pd.read_pickle(filepath)
        self._viz = NumericFeatureVisualizer()
        self._column = "sales_amount"

    def test_plot(self):

        logger.info("    Started {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        self._viz.fit(self._df[self._column])
        self._viz.plot()

        logger.info("\t\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))


if __name__ == "__main__":
    logger.info("Started VisualizerTests")
    # t = CategoricalFeatureVisualizerTests()
    # t.test_plot()
    t = NumericFeatureVisualizerTests()
    t.test_plot()
    logger.info("Completed VisualizerTests")


#%%
