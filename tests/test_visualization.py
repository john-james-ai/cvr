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
# Modified : Thursday, January 27th 2022, 8:08:50 am                                                                       #
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

from cvr.data.profile import DataProfiler
from cvr.visuals.visualize import Visual
from cvr.visuals.frequency import Frequency

# ------------------------------------------------------------------------------------------------------------------------ #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------------------------------ #


class CriteoVisualizationTests:
    def __init__(self):
        filepath = "tests\data\criteo\staged\criteo_sample.pkl"
        self._df = pd.read_pickle(filepath)
        self._profiler = DataProfiler()
        self._profiler.build(self._df)
        self._viz = Visual()

    def test_hist(self):

        logger.info("    Started {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))
        df = self._df.loc[self._df["sale"] == 1]
        self._viz.hist(df, "sales_amount")

        logger.info("\t\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_freq(self):
        logger.info("    Started {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))
        profiler = DataProfiler()
        profiler.build(self._df)
        counts = profiler.frequency_counts(column="product_country")
        logger.info("Profile Complete")
        print(counts.head())
        freq = Frequency()
        freq.analysis(
            df=counts,
            column="product_country",
            col_category="Category Rank",
            col_freq="Count",
            col_cum="Cumulative",
            col_pct_cum="Pct Cum",
            col_rank="Rank",
        )

        logger.info("\t\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))


if __name__ == "__main__":
    logger.info("Testing CriteoVisualization")
    t = CriteoVisualizationTests()
    t.test_freq()
    logger.info("Completed CriteoVisualization tests")


#%%
