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
# Modified : Thursday, January 13th 2022, 12:21:13 am                                                                      #
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

from cvr.visuals.visualize import Visual
from cvr.data.datastore import DataStore

# ------------------------------------------------------------------------------------------------------------------------ #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------------------------------ #


class CriteoVisualizationTests:
    def __init__(self):
        self._dal = DataStore(stage="staged", env="t")
        self._df = self._dal.read()
        self._viz = Visual()

    def test_hist(self):

        logger.info(
            "    Started {} {}".format(self.__class__.__name__, inspect.stack()[0][3])
        )
        df = self._df.loc[self._df["sale"] == 1]
        self._viz.hist(df, "sales_amount")

        logger.info(
            "\t\tSuccessfully completed {} {}".format(
                self.__class__.__name__, inspect.stack()[0][3]
            )
        )


if __name__ == "__main__":
    logger.info("Testing CriteoVisualization")
    t = CriteoVisualizationTests()
    t.test_hist()
    logger.info("Completed CriteoVisualization tests")


#%%
