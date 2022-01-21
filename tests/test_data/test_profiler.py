#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ======================================================================================================================== #
# Project  : Explainable Recommendation (XRec)                                                                             #
# Version  : 0.1.0                                                                                                         #
# File     : \test_profiler.py                                                                                             #
# Language : Python 3.8                                                                                                    #
# ------------------------------------------------------------------------------------------------------------------------ #
# Author   : John James                                                                                                    #
# Email    : john.james.ai.studio@gmail.com                                                                                #
# URL      : https://github.com/john-james-ai/xrec                                                                         #
# ------------------------------------------------------------------------------------------------------------------------ #
# Created  : Tuesday, December 28th 2021, 1:06:31 am                                                                       #
# Modified : Sunday, January 16th 2022, 3:19:19 pm                                                                         #
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
from datetime import datetime
import inspect

from cvr.data.profile import DataProfiler
from cvr.data import raw_dtypes

# ------------------------------------------------------------------------------------------------------------------------ #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------------------------------ #


class DataProfilerTests:
    def __init__(self):
        filepath = "tests\\test_data\staged\criteo.csv"
        self._df = pd.read_csv(filepath, dtype=raw_dtypes)
        self._profiler = DataProfiler(self._df)

    def test_summary(self):
        logger.info("    Started {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        d = self._profiler.summary
        assert d["rows"] == self._df.shape[0], logger.error("Failure in {}.".format(inspect.stack()[0][3]))
        assert d["columns"] == self._df.shape[1], logger.error("Failure in {}.".format(inspect.stack()[0][3]))
        assert d["size"] > 100, logger.error("Failure in {}.".format(inspect.stack()[0][3]))

        logger.info("\t\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_numerics(self):
        logger.info("    Started {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        n = self._profiler.numerics
        print(n)
        assert isinstance(n, pd.DataFrame), logger.error("Failure in {}.".format(inspect.stack()[0][3]))

        logger.info("\t\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_categoricals(self):
        logger.info("    Started {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        n = self._profiler.categoricals
        print(n)
        assert isinstance(n, pd.DataFrame), logger.error("Failure in {}.".format(inspect.stack()[0][3]))

        logger.info("\t\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_missing(self):
        logger.info("    Started {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        m = self._profiler.missing
        print(m)
        assert isinstance(m, dict), logger.error("Failure in {}.".format(inspect.stack()[0][3]))

        logger.info("\t\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_cardinality(self):
        logger.info("    Started {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        c = self._profiler.cardinality
        print(c)
        assert isinstance(c, pd.DataFrame), logger.error("Failure in {}.".format(inspect.stack()[0][3]))

        logger.info("\t\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_metrics(self):
        logger.info("    Started {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        c = self._profiler.metrics
        print(c)
        assert isinstance(c, dict), logger.error("Failure in {}.".format(inspect.stack()[0][3]))

        logger.info("\t\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_datatypes(self):
        logger.info("    Started {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        c = self._profiler.datatypes
        print(c)
        assert isinstance(c, pd.DataFrame), logger.error("Failure in {}.".format(inspect.stack()[0][3]))

        logger.info("\t\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))


if __name__ == "__main__":
    t = DataProfilerTests()
    t.test_summary
    t.test_numerics()
    t.test_categoricals()
    t.test_missing()
    t.test_cardinality()
    t.test_metrics()
    t.test_datatypes()

#%%
