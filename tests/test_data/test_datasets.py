#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ======================================================================================================================== #
# Project  : Explainable Recommendation (XRec)                                                                             #
# Version  : 0.1.0                                                                                                         #
# File     : \test_processing.py                                                                                           #
# Language : Python 3.8                                                                                                    #
# ------------------------------------------------------------------------------------------------------------------------ #
# Author   : John James                                                                                                    #
# Email    : john.james.ai.studio@gmail.com                                                                                #
# URL      : https://github.com/john-james-ai/xrec                                                                         #
# ------------------------------------------------------------------------------------------------------------------------ #
# Created  : Monday, December 27th 2021, 4:41:43 am                                                                        #
# Modified : Tuesday, January 18th 2022, 7:40:51 pm                                                                        #
# Modifier : John James (john.james.ai.studio@gmail.com)                                                                   #
# ------------------------------------------------------------------------------------------------------------------------ #
# License  : BSD 3-clause "New" or "Revised" License                                                                       #
# Copyright: (c) 2021 Bryant St. Labs                                                                                      #
# ======================================================================================================================== #
#%%
import os
import pytest
import logging
import inspect
import shutil
import pandas as pd
from datetime import datetime

from cvr.data.datasets import Dataset, DatasetBuilder
from cvr.data.source import CriteoTransformer

# ------------------------------------------------------------------------------------------------------------------------ #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------------------------------ #


class DatasetTests:
    def __init__(self):
        filepath = "data\criteo\\raw\criteo.csv"
        t = CriteoTransformer()
        t.transform(filepath, nrows=100)
        self._df = t.data

    def test_builder(self):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        b = DatasetBuilder("test")
        self._dataset = b.set_data(self._df).set_name("test").set_stage("staged").set_creator("j2").build().dataset
        assert self._dataset.name == "test", logger.error("Failure in {}.".format(inspect.stack()[0][3]))
        assert self._dataset.workspace == "test", logger.error("Failure in {}.".format(inspect.stack()[0][3]))
        assert self._dataset.stage == "staged", logger.error("Failure in {}.".format(inspect.stack()[0][3]))
        assert isinstance(self._dataset, Dataset), logger.error("Failure in {}.".format(inspect.stack()[0][3]))

        self._dataset.verbose = True

        logger.info("\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_info(self):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))
        self._dataset.info
        logger.info("\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_summary(self):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))
        self._dataset.summary
        logger.info("\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_data_types(self):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))
        self._dataset.data_types
        logger.info("\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_numerics(self):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))
        self._dataset.numeric_statistics
        logger.info("\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_categoricals(self):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))
        self._dataset.categorical_statistics
        logger.info("\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_missing_summary(self):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))
        self._dataset.missing_summary
        logger.info("\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_missing(self):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))
        self._dataset.missing
        logger.info("\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_cardinality(self):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))
        self._dataset.cardinality
        logger.info("\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_metrics(self):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))
        self._dataset.metrics
        logger.info("\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_metadata(self):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))
        self._dataset.metadata
        logger.info("\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_numeric_analysis(self):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))
        self._dataset.numeric_analysis("sales_amount")
        logger.info("\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_categorical_analysis(self):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))
        # self._dataset.categorical_analysis("device_type")
        # self._dataset.categorical_analysis("product_country")
        self._dataset.categorical_analysis("product_brand")
        logger.info("\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))


class DatasetBuilderTests:
    def __init__(self):
        filepath = "data\criteo\\raw\criteo.csv"
        t = CriteoTransformer()
        t.transform(filepath, nrows=100)
        self._df = t.data

    def test_builder(self):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        b = DatasetBuilder("test")
        self._dataset = b.set_data(self._df).set_name("test").set_stage("staged").set_creator("j2").build().dataset
        assert self._dataset.name == "test", logger.error("Failure in {}.".format(inspect.stack()[0][3]))
        assert self._dataset.workspace == "test", logger.error("Failure in {}.".format(inspect.stack()[0][3]))
        assert self._dataset.stage == "staged", logger.error("Failure in {}.".format(inspect.stack()[0][3]))
        assert isinstance(self._dataset, Dataset), logger.error("Failure in {}.".format(inspect.stack()[0][3]))

        self._dataset.verbose = True

        logger.info("\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))


if __name__ == "__main__":
    t = DatasetTests()
    t.test_builder()
    # t.test_info()
    # t.test_summary()
    # t.test_data_types()
    # t.test_numerics()
    # t.test_categoricals()
    # t.test_missing_summary()
    # t.test_missing()
    # t.test_cardinality()
    # t.test_metrics()
    # t.test_metadata()
    # t.test_numeric_analysis()
    t.test_categorical_analysis()

    t = DatasetBuilderTests()
    t.test_builder()

#%%
