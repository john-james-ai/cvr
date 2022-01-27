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
# Modified : Wednesday, January 26th 2022, 9:34:25 pm                                                                      #
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

from cvr.core.dataset import Dataset, DatasetBuilder, DatasetRequest
from cvr.utils.sampling import sample_df
from cvr.data import criteo_columns, criteo_dtypes

# ------------------------------------------------------------------------------------------------------------------------ #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------------------------------ #


class DatasetTests:
    def __init__(self):
        start = datetime.now()
        filepath = "tests\data\criteo\staged\criteo_sample.pkl"
        self.df = pd.read_pickle(filepath)
        self.request = DatasetRequest(
            name="After Hours", description="Have a Margarita and Keep Fighting", stage="Left", data=self.df
        )
        self.name = "After Hours"
        self.description = "Have a Margarita and Keep Fighting"
        self.stage = "Left"
        end = datetime.now()
        duration = end - start
        logger.info("Load time {}".format(duration))

    def test_builder(self):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))
        start = datetime.now()
        b = DatasetBuilder()
        self.ds = b.make_request(self.request).build()

        assert self.ds.name == self.name, logger.error("Failure in {}.".format(inspect.stack()[0][3]))
        assert self.ds.stage == self.stage, logger.error("Failure in {}.".format(inspect.stack()[0][3]))
        assert isinstance(self.ds, Dataset), logger.error("Failure in {}.".format(inspect.stack()[0][3]))

        end = datetime.now()
        duration = end - start
        logger.info("Build time {}".format(duration))
        logger.info("\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_info(self):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))
        self.ds.info
        logger.info("\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_summary(self):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))
        self.ds.summary
        logger.info("\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def datatypes(self):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))
        self.ds.datatypes
        logger.info("\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_numerics(self):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))
        self.ds.numerics
        logger.info("\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_categoricals(self):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))
        self.ds.categoricals
        logger.info("\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_missing_summary(self):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))
        self.ds.missing_summary
        logger.info("\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_missing(self):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))
        self.ds.missing
        logger.info("\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_cardinality(self):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))
        self.ds.cardinality
        logger.info("\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_metrics(self):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))
        self.ds.metrics
        logger.info("\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_numeric_analysis(self):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))
        self.ds.numeric_analysis("sales_amount")
        logger.info("\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_categorical_analysis(self):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))
        # self.ds.categorical_analysis("device_type")
        # self.ds.categorical_analysis("product_country")
        self.ds.categorical_analysis("product_brand")
        logger.info("\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))


if __name__ == "__main__":
    t = DatasetTests()
    t.test_builder()
    t.test_info()
    t.test_summary()
    t.datatypes()
    t.test_numerics()
    t.test_categoricals()
    t.test_missing_summary()
    t.test_missing()
    t.test_cardinality()
    t.test_metrics()
    t.test_numeric_analysis()
    t.test_categorical_analysis()


#%%
