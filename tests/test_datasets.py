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
# Modified : Sunday, January 30th 2022, 5:48:00 pm                                                                         #
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

pd.options.display.float_format = "{:,.2f}".format
pd.set_option("display.width", 1000)
from datetime import datetime

from cvr.core.workspace import WorkspaceManager, Workspace
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

        wsm = WorkspaceManager()
        wsm.delete_workspace("dataset_tests")
        workspace = wsm.create_workspace(name="dataset_tests", description="Testing Dataset builder and Datasets")

        self.name = "After Hours"
        self.description = "Have a Margarita and Keep Fighting"
        self.stage = "Irepressible"
        self.request = DatasetRequest(
            name=self.name, description=self.description, stage=self.stage, workspace=workspace, data=self.df
        )
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
        assert self.ds.version == 0, logger.error("Failure in {}.".format(inspect.stack()[0][3]))
        assert isinstance(self.ds, Dataset), logger.error("Failure in {}.".format(inspect.stack()[0][3]))

        b.reset()
        self.ds = b.make_request(self.request).build()
        assert self.ds.version == 1, logger.error("Failure in {}.".format(inspect.stack()[0][3]))

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
        self.ds.profile.summary
        logger.info("\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def datatypes(self):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))
        self.ds.profile.datatypes
        logger.info("\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_numerics(self):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))
        self.ds.profile.analyze("sales_amount")
        logger.info("\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_categoricals(self):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))
        self.ds.profile.analyze("product_brand")
        logger.info("\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_missing(self):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))
        self.ds.profile.missing
        logger.info("\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_cardinality(self):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))
        self.ds.profile.cardinality
        logger.info("\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))


if __name__ == "__main__":
    t = DatasetTests()
    t.test_builder()
    t.test_info()
    t.test_summary()
    t.datatypes()
    t.test_numerics()
    t.test_categoricals()
    t.test_missing()
    t.test_cardinality()


#%%
