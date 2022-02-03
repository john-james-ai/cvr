#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ============================================================================ #
# Project  : Deep Learning for Conversion Rate Prediction (CVR)                #
# Version  : 0.1.0                                                             #
# File     : \test_datasets.py                                                 #
# Language : Python 3.7.12                                                     #
# ---------------------------------------------------------------------------- #
# Author   : John James                                                        #
# Email    : john.james.ai.studio@gmail.com                                    #
# URL      : https://github.com/john-james-ai/cvr                              #
# ---------------------------------------------------------------------------- #
# Created  : Tuesday, January 18th 2022, 11:09:05 am                           #
# Modified : Tuesday, February 1st 2022, 7:31:10 am                            #
# Modifier : John James (john.james.ai.studio@gmail.com)                       #
# ---------------------------------------------------------------------------- #
# License  : BSD 3-clause "New" or "Revised" License                           #
# Copyright: (c) 2022 Bryant St. Labs                                          #
# ============================================================================ #
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

from cvr.core.workspace import Project, Workspace
from cvr.core.dataset import Dataset, DatasetBuilder, DatasetRequest
from cvr.utils.sampling import sample_df
from cvr.data import criteo_columns, criteo_dtypes

# ---------------------------------------------------------------------------- #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# ---------------------------------------------------------------------------- #


class DatasetTests:
    def __init__(self):
        start = datetime.now()
        filepath = "tests\data\criteo\staged\criteo_sample.pkl"
        self.df = pd.read_pickle(filepath)

        wsm = Project()
        wsm.delete_workspace("dataset_tests")
        workspace = wsm.create_workspace(
            name="dataset_tests",
            description="Testing Dataset builder and Datasets",
        )

        self.name = "After Hours"
        self.description = "Have a Margarita and Keep Fighting"
        self.stage = "Irepressible"
        self.request = DatasetRequest(
            name=self.name,
            description=self.description,
            stage=self.stage,
            workspace_name=workspace.name,
            workspace_directory=workspace.directory,
            data=self.df,
        )
        end = datetime.now()
        duration = end - start

    def test_builder(self):
        logger.info(
            "\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3])
        )
        start = datetime.now()
        b = DatasetBuilder()
        self.ds = b.make_request(self.request).build().dataset

        assert self.ds.name == self.name, logger.error(
            "Failure in {}.".format(inspect.stack()[0][3])
        )
        assert self.ds.stage == self.stage, logger.error(
            "Failure in {}.".format(inspect.stack()[0][3])
        )
        assert self.ds.version == 0, logger.error(
            "Failure in {}.".format(inspect.stack()[0][3])
        )
        assert isinstance(self.ds, Dataset), logger.error(
            "Failure in {}.".format(inspect.stack()[0][3])
        )

        b.reset()
        self.ds = b.make_request(self.request).build().dataset
        assert self.ds.version == 1, logger.error(
            "Failure in {}.".format(inspect.stack()[0][3])
        )

        end = datetime.now()
        duration = end - start
        logger.info(
            "\tSuccessfully completed {} {}".format(
                self.__class__.__name__, inspect.stack()[0][3]
            )
        )

    def test_info(self):
        logger.info(
            "\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3])
        )
        self.ds.info
        logger.info(
            "\tSuccessfully completed {} {}".format(
                self.__class__.__name__, inspect.stack()[0][3]
            )
        )

    def test_stats(self):
        logger.info(
            "\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3])
        )
        rf = self.ds.rank_frequencies(column="product_brand")
        crf = self.ds.rank_frequencies(column="product_brand")

        assert len(rf) > 100, logger.error(
            "Failure in {}.".format(inspect.stack()[0][3])
        )
        assert len(rf) == len(crf), logger.error(
            "Failure in {}.".format(inspect.stack()[0][3])
        )
        logger.info(
            "\tSuccessfully completed {} {}".format(
                self.__class__.__name__, inspect.stack()[0][3]
            )
        )

    def test_teardown(self):
        logger.info(
            "\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3])
        )
        wsm = Project()
        wsm.delete_workspace("dataset_tests")
        logger.info(
            "\tSuccessfully completed {} {}".format(
                self.__class__.__name__, inspect.stack()[0][3]
            )
        )


if __name__ == "__main__":
    logger.info("Started Datasest Tests")
    t = DatasetTests()
    t.test_builder()
    t.test_info()
    t.test_stats()
    t.test_teardown()
    logger.info("Completed Datasest Tests")


#%%
