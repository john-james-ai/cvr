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
# Modified : Saturday, February 5th 2022, 3:31:38 am                           #
# Modifier : John James (john.james.ai.studio@gmail.com)                       #
# ---------------------------------------------------------------------------- #
# License  : BSD 3-clause "New" or "Revised" License                           #
# Copyright: (c) 2022 Bryant St. Labs                                          #
# ============================================================================ #
#%%
import logging
import inspect
import shutil
import pandas as pd

from cvr.core.dataset import Dataset, DatasetRepo
from cvr.utils.config import AIDConfig


pd.options.display.float_format = "{:,.2f}".format
pd.set_option("display.width", 1000)


# ---------------------------------------------------------------------------- #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# ---------------------------------------------------------------------------- #


class DatasetTests:
    def __init__(self):
        filepath = "tests\data\criteo\staged\criteo_sample.pkl"
        self._repo_directory = "tests\\test_datasets"
        shutil.rmtree(self._repo_directory, ignore_errors=True)

        self.df = pd.read_pickle(filepath)
        self.asset_type = "dataset"
        self.name = "spiers"
        self.stage = "moh"
        self.description = "German's couldn't believe what they were seeing"
        self.creator = "rangers"
        aid = AIDConfig()
        aid.reset()

    def test_factory(self):

        logger.info(
            "\tStarted {} {}".format(
                self.__class__.__name__, inspect.stack()[0][3]
            )
        )

        repo = DatasetRepo(self._repo_directory)
        self.ds = repo.create(
            name=self.name,
            stage=self.stage,
            creator=self.creator,
            description=self.description,
            data=self.df,
        )

        assert self.ds.name == self.name, logger.error(
            "Failure in {}.".format(inspect.stack()[0][3])
        )
        assert self.ds.stage == self.stage, logger.error(
            "Failure in {}.".format(inspect.stack()[0][3])
        )
        assert self.ds.asset_type == self.asset_type, logger.error(
            "Failure in {}.".format(inspect.stack()[0][3])
        )
        assert self.ds.creator == self.creator, logger.error(
            "Failure in {}.".format(inspect.stack()[0][3])
        )
        assert self.ds.version == 0, logger.error(
            "Failure in {}.".format(inspect.stack()[0][3])
        )
        assert self.ds.aid == "0000", logger.error(
            "Failure in {}.".format(inspect.stack()[0][3])
        )
        assert isinstance(self.ds, Dataset), logger.error(
            "Failure in {}.".format(inspect.stack()[0][3])
        )

        logger.info(
            "\tSuccessfully completed {} {}".format(
                self.__class__.__name__, inspect.stack()[0][3]
            )
        )

    def test_properties(self):
        logger.info(
            "\tStarted {} {}".format(
                self.__class__.__name__, inspect.stack()[0][3]
            )
        )
        assert self.ds.size > 10, logger.error(
            "Failure in {}.".format(inspect.stack()[0][3])
        )
        self.ds.info
        self.ds.summary

        logger.info(
            "\tSuccessfully completed {} {}".format(
                self.__class__.__name__, inspect.stack()[0][3]
            )
        )

    def test_data_access(self):
        logger.info(
            "\tStarted {} {}".format(
                self.__class__.__name__, inspect.stack()[0][3]
            )
        )

        self.ds.head(5)
        self.ds.tail(3)
        self.ds.sample(100)

        logger.info(
            "\tSuccessfully completed {} {}".format(
                self.__class__.__name__, inspect.stack()[0][3]
            )
        )

    def test_stats(self):
        logger.info(
            "\tStarted {} {}".format(
                self.__class__.__name__, inspect.stack()[0][3]
            )
        )

        print(self.ds.describe("product_brand"))
        print(self.ds.describe("sales_amount"))
        print(self.ds.rank_frequencies("product_country"))
        print(self.ds.cum_rank_frequencies("product_category_1", n=10))

        logger.info(
            "\tSuccessfully completed {} {}".format(
                self.__class__.__name__, inspect.stack()[0][3]
            )
        )


if __name__ == "__main__":
    logger.info("Started Datasest Tests")
    t = DatasetTests()
    t.test_factory()
    t.test_properties()
    t.test_data_access()
    t.test_stats()

    logger.info("Completed Datasest Tests")


#%%
