#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ============================================================================ #
# Project  : Deep Learning for Conversion Rate Prediction (CVR)                #
# Version  : 0.1.0                                                             #
# File     : \test_assets.py                                                   #
# Language : Python 3.7.12                                                     #
# ---------------------------------------------------------------------------- #
# Author   : John James                                                        #
# Email    : john.james.ai.studio@gmail.com                                    #
# URL      : https://github.com/john-james-ai/cvr                              #
# ---------------------------------------------------------------------------- #
# Created  : Wednesday, February 2nd 2022, 9:47:17 pm                          #
# Modified : Thursday, February 3rd 2022, 10:36:50 am                          #
# Modifier : John James (john.james.ai.studio@gmail.com)                       #
# ---------------------------------------------------------------------------- #
# License  : BSD 3-clause "New" or "Revised" License                           #
# Copyright: (c) 2022 Bryant St. Labs                                          #
# ============================================================================ #
#%%
import os
from datetime import datetime
import pytest
import logging
import inspect
import pandas as pd
import shutil

pd.set_option("display.width", 1000)
pd.options.display.float_format = "{:,.2f}".format

from cvr.core.asset import AssetRepo
from cvr.core.dataset import Dataset, DatasetFactory

# ---------------------------------------------------------------------------- #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# ---------------------------------------------------------------------------- #


class AssetTests:
    def __init__(self):

        self.names = ["name_1", "name_2", "name_3"]
        self.descriptions = ["desc_1", "desc_2", "desc_3"]
        self.stages = ["stage_1", "stage_2", "stage_3"]
        self.creators = ["kruder", "dorfmeister", "bukem"]
        self.asset_type = "dataset"
        self.start = datetime.now()
        self.end = None
        self.duration = None
        self.minutes = None
        logger.info("Started AssetTests at {}".format(self.start))

        self.repo = None

        filepath = "tests\data\criteo\staged\criteo_sample.pkl"
        self.df = pd.read_pickle(filepath)[0:100]
        print(self.df.shape)

        filepath = "tests\data\\asset_repo"
        shutil.rmtree(filepath, ignore_errors=True)

        self.factory = DatasetFactory(workspace_directory=filepath)
        self.repo = AssetRepo(workspace_directory=filepath)

    def test_create_repo(self):
        logger.info(
            "\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3])
        )

        for i in range(3):

            for name, stage, creator, description in zip(
                self.names, self.stages, self.creators, self.descriptions
            ):
                # Test DatasetFactory
                dataset = self.factory.create(
                    name=name,
                    stage=stage,
                    creator=creator,
                    description=description,
                    data=self.df,
                )

                assert dataset.asset_type == "dataset", logger.error(
                    "Failure in {}.".format(inspect.stack()[0][3])
                )
                assert dataset.name == name, logger.error(
                    "Failure in {}.".format(inspect.stack()[0][3])
                )
                assert dataset.stage == stage, logger.error(
                    "Failure in {}.".format(inspect.stack()[0][3])
                )
                assert dataset.description == description, logger.error(
                    "Failure in {}.".format(inspect.stack()[0][3])
                )
                assert dataset.creator == creator, logger.error(
                    "Failure in {}.".format(inspect.stack()[0][3])
                )
                assert dataset.version == i + 1, logger.error(
                    "Failure in {}.".format(inspect.stack()[0][3])
                )

                logger.info(
                    "\t\tCreated\t{}\t{} of {}.\tCreator: {}\tVersion: {}.".format(
                        dataset.asset_type,
                        dataset.name,
                        dataset.stage,
                        dataset.creator,
                        str(dataset.version),
                    )
                )

                # Test AssetRepo
                self.repo.add(dataset)

                # Test Exists
                assert self.repo.asset_exists(
                    asset_type=dataset.asset_type,
                    name=dataset.name,
                    stage=dataset.stage,
                    version=dataset.version,
                ), logger.error("Failure in {}.".format(inspect.stack()[0][3]))

                logger.info(
                    "\t\tTested\t{}\t{} of {}.\tCreator: {}\tVersion: {}.".format(
                        dataset.asset_type,
                        dataset.name,
                        dataset.stage,
                        dataset.creator,
                        str(dataset.version),
                    )
                )

                # Test Reporting
                ds2 = self.repo.get(
                    asset_type=dataset.asset_type,
                    name=dataset.name,
                    stage=dataset.stage,
                    version=dataset.version,
                )

                # Test Get
                assert dataset.asset_type == ds2.asset_type, logger.error(
                    "Failure in {}.".format(inspect.stack()[0][3])
                )
                assert dataset.name == ds2.name, logger.error(
                    "Failure in {}.".format(inspect.stack()[0][3])
                )
                assert dataset.stage == ds2.stage, logger.error(
                    "Failure in {}.".format(inspect.stack()[0][3])
                )
                assert dataset.description == ds2.description, logger.error(
                    "Failure in {}.".format(inspect.stack()[0][3])
                )
                assert dataset.creator == ds2.creator, logger.error(
                    "Failure in {}.".format(inspect.stack()[0][3])
                )
                assert dataset.version == ds2.version, logger.error(
                    "Failure in {}.".format(inspect.stack()[0][3])
                )

                logger.info(
                    "\t\tRetrieved\t{}\t{} of {}.\tCreator: {}\tVersion: {}.".format(
                        dataset.asset_type,
                        dataset.name,
                        dataset.stage,
                        dataset.creator,
                        str(dataset.version),
                    )
                )

        logger.info(
            "\tSuccessfully completed {} {}".format(
                self.__class__.__name__, inspect.stack()[0][3]
            )
        )

    def test_delete_repo(self):
        logger.info(
            "\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3])
        )

        for name, stage in zip(self.names, self.stages):

            for i in range(3):

                # Test DatasetFactory
                assert self.repo.asset_exists(
                    asset_type=self.asset_type,
                    name=name,
                    stage=stage,
                    version=i + 1,
                ), logger.error("Failure in {}.".format(inspect.stack()[0][3]))

            # Delete
            self.repo.delete_asset(asset_type=self.asset_type, name=name, stage=stage)

            for i in range(5):

                # Test DatasetFactory
                assert not self.repo.asset_exists(
                    asset_type=self.asset_type,
                    name=name,
                    stage=stage,
                    version=i + 1,
                ), logger.error("Failure in {}.".format(inspect.stack()[0][3]))

            print(self.repo.get_assets())

        logger.info(
            "\tSuccessfully completed {} {}".format(
                self.__class__.__name__, inspect.stack()[0][3]
            )
        )

    def test_teardown(self):
        filepath = "tests\data\\asset_repo"
        shutil.rmtree(filepath)
        self.end = datetime.now()
        self.duration = self.end - self.start
        self.minutes = round(self.duration.total_seconds() / 60, 2)
        logger.info(
            "Completed Asset Tests at {}. Duration: {} minutes".format(
                self.end, str(self.minutes)
            )
        )


if __name__ == "__main__":

    t = AssetTests()
    t.test_create_repo()
    t.test_delete_repo()
    t.test_teardown()

#%%
