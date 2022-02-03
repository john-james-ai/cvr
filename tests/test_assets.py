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
# Modified : Thursday, February 3rd 2022, 4:59:03 am                           #
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


from cvr.core.dataset import Dataset, DatasetRepo

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
        self.df = pd.read_pickle(filepath)

        filepath = "tests\data\\asset_repo"
        shutil.rmtree(filepath, ignore_errors=True)
        self.repo = DatasetRepo(filepath)

    def test_asset(self):
        logger.info(
            "\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3])
        )

        for name, stage, creator, description in zip(
            self.names, self.stages, self.creators, self.descriptions
        ):
            asset = Dataset(
                name=name,
                stage=stage,
                creator=creator,
                description=description,
                data=self.df,
            )
            assert asset.asset_type == "dataset", logger.error(
                "Failure in {}.".format(inspect.stack()[0][3])
            )
            assert asset.name == name, logger.error(
                "Failure in {}.".format(inspect.stack()[0][3])
            )
            assert asset.stage == stage, logger.error(
                "Failure in {}.".format(inspect.stack()[0][3])
            )
            assert asset.description == description, logger.error(
                "Failure in {}.".format(inspect.stack()[0][3])
            )
            assert asset.creator == creator, logger.error(
                "Failure in {}.".format(inspect.stack()[0][3])
            )

        logger.info(
            "\tSuccessfully completed {} {}".format(
                self.__class__.__name__, inspect.stack()[0][3]
            )
        )

    def test_repo_add_versions(self):
        logger.info(
            "\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3])
        )

        name, stage, creator, description, data = (
            "triple",
            "back",
            "hack",
            "seeing 3s",
            self.df,
        )

        for i in range(3):
            asset = Dataset(
                name=name,
                stage=stage,
                creator=creator,
                description=description,
                data=self.df,
            )
            self.repo.add(asset)

            asset2 = self.repo.get(
                asset_type=self.asset_type, name=name, stage=stage, version=i + 1
            )

            assert asset.name == asset2.name, logger.error(
                "Failure in {}.".format(inspect.stack()[0][3])
            )
            assert asset.creator == asset2.creator, logger.error(
                "Failure in {}.".format(inspect.stack()[0][3])
            )
            assert asset.description == asset2.description, logger.error(
                "Failure in {}.".format(inspect.stack()[0][3])
            )
            assert asset.stage == asset2.stage, logger.error(
                "Failure in {}.".format(inspect.stack()[0][3])
            )
            assert asset.version == i + 1, logger.error(
                "Failure in {}.".format(inspect.stack()[0][3])
            )

        print(self.repo.get_assets())

        logger.info(
            "\tSuccessfully completed {} {}".format(
                self.__class__.__name__, inspect.stack()[0][3]
            )
        )

    def test_repo_add_get_assets(self):
        logger.info(
            "\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3])
        )

        for i in range(5):
            for name, stage, creator, description in zip(
                self.names, self.stages, self.creators, self.descriptions
            ):
                asset = Dataset(
                    name=name,
                    stage=stage,
                    creator=creator,
                    description=description,
                    data=self.df,
                )
                self.repo.add(asset)
                asset2 = self.repo.get(
                    asset_type=self.asset_type, name=name, stage=stage, version=i + 1
                )

                assert asset.name == asset2.name, logger.error(
                    "Failure in {}.".format(inspect.stack()[0][3])
                )
                assert asset.creator == asset2.creator, logger.error(
                    "Failure in {}.".format(inspect.stack()[0][3])
                )
                assert asset.description == asset2.description, logger.error(
                    "Failure in {}.".format(inspect.stack()[0][3])
                )
                assert asset.stage == asset2.stage, logger.error(
                    "Failure in {}.".format(inspect.stack()[0][3])
                )
                assert asset.version == i + 1, logger.error(
                    "Failure in {}.".format(inspect.stack()[0][3])
                )

        print(self.repo.get_assets())

        logger.info(
            "\tSuccessfully completed {} {}".format(
                self.__class__.__name__, inspect.stack()[0][3]
            )
        )

    def test_repo_delete_most_recent_version(self):
        logger.info(
            "\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3])
        )

        for name, stage, creator, description in zip(
            self.names, self.stages, self.creators, self.descriptions
        ):
            self.repo.delete_version(asset_type=self.asset_type, stage=stage, name=name)

            assert self.repo.asset_exists(
                asset_type=self.asset_type, stage=stage, name=name
            ), logger.error("Failure in {}.".format(inspect.stack()[0][3]))

            for i in range(4):
                assert self.repo.version_exists(
                    asset_type=self.asset_type, stage=stage, name=name, version=i + 1
                ), logger.error("Failure in {}.".format(inspect.stack()[0][3]))

            assert not self.repo.version_exists(
                asset_type=self.asset_type, stage=stage, name=name, version=5
            ), logger.error("Failure in {}.".format(inspect.stack()[0][3]))

        print(self.repo.get_assets())

        logger.info(
            "\tSuccessfully completed {} {}".format(
                self.__class__.__name__, inspect.stack()[0][3]
            )
        )

    def test_repo_delete_version(self):
        logger.info(
            "\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3])
        )

        for name, stage, creator, description in zip(
            self.names, self.stages, self.creators, self.descriptions
        ):
            self.repo.delete_version(
                asset_type=self.asset_type, stage=stage, name=name, version=1
            )

            assert self.repo.asset_exists(
                asset_type=self.asset_type, stage=stage, name=name
            ), logger.error("Failure in {}.".format(inspect.stack()[0][3]))

            assert not self.repo.version_exists(
                asset_type=self.asset_type, stage=stage, name=name, version=1
            ), logger.error("Failure in {}.".format(inspect.stack()[0][3]))

            for i in np.arange(1, 4):
                assert self.repo.version_exists(
                    asset_type=self.asset_type, stage=stage, name=name, version=i + 1
                ), logger.error("Failure in {}.".format(inspect.stack()[0][3]))

            assert not self.repo.version_exists(
                asset_type=self.asset_type, stage=stage, name=name, version=5
            ), logger.error("Failure in {}.".format(inspect.stack()[0][3]))

        print(self.repo.get_assets())

        logger.info(
            "\tSuccessfully completed {} {}".format(
                self.__class__.__name__, inspect.stack()[0][3]
            )
        )

    def test_repo_delete_asset(self):
        logger.info(
            "\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3])
        )

        for name, stage, creator, description in zip(
            self.names, self.stages, self.creators, self.descriptions
        ):
            self.repo.delete_asset(asset_type=self.asset_type, stage=stage, name=name)

            assert not self.repo.asset_exists(
                asset_type=self.asset_type, stage=stage, name=name
            ), logger.error("Failure in {}.".format(inspect.stack()[0][3]))

            for i in np.arange(5):
                assert not self.repo.version_exists(
                    asset_type=self.asset_type, stage=stage, name=name, version=i + 1
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
    t.test_asset()
    t.test_repo_add_versions()
    t.test_repo_add_get_assets()
    t.test_repo_delete_most_recent_version()
    t.test_repo_delete_version()
    t.test_repo_delete_asset()
    t.test_teardown()

#%%
