#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ============================================================================ #
# Project  : Deep Learning for Conversion Rate Prediction (CVR)                #
# Version  : 0.1.0                                                             #
# File     : \test_lab.py                                                      #
# Language : Python 3.7.12                                                     #
# ---------------------------------------------------------------------------- #
# Author   : John James                                                        #
# Email    : john.james.ai.studio@gmail.com                                    #
# URL      : https://github.com/john-james-ai/cvr                              #
# ---------------------------------------------------------------------------- #
# Created  : Tuesday, January 18th 2022, 6:12:41 am                            #
# Modified : Saturday, February 5th 2022, 1:34:41 am                           #
# Modifier : John James (john.james.ai.studio@gmail.com)                       #
# ---------------------------------------------------------------------------- #
# License  : BSD 3-clause "New" or "Revised" License                           #
# Copyright: (c) 2022 Bryant St. Labs                                          #
# ============================================================================ #
#%%
import os
import inspect
import pandas as pd
import shutil
import logging
from datetime import datetime

from cvr.core.lab import LabAdmin
from cvr.core.dataset import DatasetFactory

# ---------------------------------------------------------------------------- #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# ---------------------------------------------------------------------------- #


class LabTests:
    def __init__(self):

        self.lab = None
        self.lab_name = "LabTests"
        self.lab_description = "Unit Testing for Labs"
        self.base_directory = "tests\data\labs"
        # Datasets
        self.asset_type = "dataset"
        self.names = ["name_1", "name_2", "name_3"]
        self.descriptions = ["desc_1", "desc_2", "desc_3"]
        self.stages = ["stage_1", "stage_2", "stage_3"]
        self.creators = ["feinmann", "diamond", "sting"]

        # Reset lab
        shutil.rmtree(self.base_directory, ignore_errors=True)

        # Data
        filepath = "tests\data\criteo\staged\criteo_sample.pkl"
        self.df = pd.read_pickle(filepath)
        self.df = self.df[0:100]

        # Create the admin
        self.admin = LabAdmin(base_directory=self.base_directory)

        directory = os.path.join(self.base_directory, self.lab_name)
        # Create the lab
        self.lab = self.admin.create(
            name=self.lab_name,
            description=self.lab_description,
        )

        self.factory = DatasetFactory(
            directory=directory,
        )
        # Metrics
        self.start = datetime.now()
        self.end = None
        self.duration = None
        self.minutes = None

    def test_build_lab(self):

        logger = self.lab.logger
        logger.info(
            "\tStarted {} {}".format(
                self.__class__.__name__, inspect.stack()[0][3]
            )
        )

        for i in range(2):

            logger.info("\t\tTest Build Lab Interation {}".format(str(i + 1)))

            # Load assets
            for name, stage, creator, description in zip(
                self.names, self.stages, self.creators, self.descriptions
            ):
                asset = self.factory.create(
                    name, stage, creator, description, self.df
                )

                logger.info(
                    "\t\t\tCreated Dataset {} Stage: {} \
                        Creator: {}. Version {}.".format(
                        asset.name,
                        asset.stage,
                        asset.creator,
                        asset.version,
                    )
                )

                self.lab.add(asset)

                logger.info(
                    "\t\t\tAdded Dataset {} Stage: {}\
                        Creator: {}. Version {}.".format(
                        asset.name,
                        asset.stage,
                        asset.creator,
                        asset.version,
                    )
                )

                assert asset.version == i + 1, logger.error(
                    "Failure in {}.".format(inspect.stack()[0][3])
                )

        # Print asset inventory
        print(self.lab.assets)

        logger.info(
            "\tSuccessfully completed {} {}".format(
                self.__class__.__name__, inspect.stack()[0][3]
            )
        )

    def test_get_asset(self):
        logger = self.lab.logger
        logger.info(
            "\tStarted {} {}".format(
                self.__class__.__name__, inspect.stack()[0][3]
            )
        )

        for i in range(2):
            logger.info("\t\tTest Get Dataset {}".format(str(i + 1)))

            # Iterate through assets
            for name, stage, creator, description in zip(
                self.names, self.stages, self.creators, self.descriptions
            ):

                aid = (
                    stage
                    + "_"
                    + "dataset"
                    + "_"
                    + name
                    + "_v"
                    + str(i + 1).zfill(3)
                )

                asset = self.lab.get(aid)

                # Check contents
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
                    "\t\t\tRetrieved Dataset {} Stage: \
                        {} Creator: {}. Version {}.".format(
                        asset.name,
                        asset.stage,
                        asset.creator,
                        asset.version,
                    )
                )

        logger.info(
            "\tSuccessfully completed {} {}".format(
                self.__class__.__name__, inspect.stack()[0][3]
            )
        )

    def test_exists(self):
        logger = self.lab.logger
        logger.info(
            "\tStarted {} {}".format(
                self.__class__.__name__, inspect.stack()[0][3]
            )
        )

        for i in range(2):
            logger.info("\t\tTest Exists {}".format(str(i + 1)))

            # Iterate through assets
            for name, stage, creator, description in zip(
                self.names, self.stages, self.creators, self.descriptions
            ):

                aid = (
                    stage
                    + "_"
                    + "dataset"
                    + "_"
                    + name
                    + "_v"
                    + str(i + 1).zfill(3)
                )

                # Check contents
                assert self.lab.exists(aid), logger.error(
                    "Failure in {}.".format(inspect.stack()[0][3])
                )

        logger.info(
            "\tSuccessfully completed {} {}".format(
                self.__class__.__name__, inspect.stack()[0][3]
            )
        )

    def test_list_assets(self):
        logger = self.lab.logger
        logger.info(
            "\tStarted {} {}".format(
                self.__class__.__name__, inspect.stack()[0][3]
            )
        )

        assets = self.lab.assets

        assert len(assets) == 6, logger.error(
            "Failure in {}.".format(inspect.stack()[0][3])
        )

        print(assets)

        logger.info(
            "\tSuccessfully completed {} {}".format(
                self.__class__.__name__, inspect.stack()[0][3]
            )
        )

    def test_delete(self):
        logger.info(
            "\tStarted {} {}".format(
                self.__class__.__name__, inspect.stack()[0][3]
            )
        )

        for i in range(2):
            logger.info("\t\tTest Exists {}".format(str(i + 1)))

            # Iterate through assets
            for name, stage, creator, description in zip(
                self.names, self.stages, self.creators, self.descriptions
            ):

                aid = (
                    stage
                    + "_"
                    + "dataset"
                    + "_"
                    + name
                    + "_v"
                    + str(i + 1).zfill(3)
                )

                self.lab.remove(aid)

                # Check contents
                assert not self.lab.exists(aid), logger.error(
                    "Failure in {}.".format(inspect.stack()[0][3])
                )

        logger.info(
            "\tSuccessfully completed {} {}".format(
                self.__class__.__name__, inspect.stack()[0][3]
            )
        )


if __name__ == "__main__":
    logger.info("Starting LabTests")
    t = LabTests()
    t.test_build_lab()
    t.test_get_asset()
    t.test_exists()
    t.test_list_assets()
    t.test_delete()
    logger.info("Completed LabTests in {} minutes".format(str(t.minutes)))
#%%
