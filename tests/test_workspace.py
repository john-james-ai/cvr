#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ============================================================================ #
# Project  : Deep Learning for Conversion Rate Prediction (CVR)                #
# Version  : 0.1.0                                                             #
# File     : \test_workspace.py                                                #
# Language : Python 3.7.12                                                     #
# ---------------------------------------------------------------------------- #
# Author   : John James                                                        #
# Email    : john.james.ai.studio@gmail.com                                    #
# URL      : https://github.com/john-james-ai/cvr                              #
# ---------------------------------------------------------------------------- #
# Created  : Tuesday, January 18th 2022, 6:12:41 am                            #
# Modified : Thursday, February 3rd 2022, 1:51:06 pm                           #
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

pd.set_option("display.width", 1000)
pd.options.display.float_format = "{:,.2f}".format
from datetime import datetime

from cvr.core.workspace import Workspace, WorkspaceAdmin
from cvr.core.dataset import DatasetFactory, Dataset

# ---------------------------------------------------------------------------- #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# ---------------------------------------------------------------------------- #


class WorkspaceTests:
    def __init__(self):

        self.workspace = None
        self.workspace_name = "WorkspaceTests"
        self.workspace_description = "Unit Testing for Workspaces"
        self.workspace_directory = "tests\data\workspaces"
        # Datasets
        self.asset_type = "dataset"
        self.names = ["name_1", "name_2", "name_3"]
        self.descriptions = ["desc_1", "desc_2", "desc_3"]
        self.stages = ["stage_1", "stage_2", "stage_3"]
        self.creators = ["feinmann", "diamond", "sting"]

        # Reset workspace
        shutil.rmtree(self.workspace_directory, ignore_errors=True)

        # Data
        filepath = "tests\data\criteo\staged\criteo_sample.pkl"
        self.df = pd.read_pickle(filepath)
        self.df = self.df[0:100]

        # Create the admin
        self.admin = WorkspaceAdmin(workspace_directory=self.workspace_directory)

        # Create the workspace
        self.workspace = self.admin.create(
            name=self.workspace_name,
            workspace_directory=self.workspace_directory,
            description=self.workspace_description,
        )
        logger = self.workspace.logger
        self.factory = DatasetFactory(
            workspace_directory=os.path.join(
                self.workspace_directory, self.workspace_name
            ),
            logger=logger,
        )
        # Metrics
        self.start = datetime.now()
        self.end = None
        self.duration = None
        self.minutes = None

    def test_build_workspace(self):

        logger = self.workspace.logger
        logger.info(
            "\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3])
        )

        for i in range(2):

            logger.info("\t\tTest Build Workspace Interation {}".format(str(i + 1)))

            # Load datasets
            for name, stage, creator, description in zip(
                self.names, self.stages, self.creators, self.descriptions
            ):
                dataset = self.factory.create(
                    name, stage, creator, description, self.df
                )

                logger.info(
                    "\t\t\tCreated Dataset {} Stage: {} Creator: {}. Version {}.".format(
                        dataset.name, dataset.stage, dataset.creator, dataset.version
                    )
                )

                self.workspace.add_dataset(dataset)

                logger.info(
                    "\t\t\tAdded Dataset {} Stage: {} Creator: {}. Version {}.".format(
                        dataset.name, dataset.stage, dataset.creator, dataset.version
                    )
                )

                assert dataset.version == i + 1, logger.error(
                    "Failure in {}.".format(inspect.stack()[0][3])
                )

        # Print dataset inventory
        print(self.workspace.list_datasets())

        logger.info(
            "\tSuccessfully completed {} {}".format(
                self.__class__.__name__, inspect.stack()[0][3]
            )
        )

    def test_get_dataset(self):
        logger = self.workspace.logger
        logger.info(
            "\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3])
        )

        for i in range(2):
            logger.info("\t\tTest Get Dataset {}".format(str(i + 1)))

            # Iterate through datasets
            for name, stage, creator, description in zip(
                self.names, self.stages, self.creators, self.descriptions
            ):

                dataset = self.workspace.get_dataset(
                    stage=stage, name=name, version=i + 1
                )

                # Check contents
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

                logger.info(
                    "\t\t\tRetrieved Dataset {} Stage: {} Creator: {}. Version {}.".format(
                        dataset.name, dataset.stage, dataset.creator, dataset.version
                    )
                )

        logger.info(
            "\tSuccessfully completed {} {}".format(
                self.__class__.__name__, inspect.stack()[0][3]
            )
        )

    def test_exists(self):
        logger = self.workspace.logger
        logger.info(
            "\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3])
        )

        for i in range(2):
            logger.info("\t\tTest Exists {}".format(str(i + 1)))

            # Iterate through datasets
            for name, stage, creator, description in zip(
                self.names, self.stages, self.creators, self.descriptions
            ):

                # Check contents
                assert self.workspace.dataset_exists(name, stage, i + 1), logger.error(
                    "Failure in {}.".format(inspect.stack()[0][3])
                )

        logger.info(
            "\tSuccessfully completed {} {}".format(
                self.__class__.__name__, inspect.stack()[0][3]
            )
        )

    def test_list_datasets(self):
        logger = self.workspace.logger
        logger.info(
            "\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3])
        )

        datasets = self.workspace.list_datasets()

        assert len(datasets) == 6, logger.error(
            "Failure in {}.".format(inspect.stack()[0][3])
        )

        print(datasets)

        logger.info(
            "\tSuccessfully completed {} {}".format(
                self.__class__.__name__, inspect.stack()[0][3]
            )
        )

    def test_delete_dataset(self):
        logger.info(
            "\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3])
        )

        for i in range(2):
            logger.info("\t\tTest Exists {}".format(str(i + 1)))

            # Iterate through datasets
            for name, stage, creator, description in zip(
                self.names, self.stages, self.creators, self.descriptions
            ):

                self.workspace.remove_dataset(name=name, stage=stage, version=i + 1)

                # Check contents
                assert not self.workspace.dataset_exists(
                    name, stage, i + 1
                ), logger.error("Failure in {}.".format(inspect.stack()[0][3]))

        logger.info(
            "\tSuccessfully completed {} {}".format(
                self.__class__.__name__, inspect.stack()[0][3]
            )
        )


if __name__ == "__main__":
    logger.info("Starting WorkspaceTests")
    t = WorkspaceTests()
    t.test_build_workspace()
    t.test_get_dataset()
    t.test_exists()
    t.test_list_datasets()
    t.test_delete_dataset()
    logger.info("Completed WorkspaceTests in {} minutes".format(str(t.minutes)))

#%%
