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
# Modified : Tuesday, January 25th 2022, 4:45:22 pm                                                                        #
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

pd.set_option("display.width", 1000)
pd.options.display.float_format = "{:,.2f}".format
from datetime import datetime

from cvr.core.workspace import Workspace, WorkspaceManager
from cvr.data.datasets import Dataset, DatasetBuilder

# ------------------------------------------------------------------------------------------------------------------------ #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------------------------------ #


class WorkspaceTests:
    directory = "tests\test_core\workspaces"

    def __init__(self):
        # Build Dataset
        shutil.rmtree(WorkspaceTests.directory, ignore_errors=True)

        # Get data and create Dataset builder object
        filepath = "tests\\test_data\criteo\staged\criteo_sample.pkl"
        self.df = pd.read_pickle(filepath)

        self.workspace = None
        self.names = ["name_1", "name_2", "name_3"]
        self.descriptions = ["desc_1", "desc_2", "desc_3"]
        self.stages = ["stage_1", "stage_2", "stage_3"]

        self.start = datetime.now()
        self.end = None
        self.duration = None
        self.minutes = None

    def test_build_workspace(self):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))
        name = "space"
        description = "space_test"
        random_state = 22
        self.workspace = Workspace(name=name, description=description, random_state=random_state)
        assert self.workspace.name == name, logger.error("Failure in {}.".format(inspect.stack()[0][3]))
        assert self.workspace.description == description, logger.error("Failure in {}.".format(inspect.stack()[0][3]))
        assert self.workspace.random_state == random_state, logger.error("Failure in {}.".format(inspect.stack()[0][3]))

        logger.info("\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_add_datasets(self):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        builder = DatasetBuilder()
        for name, desc, stage in zip(self.names, self.descriptions, self.stages):
            dataset = builder.set_data(self.df).set_name(name).set_description(desc).set_stage(stage).build()
            builder.reset()
            logger.info("\t\tStoring {}...".format(name))
            self.workspace.add_dataset(dataset)
        assert self.workspace.dataset_count == 3, logger.error("Failure in {}.".format(inspect.stack()[0][3]))
        self.workspace.print()
        print("\n")

        logger.info("\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_get_datasets(self):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        for name, desc, stage in zip(self.names, self.descriptions, self.stages):
            ds = self.workspace.get_dataset(name=name, stage=stage)
            assert ds.name == name, logger.error("Failure in {}.".format(inspect.stack()[0][3]))
            assert ds.stage == stage, logger.error("Failure in {}.".format(inspect.stack()[0][3]))
            assert ds.shape == self.df.shape, logger.error("Failure in {}.".format(inspect.stack()[0][3]))
            logger.info("\t\tRecovered {}".format(name))

        logger.info("\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_delete_datasets(self):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        for name, stage in zip(self.names, self.stages):
            self.workspace.delete_dataset(name=name, stage=stage)
        assert self.workspace.dataset_count == 0, logger.error("Failure in {}.".format(inspect.stack()[0][3]))

        logger.info("\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_teardown(self):
        self.end = datetime.now()
        self.duration = self.end - self.start
        self.minutes = round(self.duration.total_seconds() / 60, 2)


class WorkspaceManagerTests:
    def __init__(self):
        self.wsm = WorkspaceManager()
        # Get data and create Dataset builder object
        filepath = "tests\\test_data\criteo\staged\criteo_sample.pkl"
        df = pd.read_pickle(filepath)
        builder = DatasetBuilder()
        self._dataset = (
            builder.set_data(df)
            .set_name("wsm_test")
            .set_description("Workspace Manager Test Dataset")
            .set_stage("Stage of Disbelief")
            .build()
        )
        self.name = "monroe"
        self.description = "place of birth"
        self.sample_size = 10000
        self.random_state = 32

        self.start = datetime.now()
        self.end = None
        self.duration = None
        self.minutes = None

    def test_create_workspace(self):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        self._workspace = self.wsm.create_workspace(
            name=self.name, description=self.description, sample_size=self.sample_size, random_state=self.random_state
        )
        assert self._workspace.name == self.name, logger.error("Failure in {}.".format(inspect.stack()[0][3]))
        assert self._workspace.description == self.description, logger.error("Failure in {}.".format(inspect.stack()[0][3]))
        assert self._workspace.sample_size == self.sample_size, logger.error("Failure in {}.".format(inspect.stack()[0][3]))
        assert self._workspace.random_state == self.random_state, logger.error(
            "Failure in {}.".format(inspect.stack()[0][3])
        )
        workspace = self.wsm.get_current_workspace()
        assert workspace.name == self._workspace.name, logger.error("Failure in {}.".format(inspect.stack()[0][3]))
        assert workspace.description == self._workspace.description, logger.error(
            "Failure in {}.".format(inspect.stack()[0][3])
        )
        assert workspace.dataset_count == 0, logger.error("Failure in {}.".format(inspect.stack()[0][3]))

        logger.info("\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_get_workspace(self):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        workspace = self.wsm.get_workspace(name=self.name)
        assert self._workspace.name == workspace.name, logger.error("Failure in {}.".format(inspect.stack()[0][3]))
        assert self._workspace.description == workspace.description, logger.error(
            "Failure in {}.".format(inspect.stack()[0][3])
        )
        assert self._workspace.sample_size == workspace.sample_size, logger.error(
            "Failure in {}.".format(inspect.stack()[0][3])
        )
        assert self._workspace.random_state == workspace.random_state, logger.error(
            "Failure in {}.".format(inspect.stack()[0][3])
        )

        logger.info("\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_update_workspace(self):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        workspace = self.wsm.get_workspace(name=self.name)
        assert workspace.dataset_count == 0, logger.error("Failure in {}.".format(inspect.stack()[0][3]))

        workspace.add_dataset(self._dataset)
        assert workspace.dataset_count == 1, logger.error("Failure in {}.".format(inspect.stack()[0][3]))

        self.wsm.update_workspace(workspace)
        workspace = self.wsm.get_workspace(name=self.name)
        assert workspace.dataset_count == 1, logger.error("Failure in {}.".format(inspect.stack()[0][3]))

        logger.info("\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_teardown(self):
        self.end = datetime.now()
        self.duration = self.end - self.start
        self.minutes = round(self.duration.total_seconds() / 60, 2)


if __name__ == "__main__":
    # logger.info("Starting WorkspaceTests")
    # t = WorkspaceTests()
    # t.test_build_workspace()
    # t.test_add_datasets()
    # t.test_get_datasets()
    # t.test_delete_datasets()
    # t.test_teardown()
    # logger.info("Completed WorkspaceTests in {} minutes".format(str(t.minutes)))

    logger.info("Starting WorkspaceManagerTests")
    t = WorkspaceManagerTests()
    t.test_create_workspace()
    t.test_get_workspace()
    t.test_update_workspace()
    t.test_teardown()
    logger.info("Completed WorkspaceManagerTests in {} minutes".format(str(t.minutes)))

#%%
