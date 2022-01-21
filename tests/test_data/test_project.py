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
# Modified : Wednesday, January 19th 2022, 2:52:35 am                                                                      #
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

from cvr.core.workspace import Workspace
from cvr.core.project import Project
from cvr.data.datasets import Dataset, DatasetBuilder

# ------------------------------------------------------------------------------------------------------------------------ #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------------------------------ #


class ProjectTests:
    def __init__(self):
        self._project = Project("Test")

    def test_source_data(self):
        logger.info("    Started {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        self._project.source_data()

        logger.info("\t\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_create_workspace(self):
        logger.info("    Started {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        self._workspace = self._project.create_workspace("Tom").with_description("Toms Pad").add_sample_data(0.01).build()
        assert isinstance(self._workspace, Workspace)

        logger.info("\t\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_add_dataset(self):
        logger.info("    Started {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))
        # Grab some data
        filepath = "tests\\test_data\staged\criteo.pkl"
        df = pd.read_pickle(filepath)
        # Build the builder
        builder = DatasetBuilder("Tom")
        # Create the dataset
        ds2 = builder.set_data(df).set_name("2nd Dataset").set_stage("interim").set_creator("ProjectTests").build()
        # Add the dataset to the workspace
        self._workspace.add_dataset(ds2)
        # Retrieve dataset from workspace
        ds3 = self._workspace.get_dataset(ds2.id)
        # Confirm dataset was persistedd
        assert os.path.exists(filepath), logger.error("Failure in {}.".format(inspect.stack()[0][3]))
        # Compare the dataset objects
        assert ds2.id == ds3.id, logger.error("Failure in {}.".format(inspect.stack()[0][3]))
        assert ds2.workspace == ds3.workspace, logger.error("Failure in {}.".format(inspect.stack()[0][3]))
        assert ds2.stage == ds3.stage, logger.error("Failure in {}.".format(inspect.stack()[0][3]))

        logger.info("\t\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_reporting(self):
        logger.info("    Started {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        self._project.inventory

        logger.info("\t\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))


if __name__ == "__main__":
    t = ProjectTests()
    t.test_source_data()
    t.test_create_workspace()
    t.test_add_dataset()

#%%
