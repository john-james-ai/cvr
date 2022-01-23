#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ======================================================================================================================== #
# Project  : Explainable Recommendation (XRec)                                                                             #
# Version  : 0.1.0                                                                                                         #
# File     : \test_config.py                                                                                               #
# Language : Python 3.8                                                                                                    #
# --------------------------------------------  ---------------------------------------------------------------------------- #
# Author   : John James                                                                                                    #
# Email    : john.james.ai.studio@gmail.com                                                                                #
# URL      : https://github.com/john-james-ai/xrec                                                                         #
# ------------------------------------------------------------------------------------------------------------------------ #
# Created  : Saturday, December 25th 2021, 11:16:00 am                                                                     #
# Modified : Saturday, January 22nd 2022, 2:14:56 pm                                                                       #
# Modifier : John James (john.james.ai.studio@gmail.com)                                                                   #
# ------------------------------------------------------------------------------------------------------------------------ #
# License  : BSD 3-clause "New" or "Revised" License                                                                       #
# Copyright: (c) 2021 Bryant St. Labs                                                                                      #
# ======================================================================================================================== #
#%%
import pytest
import logging
import numpy as np
import inspect
from itertools import product

from cvr.utils.config import DatastoreConfig

# ------------------------------------------------------------------------------------------------------------------------ #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatastoreConfigTests:
    def __init__(self):
        self.ds = DatastoreConfig()

    def test_workspace(self):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        workspace = "dev"
        self.ds.workspace = workspace
        assert self.ds.workspace == "dev", logger.error("Failure in {}.".format(inspect.stack()[0][3]))

        workspace = "prod"
        self.ds.datastore = workspace
        assert self.ds.workspace == "prod", logger.error("Failure in {}.".format(inspect.stack()[0][3]))

        logger.info("\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_add_dataset_pipeline(self):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        pipeline = "test_pipe"
        version = 22
        filename_parts = [self.ds.workspace, str(version).zfill(3), pipeline]
        filename = "_".join(filename_parts) + ".pkl"
        filepath = os.path.join("workspaces", self.ds.workspace, pipeline, version, filename)
        self.ds.add_dataset(pipeline, version)
        assert ds.get_dataset_filepath == filepath, logger.error("Failure in {}.".format(inspect.stack()[0][3]))

        logger.info("\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_add_dataset_task(self):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        pipeline = "test_pipe"
        version = 22
        task = "grocery_store"
        task_seq = 32
        filename_parts = [self.ds.workspace, str(version).zfill(3), pipeline, task_seq, task]
        filename = "_".join(filename_parts) + ".pkl"
        filepath = os.path.join("workspaces", self.ds.workspace, pipeline, version, filename)
        self.ds.add_dataset(pipeline, version, task_seq, task)
        assert ds.get_dataset_filepath == filepath, logger.error("Failure in {}.".format(inspect.stack()[0][3]))

        logger.info("\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_get_dataset_pipeline(self):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        df = self.ds.get_datasets(pipeline="test_pipe")
        assert df.shape[0] == 2, logger.error("Failure in {}.".format(inspect.stack()[0][3]))

        logger.info("\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_remove_dataset(self):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        pipeline = "test_pipe"
        version = 22
        task = "grocery_store"
        task_seq = 32
        df = self.ds.get_datasets(pipeline="test_pipe")
        assert df.shape[0] == 2, logger.error("Failure in {}.".format(inspect.stack()[0][3]))

        logger.info("\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))


if __name__ == "__main__":
    logger.info("Started DatastoreConfigTests")
    t = DatastoreConfigTests()
    t.test_workspace()
    t.test_add_dataset_pipeline()
    t.test_add_dataset_task()
    t.test_get_dataset_pipeline()
    t.test_remove_dataset()
    logger.info("Completed DatastoreConfigTests")


# %%
