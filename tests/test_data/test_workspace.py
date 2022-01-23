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
# Modified : Sunday, January 23rd 2022, 4:06:33 am                                                                         #
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
from cvr.data.datasets import Dataset, DatasetBuilder
from cvr.utils.config import WorkspaceConfig

# ------------------------------------------------------------------------------------------------------------------------ #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------------------------------ #


class WorkspaceTests:
    def __init__(self):
        # Build Dataset
        shutil.rmtree("workspaces", ignore_errors=True)
        start = datetime.now()
        filepath = "tests\\test_data\criteo\staged\criteo_sample.pkl"
        self.df = pd.read_pickle(filepath)
        self.name = "After Hours"
        self.description = "Have a Margarita and Keep Fighting"
        self.stage = "Left"
        self.b = DatasetBuilder()
        end = datetime.now()
        duration = end - start
        logger.info("Build time {}".format(duration))

    def test_build_workspace(self):
        self.name = "Cafe_Trieste"
        self.description = "Any Given Thursday, it could Happen!"
        self.ws = Workspace(self.name, self.description)
        assert WorkspaceConfig().get_workspace() == self.name, logger.error("Failure in {}.".format(inspect.stack()[0][3]))
        assert self.ws.name == self.name, logger.error("Failure in {}.".format(inspect.stack()[0][3]))
        assert self.ws.description == self.description, logger.error("Failure in {}.".format(inspect.stack()[0][3]))

    def test_datasets(self):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))
        names = ["Candyland", "The_Clown", "Vesuvios"]
        for name in names:
            self.b = DatasetBuilder()
            self.ds = self.b.set_data(self.df).set_name(name).set_description(self.description).set_stage(self.stage).build()
            logger.info("\t\tStoring {}...".format(name))
            self.ws.add_dataset(self.ds)
        self.ws.print()
        print("\n")

        for name in names:
            ds = self.ws.get_dataset(self.stage, name)
            assert ds.name == name, logger.error("Failure in {}.".format(inspect.stack()[0][3]))
            logger.info("\t\tRecovered {}".format(name))

        for name in names:
            ds = self.ws.delete_dataset(self.stage, name)
        self.ws.print()

        logger.info("\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))


if __name__ == "__main__":
    t = WorkspaceTests()
    t.test_build_workspace()
    t.test_datasets()
#%%
