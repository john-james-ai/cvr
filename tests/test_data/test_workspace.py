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
# Modified : Tuesday, January 18th 2022, 10:35:38 pm                                                                       #
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
from cvr.data.datasets import Dataset

# ------------------------------------------------------------------------------------------------------------------------ #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------------------------------ #


class WorkspaceTests:
    def test_build_workspace(self):
        filepath = "tests\\test_data\staged\criteo.csv"
        self._df = pd.read_csv(filepath, index_col=False)
        self._workspace = Workspace("Leviathan", "Moby Dick")

    def test_initialize(self):
        logger.info("    Started {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        self._workspace.initialize()

        logger.info("\t\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_add_dataset(self):
        logger.info("    Started {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        self._workspace.add_dataset()

        logger.info("\t\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))


if __name__ == "__main__":
    t = WorkspaceTests()
    t.test_initialize()
#%%
