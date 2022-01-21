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
# Modified : Tuesday, January 18th 2022, 9:42:04 pm                                                                        #
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
from datetime import datetime

from cvr.utils.config import DataSourceConfig, ProjectConfig
from cvr.data.source import CriteoETL

# ------------------------------------------------------------------------------------------------------------------------ #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------------------------------ #


class DataSourceTests:
    def test_etl(self):
        logger.info("    Started {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        filepath = os.path.join("data", "criteo", "raw", "criteo.csv")

        datasource_config = DataSourceConfig()
        project_config = ProjectConfig()

        source = CriteoETL(
            datasource_config=datasource_config, project_config=project_config, source="criteo", force=False, verbose=True
        )
        source.etl()

        logger.info("\t\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))


if __name__ == "__main__":
    t = DataSourceTests()
    t.test_etl()
#%%
