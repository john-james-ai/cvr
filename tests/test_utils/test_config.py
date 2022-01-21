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
# Modified : Sunday, January 16th 2022, 12:58:42 am                                                                        #
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

from cvr.utils.config import DataSourceConfig, ProjectConfig

# ------------------------------------------------------------------------------------------------------------------------ #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProjectConfigTests:

    config = {"workspace": "dev", "random_state": 55, "sample_size": 0.01}

    def test_setup(self):
        c = ProjectConfig()
        c.save_config(ProjectConfigTests.config)

    def test_get_config(self):
        logger.info("    Started {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        c = ProjectConfig()
        config = c.get_config()
        for k, v in config.items():
            assert ProjectConfigTests.config[k] == v, logger.error(
                "Failure in {}. k={}, v={}".format(inspect.stack()[0][3], k, v)
            )

        logger.info("    Successfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_properties(self):
        logger.info("    Started {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        c = ProjectConfig()
        assert c.random_state == ProjectConfigTests().config["random_state"], logger.error(
            "Failure in {}.".format(inspect.stack()[0][3])
        )

        c.random_state = 33
        assert c.random_state == 33, logger.error("Failure in {}.".format(inspect.stack()[0][3]))

        assert c.sample_size == ProjectConfigTests().config["sample_size"], logger.error(
            "Failure in {}.".format(inspect.stack()[0][3])
        )

        c.sample_size = 0.1
        assert c.sample_size == 0.1, logger.error("Failure in {}.".format(inspect.stack()[0][3]))

        assert c.workspace == "dev", logger.error("Failure in {}.".format(inspect.stack()[0][3]))

        c.set_prod_workspace()
        assert c.workspace == "prod", logger.error("Failure in {}.".format(inspect.stack()[0][3]))

        c.set_dev_workspace()
        assert c.workspace == "dev", logger.error("Failure in {}.".format(inspect.stack()[0][3]))

        logger.info("    Successfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))


class DataSourceConfigTests:

    criteo = {
        "url": "http://go.criteo.net/criteo-research-search-conversion.tar.gz",
        "destination": "data/files/raw/",
        "compressed": "criteo.tar.gz",
        "decompressed": "Criteo_Conversion_Search/CriteoSearchData",
        "filename": "criteo.csv",
        "sep": "\\t",
        "missing": -1,
        "name": "Criteo Sponsored Search Conversion Log Dataset",
    }

    def test_get_config(self):
        logger.info("    Started {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        c = DataSourceConfig()
        config = c.get_config()
        criteo = config["criteo"]
        for k, v in criteo.items():
            assert DataSourceConfigTests.criteo[k] == v, logger.error(
                "Failure in {}. k={}, v={}".format(inspect.stack()[0][3], k, v)
            )

        config = c.get_config("criteo")
        for k, v in config.items():
            assert DataSourceConfigTests.criteo[k] == v, logger.error(
                "Failure in {}. k={}, v={}".format(inspect.stack()[0][3], k, v)
            )

        logger.info("    Successfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_print_config(self):
        logger.info("    Started {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        c = DataSourceConfig()
        c.print_config()

        logger.info("    Successfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))


if __name__ == "__main__":
    t = ProjectConfigTests()
    t.test_setup()
    t.test_get_config()
    t.test_properties()

    t = DataSourceConfigTests()
    t.test_get_config()
    t.test_print_config()


# %%
