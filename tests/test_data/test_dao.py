#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ======================================================================================================================== #
# Project  : Explainable Recommendation (XRec)                                                                             #
# Version  : 0.1.0                                                                                                         #
# File     : \test_config.py                                                                                               #
# Language : Python 3.8                                                                                                    #
# ------------------------------------------------------------------------------------------------------------------------ #
# Author   : John James                                                                                                    #
# Email    : john.james.ai.studio@gmail.com                                                                                #
# URL      : https://github.com/john-james-ai/xrec                                                                         #
# ------------------------------------------------------------------------------------------------------------------------ #
# Created  : Saturday, December 25th 2021, 11:16:00 am                                                                     #
# Modified : Thursday, January 13th 2022, 12:21:13 am                                                                      #
# Modifier : John James (john.james.ai.studio@gmail.com)                                                                   #
# ------------------------------------------------------------------------------------------------------------------------ #
# License  : BSD 3-clause "New" or "Revised" License                                                                       #
# Copyright: (c) 2021 Bryant St. Labs                                                                                      #
# ======================================================================================================================== #
#%%
import pytest
import logging
import pandas as pd
import inspect
from itertools import product

from cvr.data.datastore import DataStore

# ------------------------------------------------------------------------------------------------------------------------ #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataStoreTests:

    environment = {"p": "Production", "d": "Development", "t": "Test"}

    filename = "criteo.csv"
    directories = {
        "d": {
            "raw": "data/dev/raw/",
            "staged": "data/dev/staged/",
            "cleaned": "data/dev/cleaned/",
            "cooked": "data/dev/cooked/",
        },
        "p": {
            "raw": "data/prod/raw/",
            "staged": "data/prod/staged/",
            "cleaned": "data/prod/cleaned/",
            "cooked": "data/prod/cooked/",
        },
        "t": {
            "raw": "tests/test_data/raw/",
            "staged": "tests/test_data/staged/",
            "cleaned": "tests/test_data/cleaned/",
            "cooked": "tests/test_data/cooked/",
        },
    }

    def test_setup(self):
        c = DataStore("T")
        if c.exists():
            c.delete()

    def _test_properties(self, env):
        logger.info(
            "    Started {} {}".format(self.__class__.__name__, inspect.stack()[0][3])
        )

        env = env.lower()[0]

        c = DataStore(env)

        assert c.environment == DataStoreTests.environment[env], logger.error(
            "Failure in {}.".format(inspect.stack()[0][3])
        )
        assert c.filename == DataStoreTests.filename, logger.error(
            "Failure in {}.".format(inspect.stack()[0][3])
        )
        assert c.directory == DataStoreTests.directories[env][stage], logger.error(
            "Failure in {}.".format(inspect.stack()[0][3])
        )

        assert c.filepath == os.path.join(
            DataStoreTests.directories[env][stage], DataStoreTests.filename
        ), logger.error("Failure in {}.".format(inspect.stack()[0][3]))

        if os.path.exists(c.filepath):
            assert isinstance(c.filesize, int), logger.error(
                "Failure in {}.".format(inspect.stack()[0][3])
            )
            assert isinstance(c.created, str), logger.error(
                "Failure in {}.".format(inspect.stack()[0][3])
            )

            assert isinstance(c.modified, str), logger.error(
                "Failure in {}.".format(inspect.stack()[0][3])
            )

        logger.info(
            "    Successfully completed {} {}".format(
                self.__class__.__name__, inspect.stack()[0][3]
            )
        )

    def test_properties(self):
        envs = ["dev", "PR", "tESTS"]
        stages = ["raw", "staged", "cleaned", "cooked"]
        for e, f in list(product(envs, stages)):
            self._test_properties(e, f)

    def test_read(self):
        logger.info(
            "    Started {} {}".format(self.__class__.__name__, inspect.stack()[0][3])
        )

        c = DataStore("staged", "TE")
        self._df = c.read()

        assert isinstance(self._df, pd.DataFrame), logger.error(
            "Failure in {}.".format(inspect.stack()[0][3])
        )
        assert self._df.shape[0] > 100000, logger.error(
            "Failure in {}.".format(inspect.stack()[0][3])
        )

        logger.info(
            "    Successfully completed {} {}".format(
                self.__class__.__name__, inspect.stack()[0][3]
            )
        )

    def test_write(self):
        logger.info(
            "    Started {} {}".format(self.__class__.__name__, inspect.stack()[0][3])
        )

        c = DataStore("cleaned", "T")
        c.write(self._df)

        assert c.filepath, logger.error("Failure in {}.".format(inspect.stack()[0][3]))

        logger.info(
            "    Successfully completed {} {}".format(
                self.__class__.__name__, inspect.stack()[0][3]
            )
        )

    def test_exists(self):
        logger.info(
            "    Started {} {}".format(self.__class__.__name__, inspect.stack()[0][3])
        )

        c = DataStore("cleaned", "T")
        assert c.exists(), logger.error("Failure in {}.".format(inspect.stack()[0][3]))

        logger.info(
            "    Successfully completed {} {}".format(
                self.__class__.__name__, inspect.stack()[0][3]
            )
        )

    def test_delete(self):
        logger.info(
            "    Started {} {}".format(self.__class__.__name__, inspect.stack()[0][3])
        )

        c = DataStore("cleaned", "T")
        c.delete()

        assert ~c.exists(), logger.error("Failure in {}.".format(inspect.stack()[0][3]))

        logger.info(
            "    Successfully completed {} {}".format(
                self.__class__.__name__, inspect.stack()[0][3]
            )
        )

    def test_io_with_filename(self):

        logger.info(
            "    Started {} {}".format(self.__class__.__name__, inspect.stack()[0][3])
        )

        c = DataStore("staged", "T")
        f = c.read()
        c.write(f, filename="test.csv")
        f2 = c.read("test.csv")

        assert isinstance(f2, pd.DataFrame), logger.error(
            "Failure in {}.".format(inspect.stack()[0][3])
        )

        assert c.exists("test.csv"), logger.error(
            "Failure in {}.".format(inspect.stack()[0][3])
        )

        c.delete("test.csv")

        assert ~c.exists("test.csv"), logger.error(
            "Failure in {}.".format(inspect.stack()[0][3])
        )

        logger.info(
            "    Successfully completed {} {}".format(
                self.__class__.__name__, inspect.stack()[0][3]
            )
        )


if __name__ == "__main__":
    logger.info(" Started ConfigIO Tests")
    t = DataStoreTests()
    t.test_setup()
    t.test_properties()
    t.test_read()
    t.test_write()
    t.test_exists()
    t.test_delete()
    t.test_io_with_filename()
    logger.info(" Completed ConfigIO Tests")

# %%
