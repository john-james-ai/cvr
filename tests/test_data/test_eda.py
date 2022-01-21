#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ======================================================================================================================== #
# Project  : Explainable Recommendation (XRec)                                                                             #
# Version  : 0.1.0                                                                                                         #
# File     : \test_eda.py                                                                                                  #
# Language : Python 3.8                                                                                                    #
# ------------------------------------------------------------------------------------------------------------------------ #
# Author   : John James                                                                                                    #
# Email    : john.james.ai.studio@gmail.com                                                                                #
# URL      : https://github.com/john-james-ai/xrec                                                                         #
# ------------------------------------------------------------------------------------------------------------------------ #
# Created  : Monday, December 27th 2021, 3:29:07 pm                                                                        #
# Modified : Wednesday, January 12th 2022, 3:29:33 pm                                                                      #
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

from cvr.data.eda import CriteoEDADataQuality
from cvr.utils.config import DataConfig
from cvr.utils.file import sample

# ------------------------------------------------------------------------------------------------------------------------ #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------------------------------ #


class CriteoEDADataQualityTests:
    def __init__(self):
        self._filepath = "data\external\criteo.txt"
        self._test_filepath = "tests\test_data\test_eda.csv"

    def test_setup_teardown(self):
        logger.info(
            "    Started {} {}".format(self.__class__.__name__, inspect.stack()[0][3])
        )

        if not os.path.exists(self._test_filepath):
            df = sample(self._filepath, size=0.01, sep="\t", random_state=55)
            df.to_csv(self._test_filepath)

        logger.info(
            "    Successfully completed {} {}".format(
                self.__class__.__name__, inspect.stack()[0][3]
            )
        )

    def test_prod(self):
        logger.info(
            "    Started {} {}".format(self.__class__.__name__, inspect.stack()[0][3])
        )

        c = Data(
            filepath=self._filepath, dev=False, sep="\t", sample=0.1, random_state=55
        )

        # Check raw data in development
        directory = self._version_config.get_directory("raw")
        filepath_dev = os.path.join(directory, self._filename)
        assert os.path.exists(filepath_dev), logger.error(
            "Failure in {}.".format(inspect.stack()[0][3])
        )

        # Check raw data in production
        directory = self._prod_config.get_directory("raw")
        filepath_prod = os.path.join(directory, self._filename)
        assert os.path.exists(filepath_prod), logger.error(
            "Failure in {}.".format(inspect.stack()[0][3])
        )

        # Confirm dev file size is smaller
        assert (
            os.path.getsize(filepath_dev) < os.path.getsize(filepath_prod) / 2
        ), logger.error("Failure in {}.".format(inspect.stack()[0][3]))

        assert (
            os.path.getsize(filepath_dev) < os.path.getsize(filepath_prod) / 2
        ), logger.error("Failure in {}.".format(inspect.stack()[0][3]))

        logger.info(
            "    Successfully completed {} {}".format(
                self.__class__.__name__, inspect.stack()[0][3]
            )
        )

    def test_dev(self):
        logger.info(
            "    Started {} {}".format(self.__class__.__name__, inspect.stack()[0][3])
        )

        c = Data(
            filepath=self._filepath, dev=True, sep="\t", sample=0.1, random_state=55
        )

        # Check raw data in development
        directory = self._version_config.get_directory("raw")
        filepath_dev = os.path.join(directory, self._filename)
        assert os.path.exists(filepath_dev), logger.error(
            "Failure in {}.".format(inspect.stack()[0][3])
        )

        # Check raw data in production
        directory = self._prod_config.get_directory("raw")
        filepath_prod = os.path.join(directory, self._filename)
        assert os.path.exists(filepath_prod), logger.error(
            "Failure in {}.".format(inspect.stack()[0][3])
        )

        # Confirm dev file size is smaller
        assert (
            os.path.getsize(filepath_dev) < os.path.getsize(filepath_prod) / 2
        ), logger.error("Failure in {}.".format(inspect.stack()[0][3]))

        logger.info(
            "    Successfully completed {} {}".format(
                self.__class__.__name__, inspect.stack()[0][3]
            )
        )


if __name__ == "__main__":
    t = DataAcquisitionTests()
    t.test_setup_teardown()
    t.test_prod()
    t.test_dev()
    t.test_setup_teardown()
#%%
