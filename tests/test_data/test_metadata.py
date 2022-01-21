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

from cvr.data.profile import CriteoMD
from cvr.data.datastore import DataStore

# ------------------------------------------------------------------------------------------------------------------------ #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CriteoMDTests:
    def test_statistics(self):
        logger.info(
            "    Started {} {}".format(self.__class__.__name__, inspect.stack()[0][3])
        )
        c = DataStore(stage="staged", env="t")
        df = c.read()
        c = CriteoMD()
        s = c.statistics(df)

        assert isinstance(s, dict), logger.error(
            "Failure in {}.".format(inspect.stack()[0][3])
        )
        assert isinstance(s["numeric"], pd.DataFrame), logger.error(
            "Failure in {}.".format(inspect.stack()[0][3])
        )
        assert isinstance(s["categorical"], pd.DataFrame), logger.error(
            "Failure in {}.".format(inspect.stack()[0][3])
        )

        print(s["numeric"])
        print(s["categorical"])

        logger.info(
            "    Successfully completed {} {}".format(
                self.__class__.__name__, inspect.stack()[0][3]
            )
        )


if __name__ == "__main__":
    logger.info(" Started CriteoMD Tests")
    t = CriteoMDTests()
    t.test_statistics()
    logger.info(" Completed CriteoMD Tests")

# %%
