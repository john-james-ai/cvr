#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ============================================================================ #
# Project  : Deep Learning for Conversion Rate Prediction (CVR)                #
# Version  : 0.1.0                                                             #
# File     : \test_config.py                                                   #
# Language : Python 3.7.12                                                     #
# ---------------------------------------------------------------------------- #
# Author   : John James                                                        #
# Email    : john.james.ai.studio@gmail.com                                    #
# URL      : https://github.com/john-james-ai/cvr                              #
# ---------------------------------------------------------------------------- #
# Created  : Wednesday, January 26th 2022, 8:24:51 pm                          #
# Modified : Saturday, February 5th 2022, 2:53:49 am                           #
# Modifier : John James (john.james.ai.studio@gmail.com)                       #
# ---------------------------------------------------------------------------- #
# License  : BSD 3-clause "New" or "Revised" License                           #
# Copyright: (c) 2022 Bryant St. Labs                                          #
# ============================================================================ #
#%%
import logging
import inspect

from cvr.utils.config import AIDConfig

# ---------------------------------------------------------------------------- #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# ---------------------------------------------------------------------------- #


class AIDConfigTest:
    def __init__(self):
        filepath = "tests\\aid.yaml"
        self.config = AIDConfig(aid_filepath=filepath)
        self.config.reset()

    def test_gen(self):
        logger.info(
            "\tStarted {} {}".format(
                self.__class__.__name__, inspect.stack()[0][3]
            )
        )

        aid = self.config.gen()
        print(aid)
        assert aid == "0000", logger.error(
            "\t\tFailure in {}.".format(inspect.stack()[0][3])
        )

        aid = self.config.gen()
        print(aid)
        assert aid == "0001", logger.error(
            "\t\tFailure in {}.".format(inspect.stack()[0][3])
        )

        aid = self.config.gen()
        print(aid)
        assert aid == "0002", logger.error(
            "\t\tFailure in {}.".format(inspect.stack()[0][3])
        )

        self.config.reset()
        aid = self.config.gen()
        print(aid)
        assert aid == "0000", logger.error(
            "\t\tFailure in {}.".format(inspect.stack()[0][3])
        )
        logger.info(
            "\tSuccessfully completed {} {}".format(
                self.__class__.__name__, inspect.stack()[0][3]
            )
        )


if __name__ == "__main__":
    logger.info("Started AIDConfigTest")
    t = AIDConfigTest()
    t.test_gen()
    logger.info("Completed DatastoreConfigTests")


# %%
