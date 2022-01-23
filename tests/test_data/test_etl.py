#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ======================================================================================================================== #
# Project  : Conversion Rate Prediction (CVR)                                                                              #
# Version  : 0.1.0                                                                                                         #
# File     : \test_etl.py                                                                                                  #
# Language : Python 3.10.1                                                                                                 #
# ------------------------------------------------------------------------------------------------------------------------ #
# Author   : John James                                                                                                    #
# Email    : john.james.ai.studio@gmail.com                                                                                #
# URL      : https://github.com/john-james-ai/cvr                                                                          #
# ------------------------------------------------------------------------------------------------------------------------ #
# Created  : Thursday, January 20th 2022, 1:24:43 pm                                                                       #
# Modified : Sunday, January 23rd 2022, 5:32:28 am                                                                         #
# Modifier : John James (john.james.ai.studio@gmail.com)                                                                   #
# ------------------------------------------------------------------------------------------------------------------------ #
# License  : BSD 3-clause "New" or "Revised" License                                                                       #
# Copyright: (c) 2022 Bryant St. Labs                                                                                      #
# ======================================================================================================================== #
#%%
import os
import pytest
import logging
import pandas as pd
from datetime import datetime
import inspect
import time

from cvr.data.etl import Extract, TransformMissing, LoadDataset
from cvr.core.pipeline import PipelineCommand, DataPipelineBuilder, DataPipeline
from cvr.data import criteo_columns, criteo_dtypes
from cvr.utils.config import CriteoConfig

# ------------------------------------------------------------------------------------------------------------------------ #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------------------------------ #
class ETLTests:
    def __init__(self):
        config_filepath = "tests\\test_config\criteo.yaml"
        self._config = CriteoConfig(config_filepath).get_config()

        if os.path.exists(self._config["destination"]):
            x = input("Delete existing download?")
            if "y" in x:
                os.remove(self._config["destination"])
                time.sleep(3)

    def test_tasks(self):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        self._extract = Extract(config=self._config)
        self._transform = TransformMissing(value=[-1, "-1"])
        self._load = LoadDataset()

        logger.info("\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_builder(self):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        self._builder = DataPipelineBuilder()
        self._builder.create()
        self._builder.set_name(name="11th_street").set_stage("disenchanted").set_force(False).set_keep_interim(
            True
        ).set_verbose(True)
        self._builder.add_task(self._extract)
        self._builder.add_task(self._transform)
        self._builder.add_task(self._load)
        self._builder.build()
        self._pipeline = self._builder.pipeline

        logger.info("\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_pipeline(self):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        self._pipeline.run()
        self._pipeline.summary

        logger.info("\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_summary(self):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        d = self._profiler.summary
        assert d["rows"] == self._df.shape[0], logger.error("Failure in {}.".format(inspect.stack()[0][3]))
        assert d["columns"] == self._df.shape[1], logger.error("Failure in {}.".format(inspect.stack()[0][3]))
        assert d["size"] > 100, logger.error("Failure in {}.".format(inspect.stack()[0][3]))

        logger.info("\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))


if __name__ == "__main__":
    logger.info(" Started ETL Pipeline Tests ")
    t = ETLTests()
    t.test_tasks()
    t.test_builder()
    t.test_pipeline()
    t.test_summary()
    logger.info(" Completed ETL Pipeline Tests ")

#%%
