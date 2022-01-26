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
# Modified : Tuesday, January 25th 2022, 5:44:18 pm                                                                        #
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

from cvr.data.etl import Extract, TransformETL, LoadDataset
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
        self._config = CriteoConfig()

        if os.path.exists(self._config.destination):
            x = input("Delete existing download?")
            if "y" in x:
                os.remove(self._config.destination)
                time.sleep(3)

    def test_tasks(self):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        self._extract = Extract(config=self._config, sample_size=1000, random_state=55)
        self._transform = TransformETL(value=[-1, "-1"])
        self._load = LoadDataset()

        logger.info("\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_builder(self):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        self._builder = DataPipelineBuilder()
        self._builder.set_name(name="11th_street").set_stage("seed").set_force(False).set_verbose(True)
        self._builder.add_task(self._extract)
        self._builder.add_task(self._transform)
        self._builder.add_task(self._load)
        self._builder.build()
        self._pipeline = self._builder.pipeline

        logger.info("\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_pipeline(self):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        self.dataset = self._pipeline.run()
        self._pipeline.summary

        logger.info("\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_data(self):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        self.dataset.info()
        self.dataset.numerics
        self.dataset.categoricals
        self.dataset.missing_summary
        self.dataset.missing
        self.dataset.cardinality
        self.dataset.metrics
        self.dataset.datatypes

        logger.info("\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))


if __name__ == "__main__":
    logger.info(" Started ETL Pipeline Tests ")
    t = ETLTests()
    t.test_tasks()
    t.test_builder()
    t.test_pipeline()
    t.test_data()
    logger.info(" Completed ETL Pipeline Tests ")

#%%
