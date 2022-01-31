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
# Modified : Sunday, January 30th 2022, 11:23:10 pm                                                                        #
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
import shutil

from cvr.core.workspace import WorkspaceManager
from cvr.core.pipeline import PipelineRequest
from cvr.core.dataset import DatasetRequest
from cvr.data.etl import Extract, TransformETL, LoadDataset
from cvr.core.pipeline import DataPipelineBuilder, DataPipeline, DataPipelineRequest
from cvr.data import criteo_columns, criteo_dtypes
from cvr.utils.config import CriteoConfig

# ------------------------------------------------------------------------------------------------------------------------ #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------------------------------ #
class ETLTests:
    def __init__(self):
        # Create Test Workspace if it doesn't already exist
        wsm = WorkspaceManager()
        if wsm.exists("test_etl"):
            self._workspace = wsm.get_workspace("test_etl")
        else:
            self._workspace = wsm.create_workspace(name="test_etl", description="Test ETL", current=True)

        # Get configuration for data source
        self._config = CriteoConfig()

    def test_tasks(self):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        self._extract = Extract(datasource_config=self._config)
        self._transform = TransformETL(value=[-1, "-1"])
        self._load = LoadDataset()

        logger.info("\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_builder(self):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        dataset_request = DatasetRequest(
            name="test_etl_dataset",
            description="Sample Dataset for ETL Test",
            stage="test",
            sample_size=1000,
            workspace_name=self._workspace.name,
            workspace_directory=self._workspace.directory,
        )
        data_pipeline_request = DataPipelineRequest(
            name="test_etl_pipeline",
            description="Testing The Sax Man",
            stage="mezzanine",
            workspace_name=self._workspace.name,
            workspace_directory=self._workspace.directory,
            logging_level="info",
            force=True,
            verbose=True,
            progress=False,
            random_state=602,
            dataset_request=dataset_request,
        )

        self._builder = DataPipelineBuilder()
        self._pipeline = (
            self._builder.make_request(data_pipeline_request)
            .add_task(self._extract)
            .add_task(self._transform)
            .add_task(self._load)
            .build()
            .pipeline
        )

        logger.info("\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_pipeline(self):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        self.dataset = self._pipeline.run()
        self._pipeline.summary

        logger.info("\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def data(self):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        self.dataset.info()
        self.dataset.profile.summary
        self.dataset.profile.datatypes
        self.dataset.profile.missing
        self.dataset.profile.cardinality
        self.dataset.profile.frequency_stats
        self.dataset.profile.analyze("product_brand")
        self.dataset.profile.analyze("sales_amount")

        logger.info("\tSuccessfully completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))


if __name__ == "__main__":
    logger.info(" Started ETL Pipeline Tests ")
    t = ETLTests()
    t.test_tasks()
    t.test_builder()
    t.test_pipeline()
    t.data()
    logger.info(" Completed ETL Pipeline Tests ")

#%%
