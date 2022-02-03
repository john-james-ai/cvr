#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ======================================================================================================================== #
# Project  : Conversion Rate Prediction (CVR)                                                                              #
# Version  : 0.1.0                                                                                                         #
# File     : \pipeline.py                                                                                                  #
# Language : Python 3.10.1                                                                                                 #
# ------------------------------------------------------------------------------------------------------------------------ #
# Author   : John James                                                                                                    #
# Email    : john.james.ai.studio@gmail.com                                                                                #
# URL      : https://github.com/john-james-ai/cvr                                                                          #
# ------------------------------------------------------------------------------------------------------------------------ #
# Created  : Wednesday, January 19th 2022, 5:46:57 pm                                                                      #
# Modified : Sunday, January 30th 2022, 11:00:34 pm                                                                        #
# Modifier : John James (john.james.ai.studio@gmail.com)                                                                   #
# ------------------------------------------------------------------------------------------------------------------------ #
# License  : BSD 3-clause "New" or "Revised" License                                                                       #
# Copyright: (c) 2022 Bryant St. Labs                                                                                      #
# ======================================================================================================================== #
"""Defines the pipeline construction and operation classes."""
from abc import ABC, abstractmethod
import os
from dataclasses import dataclass, field
from datetime import datetime
import logging
import pandas as pd
from collections import OrderedDict
from typing import Union

from cvr.core.asset import Asset, AssetBuilder, AssetRequest, AssetConfig
from cvr.core.dataset import DatasetRequest, DatasetConfig
from cvr.core.workspace import Project, Workspace
from cvr.utils.logger import LoggerFactory
from cvr.utils.printing import Printer

# ------------------------------------------------------------------------------------------------------------------------ #
@dataclass(frozen=True)
class PipelineRequest(AssetRequest):
    """Encapsules instructions for parameters a pipeline"""

    name: str
    description: str
    stage: str
    workspace_name: str
    workspace_directory: str
    logging_level: str
    force: bool
    verbose: bool
    progress: bool
    random_state: int


# ------------------------------------------------------------------------------------------------------------------------ #
@dataclass(frozen=True)
class DataPipelineRequest(PipelineRequest):
    dataset_request: DatasetRequest


# ------------------------------------------------------------------------------------------------------------------------ #
@dataclass
class PipelineConfig(AssetConfig):

    force: bool = False
    verbose: bool = True
    progress: bool = False
    random_state: int = None
    logging_level: str = "info"
    aid: str = field(init=False)
    version: int = field(init=False)
    logger: logging = field(init=False)
    filepath: str = field(init=False)
    printer: Printer = Printer()

    tasks: [] = field(init=False)

    def set_logger(self) -> None:
        self.logger = LoggerFactory().get_logger(
            name=self.workspace_name,
            directory=self.workspace_directory,
            logging_level=self.logging_level,
            verbose=self.verbose,
        )

    def set_tasks(self, tasks) -> None:
        self.tasks = tasks


# ------------------------------------------------------------------------------------------------------------------------ #
@dataclass
class DataPipelineConfig(PipelineConfig):
    dataset_config: DatasetConfig = None


# ======================================================================================================================== #
class Pipeline(Asset):
    """Defines interface for pipelines."""

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        # Unpack necessary parameters
        self._name = config.name
        self._stage = config.stage
        self._force = config.force
        self._verbose = config.verbose
        # Unpack Tasks
        self._tasks = config.tasks
        # Unpack operational dependencies
        self._logger = config.logger

        # Initialize instance variables.
        self._result = pd.DataFrame()
        self._data = None
        self._start = None
        self._end = None
        self._duration = None

    @property
    def summary(self) -> None:
        self._config.printer.print_title(
            "Pipeline {} Summary".format(self._config.name)
        )
        self._config.printer.print_dataframe(self._result)

    def run(self) -> None:
        self._setup()
        self._run()
        self._teardown()
        return self._data

    def _setup(self) -> None:
        self._start = datetime.now()
        self._logger.info("Started {}".format(self._name))

    def _teardown(self) -> None:
        self._end = datetime.now()
        self._duration = self._end - self._start
        self._logger.info("Completed {}".format(self._name))

    def _run(self) -> None:
        for task in self._config.tasks:
            # Run the task
            self._data = task.run(config=self._config, data=self._data)

            # Capture result
            d = OrderedDict()
            d["Task"] = task.__class__.__name__
            d["Start"] = task.start
            d["End"] = task.end
            d["Minutes"] = round(task.duration.total_seconds() / 60.0, 2)
            d["Status"] = task.status
            df = pd.DataFrame(d, index=[task.task_seq])
            self._result = pd.concat([self._result, df], axis=0)


# ------------------------------------------------------------------------------------------------------------------------ #
class DataPipeline(Pipeline):
    def __init__(self, config: DataPipelineConfig) -> None:
        super(DataPipeline, self).__init__(config)


# ------------------------------------------------------------------------------------------------------------------------ #
class PipelineBuilder(AssetBuilder):
    """Abstract pipeline builder. Defines interface."""

    def __init__(self) -> None:
        self._request = None
        self._config = None

        self._pipeline = None
        self._tasks = []
        self._task_seq = 0

    @property
    def pipeline(self) -> Pipeline:
        pipeline = self._pipeline
        self.reset()
        return pipeline

    def reset(self) -> None:
        self._pipeline = None
        self._tasks = []
        return self

    def make_request(self, request: PipelineRequest) -> None:
        self._request = request
        return self

    def add_task(self, task) -> None:
        task.task_seq = self._task_seq
        self._task_seq += 1
        self._tasks.append(task)
        return self

    def build(self) -> Pipeline:
        config = self._build_config()
        self._pipeline = self._build(config)
        return self

    @abstractmethod
    def _build_config(self, classname: str) -> None:
        """Builds the configuration object that parameterizes the Pipeline."""
        pass

    @abstractmethod
    def _build(self, config: PipelineConfig) -> Pipeline:
        """Takes a PipelineConfig object and returns a Pipeline object."""
        pass


# ------------------------------------------------------------------------------------------------------------------------ #
class DataPipelineBuilder(PipelineBuilder):
    """Data pipeline builder"""

    def __init__(self) -> None:
        super(DataPipelineBuilder, self).__init__()

    def _build_config(self) -> None:
        dataset_config = DatasetConfig(
            name=self._request.dataset_request.name,
            description=self._request.dataset_request.description,
            stage=self._request.dataset_request.stage,
            sample_size=self._request.dataset_request.sample_size,
            workspace_name=self._request.dataset_request.workspace_name,
            workspace_directory=self._request.dataset_request.workspace_directory,
        )
        dataset_config.set_config_filepath(classname="dataset")
        dataset_config.set_version()
        dataset_config.set_aid(classname="dataset")
        dataset_config.set_filepath(classname="dataset")
        dataset_config.set_profiler()

        config = DataPipelineConfig(
            name=self._request.name,
            description=self._request.description,
            stage=self._request.stage,
            workspace_name=self._request.workspace_name,
            workspace_directory=self._request.workspace_directory,
            force=self._request.force,
            verbose=self._request.verbose,
            progress=self._request.progress,
            random_state=self._request.random_state,
            dataset_config=dataset_config,
        )
        config.set_config_filepath(classname=self.__class__.__name__.lower())
        config.set_version()
        config.set_aid(classname=self.__class__.__name__.lower())
        config.set_filepath(classname=self.__class__.__name__.lower())
        config.set_logger()
        config.set_tasks(self._tasks)
        return config

    def _build(self, config: PipelineConfig) -> Pipeline:
        """Builds and returns a Pipeline object."""
        return DataPipeline(config=config)
