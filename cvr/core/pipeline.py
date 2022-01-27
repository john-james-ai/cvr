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
# Modified : Thursday, January 27th 2022, 1:56:30 am                                                                       #
# Modifier : John James (john.james.ai.studio@gmail.com)                                                                   #
# ------------------------------------------------------------------------------------------------------------------------ #
# License  : BSD 3-clause "New" or "Revised" License                                                                       #
# Copyright: (c) 2022 Bryant St. Labs                                                                                      #
# ======================================================================================================================== #
"""Defines the pipeline construction and operation classes."""
from abc import ABC, abstractmethod
import os
from dataclasses import dataclass
from datetime import datetime
import logging
import pandas as pd
from collections import OrderedDict
from typing import Union

from cvr.core.asset import Asset
from cvr.core.dataset import DatasetRequest
from cvr.core.workspace import WorkspaceManager, Workspace
from cvr.utils.logger import LoggerFactory
from cvr.utils.printing import Printer

# ======================================================================================================================== #
@dataclass(frozen=True)
class PipelineRequest(ABC):
    name: str
    stage: str
    workspace: Workspace


# ------------------------------------------------------------------------------------------------------------------------ #
@dataclass(frozen=True)
class DataPipelineRequest(PipelineRequest):
    dataset_request: DatasetRequest
    logging_level: str = "info"
    force: bool = False
    verbose: bool = True
    progress: bool = False
    random_state: int = None


# ------------------------------------------------------------------------------------------------------------------------ #
@dataclass(frozen=True)
class PipelineConfig(ABC):
    name: str
    stage: str
    workspace: Workspace
    tasks: []
    logger: logging
    force: bool
    verbose: bool
    progress: bool
    random_state: int


# ------------------------------------------------------------------------------------------------------------------------ #
@dataclass(frozen=True)
class DataPipelineConfig(PipelineConfig):
    dataset_config: DatasetRequest


# ======================================================================================================================== #
class Pipeline(Asset):
    """Defines interface for pipelines."""

    def __init__(self, config: PipelineConfig) -> None:
        super(Pipeline, self).__init__(config.name, config.stage)
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

        # Implementation dependencies
        self._printer = Printer()

        # Initialize instance variables.
        self._result = pd.DataFrame()
        self._data = None
        self._start = None
        self._end = None
        self._duration = None

    @property
    def aid(self) -> str:
        return self._aid

    @property
    def name(self) -> str:
        return self._name

    @property
    def stage(self) -> str:
        return self._stage

    @property
    def force(self) -> bool:
        return self._force

    @property
    def verbose(self) -> bool:
        return self._verbose

    @property
    def summary(self) -> None:
        self._printer.print_title("Pipeline {} Summary".format(self._config.name))
        self._printer.print_dataframe(self._result)

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
        for task in self._tasks:
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
class PipelineBuilder(ABC):
    """Abstract pipeline builder. Defines interface."""

    def __init__(self) -> None:
        self._request = None
        self._config = None

        self._pipeline = None
        self._tasks = []
        self._task_seq = 0

    @property
    def pipeline(self) -> Pipeline:
        return self._pipeline

    def make_request(self, request: PipelineRequest) -> None:
        self._request = request
        return self

    def reset(self) -> None:
        self._pipeline = None
        self._tasks = []
        return self

    def add_task(self, task) -> None:
        task.task_seq = self._task_seq
        self._task_seq += 1
        self._tasks.append(task)
        return self

    def build(self) -> Pipeline:
        self._pipeline = self._build()
        return self

    @abstractmethod
    def _build_configuration(self) -> None:
        """Returns a PipelineConfig of the appropriate type object."""
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

    def _build_configuration(self) -> None:
        factory = LoggerFactory()
        logger = factory.get_logger(
            name=self._request.workspace.name,
            directory=self._request.workspace.directory,
            logging_level=self._request.logging_level,
            verbose=self._request.verbose,
        )
        return DataPipelineConfig(
            name=self._request.name,
            stage=self._request.stage,
            workspace=self._request.workspace,
            tasks=self._tasks,
            logger=logger,
            force=self._request.force,
            verbose=self._request.verbose,
            progress=self._request.progress,
            random_state=self._request.random_state,
            dataset_config=self._request.dataset_request,
        )

    def _build(self) -> Pipeline:
        """Builds and returns a Pipeline object."""
        config = self._build_configuration()
        return DataPipeline(config=config)
