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
# Modified : Monday, January 24th 2022, 11:18:48 am                                                                        #
# Modifier : John James (john.james.ai.studio@gmail.com)                                                                   #
# ------------------------------------------------------------------------------------------------------------------------ #
# License  : BSD 3-clause "New" or "Revised" License                                                                       #
# Copyright: (c) 2022 Bryant St. Labs                                                                                      #
# ======================================================================================================================== #
"""Defines the pipeline construction and operation classes."""
from abc import ABC, abstractmethod
import os
from datetime import datetime
import logging
import pandas as pd
from collections import OrderedDict
from typing import Union

from cvr.core.asset import Asset
from cvr.utils.logger import LoggerFactory
from cvr.utils.config import WorkspaceConfig
from cvr.utils.printing import Printer


# ======================================================================================================================== #
class PipelineCommand:
    def __init__(
        self,
        aid: str,
        workspace: str,
        sample_size: float,
        random_state: int,
        stage: str,
        name: str,
        logger: logging,
        force: bool = False,
        keep_interim: bool = True,
        verbose: bool = True,
        progress: bool = False,
        check_download: int = 20,
    ) -> None:
        self._aid = aid
        self._workspace = workspace
        self._sample_size = sample_size
        self._random_state = random_state
        self._stage = stage
        self._name = name
        self._logger = logger
        self._force = force
        self._keep_interim = keep_interim
        self._verbose = verbose
        self._progress = progress
        self._check_download = check_download

    @property
    def aid(self) -> str:
        return self._aid

    @property
    def workspace(self) -> str:
        return self._workspace

    @property
    def sample_size(self) -> float:
        return self._sample_size

    @property
    def random_state(self) -> int:
        return self._random_state

    @property
    def name(self) -> str:
        return self._name

    @property
    def stage(self) -> str:
        return self._stage

    @property
    def logger(self) -> logging:
        return self._logger

    @property
    def force(self) -> str:
        return self._force

    @property
    def keep_interim(self) -> str:
        return self._keep_interim

    @property
    def verbose(self) -> str:
        return self._verbose

    @property
    def progress(self) -> str:
        return self._progress

    @property
    def check_download(self) -> str:
        return self._check_download


# ======================================================================================================================== #
class Pipeline(Asset):
    """Defines interface for pipelines."""

    def __init__(self, command: PipelineCommand, tasks: list) -> None:
        super(Pipeline, self).__init__(command.name, command.stage)
        self._command = command
        self._tasks = tasks
        self._logger = command.logger

        self._data = None

        self._result = pd.DataFrame()

        self._printer = Printer()
        self._start = None
        self._end = None
        self._duration = None

    @property
    def aid(self) -> str:
        return self._command.aid

    @property
    def name(self) -> str:
        return self._command.name

    @property
    def workspace(self) -> str:
        return self._command.workspace

    @property
    def sample_size(self) -> str:
        return self._command.sample_size

    @property
    def random_state(self) -> str:
        return self._command.random_state

    @property
    def stage(self) -> str:
        return self._command.stage

    @property
    def force(self) -> bool:
        return self._command.force

    @property
    def keep_interim(self) -> bool:
        return self._command.keep_interim

    @property
    def verbose(self) -> bool:
        return self._command.verbose

    @property
    def check_download(self) -> bool:
        return self._command.check_download

    @property
    def summary(self) -> None:
        self._printer.print_title("DataPipeline {} Summary".format(self._command.name))
        self._printer.print_dataframe(self._result)

    def run(self) -> None:
        self._setup()
        self._run(command=self._command)
        self._teardown()
        return self._data

    def _setup(self) -> None:
        self._start = datetime.now()
        self._logger.info("Started {}".format(self._name))

    def _teardown(self) -> None:
        self._end = datetime.now()
        self._duration = self._end - self._start
        self._logger.info("Completed {}".format(self._name))

    def _run(self, command: PipelineCommand) -> None:
        for task in self._tasks:
            # Run the task
            self._data = task.run(command=command, data=self._data)

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
    def __init__(self, command: PipelineCommand, tasks: list) -> None:
        super(DataPipeline, self).__init__(command=command, tasks=tasks)


# ------------------------------------------------------------------------------------------------------------------------ #
class PipelineBuilder(ABC):
    """Abstract pipeline builder. Defines interface."""

    def __init__(self) -> None:
        self._name = None
        self._stage = None
        self._force = False
        self._keep_interim = None
        self._verbose = True
        self._command = None
        self._progress = False
        self._check_download = 20

        self._config = WorkspaceConfig()
        self._workspace = self._config.get_workspace()
        self._sample_size = self._config.get_sample_size()
        self._random_state = self._config.get_random_state()

        self._pipeline = None
        self._tasks = []
        self._task_seq = 0
        self._printer = Printer()

    @property
    def pipeline(self) -> None:
        return self._pipeline

    def set_name(self, name: str) -> None:
        self._name = name
        return self

    def set_stage(self, stage: str) -> None:
        self._stage = stage
        return self

    def set_force(self, force: bool = False) -> None:
        self._force = force
        return self

    def set_verbose(self, verbose: bool = True) -> None:
        self._verbose = verbose
        return self

    def set_progress(self, progress: bool = False) -> None:
        self._progress = progress
        return self

    def set_check_download(self, check_download: int = 20) -> None:
        self._check_download = check_download
        return self

    def build_command(self) -> PipelineCommand:
        self._command = PipelineCommand(
            aid=self._workspace + "_" + self._pipeline.__class__.__name__ + "_" + self._stage + "_" + self._name,
            workspace=self._workspace,
            sample_size=self._sample_size,
            random_state=self._random_state,
            stage=self._stage,
            name=self._name,
            logger=self._logger,
            force=self._force,
            keep_interim=self._keep_interim,
            verbose=self._verbose,
            progress=self._progress,
            check_download=self._check_download,
        )

    def build_log(self) -> None:
        factory = LoggerFactory()
        self._logger = factory.get_logger(
            workspace=self._workspace, stage=self._stage, name=self._name, verbose=self._verbose
        )

    @abstractmethod
    def create(self) -> None:
        pass

    def add_task(self, task) -> None:
        task.task_seq = self._task_seq
        self._task_seq += 1
        self._tasks.append(task)

    def build(self) -> Pipeline:
        self.build_log()
        self.build_command()
        self._pipeline = DataPipeline(command=self._command, tasks=self._tasks)


# ------------------------------------------------------------------------------------------------------------------------ #
class DataPipelineBuilder(PipelineBuilder):
    """Data pipeline builder"""

    def __init__(self) -> None:
        super(DataPipelineBuilder, self).__init__()

    def create(self) -> None:
        self._pipeline = None
        self._tasks = []
