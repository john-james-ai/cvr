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
# Modified : Saturday, January 22nd 2022, 9:02:59 pm                                                                       #
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
from cvr.core.task import Task
from cvr.utils.logger import LoggerFactory
from cvr.utils.config import WorkspaceConfig, PipelineConfig
from cvr.utils.printing import Printer


# ======================================================================================================================== #
class Pipeline(Asset):
    """Defines interface for pipelines."""

    def __init__(self, command: PipelineCommand, tasks: list) -> None:
        self._command = command
        self._tasks = tasks
        self._logger = command.logger

        self._data = None

        self._summary = OrderedDict()
        self._task_summaries = OrderedDict()

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
    def stage(self) -> str:
        return self._command.stage

    @property
    def force(self) -> logging:
        return self._command.force

    @property
    def keep_interim(self) -> logging:
        return self._command.keep_interim

    @property
    def verbose(self) -> logging:
        return self._command.verbose

    @property
    def summary(self) -> None:
        self._printer.print_title("DataPipeline {} Summary".format(self._command.name))
        self._printer.print_dictionary(self._summary)

    def run(self) -> None:
        self._setup()
        self._run(command=self._command)
        self._teardown()

    def _setup(self) -> None:
        self._start = datetime.now()
        self._logger.info("Started {}".format(self._name))

    def _teardown(self) -> None:
        self._end = datetime.now()
        self._duration = self._end - self._start
        self._logger.info("Completed {}. Duration {}".format(self._name, self._duration))
        self._summary["Start"] = self._start
        self._summary["End"] = self._end
        self._summary["Duration"] = self._duration

    @abstractmethod
    def _run(self, command: PipelineCommand) -> None:
        pass


# ------------------------------------------------------------------------------------------------------------------------ #
class DataPipeline(Pipeline):
    def __init__(self, command: PipelineCommand, tasks: list) -> None:
        super(DataPipeline, self).__init__(command=command, tasks=tasks)

    def _run(self, command: PipelineCommand) -> None:
        for task in self._tasks:
            # Run the task
            self._data = task.run(command=command, data=self._data)

            # Update the pipeline summary with the task status
            self._summary.update({task.__class__.__name__: task.status})

            # Add the task summary to the list of task summaries for reporting and debugging.
            self._task_summaries[task.name] = task.summary

            # Print the task summary if verbose
            if self._command.verbose:
                self._printer.print_title(
                    "DataPipeline {} Summary".format(self._command.name), "{} Step".format(task.__class__.__name__)
                )
                self._printer.print_dictionary(task.summary)

            logger.info("{} Complete. Status: {}".format(task.__class__.__name__, task.status))


# ======================================================================================================================== #
class PipelineCommand:
    def __init__(
        self,
        aid: str,
        workspace: str,
        stage: str,
        name: str,
        logger: logging,
        force: bool = False,
        keep_interim: bool = True,
        verbose: bool = True,
    ) -> None:
        self._aid = aid
        self._workspace = workspace
        self._stage = stage
        self._name = name
        self._logger = logger
        self._force = force
        self._keep_interim = keep_interim
        self._verbose = verbose

    @property
    def aid(self) -> str:
        return self._aid

    @property
    def workspace(self) -> str:
        return self._workspace

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


# ------------------------------------------------------------------------------------------------------------------------ #
class PipelineBuilder(ABC):
    """Abstract pipeline builder. Defines interface."""

    def __init__(self, name: str, stage: str, force: bool = False, keep_interim: bool = True, verbose: bool = True) -> None:
        self._name = name
        self._stage = stage
        self._force = force
        self._keep_interim = keep_interim
        self._verbose = verbose
        self._command = None

        self._workspace = WorkspaceConfig().get_workspace()

        self._pipeline = None
        self._tasks = []
        self._printer = Printer()

    @property
    def pipeline(self) -> None:
        return self._pipeline

    def build_command(self) -> PipelineCommand:
        self._command = PipelineCommand(
            aid=self._workspace + "_" + self._pipeline.__class__.__name__ + "_" + self._stage + "_" + self._name,
            workspace=self._workspace,
            stage=self._stage,
            name=self._name,
            logger=self._logger,
            force=self._force,
            keep_interim=self._keep_interim,
            verbose=self._verbose,
        )

    def build_log(self) -> None:
        factory = LoggerFactory()
        self._logger = factor.get_logger(
            workspace=self._workspace, classname=self._pipeline.__class__.__name__, name=self._name, verbose=self._verbose
        )

    @abstractmethod
    def create(self) -> None:
        pass

    def add_task(self, task: Task) -> None:
        task.task_seq = self._task_seq
        self._task_seq += 1
        self._tasks.append(task)

    def build(self) -> Pipeline:
        self.build_log()
        self.build_command()
        self.pipeline = DataPipeline(command=self._command, tasks=self._tasks)


# ------------------------------------------------------------------------------------------------------------------------ #
class DataPipelineBuilder(PipelineBuilder):
    """Data pipeline builder"""

    def __init__(self) -> None:
        super(DataPipelineBuilder, self).__init__()

    def create(self) -> None:
        self._pipeline = DataPipeline()
        self._tasks = []
