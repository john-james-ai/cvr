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
# Modified : Friday, January 21st 2022, 12:59:55 am                                                                        #
# Modifier : John James (john.james.ai.studio@gmail.com)                                                                   #
# ------------------------------------------------------------------------------------------------------------------------ #
# License  : BSD 3-clause "New" or "Revised" License                                                                       #
# Copyright: (c) 2022 Bryant St. Labs                                                                                      #
# ======================================================================================================================== #
"""Defines the pipeline construction and operation classes."""
from abc import ABC, abstractmethod
from datetime import datetime
import logging
import pandas as pd
from collections import OrderedDict
from typing import Union

from cvr.core.task import Task
from cvr.utils.printing import Printer

# ------------------------------------------------------------------------------------------------------------------------ #
class PipelineCommand:
    def __init__(self, name: str, force: bool = False, keep_interim: bool = True, verbose: bool = True) -> None:
        self.name = name
        self.force = force
        self.keep_interim = keep_interim
        self.verbose = verbose


# ======================================================================================================================== #
class Pipeline(ABC):
    """Defines interface for pipelines."""

    def __init__(self, command: PipelineCommand, tasks: list) -> None:
        self._command = command
        self._tasks = tasks

        self._data = None

        self._summary = OrderedDict()
        self._task_summaries = OrderedDict()
        self._workspace = "root"
        self._name = self._workspace + "_" + str(__class__.__name__).lower()
        self._logger = None
        self._printer = Printer()
        self._start = None
        self._end = None
        self._duration = None

    @property
    def name(self) -> str:
        self._name = self._workspace + "_" + str(__class__.__name__).lower()
        return self._name

    @property
    def summary(self) -> None:
        self._printer.print_title("DataPipeline {} Summary".format(self._command.name))
        self._printer.print_dictionary(self._summary)

    @property
    def workspace(self) -> str:
        return self._workspace

    @workspace.setter
    def workspace(self, workspace: str) -> None:
        self._workspace = workspace
        self._name = self._workspace + "_" + str(__class__.__name__).lower()

    @property
    def name(self) -> str:
        return self._name

    def run(self) -> None:
        self._setup()
        self._run(command=self._command, logger=self._logger)
        self._teardown()

    def _set_logger(self) -> logging:
        logname = self.__class__.__name__.lower() + "_" + self._name
        logfilename = logname + ".log"
        logfilepath = os.path.join("workspaces", self._workspace, "logs", logfilename)
        logfile
        logging.root.handlers = []
        self._logger = logging.getLogger(self._name)
        self._logger.setLevel(logging.DEBUG)
        format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        ch = logging.StreamHandler()
        ch.setFormatter(format)
        self._logger.addHandler(ch)

        fh = logging.handlers.TimedRotatingFileHandler(logfilename, when="d", encoding=None, delay=False)
        fh.setFormatter(format)
        self._logger.addHandler(fh)

    def _setup(self) -> None:
        self._set_logger()
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
    def _run(self, command: PipelineCommand, logger: logging) -> None:
        pass


# ------------------------------------------------------------------------------------------------------------------------ #
class DataPipeline(Pipeline):
    def __init__(self, command: PipelineCommand, tasks: list) -> None:
        super(DataPipeline, self).__init__(command=command, tasks=tasks)

    def _run(self, command: PipelineCommand, logger: logging) -> None:
        for task in self._tasks:
            # Run the task
            self._data = task.run(logger=logger, data=self._data, force=self._command.force)

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
            else:
                logger.info("{} Complete. Status: {}".format(task.__class__.__name__, task.status))


# ------------------------------------------------------------------------------------------------------------------------ #
class PipelineBuilder(ABC):
    """Abstract pipeline builder. Defines interface."""

    def __init__(self) -> None:
        self._pipeline = None
        self._command = None
        self._tasks = []
        self._printer = Printer()

    @property
    def pipeline(self) -> None:
        return self._pipeline

    def create(self, command: PipelineCommand) -> None:
        self._command = command
        self._pipeline = None
        self._tasks = []

    def add_task(self, task: Task) -> None:
        self._tasks.append(task)

    @abstractmethod
    def build(self) -> Pipeline:
        pass


# ------------------------------------------------------------------------------------------------------------------------ #
class DataPipelineBuilder(PipelineBuilder):
    """Data pipeline builder"""

    def __init__(self) -> None:
        super(DataPipelineBuilder, self).__init__()

    def build(self) -> None:
        self._pipeline = DataPipeline(command=self._command, tasks=self._tasks)
