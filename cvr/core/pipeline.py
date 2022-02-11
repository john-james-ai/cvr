#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project  : Deep Learning for Conversion Rate Prediction (CVR)                                    #
# Version  : 0.1.0                                                                                 #
# File     : \pipeline.py                                                                          #
# Language : Python 3.7.12                                                                         #
# ------------------------------------------------------------------------------------------------ #
# Author   : John James                                                                            #
# Email    : john.james.ai.studio@gmail.com                                                        #
# URL      : https://github.com/john-james-ai/cvr                                                  #
# ------------------------------------------------------------------------------------------------ #
# Created  : Wednesday, January 19th 2022, 5:46:57 pm                                              #
# Modified :                                                                                       #
# Modifier : John James (john.james.ai.studio@gmail.com)                                           #
# ------------------------------------------------------------------------------------------------ #
# License  : BSD 3-clause "New" or "Revised" License                                               #
# Copyright: (c) 2022 Bryant St. Labs                                                              #
# ================================================================================================ #
#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ============================================================================ #
# Project  : Deep Learning for Conversion Rate Prediction (CVR)                #
# Version  : 0.1.0                                                             #
# File     : \pipeline.py                                                      #
# Language : Python 3.7.12                                                     #
# ---------------------------------------------------------------------------- #
# Author   : John James                                                        #
# Email    : john.james.ai.studio@gmail.com                                    #
# URL      : https://github.com/john-james-ai/cvr                              #
# ---------------------------------------------------------------------------- #
# Created  : Wednesday, January 19th 2022, 5:46:57 pm                          #
# Modified : Wednesday, February 9th 2022, 6:58:05 pm                          #
# Modifier : John James (john.james.ai.studio@gmail.com)                       #
# ---------------------------------------------------------------------------- #
# License  : BSD 3-clause "New" or "Revised" License                           #
# Copyright: (c) 2022 Bryant St. Labs                                          #
# ============================================================================ #

"""Defines the pipeline construction and operation classes."""
from abc import ABC, abstractmethod
from collections import OrderedDict
from datetime import datetime
import pandas as pd
from dataclasses import dataclass, field
from typing import Union
import logging

from cvr.core.task import Task
from cvr.core.dataset import Dataset, DatasetRepo
from cvr.core.asset import Asset, AssetBuilder, AssetPassport
from cvr.utils.printing import Printer

# ---------------------------------------------------------------------------- #
#                           PIPELINE CONFIG                                    #
# ---------------------------------------------------------------------------- #


@dataclass
class PipelineConfig:
    """Configuration for an Event

    Args:
        directory (str): The home directory for the studio.
        logger (logging): Logging object
        dataset_repo (DatasetRepo): Dataset repository
        force (bool): If True. All tasks are executed, overwriting any
                existing cache. If False, only tasks for which no cache is
                available are executed.
        verbose (bool): Controls the level of log messaging during
                pipeline execution
        progress (bool): If True, a progress bar is rendered.
        random_state (int): Seed for pseudo random number generation


    """

    dataset_repo: DatasetRepo
    directory: str
    logger: logging
    verbose: bool = True
    progress: bool = False
    force: bool = False
    random_state: int = None


# ---------------------------------------------------------------------------- #
#                             PIPELINE SUMMARY                                 #
# ---------------------------------------------------------------------------- #


@dataclass
class PipelineSummary(ABC):
    """Summarizes a Pipeline.

    Args:
        passport (AssetPassport): Identity object
        start (datetime): start time for event
        end (datetime): end time for event
        duration (timedelta): the duration of the event in minutes
        passed (bool): True if the event passed
        executed (bool): True if the event was executed. An event
            may be skipped if its endpoint already exists
        response (dict): Event specific response
        result (str): result of the event
    """

    passport: AssetPassport
    start: datetime = field(init=False)
    end: datetime = field(init=False)
    duration: int = field(init=False)
    passed: bool = field(default=False)
    executed: bool = field(default=False)
    response: dict = field(init=False)
    result: str = field(default="Ok")

    def __post_init__(self) -> None:
        self._printer = Printer()

    def begin(self) -> None:
        self.start = datetime.now()
        self.result = "In-Progress"

    def stop(self) -> None:
        self.end = datetime.now()
        self.duration = (self.end - self.start).total_seconds() / 60
        self.result = "Complete"

    def asdict(self) -> dict:
        result = {
            "Start": self.start,
            "End": self.end,
            "Duration": self.duration,
            "Passed": self.passed,
            "Executed": self.executed,
            "Result": self.result,
        }
        self.response.update(result)

        return result

    def print(self) -> None:
        result = {
            "Start": self.start,
            "End": self.end,
            "Duration": self.duration,
            "Passed": self.passed,
            "Executed": self.executed,
            "Result": self.result,
        }
        self._printer.print_title("Pipeline Event Summary")
        self._printer.print_dictionary(self.response)
        self._printer.print_blank_line()
        self._printer.print_dictionary(result)


# ---------------------------------------------------------------------------- #
#                                PIPELINE                                      #
# ---------------------------------------------------------------------------- #


class Pipeline(Asset):
    """Defines interface for pipelines.

    Args:
        tasks (list): List of task objects
    """

    def __init__(self, passport: AssetPassport, config: PipelineConfig, tasks: list) -> None:
        self._passport = passport
        self._tasks = tasks
        self._config = config
        self._summary = PipelineSummary(passport=passport)
        self._task_summaries = OrderedDict()

    @property
    def passport(self) -> AssetPassport:
        return self._passport

    @property
    def config(self) -> PipelineConfig:
        return self._config

    @abstractmethod
    def run(self) -> Union[pd.DataFrame, dict, Dataset]:
        pass

    @property
    def summary(self) -> PipelineSummary:
        return self._summary

    def summarize(self) -> None:
        self._summary.print()


# ---------------------------------------------------------------------------- #
#                              DATA PIPELINE                                   #
# ---------------------------------------------------------------------------- #


class DataPipeline(Pipeline):
    """Defines interface for DataPipeline

    Args:
        tasks (list): List of task objects
    """

    def __init__(self, passport: AssetPassport, config: PipelineConfig, tasks: list) -> None:
        super(DataPipeline, self).__init__(passport=passport, config=config, tasks=tasks)
        self._data = None

    def run(self) -> Union[pd.DataFrame, dict, Dataset]:
        self.setup()
        for task in self._tasks:
            self._data = task.run(self._data)
            self._task_summaries[task.__class__.__name__.lower()] = task.summary
            task.summarize()

        self.teardown()
        return self._data

    def setup(self) -> None:
        self._summary.begin()

    def teardown(self) -> None:
        self._summary.stop()


# ---------------------------------------------------------------------------- #
#                        DATA PIPELINE BUILDER                                 #
# ---------------------------------------------------------------------------- #
class DataPipelineBuilder(AssetBuilder):
    def __init__(self) -> None:
        self.reset()
        self._config = None
        self._passport = None
        self._tasks = []

    def reset(self) -> None:
        self._data_pipeline = None
        self._passport = None
        self._config = None
        self._tasks = []
        return self

    @property
    def data_pipeline(self) -> DataPipeline:
        data_pipeline = self._data_pipeline
        self.reset()
        return data_pipeline

    def set_passport(self, passport) -> None:
        self._passport = passport
        return self

    def set_config(self, config) -> None:
        self._config = config
        return self

    def add_task(self, task: Task) -> None:
        task.config = self._config
        self._tasks.append(task)
        return self

    def build(self) -> None:
        self._data_pipeline = DataPipeline(
            passport=self._passport, config=self._config, tasks=self._tasks
        )
        return self
