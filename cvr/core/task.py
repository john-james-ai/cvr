#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project  : Deep Learning for Conversion Rate Prediction (CVR)                                    #
# Version  : 0.1.0                                                                                 #
# File     : \task.py                                                                              #
# Language : Python 3.7.12                                                                         #
# ------------------------------------------------------------------------------------------------ #
# Author   : John James                                                                            #
# Email    : john.james.ai.studio@gmail.com                                                        #
# URL      : https://github.com/john-james-ai/cvr                                                  #
# ------------------------------------------------------------------------------------------------ #
# Created  : Wednesday, January 19th 2022, 5:34:06 pm                                              #
# Modified : Thursday, February 10th 2022, 9:28:37 pm                                              #
# Modifier : John James (john.james.ai.studio@gmail.com)                                           #
# ------------------------------------------------------------------------------------------------ #
# License  : BSD 3-clause "New" or "Revised" License                                               #
# Copyright: (c) 2022 Bryant St. Labs                                                              #
# ================================================================================================ #
from abc import ABC, abstractmethod
import pandas as pd
import inspect
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from cvr.utils.printing import Printer
from cvr.core.asset import AssetPassport
from cvr.core.dataset import Dataset

# ---------------------------------------------------------------------------- #
#                               TASK RESULT                                    #
# ---------------------------------------------------------------------------- #


@dataclass
class TaskResult(ABC):
    """Standardized result object for all tasks"""

    executed: bool = field(default=False)
    passed: bool = field(default=False)
    complete: str = field(default=False)
    completed: datetime = field(init=False)
    comment: str = field(default="")

    def __post_init__(self) -> None:
        self._printer = Printer()

    def todict(self) -> dict:
        d = {
            "Executed": self.executed,
            "Passed": self.passed,
            "Complete": self.complete,
            "Completed": self.completed,
            "Comment": self.comment,
        }
        return d

    def print(self) -> None:
        d = self.todict()
        self._printer.print_title("Task Result")
        self._printer.print_dictionary(d)


# ---------------------------------------------------------------------------- #
#                               TASK RESPONSE                                  #
# ---------------------------------------------------------------------------- #


@dataclass
class TaskResponse(ABC):
    """Task specific metrics to be overridden by subclasses."""

    start: datetime = field(init=False)
    end: datetime = field(init=False)
    duration: timedelta = field(init=False)

    def __post_init__(self) -> None:
        pass

    def begin(self) -> None:
        self.start = datetime.now()

    def stop(self) -> None:
        self.end = datetime.now()
        self.duration = self.end - self.start

    def __post_init__(self) -> None:
        self._printer = Printer()

    def todict(self) -> dict:
        d = {"Start": self.start, "End": self.end, "Duration": self.duration}
        return d

    def print(self) -> None:
        title = "Task Response"
        self._printer.print_title(title)
        d = self.todict()
        self._printer.print_dictionary(d)


# ---------------------------------------------------------------------------- #
#                               TASK SUMMARY                                   #
# ---------------------------------------------------------------------------- #


@dataclass
class TaskSummary(ABC):
    """Summarizes a Task.

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
    response: TaskResponse
    result: TaskResult

    def __post_init__(self) -> None:
        self._printer = Printer()

    def print(self) -> None:
        self.passport.print()
        self.response.print()
        self.result.print()


# ---------------------------------------------------------------------------- #
#                                 TASK                                         #
# ---------------------------------------------------------------------------- #


class Task(ABC):
    """Defines interface for task classes."""

    def __init__(self, passport: AssetPassport, **kwargs) -> None:
        self._passport = passport
        self._config = None

    @property
    def config(self):
        return self._config

    @property
    @abstractmethod
    def passport(self):
        pass

    @property
    @abstractmethod
    def response(self) -> TaskResponse:
        pass

    @property
    @abstractmethod
    def result(self) -> TaskResult:
        pass

    @config.setter
    def config(self, config) -> None:
        self._config = config

    def setup(self, **kwargs) -> None:
        # Logging facility
        self._logger = self._config.logger
        # Subclass specific setup
        self._setup()

    def _setup(self) -> None:
        pass

    def teardown(self, **kwargs) -> None:
        # Subclass specific teardown.
        self._teardown()
        # Base class gets last word
        self._result.executed = "No" if self._result.executed is False else "Yes"
        self._result.passed = "No" if self._result.passed is False else "Yes"
        self._result.complete = "No" if self._result.complete is False else "Yes"
        self._summary = TaskSummary(
            passport=self.passport,
            response=self.response,
            result=self.result,
        )

    def _teardown(self, **kwargs) -> None:
        pass

    @abstractmethod
    def run(self, data: pd.DataFrame = None) -> pd.DataFrame:
        """Runs the task through delegation to a private member on the subclass

        Args:
            df (pd.DataFrame): Input DataFrame object.

        Returns:
            df (pd.DataFrame): DataFrame object
            response (dict): Dictionary containing task response information.

        """
        pass

    @abstractmethod
    def passed(self) -> bool:
        """Checks success of task. Returns True if conditions pass."""
        pass

    def summary(self) -> TaskSummary:
        return self._summary

    def summarize(self) -> None:
        self._summary.print()


# ============================================================================ #
#                            DATASET FACTORY                                   #
# ============================================================================ #
class DatasetFactory(Task):
    """Creates Dataset objects."""

    def __init__(self, passport: AssetPassport, dataset_passport: AssetPassport) -> None:
        super(DatasetFactory, self).__init__(passport=passport)
        self._dataset_passport = dataset_passport

    def run(self, data: pd.DataFrame) -> Dataset:

        self.setup()

        self._logger.debug("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        aid = self._config.dataset_repo.aid_gen()
        self._dataset_passport.aid = aid

        dataset = Dataset(
            passport=self._dataset_passport,
            data=data,
        )

        dataset = self._config.dataset_repo.set_version(dataset)

        self._logger.debug(
            "\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3])
        )

        self.teardown()

        return dataset

    def _setup(self) -> None:
        self._response = TaskResponse()
        self._response.begin()
        self._result = TaskResult()

    def _teardown(self) -> None:
        self._response.stop()
        self._result.executed = True
        self._result.passed = self.passed()
        self._result.complete = True
        self._result.completed = datetime.now()

    @property
    def passport(self) -> AssetPassport:
        return self._passport

    @property
    def response(self) -> TaskResponse:
        return self._response

    @property
    def result(self) -> TaskResult:
        return self._result

    def passed(self) -> bool:
        return True
