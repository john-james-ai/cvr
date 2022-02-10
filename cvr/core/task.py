#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ============================================================================ #
# Project  : Deep Learning for Conversion Rate Prediction (CVR)                #
# Version  : 0.1.0                                                             #
# File     : \task.py                                                          #
# Language : Python 3.7.12                                                     #
# ---------------------------------------------------------------------------- #
# Author   : John James                                                        #
# Email    : john.james.ai.studio@gmail.com                                    #
# URL      : https://github.com/john-james-ai/cvr                              #
# ---------------------------------------------------------------------------- #
# Created  : Wednesday, January 19th 2022, 5:34:06 pm                          #
# Modified : Wednesday, February 9th 2022, 5:39:17 pm                          #
# Modifier : John James (john.james.ai.studio@gmail.com)                       #
# ---------------------------------------------------------------------------- #
# License  : BSD 3-clause "New" or "Revised" License                           #
# Copyright: (c) 2022 Bryant St. Labs                                          #
# ============================================================================ #
"""Abstract base class for pipeline tasks."""
from abc import ABC, abstractmethod
from datetime import datetime
from dataclasses import dataclass, field
from collections import OrderedDict
import pandas as pd

from cvr.utils.printing import Printer
from cvr.core.asset import AssetPassport

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
    start: datetime = field(init=False)
    end: datetime = field(init=False)
    duration: int = field(init=False)
    passed: bool = field(default=False)
    executed: bool = field(default=False)
    response: dict = field(init=False)
    result: str = field(default="Ok")

    def __post_init__(self) -> None:
        self._printer = Printer()

    def start(self) -> None:
        self.start = datetime.now()
        self.result = "In-Progress"

    def stop(self) -> None:
        self.end = datetime.now()
        self.duration = (self.end - self.start).total_seconds() / 60
        self.result = "Complete"

    def asdict(self) -> dict:
        passport = {
            "aid": self.passport.aid,
            "name": self.passport.name,
            "description": self.passport.description,
            "stage": self.passport.stage,
        }
        result = {
            "Start": self.start,
            "End": self.end,
            "Duration": self.duration,
            "Passed": self.passed,
            "Executed": self.executed,
            "Result": self.result,
        }
        passport.update(self.response)
        passport.update(result)

        return passport

    def print(self) -> None:
        d = self.asdict()
        self._printer.print_title("Pipeline Event Summary")
        self._printer.print_dictionary(d)


# ---------------------------------------------------------------------------- #
#                                 TASK                                         #
# ---------------------------------------------------------------------------- #


class Task(ABC):
    """Defines interface for task classes."""

    def __init__(self, passport: AssetPassport, **kwargs) -> None:
        self._passport = passport
        self._response = OrderedDict()
        self._summary = TaskSummary(passport)
        self._config = None

    @property
    def passport(self):
        return self._passport

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config) -> None:
        self._config = config

    def setup(self, **kwargs) -> None:
        self._summary.begin()
        self._setup(kwargs)

    def _setup(self) -> None:
        pass

    def teardown(self, **kwargs) -> None:
        self._summary.stop()
        self._teardown(kwargs)

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

    def summary(self) -> TaskSummary:
        return self._summary

    def summarize(self) -> None:
        self._summary.print()
