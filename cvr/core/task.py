#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ======================================================================================================================== #
# Project  : Conversion Rate Prediction (CVR)                                                                              #
# Version  : 0.1.0                                                                                                         #
# File     : \task.py                                                                                                      #
# Language : Python 3.10.1                                                                                                 #
# ------------------------------------------------------------------------------------------------------------------------ #
# Author   : John James                                                                                                    #
# Email    : john.james.ai.studio@gmail.com                                                                                #
# URL      : https://github.com/john-james-ai/cvr                                                                          #
# ------------------------------------------------------------------------------------------------------------------------ #
# Created  : Wednesday, January 19th 2022, 5:34:06 pm                                                                      #
# Modified : Tuesday, January 25th 2022, 11:45:35 pm                                                                       #
# Modifier : John James (john.james.ai.studio@gmail.com)                                                                   #
# ------------------------------------------------------------------------------------------------------------------------ #
# License  : BSD 3-clause "New" or "Revised" License                                                                       #
# Copyright: (c) 2022 Bryant St. Labs                                                                                      #
# ======================================================================================================================== #
"""Abstract base class for all behavioral classes that performing operations on data."""
from abc import ABC, abstractmethod
from collections import OrderedDict
from datetime import datetime, date
import logging
import pandas as pd
from typing import Union

from cvr.data.dataset import DatasetBuilder
from cvr.data.dataset import Dataset
from cvr.core.pipeline import PipelineCommand
from cvr.utils.printing import Printer
from cvr.utils.format import titlelize

# ------------------------------------------------------------------------------------------------------------------------ #
STATUS_CODES = {
    "102": "Processing",
    "200": "OK",
    "202": "Accepted",
    "215": "Complete - Not Executed: Output Data Already Exists",
    "404": "Bad Request",
    "500": "Internal Server Error. See HTTP Response Code",
}

# ======================================================================================================================== #
class Task(ABC):
    """Defines interface for task classes."""

    def __init__(self, **kwargs) -> None:
        self._name = self.__class__.__name__
        self._task_seq = None

        self._start = None
        self._end = None
        self._duration = None

        self._logger = None

        self._status_code = "202"
        self._status_text = STATUS_CODES[self._status_code]
        self._status = self._status_code + ": " + self._status_text

        self._summary = OrderedDict()
        self._command = None

        self._dataset_builder = DatasetBuilder()
        self._printer = Printer()

    @property
    def name(self) -> None:
        return self._name

    @property
    def task_seq(self) -> int:
        return self._task_seq

    @task_seq.setter
    def task_seq(self, task_seq) -> None:
        self._task_seq = task_seq

    @property
    def start(self) -> datetime:
        return self._start

    @property
    def end(self) -> datetime:
        return self._end

    @property
    def duration(self) -> datetime:
        return self._duration

    @property
    def status(self) -> str:
        self._status_text = STATUS_CODES[self._status_code]
        self._status = self._status_code + ": " + self._status_text
        return self._status

    def set_data(self, df: pd.DataFrame) -> None:
        self._df = df

    def _setup(self) -> None:
        self._start = datetime.now()
        self._status_code = "102"

    def _teardown(self) -> None:
        self._end = datetime.now()
        self._duration = self._end - self._start
        d = OrderedDict()
        d["Start"] = self._start
        d["End"] = self._end
        d["Duration"] = self._duration
        d["Status"] = self.status
        d["Status Date"] = date.today()
        d["Status Time"] = self._end.strftime("%H:%M:%S")
        self._summary.update(d)

    def run(self, command: PipelineCommand, data: Dataset = None) -> Dataset:
        """Runs the task through delegation to a private member on the subclass

        Args:
            command (PipelineCommand): Container for Pipeline configuration
            data (Dataset): Optional. Input Dataset object. The optional exception is for original sourcing.

        """
        self._command = command
        self._logger = command.logger

        self._setup()
        dataset = self._run(data=data)
        self._teardown()
        return dataset

    def _abort_existence(self) -> None:
        """Status to report when task not executed because output data already exists."""
        self._status_code = "215"

    def _build_dataset(self, data: pd.DataFrame) -> Dataset:
        dataset_name = self._command.name
        self._dataset_builder.create()
        self._dataset_builder.set_data(data)
        self._dataset_builder.set_name(dataset_name)
        self._dataset_builder.set_stage(self._command.stage)
        dataset = self._dataset_builder.build()
        return dataset

    @property
    @abstractmethod
    def summary(self) -> None:
        """Can print a result or return a summary to the caller."""
        pass

    def _print_summary_dict(self, summary: dict, subtitle: str = None) -> None:
        """Prints the result of the task operation"""
        title = "{} Task Summary".format(self.__class__.__name__)
        if not subtitle:
            subtitle = "Dataset: {} / {} Stage".format(self._command.name, self._command.stage)
        subtitle = titlelize(subtitle)
        self._printer.print_title(title, subtitle)
        self._printer.print_dictionary(summary)

    def _print_summary_df(self, summary: pd.DataFrame, subtitle: str = None) -> None:
        """Prints the result of the task operation"""
        title = "{} Task Summary".format(self.__class__.__name__)
        if not subtitle:
            subtitle = "Dataset: {} / {} Stage".format(self._command.name, self._command.stage)
        subtitle = titlelize(subtitle)
        self._printer.print_title(title, subtitle)
        self._printer.print_dataframe(summary)

    @abstractmethod
    def _run(self, data: Dataset = None) -> Dataset:
        pass
