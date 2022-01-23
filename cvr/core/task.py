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
# Modified : Sunday, January 23rd 2022, 5:31:19 am                                                                         #
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

from cvr.data.datasets import DatasetBuilder
from cvr.data.datasets import Dataset
from cvr.core.pipeline import PipelineCommand

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

        self._dataset_builder = DatasetBuilder()

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

    def _setup(self, logger: logging) -> None:
        self._start = datetime.now()
        self._status_code = "102"
        logger.info("Started Task: {}".format(self._name))

    def _teardown(self, logger: logging) -> None:
        self._end = datetime.now()
        self._duration = self._end - self._start
        self._summary["Start"] = self._start
        self._summary["End"] = self._end
        self._summary["Duration"] = self._duration
        self._summary["Status"] = self.status
        self._summary["Status Date"] = date.today()
        self._summary["Status Time"] = self._end.strftime("%H:%M:%S")
        logger.info("Ended Task: {}. Status: {}".format(self._name, self.status))

    def run(self, command: PipelineCommand, data: Dataset = None) -> Dataset:
        """Runs the task through delegation to a private member on the subclass

        Args:
            command (PipelineCommand): Container for Pipeline configuration
            data (Dataset): Optional. Input Dataset object. The optional exception is for original sourcing.

        """
        self._setup(command.logger)
        dataset = self._run(command=command, data=data)
        self._teardown(command.logger)
        return dataset

    def _abort_existence(self) -> None:
        """Status to report when task not executed because output data already exists."""
        self._status_code = "215"

    def _build_dataset(self, command: PipelineCommand, data: pd.DataFrame) -> Dataset:
        dataset_name = command.name + "_" + self.__class__.__name__
        self._dataset_builder.create()
        self._dataset_builder.set_data(data)
        self._dataset_builder.set_name(dataset_name)
        self._dataset_builder.set_stage(command.stage)
        dataset = self._dataset_builder.build()
        return dataset

    @property
    @abstractmethod
    def summary(self) -> None:
        pass

    @abstractmethod
    def _run(command: PipelineCommand, data: Dataset = None) -> Dataset:
        pass
