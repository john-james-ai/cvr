#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project  : Deep Learning for Conversion Rate Prediction (CVR)                                    #
# Version  : 0.1.0                                                                                 #
# File     : \io.py                                                                                #
# Language : Python 3.7.12                                                                         #
# ------------------------------------------------------------------------------------------------ #
# Author   : John James                                                                            #
# Email    : john.james.ai.studio@gmail.com                                                        #
# URL      : https://github.com/john-james-ai/cvr                                                  #
# ------------------------------------------------------------------------------------------------ #
# Created  : Thursday, February 10th 2022, 1:12:03 am                                              #
# Modified : Thursday, February 10th 2022, 8:18:34 pm                                              #
# Modifier : John James (john.james.ai.studio@gmail.com)                                           #
# ------------------------------------------------------------------------------------------------ #
# License  : BSD 3-clause "New" or "Revised" License                                               #
# Copyright: (c) 2022 Bryant St. Labs                                                              #
# ================================================================================================ #

"""IO Related Tasks"""
import inspect
from datetime import datetime

from cvr.core.asset import AssetPassport
from cvr.core.dataset import Dataset
from cvr.core.task import Task, TaskResult, TaskResponse


# ============================================================================ #
#                            DATASET READER                                    #
# ============================================================================ #
class DatasetReader(Task):
    """Reads Dataset objects from the asset repository

    Args:
        dataset_repo (DatasetRepo): Asset repository

    """

    def __init__(self, passport: AssetPassport) -> None:
        super(DatasetReader, self).__init__(passport=passport)

    def run(self, passport: AssetPassport) -> Dataset:

        self.setup()
        self._logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        dataset = self._config.dataset_repo.get(
            asset_type=passport.asset_type,
            stage=passport.stage,
            name=passport.name,
        )

        self._logger.info(
            "\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3])
        )

        self.teardown()

        return dataset

    def _setup(self) -> None:
        self._response = TaskResponse()
        self._response.begin()
        self._result = TaskResult()

    def _teardown(self) -> None:
        self._response.end()
        self._result.executed = True
        self._result.passed = True
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


# ============================================================================ #
#                            DATASET WRITER                                    #
# ============================================================================ #
class DatasetWriter(Task):
    """Writes Dataset objects to the asset repository

    Args:
        dataset_repo (DatasetRepo): Asset repository

    """

    def __init__(self, passport: AssetPassport) -> None:
        super(DatasetWriter, self).__init__(passport=passport)

    def run(self, dataset: Dataset) -> None:

        self.setup()

        self._logger.debug("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        self._config.dataset_repo.add(dataset)

        self._logger.debug(
            "\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3])
        )

        self.teardown()

    def _setup(self) -> None:
        self._response = TaskResponse()
        self._response.begin()
        self._result = TaskResult()

    def _teardown(self) -> None:
        self._response.stop()
        self._result.executed = True
        self._result.passed = True
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
