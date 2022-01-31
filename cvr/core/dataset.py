#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ======================================================================================================================== #
# Project  : Conversion Rate Prediction (CVR)                                                                              #
# Version  : 0.1.0                                                                                                         #
# File     : \dataset.py                                                                                                   #
# Language : Python 3.7.12                                                                                                 #
# ------------------------------------------------------------------------------------------------------------------------ #
# Author   : John James                                                                                                    #
# Email    : john.james.ai.studio@gmail.com                                                                                #
# URL      : https://github.com/john-james-ai/cvr                                                                          #
# ------------------------------------------------------------------------------------------------------------------------ #
# Created  : Thursday, January 13th 2022, 2:22:59 am                                                                       #
# Modified : Sunday, January 30th 2022, 11:36:16 pm                                                                        #
# Modifier : John James (john.james.ai.studio@gmail.com)                                                                   #
# ------------------------------------------------------------------------------------------------------------------------ #
# License  : BSD 3-clause "New" or "Revised" License                                                                       #
# Copyright: (c) 2022 Bryant St. Labs                                                                                      #
# ======================================================================================================================== #
"""Dataset Module"""
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from collections import OrderedDict
from datetime import datetime
import pandas as pd
import logging

from cvr.core.asset import Asset, AssetRequest, AssetBuilder, AssetConfig
from cvr.core.profile import DataProfiler
from cvr.utils.printing import Printer
from cvr.utils.format import titlelize
from cvr.utils.io import Pickler

# ------------------------------------------------------------------------------------------------------------------------ #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------------------------------------------------ #
@dataclass(frozen=True)
class DatasetRequest(AssetRequest):
    """Dataclass encapsulating the parameters sent to a DatasetBuilder object."""

    sample_size: int = None
    data: pd.DataFrame = pd.DataFrame()


# ------------------------------------------------------------------------------------------------------------------------ #
@dataclass
class DatasetConfig(AssetConfig):
    sample_size: int = None
    data: pd.DataFrame = pd.DataFrame()
    printer: Printer = Printer()
    profile: DataProfiler = field(init=False)

    def set_profiler(self) -> None:
        self.profile = DataProfiler(name=self.name, description=self.description, stage=self.stage, data=self.data)


# ------------------------------------------------------------------------------------------------------------------------ #
class AbstractDataset(Asset):
    def __init__(self, config: DatasetConfig) -> None:
        self._config = config

        self._df = self._config.data
        self._profile = self._config.profile

        self._printer = self._config.printer

    def set_task_data(self, task):
        """Injects a task object with the data from this dataset.

        Args:
            task (Task): The task requiring the data
        """

        task.set_data(self._df)
        return task

    def info(self) -> dict:
        """Prints column i.e. structural metadata."""
        self._printer.print_title("Dataset {}".format(self.name))
        self._df.info(show_counts=True)

    # ------------------------------------------------------------------------------------------------------------------- #
    #                                               PROPERTIES                                                            #
    # ------------------------------------------------------------------------------------------------------------------- #
    @property
    def name(self) -> str:
        return self._config.name

    @property
    def description(self) -> str:
        return self._config.description

    @property
    def size(self) -> str:
        return self._df.memory_usage(deep=True)

    @property
    def shape(self) -> str:
        return self._df.shape

    @property
    def profile(self) -> DataProfiler:
        return self._profile

    # ------------------------------------------------------------------------------------------------------------------- #
    #                                           DATA ACCESS FUNCTIONS                                                     #
    # ------------------------------------------------------------------------------------------------------------------- #
    def head(self, n: int = 5) -> pd.DataFrame:
        """Prints and returns the top n rows from a dataset.

        Args:
            n (int): Number of observations to print/return
        """
        df = self._df.head(n)
        subtitle = "First {} Rows".format(str(n))
        self._printer.print_title(self._name, subtitle)
        self._printer.print_dataframe(df)

    def tail(self, n: int = 5) -> pd.DataFrame:
        """Prints and returns the last n rows from a dataset.

        Args:
            n (int): Number of observations to print/return
        """
        df = self._df.tail(n)
        subtitle = "Last {} Rows".format(str(n))
        self._printer.print_title(self._name, subtitle)
        self._printer.print_dataframe(df)

    def sample(self, n: int = 5, as_dict: bool = True, random_state: int = None) -> pd.DataFrame:
        """Prints and returns n randomly selected rows from a dataset.

        Args:
            n (int): Number of randomly selected observations to print/return
            as_dict (bool): Prints each sample as a dictionary
            random_stage (int): Seed for pseudo-random generator
        """
        df = self._df.sample(n=n, replace=False, random_state=random_state)
        subtitle = "{} Randomly Selected Samples".format(str(n))
        self._printer.print_title(self._name, subtitle)
        if as_dict is True:
            d = df.to_dict(orient="index")
            for index, data in d.items():
                subtitle = "Index = {}".format(index)
                self._printer.print_dictionary(data, subtitle)
                self._printer.print_blank_line()
        else:
            self._printer.print_dataframe(df)


# ======================================================================================================================== #
class Dataset(AbstractDataset):
    def __init__(self, config: DatasetConfig) -> None:
        super(Dataset, self).__init__(config)


# ======================================================================================================================== #
#                                                 DATASET BUILDERS                                                         #
# ======================================================================================================================== #
class AbstractDatasetBuilder(AssetBuilder):
    """Abstract base class for all Dataset builders."""

    def __init__(self) -> None:

        self._request = None
        self._dataset = None
        self._io = Pickler()  # Tracks versions of datasets
        self._builder_configfile = None

    def reset(self) -> None:
        self._request = None
        self._dataset = None
        self._profiler = None
        return self

    @property
    def dataset(self) -> Dataset:
        dataset = self._dataset
        self.reset()
        return dataset

    def build(self) -> None:
        config = self._build_config()
        self._dataset = Dataset(config)
        return self

    def _build_config(self) -> None:
        config = DatasetConfig(
            name=self._request.name,
            description=self._request.description,
            stage=self._request.stage,
            data=self._request.data,
            workspace_name=self._request.workspace_name,
            workspace_directory=self._request.workspace_directory,
        )
        config.set_config_filepath(classname="dataset")
        config.set_version()
        config.set_aid(classname="dataset")
        config.set_filepath(classname="dataset")
        config.set_profiler()
        return config


# ======================================================================================================================== #
class DatasetBuilder(AbstractDatasetBuilder):
    def __init__(self) -> None:
        super(DatasetBuilder, self).__init__()
