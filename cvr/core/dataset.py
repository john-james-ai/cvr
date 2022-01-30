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
# Modified : Sunday, January 30th 2022, 5:35:22 pm                                                                         #
# Modifier : John James (john.james.ai.studio@gmail.com)                                                                   #
# ------------------------------------------------------------------------------------------------------------------------ #
# License  : BSD 3-clause "New" or "Revised" License                                                                       #
# Copyright: (c) 2022 Bryant St. Labs                                                                                      #
# ======================================================================================================================== #
"""Dataset Module"""
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections import OrderedDict
from datetime import datetime
import pandas as pd
import logging

from cvr.core.asset import Asset
from cvr.core.workspace import Workspace
from cvr.core.profile import DataProfiler
from cvr.utils.printing import Printer
from cvr.utils.format import titlelize
from cvr.utils.io import Pickler

# ------------------------------------------------------------------------------------------------------------------------ #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------------------------------------------------ #
@dataclass(frozen=True)
class DatasetRequest:
    """Class used to define Dataset order for the Dataset building class."""

    name: str
    description: str
    stage: str
    workspace: Workspace
    sample_size: int = None
    data: pd.DataFrame = pd.DataFrame()


# ------------------------------------------------------------------------------------------------------------------------ #
class AbstractDataset(Asset):
    def __init__(
        self,
        aid: str,
        name: str,
        description: str,
        stage: str,
        version: int,
        filepath: str,
        df: pd.DataFrame,
        profile: DataProfiler,
    ) -> None:
        super(AbstractDataset, self).__init__(aid, name, stage, version, filepath)

        self._description = description if description is not None else name + " " + stage

        self._df = df
        self._profile = profile

        self._printer = Printer()

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
        return self._name

    @property
    def description(self) -> str:
        return self._description

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
    def __init__(
        self,
        aid: str,
        name: str,
        description: str,
        stage: str,
        version: int,
        filepath: str,
        df: pd.DataFrame,
        profile: DataProfiler,
    ) -> None:
        super(Dataset, self).__init__(aid, name, description, stage, version, filepath, df, profile)


# ======================================================================================================================== #
#                                                 DATASET BUILDERS                                                         #
# ======================================================================================================================== #
class AbstractDatasetBuilder(ABC):
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

    def make_request(self, request: DatasetRequest) -> None:
        self._request = request
        return self

    @property
    def dataset(self) -> Dataset:
        return self._dataset

    def build(self) -> None:
        version = self._get_version(name=self._request.name, stage=self._request.stage)
        aid = self._get_aid(
            workspace=self._request.workspace,
            classname="dataset",
            name=self._request.name,
            stage=self._request.stage,
            version=version,
        )
        filename = aid + ".pkl"
        filepath = os.path.join(self._request.workspace.directory, "dataset", self._request.stage, filename)

        profiler = DataProfiler(
            name=self._request.name,
            description=self._request.description,
            stage=self._request.stage,
            data=self._request.data,
        )
        self._dataset = Dataset(
            aid=aid,
            name=self._request.name,
            description=self._request.description,
            stage=self._request.stage,
            version=version,
            filepath=filepath,
            df=self._request.data,
            profile=profiler,
        )
        return self._dataset

    def _get_aid(self, workspace: Workspace, classname: str, name: str, stage: str, version: int) -> str:
        """Returns the filepath for the asset based upon the workspace, stage, name and version"""
        return (
            workspace.name.lower()
            + "_"
            + classname.lower()
            + "_"
            + stage.lower()
            + "_"
            + name.lower()
            + "_"
            + "v_"
            + str(version).zfill(3)
        )

    def _get_version(self, name: str, stage: str) -> int:
        self._builder_configfile = os.path.join(self._request.workspace.directory, self.__class__.__name__.lower() + ".pkl")
        config = self._io.load(self._builder_configfile)
        if config is None:
            version = 0
            config = {}
            config[name] = {stage: 0}
        elif name in config.keys():
            if stage in config[name].keys():
                config[name][stage] += 1
                version = config[name][stage]
            else:
                config[name][stage] = 0
                version = config[name][stage]
        self._io.save(config, self._builder_configfile)
        return version


# ======================================================================================================================== #
class DatasetBuilder(AbstractDatasetBuilder):
    def __init__(self) -> None:
        super(DatasetBuilder, self).__init__()
