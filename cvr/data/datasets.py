#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ======================================================================================================================== #
# Project  : Conversion Rate Prediction (CVR)                                                                              #
# Version  : 0.1.0                                                                                                         #
# File     : \datasets.py                                                                                                  #
# Language : Python 3.10.1                                                                                                 #
# ------------------------------------------------------------------------------------------------------------------------ #
# Author   : John James                                                                                                    #
# Email    : john.james.ai.studio@gmail.com                                                                                #
# URL      : https://github.com/john-james-ai/cvr                                                                          #
# ------------------------------------------------------------------------------------------------------------------------ #
# Created  : Thursday, January 13th 2022, 2:22:59 am                                                                       #
# Modified : Tuesday, January 18th 2022, 11:57:38 pm                                                                       #
# Modifier : John James (john.james.ai.studio@gmail.com)                                                                   #
# ------------------------------------------------------------------------------------------------------------------------ #
# License  : BSD 3-clause "New" or "Revised" License                                                                       #
# Copyright: (c) 2021 Bryant St. Labs                                                                                      #
# ======================================================================================================================== #
"""Dataset Module

Three categories of classes are defined here:
    1. Dataset: Object encapsulating a dataset.
    2. Dataset Persistence: Classes responsible for managing dataset persistence
    3. Dataset Builders The construction of Dataset objects

Datasets encapsulate the data and include the following classes:
    1. Dataset: Encapsulates the data and the following two classes
    2. DatasetProfile: Profiles the dataset
    3. DatasetMetadata: Contains all administrative, descriptive, and structural metadata.

Dataset persistence classes are:
    1. DataStore: Abstracts the collection of data files and provides basic inventory management.
    2. DAO: Data access object encapsulating file IO.

Dataset Builders construct dataset objects. The builder classes include:
    1. DatasetBuilder: Defines the interface for the following concrete builders.
    2. DatasetBuilderFile: Build a Dataset object from a file.
    3. DatasetBuilderDF: Builds a Dataset object from a DataFrame.



"""
from abc import ABC, abstractmethod
from datetime import datetime
import pandas as pd
import logging
import platform
import uuid

from cvr.data.transform import Transformer
from cvr.data.profile import DataProfiler
from cvr.utils.config import WorkspaceConfig
from cvr.visuals.visualize import Visual
from cvr.utils.printing import Printer
from cvr.data import raw_dtypes

# ------------------------------------------------------------------------------------------------------------------------ #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------------------------------------------------ #


class Dataset:
    def __init__(
        self,
        id: str,
        data: pd.DataFrame,
        metadata: dict,
        profile: DataProfiler,
    ) -> None:
        self._id = id
        self._df = data
        self._metadata = metadata
        self._profile = profile
        self._printer = Printer()
        self._visual = Visual()
        self._transform = Transformer(self._visual)
        self._verbose = False
        self._filepath = None

    # ------------------------------------------------------------------------------------------------------------------- #
    #                                               PROPERTIES                                                            #
    # ------------------------------------------------------------------------------------------------------------------- #
    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._metadata["name"]

    @property
    def workspace(self) -> str:
        return self._metadata["workspace_name"]

    @property
    def stage(self) -> str:
        return self._metadata["stage"]

    @property
    def creator(self) -> str:
        return self._metadata["creator"]

    @property
    def created(self) -> str:
        return self._metadata["created"]

    @property
    def description(self) -> str:
        return "Dataset: {}\tWorkspace: {}\tStage: {}".format(self.name, self.workspace, self.stage)

    @property
    def filepath(self) -> bool:
        return self._filepath

    @filepath.setter
    def filepath(self, filepath) -> bool:
        self._filepath = filepath

    @property
    def verbose(self) -> bool:
        return self._verbose

    @verbose.setter
    def verbose(self, verbose) -> bool:
        self._verbose = verbose

    # ------------------------------------------------------------------------------------------------------------------- #
    #                                          METADATA PROPERTIES                                                        #
    # ------------------------------------------------------------------------------------------------------------------- #
    @property
    def info(self) -> dict:
        """Prints column i.e. structural metadata."""
        self._printer.print_title("Dataset {}".format(self.name))
        self._df.info()

    @property
    def metadata(self) -> dict:
        """Prints metadata."""
        subtitle = "Metadata"
        self._printer.print_title(self.description, subtitle)
        self._printer.print_dictionary(self._metadata)

    # ------------------------------------------------------------------------------------------------------------------- #
    #                                  PROFILE PROPERTIES AND METHODS                                                     #
    # ------------------------------------------------------------------------------------------------------------------- #
    @property
    def summary(self) -> None:
        """Prints summary statistics."""
        subtitle = "Summary"
        self._printer.print_title(self.description, subtitle)
        self._printer.print_dictionary(self._profile.summary)
        return self._profile.summary

    @property
    def data_types(self) -> None:
        """Prints descriptive statistics for all numeric columns"""
        subtitle = "Data Type Analysis"
        self._printer.print_title(self.description, subtitle)
        if self._verbose:
            self._printer.print_dataframe(self._profile.datatypes)
        else:
            return self._profile.datatypes

    @property
    def numeric_statistics(self) -> None:
        """Prints descriptive statistics for all numeric columns"""
        subtitle = "Descriptive Statistics (Quantitative Variables)"
        self._printer.print_title(self.description, subtitle)
        if self._verbose:
            self._printer.print_dataframe(self._profile.numerics)
        return self._profile.numerics

    @property
    def categorical_statistics(self) -> None:
        """Prints descriptive statistics for all categorical columns"""
        subtitle = "Descriptive Statistics (Qualitative Variables)"
        self._printer.print_title(self.description, subtitle)
        if self._verbose:
            self._printer.print_dataframe(self._profile.categoricals)
        return self._profile.categoricals

    @property
    def missing_summary(self) -> dict:
        """Prints missing data statistics"""
        subtitle = "Missing Data Analysis"
        self._printer.print_title(self.description, subtitle)
        if self._verbose:
            self._printer.print_dictionary(self._profile.missing_summary)
        return self._profile.missing_summary

    @property
    def missing(self) -> dict:
        """Prints missing data statistics"""
        subtitle = "Missing Data Analysis"
        self._printer.print_title(self.description, subtitle)
        if self._verbose:
            self._printer.print_dataframe(self._profile.missing)
        return self._profile.missing

    @property
    def cardinality(self) -> None:
        """Prints / returns cardinality of the dataset."""
        subtitle = "Cardinality"
        self._printer.print_title(self.description, subtitle)
        if self._verbose:
            self._printer.print_dataframe(self._profile.cardinality)
        return self._profile.cardinality

    @property
    def metrics(self) -> None:
        """Prints conversion metrics."""
        subtitle = "Metrics"
        self._printer.print_title(self.description, subtitle)
        if self._verbose:
            self._printer.print_dictionary(self._profile.metrics)
        return self._profile.metrics

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
        self._printer.print_title(self._metadata.dataset_name, subtitle)
        self._printer.print_dataframe(df)

    def tail(self, n: int = 5) -> pd.DataFrame:
        """Prints and returns the last n rows from a dataset.

        Args:
            n (int): Number of observations to print/return
        """
        df = self._df.tail(n)
        subtitle = "Last {} Rows".format(str(n))
        self._printer.print_title(self._metadata.dataset_name, subtitle)
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
        self._printer.print_title(self._metadata.dataset_name, subtitle)
        if as_dict is True:
            d = df.to_dict(orient="index")
            for index, data in d.items():
                subtitle = "Index = {}".format(index)
                self._printer.print_dictionary(data, subtitle)
                self._printer.print_blank_line()
        else:
            self._printer.print_dataframe(df)

    # ------------------------------------------------------------------------------------------------------------------- #
    #                                            COLUMN ANALYSIS                                                          #
    # ------------------------------------------------------------------------------------------------------------------- #
    def numeric_analysis(self, column: str) -> None:
        """Provides descriptive statistics, distribution plots and transformations."""
        subtitle = "Distribution of {}".format(column)
        stats = self._df[column].describe().to_frame().T
        stats["skew"] = self._df[column].skew()
        stats["kurtosis"] = self._df[column].kurtosis()
        self._printer.print_title(self.description, subtitle)
        if self._verbose:
            self._printer.print_dataframe(stats)

        self._transform.explore(self._df, columns=[column])
        return stats

    def categorical_analysis(self, column: str) -> None:
        """Provides descriptive statistics and frequency."""
        subtitle = "Frequency Analysis of {}".format(column)
        stats = self._df[column].describe().to_frame().T
        self._printer.print_title(self.description, subtitle)
        if self._verbose:
            self._printer.print_dataframe(stats)

        self._visual.countplot(self._df, column=column, threshold=20, title=subtitle)
        return stats


# ======================================================================================================================== #
#                                                 DATASET BUILDER                                                          #
# ======================================================================================================================== #
class AbstractDatasetBuilder(ABC):
    """Abstract base class for all Dataset builders."""

    def __init__(self, workspace_name) -> None:
        self._workspace_name = workspace_name

        self._id = None
        self._data = None
        self._name = None
        self._stage = None
        self._creator = None
        self._random_state = None

        self._config = WorkspaceConfig()

        self._dataset = None

    def set_data(self, data: pd.DataFrame) -> None:
        self._data = data
        return self

    def set_name(self, name: str) -> None:
        self._name = name
        return self

    def set_stage(self, stage: str) -> None:
        self._stage = stage
        return self

    def set_creator(self, creator: str) -> None:
        self._creator = creator
        return self

    @property
    def dataset(self) -> Dataset:
        return self._dataset

    def _validate(self) -> None:
        assert self._name is not None, "self_name is None"
        assert self._workspace_name is not None, "self_workspace_name is None"
        assert self._stage is not None, "self_stage is None"
        config = self._config.get_config()
        assert self._stage in config["stages"], "{} is an invalid stage".format(self._stage)

    def _build_id(self) -> None:
        self._id = self._get_id(self._stage, self._name)

    def _build_profile(self) -> None:
        self._profile = DataProfiler(self._data)
        self._profile.build()
        return self

    def _build_metadata(self) -> None:
        self._metadata = {}
        self._metadata["id"] = self._id
        self._metadata["name"] = self._name
        self._metadata["workspace_name"] = self._workspace_name
        self._metadata["stage"] = self._stage
        self._metadata["size"] = self._data.memory_usage(deep=True).sum()
        self._metadata["creator"] = self._creator
        self._metadata["created"] = datetime.now()
        return self

    def build(self) -> None:
        self._validate()
        self._build_id()
        self._build_profile()
        self._build_metadata()
        self._dataset = Dataset(
            id=self._id,
            data=self._data,
            metadata=self._metadata,
            profile=self._profile,
        )
        return self._dataset

    def _get_id(self, stage: str, name: str) -> str:
        config = self._config.get_config()
        try:
            seq = config["workspaces"][self._workspace_name]["seq"][stage]
            config["workspaces"][self._workspace_name]["seq"][stage] += 1
            self._config.save_config(config)
            seq = str(seq).zfill(config["id_digits"])
            key = [seq, self._workspace_name, stage, name]
            key = "_".join(key)
            return key.lower()
        except KeyError as e:
            logger.error(e)


# ------------------------------------------------------------------------------------------------------------------------ #
class DatasetBuilder(AbstractDatasetBuilder):
    def __init__(self, workspace_name: str) -> None:
        super(DatasetBuilder, self).__init__(workspace_name=workspace_name)
