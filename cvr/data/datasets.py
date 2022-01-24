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
# Modified : Monday, January 24th 2022, 1:24:34 am                                                                         #
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
from collections import OrderedDict
from datetime import datetime
import pandas as pd
import logging
import platform
import uuid

from cvr.core.asset import Asset

from cvr.data.transform import Transformation
from cvr.data.profile import DataProfiler
from cvr.utils.config import WorkspaceConfig
from cvr.visuals.visualize import Visual
from cvr.utils.printing import Printer

# ------------------------------------------------------------------------------------------------------------------------ #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------------------------------------------------ #


class AbstractDataset(Asset):
    def __init__(self, name: str, description: str, stage: str, df: pd.DataFrame, profile: DataProfiler) -> None:
        super(AbstractDataset, self).__init__(name, stage)
        self._name = name
        self._description = description if description is not None else stage + "_" + name
        self._stage = stage
        self._df = df
        self._profile = profile

        self._workspace = WorkspaceConfig().get_workspace()
        self._printer = Printer()
        self._visual = Visual()
        self._transformation = Transformation(self._visual)

        self._set_id()

    def _set_id(self) -> None:
        self._id = self._workspace + "_" + self._stage + "_" + self._name + "_"

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
        self._df.info()

    # ------------------------------------------------------------------------------------------------------------------- #
    #                                               PROPERTIES                                                            #
    # ------------------------------------------------------------------------------------------------------------------- #
    @property
    def name(self) -> str:
        return self._name

    @property
    def workspace(self) -> str:
        return self._workspace

    @property
    def description(self) -> str:
        return self._description

    # ------------------------------------------------------------------------------------------------------------------- #
    #                                  PROFILE PROPERTIES AND METHODS                                                     #
    # ------------------------------------------------------------------------------------------------------------------- #
    @property
    def summary(self) -> None:
        """Prints summary statistics."""
        subtitle = "Dataset Summary"
        self._printer.print_title(subtitle, self.description)
        self._printer.print_dictionary(self._profile.summary)
        return self._profile.summary

    @property
    def datatypes(self) -> None:
        """Prints descriptive statistics for all numeric columns"""
        subtitle = "Data Type Analysis"
        self._printer.print_title(subtitle, self.description)
        self._printer.print_dictionary(self._profile.datatypes)
        return self._profile.datatypes

    @property
    def numerics(self) -> None:
        """Prints descriptive statistics for all numeric columns"""
        subtitle = "Descriptive Statistics (Quantitative Variables)"
        self._printer.print_title(subtitle, self.description)
        self._printer.print_dataframe(self._profile.numerics)
        return self._profile.numerics

    @property
    def categoricals(self) -> None:
        """Prints descriptive statistics for all categorical columns"""
        subtitle = "Descriptive Statistics (Qualitative Variables)"
        self._printer.print_title(subtitle, self.description)
        self._printer.print_dataframe(self._profile.categoricals)
        return self._profile.categoricals

    @property
    def missing_summary(self) -> dict:
        """Prints missing data statistics"""
        subtitle = "Missing Data Analysis"
        self._printer.print_title(subtitle, self.description)
        self._printer.print_dictionary(self._profile.missing_summary)
        return self._profile.missing_summary

    @property
    def missing(self) -> dict:
        """Prints missing data statistics"""
        subtitle = "Missing Data Analysis"
        self._printer.print_title(subtitle, self.description)
        self._printer.print_dataframe(self._profile.missing)
        return self._profile.missing

    @property
    def cardinality(self) -> None:
        """Prints / returns cardinality of the dataset."""
        subtitle = "Cardinality"
        self._printer.print_title(subtitle, self.description)
        self._printer.print_dataframe(self._profile.cardinality)
        return self._profile.cardinality

    @property
    def metrics(self) -> None:
        """Prints conversion metrics."""
        subtitle = "Metrics"
        self._printer.print_title(subtitle, self.description)
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
        self._printer.print_dataframe(stats)

        self._transformation.explore(self._df, columns=[column])
        return stats

    def categorical_analysis(self, column: str) -> None:
        """Provides descriptive statistics and frequency."""
        subtitle = "Frequency Analysis of {}".format(column)
        stats = self._df[column].describe().to_frame().T
        self._printer.print_title(self.description, subtitle)
        self._printer.print_dataframe(stats)

        self._visual.countplot(self._df, column=column, threshold=20, title=subtitle)
        return stats


# ======================================================================================================================== #
class Dataset(AbstractDataset):
    def __init__(self, name: str, description: str, stage: str, df: pd.DataFrame, profile: DataProfiler) -> None:
        super(Dataset, self).__init__(name, description, stage, df, profile)


# ======================================================================================================================== #
#                                                 DATASET BUILDERS                                                         #
# ======================================================================================================================== #
class AbstractDatasetBuilder(ABC):
    """Abstract base class for all Dataset builders."""

    def __init__(self) -> None:

        self._dataset_name = None
        self._stage = None
        self._description = None
        self._df = None

        self._dataset = None

        self._workspace = WorkspaceConfig().get_workspace()
        self._profiler = DataProfiler()

    def set_data(self, df: pd.DataFrame) -> None:
        self._df = df
        return self

    def set_name(self, dataset_name: str) -> None:
        self._dataset_name = dataset_name
        return self

    def set_description(self, description: str) -> None:
        self._description = description
        return self

    def set_stage(self, stage: str) -> None:
        self._stage = stage
        return self

    @property
    def dataset(self) -> Dataset:
        return self._dataset

    def create(self) -> None:
        self._dataset = None

    def _build_profile(self) -> None:
        self._profile = DataProfiler()
        self._profile.build(self._df)

    def build(self) -> None:
        self._build_profile()
        self._dataset = Dataset(
            name=self._dataset_name,
            description=self._description,
            stage=self._stage,
            df=self._df,
            profile=self._profile,
        )
        return self._dataset


# ======================================================================================================================== #
class DatasetBuilder(AbstractDatasetBuilder):
    def __init__(self) -> None:
        super(DatasetBuilder, self).__init__()
