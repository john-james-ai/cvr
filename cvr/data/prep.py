#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ======================================================================================================================== #
# Project  : Explainable Recommendation (XRec)                                                                             #
# Version  : 0.1.0                                                                                                         #
# File     : \criteo.py                                                                                                    #
# Language : Python 3.8                                                                                                    #
# ------------------------------------------------------------------------------------------------------------------------ #
# Author   : John James                                                                                                    #
# Email    : john.james.ai.studio@gmail.com                                                                                #
# URL      : https://github.com/john-james-ai/xrec                                                                         #
# ------------------------------------------------------------------------------------------------------------------------ #
# Created  : Sunday, December 26th 2021, 3:56:00 pm                                                                        #
# Modified : Thursday, January 13th 2022, 12:21:13 am                                                                      #
# Modifier : John James (john.james.ai.studio@gmail.com)                                                                   #
# ------------------------------------------------------------------------------------------------------------------------ #
# License  : BSD 3-clause "New" or "Revised" License                                                                       #
# Copyright: (c) 2021 Bryant St. Labs                                                                                      #
# ======================================================================================================================== #
import os
import math
import random
import numpy as np
import pandas as pd
from datetime import datetime
import logging

from cvr.utils.file import sample
from cvr.utils.printing import Printer
from cvr.data.datastore import DataStore
from cvr.visuals.visualize import Visual
from cvr.data.transform import Transformer
from cvr.data.outliers import OutlierDetector
from cvr.data import feature_columns, target_columns, numeric_columns

# ------------------------------------------------------------------------------------------------------------------------ #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------------------------------ #
class DataPrep:
    """Data preparation for Criteo sponsored search conversion log data."""

    def __init__(
        self,
        datastore_in: DataStore,
        datastore_out: DataStore,
        visual: Visual,
        transformer: Transformer,
        outlier_detector: OutlierDetector,
        version: str = "prod",
        sample=0.01,
        random_state=None,
    ) -> None:

        self._printer = Printer()
        self._datastore_in = datastore_in
        self._datastore_out = datastore_out
        self._visual = visual
        self._transformer = transformer
        self._outlier_detector = outlier_detector
        self._version = version
        self._sample = sample
        self._random_state = random_state

        self._dataset_name = "Criteo Sponsored Search Conversion Log Dataset"

        self._df = self._initialize()

    def _initialize(self) -> None:
        """Loads input data into the object."""
        return self._datastore_in.read()

    def _read(self) -> pd.DataFrame:
        """Reads data from file."""
        return self._datastore_out.read()

    def _save(self, df: pd.DataFrame) -> None:
        """Saves the data to file.

        Args:
            df: (pd.DataFrame) The data
        """
        self._datastore_out.write(df)
        self._df = df

    def normalize(self) -> None:
        """Normalizes all numeric variables using Robust Z-Score Normalization."""

        for col in numeric_columns:
            center = self._df[col] - self._df[col].median()
            iq = self._df[col].quantile(q=[0.25, 0.75])
            iqr = iq.values[1] - iq.values[0]
            if iqr > 0:
                self._df[col] = center.values / iqr
            else:  # If you have zero IQR using STD due to sample
                self._df[col] = center.values / np.std(self._df[col])
        self._save(self._df)

    def describe(self, columns: list) -> None:
        """Descriptive Statistics for one or more columns of the data.

        Args:
            columns (list): one or more columns to describe.
        """
        x = self._df[columns].describe().T
        subtitle = "Descriptive Statistics"
        self._printer.print_title(self._dataset_name, subtitle)
        self._printer.print_dataframe(x)

    def explore_transformations(self, columns: list) -> None:
        """Explores various transformations on one or more columns

        Args:
            columns (list): one or more column names
        """
        self._transformer.explore(self._df, columns)

    def apply_transformation(self, columns: list, transformation) -> None:
        self._df = self._transformer.apply(self._df, transformation, columns)
        self._save(self._df)

    def hist(self, columns: list = numeric_columns) -> None:
        """Renders one ore more histograms for the designated columns

        Args:
            columns (list): One or more column names
        """
        for col in columns:
            title = "Numeric Variable Analysis\n{}".format(col)
            self._visual.hist(self._df, column=col, title=title)
