#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ======================================================================================================================== #
# Project  : Conversion Rate Prediction (CVR)                                                                              #
# Version  : 0.1.0                                                                                                         #
# File     : \profile.py                                                                                                   #
# Language : Python 3.10.1                                                                                                 #
# ------------------------------------------------------------------------------------------------------------------------ #
# Author   : John James                                                                                                    #
# Email    : john.james.ai.studio@gmail.com                                                                                #
# URL      : https://github.com/john-james-ai/cvr                                                                          #
# ------------------------------------------------------------------------------------------------------------------------ #
# Created  : Sunday, January 16th 2022, 1:33:06 pm                                                                         #
# Modified : Sunday, January 30th 2022, 5:54:29 pm                                                                         #
# Modifier : John James (john.james.ai.studio@gmail.com)                                                                   #
# ------------------------------------------------------------------------------------------------------------------------ #
# License  : BSD 3-clause "New" or "Revised" License                                                                       #
# Copyright: (c) 2022 Bryant St. Labs                                                                                      #
# ======================================================================================================================== #
import pandas as pd
from pandas.api.types import is_categorical_dtype
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import numpy as np
from typing import Union

# plotting modules
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("whitegrid")
sns.set_palette("mako")

from cvr.data import categorical_columns, numeric_columns, criteo_columns
from cvr.utils.printing import Printer
from cvr.utils.format import titlelize, titlelize_df
from cvr.visualize.features import CategoricalFeatureVisualizer, NumericFeatureVisualizer

# ------------------------------------------------------------------------------------------------------------------------ #


class DataProfiler:
    """Data profiler for Dataset objects."""

    def __init__(self, name: str, description: str, stage: str, data: pd.DataFrame) -> None:
        self._name = name
        self._description = titlelize(description)
        self._stage = stage
        self._data = data
        self._profile = {}
        self._column_profiles = {}

        self._printer = Printer()
        self._categorical_visualizer = CategoricalFeatureVisualizer()
        self._numeric_visualizer = NumericFeatureVisualizer()

    @property
    def summary(self) -> None:
        title = "Dataset Summary"
        if not "summary" in self._profile.keys():
            self._profile["summary"] = self._compute_summary()
        self._printer.print_title(title, self._description)
        self._printer.print_dictionary(content=self._profile["summary"])

    @property
    def datatypes(self) -> None:
        if not "datatypes" in self._profile.keys():
            self._profile["datatypes"] = self._compute_datatypes()
        self._printer.print_dictionary(content=self._profile["datatypes"], title="Data Types")

    @property
    def missing(self) -> None:
        title = "Missing Values Analysis"
        if not "missing_summary" in self._profile.keys():
            self._profile["missing_summary"] = self._compute_missing_summary()
        if not "missing" in self._profile.keys():
            self._profile["missing"] = self._compute_missing()

        self._printer.print_title(title, self._description)
        self._printer.print_dictionary(self._profile["missing_summary"], "Summary")
        self._printer.print_blank_line()
        self._printer.print_dataframe(self._profile["missing"], "Missing Analysis by Column")

    @property
    def cardinality(self) -> None:
        title = "Cardinality Analysis"
        if not "cardinality" in self._profile.keys():
            self._profile["cardinality"] = self._compute_cardinality()
        self._printer.print_title(title, self._description)
        self._printer.print_dataframe(self._profile["cardinality"])

    @property
    def frequency_stats(self) -> pd.DataFrame:
        title = "Categorical Frequency Statistics"
        if not "frequency_stats" in self._profile.keys():
            self._profile["frequency_stats"] = self._compute_frequency_stats()
        self._printer.print_title(title, self._description)
        self._printer.print_dataframe(self._profile["frequency_stats"])

    # --------------------------------------------------------------------------------------------------------------------- #
    def analyze(self, column: str) -> pd.DataFrame:
        if is_numeric_dtype(self._data[column]):
            self.analyze_numeric(column)
        else:
            self.analyze_categorical(column)

    def analyze_numeric(self, column: str) -> None:
        if column not in self._column_profiles.keys():
            self._column_profiles[column] = {"descriptive_statistics": self._analyze_categorical_stats(column)}
        self._printer.print_title("Descriptive Statistics", column)
        self._printer.print_dataframe(self._column_profiles[column]["descriptive_statistics"])

        self._numeric_visualizer.fit(self._data[column])
        self._numeric_visualizer.plot()

    def _analyze_numeric_stats(self, column: str) -> None:
        df = self._data[column].describe().to_frame().T
        df["skew"] = self._data[column].skew()
        df["kurtosis"] = self._data[column].kurtosis()
        df["missing"] = self._data[column].isna().sum()
        df["missingness"] = round(df["missing"] / len(self._data[column]) * 100, 2)
        return df

    def analyze_categorical(self, column: str) -> None:
        if column not in self._column_profiles.keys():
            stats = self._analyze_categorical_stats(column)
            counts = self._analyze_categorical_counts(column)
            self._column_profiles[column] = {"descriptive_statistics": stats, "categorical_counts": counts}

        title = "Categorical Variable Analysis"
        self._printer.print_title(title, titlelize(column))
        self._printer.print_dataframe(self._column_profiles[column]["descriptive_statistics"])

        self._categorical_visualizer.fit(self._data[column])
        self._categorical_visualizer.plot()

    def _analyze_categorical_stats(self, column: str) -> None:
        df = self._data[column].describe().to_frame().T
        df["missing"] = self._data[column].isna().sum()
        df["missingness"] = round(df["missing"] / len(self._data[column]) * 100, 2)
        return df

    def _analyze_categorical_counts(self, column) -> pd.DataFrame:
        counts = self._data[column].value_counts().to_frame().reset_index()
        counts.columns = ["Category", "Count"]
        counts["Cumulative"] = counts["Count"].cumsum()
        counts["Percent of Data"] = counts["Cumulative"] / len(self._data[column].dropna()) * 100
        counts["Rank"] = np.arange(1, len(counts) + 1)
        counts["Category Rank"] = counts["Rank"].astype("category")
        return counts

    # --------------------------------------------------------------------------------------------------------------------- #
    def _compute_summary(self) -> dict:
        d = {}
        d["Rows"] = len(self._data)
        d["Columns"] = self._data.shape[1]
        d["Missing Cells"] = self._data.isna().sum().sum()
        d["Missing Cells %"] = round(d["Missing Cells"] / (d["Rows"] * d["Columns"]) * 100, 2)
        d["Duplicate Rows"] = self._data.duplicated(keep="first").sum()
        d["Duplicate Rows %"] = round(d["Duplicate Rows"] / d["Rows"] * 100, 2)
        d["Size (Mb)"] = round(self._data.memory_usage(deep=True).sum() / (1024 * 1024), 2)
        return d

    def _compute_numerics(self) -> pd.DataFrame:
        data = self._data.select_dtypes("number")
        df = data.describe().T
        df["skew"] = data.skew()
        df["kurtosis"] = data.kurtosis()
        df = df.round(2)
        return df

    def _compute_categoricals(self) -> pd.DataFrame:
        df = self._data.select_dtypes("category")
        df = df.describe().T
        df = df.astype({"count": int, "unique": int})
        df["cardinality"] = round(df["unique"] / df["count"] * 100, 2)
        return df

    def _compute_missing_summary(self) -> dict:
        d = {}
        d["n"] = len(self._data)
        d["columns"] = self._data.shape[1]
        d["cells"] = d["n"] * d["columns"]
        d["Missing"] = self._data.isna().sum().sum()
        d["Missingness"] = round(d["Missing"] / d["cells"] * 100, 2)
        return d

    def _compute_missing(self) -> pd.DataFrame:
        d = {}
        d["n"] = len(self._data)
        d["Missing"] = self._data.isna().sum()
        d["Missingness"] = round(d["Missing"] / d["n"] * 100, 2)
        df = pd.DataFrame(d)
        return df

    def _compute_cardinality(self) -> pd.DataFrame:
        s1 = self._data.nunique()
        s2 = self._data.count()
        s3 = round(s1 / s2 * 100, 2)
        d = {"unique": s1, "total": s2, "pct unique": s3}
        df = pd.DataFrame(d).reset_index()
        df.columns = ["Column", "Unique", "Total", "Pct Unique"]
        return df

    def _compute_datatypes(self) -> dict:
        return self._data.dtypes.astype(str).value_counts().to_frame().to_dict()[0]

    def _compute_frequency_stats(self) -> pd.DataFrame:
        return self._data[categorical_columns].nunique().describe().to_frame().T
