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
# Modified : Thursday, January 27th 2022, 8:09:31 am                                                                       #
# Modifier : John James (john.james.ai.studio@gmail.com)                                                                   #
# ------------------------------------------------------------------------------------------------------------------------ #
# License  : BSD 3-clause "New" or "Revised" License                                                                       #
# Copyright: (c) 2022 Bryant St. Labs                                                                                      #
# ======================================================================================================================== #
from datetime import datetime
import pandas as pd
import numpy as np
import logging
from scipy import stats

from cvr.data import categorical_columns

# ------------------------------------------------------------------------------------------------------------------------ #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------------------------------ #


class DataProfiler:
    """Data profiler for Dataset objects."""

    def __init__(self) -> None:
        self._data = None
        self._profile = {}
        self._column_profiles = {}

    def build(self, data: pd.DataFrame) -> None:
        self._data = data

    @property
    def summary(self) -> dict:
        if not "summary" in self._profile.keys():
            self._compute_summary()
        return self._profile["summary"]

    @property
    def numerics(self) -> pd.DataFrame:
        if not "numerics" in self._profile.keys():
            self._compute_numerics()
        return self._profile["numerics"]

    @property
    def categoricals(self) -> pd.DataFrame:
        if not "categoricals" in self._profile.keys():
            self._compute_categoricals()
        return self._profile["categoricals"]

    @property
    def missing(self) -> dict:
        if not "missing" in self._profile.keys():
            self._compute_missing()
        return self._profile["missing"]

    @property
    def missing_summary(self) -> dict:
        if not "missing_summary" in self._profile.keys():
            self._compute_missing_summary()
        return self._profile["missing_summary"]

    @property
    def cardinality(self) -> pd.DataFrame:
        if not "cardinality" in self._profile.keys():
            self._compute_cardinality()
        return self._profile["cardinality"]

    @property
    def metrics(self) -> dict:
        if not "metrics" in self._profile.keys():
            self._compute_metrics()
        return self._profile["metrics"]

    @property
    def datatypes(self) -> pd.DataFrame:
        if not "datatypes" in self._profile.keys():
            self._compute_datatypes()
        return self._profile["datatypes"]

    @property
    def frequencies(self) -> pd.DataFrame:
        if not "frequencies" in self._profile.keys():
            self._compute_frequencies()
        return self._profile["frequencies"]

    @property
    def frequency_stats(self) -> pd.DataFrame:
        if not "frequency_stats" in self._profile.keys():
            self._compute_frequency_stats()
        return self._profile["frequency_stats"]

    # --------------------------------------------------------------------------------------------------------------------- #
    def frequency_counts(self, column: str) -> pd.DataFrame:
        if not column in self._column_profiles.keys():
            self._compute_frequency_counts(column)
        return self._column_profiles[column]

    def _compute_summary(self) -> dict:
        d = {}
        d["Rows"] = len(self._data)
        d["Columns"] = self._data.shape[1]
        d["Missing Cells"] = self._data.isna().sum().sum()
        d["Missing Cells %"] = round(d["Missing Cells"] / (d["Rows"] * d["Columns"]) * 100, 2)
        d["Duplicate Rows"] = self._data.duplicated(keep="first").sum()
        d["Duplicate Rows %"] = round(d["Duplicate Rows"] / d["Rows"] * 100, 2)
        d["Size (Mb)"] = round(self._data.memory_usage(deep=True).sum() / (1024 * 1024), 2)

        self._profile["summary"] = d

    def _compute_numerics(self) -> pd.DataFrame:
        df = self._data.select_dtypes("number")
        d = df.describe().T
        d["skew"] = df.skew()
        d["kurtosis"] = df.kurtosis()
        d = d.round(2)
        self._profile["numerics"] = d

    def _compute_categoricals(self) -> pd.DataFrame:
        df = self._data.select_dtypes("category")
        df = df.describe().T
        df = df.astype({"count": int, "unique": int})
        df["cardinality"] = round(df["unique"] / df["count"] * 100, 2)
        self._profile["categoricals"] = df

    def _compute_missing_summary(self) -> dict:
        d = {}
        d["n"] = len(self._data)
        d["columns"] = self._data.shape[1]
        d["cells"] = d["n"] * d["columns"]
        d["Missing"] = self._data.isna().sum().sum()
        d["Missingness"] = round(d["Missing"] / d["cells"] * 100, 2)
        self._profile["missing_summary"] = d

    def _compute_missing(self) -> dict():
        d = {}
        d["n"] = len(self._data)
        d["Missing"] = self._data.isna().sum()
        d["Missingness"] = round(d["Missing"] / d["n"] * 100, 2)
        df = pd.DataFrame(d)

        self._profile["missing"] = df

    def _compute_cardinality(self) -> pd.DataFrame:
        s1 = self._data.nunique()
        s2 = self._data.count()
        s3 = round(s1 / s2 * 100, 2)
        d = {"unique": s1, "total": s2, "pct unique": s3}
        df = pd.DataFrame(d).reset_index()
        df.columns = ["Column", "Unique", "Total", "Pct Unique"]
        self._profile["cardinality"] = df

    def _compute_metrics(self) -> dict:
        d = {}
        d["clicks"] = self._data["click_ts"].count()
        d["conversions"] = self._data["sale"].value_counts()[1]
        d["conversion rate"] = round(d["conversions"] / d["clicks"] * 100, 2)
        d["average conversion delay"] = round(self._data["conversion_time_delay"].mean(), 2)
        self._profile["metrics"] = d

    def _compute_datatypes(self) -> pd.DataFrame:
        self._profile["datatypes"] = self._data.dtypes.astype(str).value_counts().to_frame().to_dict()[0]

    def _compute_frequencies(self) -> pd.DataFrame:
        freqs = self._data[categorical_columns].nunique().to_frame().reset_index()
        freqs.columns = ["Column", "# Categories"]
        self._profile["frequencies"] = freqs

    def _compute_frequency_stats(self) -> pd.DataFrame:
        self._profile["frequency_stats"] = self._data[categorical_columns].nunique().describe().to_frame().T

    def _compute_frequency_counts(self, column) -> None:
        counts = self._data[column].value_counts().to_frame().reset_index()
        counts.columns = ["Category", "Count"]
        counts["Cumulative"] = counts["Count"].cumsum()
        counts["Pct Cum"] = counts["Cumulative"] / len(self._data[column].dropna()) * 100
        counts["Rank"] = np.arange(1, len(counts) + 1)
        counts["Category Rank"] = counts["Rank"].astype("category")
        self._column_profiles[column] = counts
