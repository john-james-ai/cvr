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
# Modified : Saturday, January 22nd 2022, 6:14:14 am                                                                       #
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

# ------------------------------------------------------------------------------------------------------------------------ #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------------------------------ #


class DataProfiler:
    """Data profiler for Dataset objects."""

    def __init__(self) -> None:
        self._data = None
        self._profile = {}

    def build(self, data: pd.Dataframe) -> None:
        self._data = data
        self._compute_summary()
        self._compute_numerics()
        self._compute_categoricals()
        self._compute_missing()
        self._compute_missing_summary()
        self._compute_cardinality()
        self._compute_metrics()
        self._compute_datatypes(),

    @property
    def summary(self) -> None:
        return self._profile["summary"]

    @property
    def numerics(self) -> None:
        return self._profile["numerics"]

    @property
    def categoricals(self) -> None:
        return self._profile["categoricals"]

    @property
    def missing(self) -> None:
        return self._profile["missing"]

    @property
    def missing_summary(self) -> None:
        return self._profile["missing_summary"]

    @property
    def cardinality(self) -> None:
        return self._profile["cardinality"]

    @property
    def metrics(self) -> None:
        return self._profile["metrics"]

    @property
    def datatypes(self) -> None:
        return self._profile["datatypes"]

    def _compute_summary(self) -> dict:
        d = {}
        d["Rows"] = len(self._data)
        d["Columns"] = self._data.shape[1]
        d["Missing Cells"] = self._data.isna().sum().sum()
        d["Missing Cells %"] = round(d["Missing Cells"] / (d["Rows"] * d["Columns"]) * 100, 2)
        d["Duplicate Rows"] = self._data.duplicated(keep="first").sum()
        d["Duplicate Rows %"] = round(d["Duplicate Rows"] / d["Rows"] * 100, 2)
        d["Size (Mb)"] = self._data.memory_usage(deep=True).sum()

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

    def _compute_missing_summary(self) -> dict():
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
        d["Rows w/ Missing"] = self._data.isna().any(axis=1).sum()
        d["Columns w/ Missing"] = self._data.isna().any(axis=0).sum()
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

    def _compute_datatypes(self) -> pd.DataFrame():
        self._profile["datatypes"] = self._data.dtypes.astype(str).value_counts().to_frame()
        self._profile["datatypes"].columns = ["Count"]
