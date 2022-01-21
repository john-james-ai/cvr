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
# Modified : Friday, January 14th 2022, 6:45:52 pm                                                                         #
# Modifier : John James (john.james.ai.studio@gmail.com)                                                                   #
# ------------------------------------------------------------------------------------------------------------------------ #
# License  : BSD 3-clause "New" or "Revised" License                                                                       #
# Copyright: (c) 2021 Bryant St. Labs                                                                                      #
# ======================================================================================================================== #
import os
import random
import pandas as pd
import numpy as np
import logging
import pickle
from datetime import datetime


from cvr.utils.printing import Printer
from cvr.data.datastore import DataStore
from cvr.visuals.visualize import Visual
from cvr.utils.format import titlelize_df, s_to_dict
from cvr.data import (
    criteo_columns,
    feature_columns,
    target_columns,
    numeric_columns,
    categorical_columns,
)
from cvr.utils.config import DataConfig
from cvr.data import (
    numeric_descriptive_stats,
    categorical_descriptive_stats,
    numeric_columns,
    categorical_columns,
)
from cvr.data.outliers import OutlierDetector


# ------------------------------------------------------------------------------------------------------------------------ #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------------------------------ #


class DataProfiler:
    """Profiles the Criteo Data."""

    def __init__(
        self,
        datastore: DataStore,
        visual: Visual,
        stage: str = "staged",
        version: str = "prod",
    ) -> None:
        self._stage = stage
        self._version = version
        self._datastore = datastore
        self._visual = visual
        self._outlier_detector = outlier_detector
        self._printer = Printer()
        self._metadata = CriteoMD(stage=stage, version=version)
        self._dataset_name = "Criteo Sponsored Search Conversion Log Dataset"
        self._df = None
        self._stats = None
        self._stats_numeric = None
        self._stats_categorical = None
        self._stats_summary = None
        self._outlier_labels = None

    def load(self) -> None:
        self._df = self._datastore.read()
        self._compute_stats()

    def _compute_stats(self) -> None:
        """Computes stats for entire dataset and for conversions."""
        # Compute on entire dataset
        self._stats = self._metadata.get_statistics(self._df)
        self._stats_numeric = self._stats["numeric"]
        self._stats_categorical = self._stats["categorical"]
        self._stats_summary = self._stats["summary"]

    def _load_if_null(self) -> None:
        if self._df is None:
            self.load()

    @property
    def summary(self) -> None:
        """Provides overall summary at dataet level."""
        self._load_if_null()

        self._printer.print_title(self._dataset_name)
        d = {}
        d["Rows"], d["Columns"] = self._df.shape
        d["Complete Samples"] = len(self._get_complete())
        d["Size Memory"] = self._df.memory_usage().sum()
        self._printer.print_dictionary(d, "Overview")

        d = {}
        d["Users"] = self._df["user_id"].nunique()
        d["Products"] = self._df["product_id"].nunique()
        d["Clicks"] = self._df.shape[0]
        d["Conversions"] = len(self._df.loc[self._df["sale"] == 1])
        d["Conversion Rate"] = round(d["Conversions"] / d["Clicks"] * 100, 2)
        self._printer.print_dictionary(d, "Basic Statistics")

        d = {}
        d["Cells"] = self._stats_summary["cells"]
        d["Missing"] = self._stats_summary["missing"]
        d["MissingNess"] = self._stats_summary["missingness"]
        self._printer.print_dictionary(d, "Missing")

        d = {}
        d["Numeric Targets"] = self._stats_summary["numeric_targets"]
        d["Categorical Targets"] = self._stats_summary["categorical_targets"]
        d["Numeric Features"] = self._stats_summary["numeric_features"]
        d["Categorical Features"] = self._stats_summary["categorical_features"]
        self._printer.print_dictionary(d, "Data Types")

        d = {}
        d["First Click"] = self._stats_summary["first_click"]
        d["Last Click"] = self._stats_summary["last_click"]
        d["Period"] = self._stats_summary["period"]
        self._printer.print_dictionary(d, "Period")
        return self

    @property
    def info(self) -> None:
        """Prints a summary of each column."""
        self._load_if_null()
        columns = [
            "column",
            "datatype",
            "count",
            "unique",
            "uniqueness",
            "missing",
            "missingness",
            "example",
        ]
        dfn = self._stats_numeric[columns]
        dfc = self._stats_categorical[columns]
        df = pd.concat([dfn, dfc], axis=0)

        subtitle = "Column Summary"
        self._printer.print_title(self._dataset_name, subtitle)
        self._printer.print_dataframe(df)
        return self

    @property
    def dataset(self) -> None:
        """Prints basic file information and statistics."""
        self._load_if_null()
        d = {}
        d["Rows"], d["Columns"] = self._df.shape
        d["Complete Samples"] = len(self._get_complete())
        d["Size Memory"] = self._df.memory_usage().sum()
        d["Filepath"] = self._datastore.filepath
        d["Filesize"] = self._datastore.filesize
        d["Created"] = self._datastore.created
        d["Modified"] = self._datastore.modified
        subtitle = "File Summary"
        self._printer.print_title(self._dataset_name, subtitle)
        self._printer.print_dictionary(d)
        return self

    @property
    def datatypes(self) -> None:
        """Prints a summary of the datatypes in the dataset."""
        self._load_if_null()
        d = {}
        d["Numeric Targets"] = self._stats_summary["numeric_targets"]
        d["Numeric Features"] = self._stats_summary["numeric_features"]
        d["Categorical Features"] = self._stats_summary["categorical_features"]
        subtitle = "Datatype Summary"
        self._printer.print_title(self._dataset_name, subtitle)
        self._printer.print_dictionary(d)
        return self

    @property
    def codebook(self) -> None:
        """Prints column definitions."""
        df = self._datastore.codebook
        df = titlelize_df(df)
        subtitle = "Column Definitions"
        self._printer.print_title(self._dataset_name, subtitle)
        self._printer.print_dataframe(df)
        return self

    @property
    def describe(self) -> None:
        _ = self.numerics
        _ = self.categoricals
        return self

    def describe_numerics(self) -> None:
        """Prints descriptive statistics for numeric variables."""
        self._load_if_null()
        df = self._stats_numeric[numeric_descriptive_stats]
        df = titlelize_df(df)
        subtitle = "Descriptive Statistics Numeric Variables"
        self._printer.print_title(self._dataset_name, subtitle)
        self._printer.print_dataframe(df)
        return self

    def plot_numerics(self) -> None:
        """Plots histograms for numeric variables."""
        self._load_if_null()
        self._visual.multihist(
            self._df,
            columns=numeric_columns,
            rows=3,
            cols=2,
            title="Numeric Variable Analysis",
        )

    def describe_categoricals(self) -> None:
        """Prints descriptive statistics for categorical variables."""
        self._load_if_null()

        df = self._stats_categorical[categorical_descriptive_stats]

        # Split top on space and return the first element.
        tops_trunc = []
        tops = df["top"].values
        for top in tops:
            if isinstance(top, str):
                tops_trunc.append(top.split()[0])
            else:
                tops_trunc.append(top)
        df.drop("top", axis=1, inplace=True)
        df["top"] = tops_trunc

        df = titlelize_df(df)
        subtitle = "Descriptive Statistics Categorical Variables"
        self._printer.print_title(self._dataset_name, subtitle)
        self._printer.print_dataframe(df)

    def plot_categoricals(self) -> None:
        """Plots countplots for categorical variables."""
        self._load_if_null()
        for col in categorical_columns:
            self._visual.countplot(
                self._df,
                column=col,
                title="Categorical Variable Analysis\n{}".format(col),
            )

    def describe_datetimes(self) -> None:
        """Prints descriptive statistics for datetime objects."""

        self._load_if_null()

    @property
    def conversions(self) -> None:
        """Summarizes the conversions and conversion rates."""
        self._load_if_null()

        self._printer.print_title(self._dataset_name, "Conversions")
        stats = {}
        stats["Clicks"] = self._stats_summary["clicks"]
        stats["Conversions"] = self._stats_summary["conversions"]
        stats["Conversion Rate"] = round(
            stats["Conversions"] / stats["Clicks"] * 100, 2
        )
        self._printer.print_dictionary(stats)
        return self

    @property
    def entities(self) -> None:
        """Summarizes product, user, brand, device, and audience counts and cardinality"""
        self._load_if_null()

        self._printer.print_title(self._dataset_name, "Entity Summaries")

        columns = [
            "user_id",
            "product_id",
            "product_brand",
            "partner_id",
            "audience_id",
            "device_type",
        ]
        stats = [
            "column",
            "count",
            "unique",
            "uniqueness",
            "missing",
            "missingness",
            "example",
        ]

        data = self._stats_categorical.loc[
            self._stats_categorical["column"].isin(columns), stats
        ]

        data = titlelize_df(data)
        self._printer.print_dataframe(data)
        return self

    @property
    def products(self) -> None:
        self._load_if_null()

        stats = [
            "column",
            "count",
            "top",
            "freq",
            "freq_%",
            "unique",
            "uniqueness",
            "missing",
            "missingness",
            "example",
        ]

        data = self._stats_categorical.loc[
            self._stats_categorical["column"].str.contains("product"), stats
        ]

        data = titlelize_df(data)
        self._printer.print_title(self._dataset_name, "Products Summary")
        self._printer.print_dataframe(data)
        return self

    @property
    def missing(self) -> None:
        self._load_if_null()

        self._printer.print_title(self._dataset_name, "Missing Analysis")

        d = {}
        d["Cells"] = self._stats_summary["cells"]
        d["Missing"] = self._stats_summary["missing"]
        d["MissingNess"] = self._stats_summary["missingness"]
        self._printer.print_dictionary(d, "Summary")
        self._printer.print_blank_line()

        stats = [
            "column",
            "count",
            "unique",
            "uniqueness",
            "missing",
            "missingness",
        ]

        df2 = self._stats_categorical.loc[
            self._stats_categorical["column"].isin(categorical_columns), stats
        ]
        df1 = self._stats_numeric.loc[
            self._stats_numeric["column"].isin(numeric_columns), stats
        ]

        df = pd.concat([df1, df2], axis=0)

        df = titlelize_df(df)

        self._printer.print_dataframe(df)
        return self

    @property
    def outliers(self) -> pd.DataFrame:
        return self._outlier_labels

    def detect_outliers(self) -> None:
        """Returns outlier tables based upon the outlier detection ensemble."""
        self._outlier_detector.fit(X=self._df)
        self._outlier_labels = self._outlier_detector.predict(X=self._df)
        self._datastore.write(self._outlier_labels, "outliers.csv")
        return self._outlier_labels

    def analyze(self, x: str) -> None:
        """Provides counts, statistics and visual analysis of a variable

        Args:
            x (str): a column name
        """
        self._load_if_null()

        # We'll treat sale as a categorical variable for plotting.
        if x in numeric_columns and x != "sale":
            self._analyze_numeric(x)
        elif x in categorical_columns or x == "sale":
            self._analyze_categorical(x)
        else:
            raise Exception("x is not a valid column")
        return self

    def _analyze_numeric(self, x) -> None:

        df = self._stats_numeric.loc[self._stats_numeric["column"] == x][
            numeric_descriptive_stats
        ]
        df = titlelize_df(df)
        title = "Descriptive Statistics"
        subtitle = x
        plot_title = "Count Plot\n" + x
        self._printer.print_title(title, subtitle)
        self._printer.print_dataframe(df)
        self._visual.hist(self._df, column=x, title=plot_title)
        plot_title = "Boxplot\n" + x
        self._visual.boxplot(self._df, column=x, orient="h", title=plot_title)

    def _analyze_categorical(self, x) -> None:

        df = self._stats_categorical.loc[self._stats_categorical["column"] == x][
            categorical_descriptive_stats
        ]
        df = titlelize_df(df)
        title = "Descriptive Statistics"
        subtitle = x
        self._printer.print_title(title, subtitle)
        self._printer.print_dataframe(df)
        plot_title = "Count Plot\n" + x
        self._visual.countplot(self._df, column=x, title=plot_title)

    def head(self, n: int = 5) -> None:
        """Wraps pandas head method for DatFrames.

        Args:
            n (int): The number of lines to print

        Returns:
            DataFrame containing first n rows

        """
        self._load_if_null()
        result = self._df.head(n).T
        subtitle = "First {} rows".format(str(n))
        self._printer.print_title(self._dataset_name, subtitle)
        self._printer.print_dataframe(result)

    def sample(self, n: int = 5, complete=False, random_state=None) -> None:
        """Wraps pandas sample method for DatFrames.

        Args:
            n (int): The number of lines to print
            random_state (int): Pseudo random generator seed

        Returns:
            DataFrame containing n samples from the dataet

        """
        self._load_if_null()
        subtitle = "Random Samples"
        self._printer.print_title(self._dataset_name, subtitle)

        df = self._get_complete() if complete else self._df
        for i in range(n):
            result = df.sample(1, random_state=random_state)
            result = s_to_dict(df=result)
            self._printer.print_dictionary(result)
            self._printer.print_blank_line()

    def _get_complete(self) -> pd.DataFrame:
        """Returns a complete dataframe."""
        df = self._df.dropna()
        return df


# ------------------------------------------------------------------------------------------------------------------------ #
#                                              CRITEO METADATA                                                             #
# ------------------------------------------------------------------------------------------------------------------------ #
class CriteoMD:
    """Computes metadata for the Criteo dataset"""

    def __init__(self, stage: str = "staged", version: str = "prod") -> None:
        self._stage = stage
        self._version = version
        self._config = Config()
        self._filepath = self._config.get_metadata_filepath(stage, version)

        self._n_observations = None
        self._total_missing = 0

    def get_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Obtains statistics from file cache or via computation if cache not available.

        Args:
            df (pd.DataFrame): The data
        """
        if os.path.exists(self._filepath):
            return self.read_stats()
        else:
            return self._get_statistics(df)

    def _get_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        start = datetime.now()
        logger.info(
            "\tComputing statistics for {} version of {} data started at {}.".format(
                self._version, self._stage, start
            )
        )
        self._n_observations = len(df)
        numeric = self._get_stats(df, "number")
        categorical = self._get_stats(df, datatype=["category", "object"])
        summary = self._get_summary_stats(df)
        result = {"numeric": numeric, "categorical": categorical, "summary": summary}
        self.save_stats(result)

        end = datetime.now()
        duration = end - start
        logger.info(
            "\tComputed statistics for {} version of {} data completed at {}. Duration: {}".format(
                self._version, self._stage, end, duration
            )
        )

        return result

    def _get_summary_stats(self, df: pd.DataFrame) -> dict:
        """Returns dataset level summary statistics."""
        # Note: Must be run last since these summaries are based
        # upon numeric and categorical statistics.
        d = {}
        d["rows"], d["columns"] = df.shape
        d["size"] = round(df.memory_usage(deep=True).sum() / 1048576, 2)
        d["users"] = df["user_id"].nunique()
        d["products"] = df["product_id"].nunique()
        d["sellers"] = df["partner_id"].nunique()

        d["clicks"] = df.shape[0]
        d["conversions"] = len(df.loc[df["sale"] == "1"])
        d["conversion_rate"] = round(d["conversions"] / d["clicks"] * 100, 2)

        d["numeric_targets"] = (
            df[target_columns].select_dtypes(include="number").shape[1]
        )
        d["categorical_targets"] = (
            df[target_columns].select_dtypes(include="category").shape[1]
        )
        d["numeric_features"] = (
            df[feature_columns].select_dtypes(include="number").shape[1]
        )
        d["categorical_features"] = (
            df[feature_columns].select_dtypes(include=["category", "object"]).shape[1]
        )

        d["first_click"] = datetime.fromtimestamp(df["click_ts"].min())
        d["last_click"] = datetime.fromtimestamp(df["click_ts"].max())
        d["period"] = d["last_click"] - d["first_click"]

        d["cells"] = d["rows"] * d["columns"]
        d["missing"] = self._total_missing
        d["missingness"] = np.round(
            (d["missing"] / (df.shape[0] * df.shape[1])) * 100, 2
        )
        return d

    def _get_stats(self, df: pd.DataFrame, datatype) -> pd.DataFrame:
        df = df.select_dtypes(include=datatype)
        stats = df.describe().T.reset_index()
        # Add frequency % if freq is a column.
        if "freq" in stats.columns.values:
            stats["freq_%"] = stats["freq"].values / df.shape[0] * 100
        # Drop the unique column because it will be added below for all columns
        if "unique" in stats.columns.values:
            stats = stats.drop(columns=["unique"])
        dtypes = self._get_dtypes(df)
        sizes = self._get_size(df)
        uniques = self._get_unique(df)
        missing = self._get_missing(df)
        examples = self._get_examples(df)

        stats = dtypes.merge(stats, how="left", on="index")
        stats = stats.merge(uniques, how="left", on="index")
        stats = stats.merge(missing, how="left", on="index")
        stats = stats.merge(examples, how="left", on="index")
        stats = stats.merge(sizes, how="left", on="index")
        stats = stats.rename(columns={"index": "column"})
        return stats

    def _get_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        dt = df.dtypes.to_frame()
        dt.columns = ["datatype"]
        dt.reset_index(inplace=True)
        return dt

    def _get_size(self, df: pd.DataFrame) -> pd.DataFrame:
        dt = df.memory_usage(deep=True).to_frame()
        dt.columns = ["size"]
        dt.reset_index(inplace=True)
        return dt

    def _get_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = df.isna().sum().to_frame()
        missing.columns = ["missing"]
        missing.reset_index(inplace=True)
        missing["missingness"] = round(
            missing["missing"] / self._n_observations * 100, 2
        )
        self._total_missing += missing["missing"].sum()
        return missing

    def _get_unique(self, df: pd.DataFrame) -> pd.DataFrame:
        u = df.nunique().to_frame()
        u.columns = ["unique"]
        u.reset_index(inplace=True)
        u["uniqueness"] = round(u["unique"] / self._n_observations * 100, 2)
        return u

    def _get_examples(self, df: pd.DataFrame) -> pd.DataFrame:
        e = []
        for col in df.columns:
            s = df[col]
            s = s.dropna().sample(n=1).values[0]
            if isinstance(s, str):
                s = s.split()[0]
            e.append(s)
        d = {"index": df.columns.tolist(), "example": e}
        df2 = pd.DataFrame(d)
        return df2

    def read_stats(self) -> None:
        with open(self._filepath, "rb") as f:
            pkl_data = pickle.load(f)
        return pkl_data

    def save_stats(self, stats: dict) -> None:
        os.makedirs(os.path.dirname(self._filepath), exist_ok=True)
        with open(self._filepath, "wb") as f:
            pickle.dump(stats, f)
