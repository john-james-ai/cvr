#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ======================================================================================================================== #
# Project  : Explainable Recommendation (XRec)                                                                             #
# Version  : 0.1.0                                                                                                         #
# File     : \taobao.py                                                                                                    #
# Language : Python 3.8                                                                                                    #
# ------------------------------------------------------------------------------------------------------------------------ #
# Author   : John James                                                                                                    #
# Email    : john.james.ai.studio@gmail.com                                                                                #
# URL      : https://github.com/john-james-ai/xrec                                                                         #
# ------------------------------------------------------------------------------------------------------------------------ #
# Created  : Wednesday, December 22nd 2021, 2:50:27 am                                                                     #
# Modified : Friday, January 14th 2022, 6:47:11 pm                                                                         #
# Modifier : John James (john.james.ai.studio@gmail.com)                                                                   #
# ------------------------------------------------------------------------------------------------------------------------ #
# License  : BSD 3-clause "New" or "Revised" License                                                                       #
# Copyright: (c) 2021 Bryant St. Labs                                                                                      #
# ======================================================================================================================== #
from abc import ABC, abstractmethod
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import seaborn as sns
import logging

sns.set_style("whitegrid")


from cvr.utils.printing import Printer
from cvr.utils.config import DataConfig
from cvr.utils.data import to_list

# ------------------------------------------------------------------------------------------------------------------------ #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------------------------------ #
class Taobao:
    """Taobao data and methods for basic data exploration and preparation."""

    start = "11/25/2017 00:00:01"
    end = "12/03/2017 23:59:59"
    date_format = "%m/%d/%Y %H:%M:%S"
    section = "TAOBAO"

    def __init__(self, filepath, clear_data=False):
        """Initiates the object

        Args:
            filepath (str): String containing path to the data file.
            clear (bool): If True, remnants of the file are deleted from the data directories.
        """
        self._filename = os.path.basename(filepath)
        self._clear_data = clear_data
        # Required printer and configuration objects
        self._printer = Printer()
        self._config = Config(Taobao.section)
        self._analysis = TaobaoAnalysis()
        # The data and the data stage
        self._df = None
        # Clear old files
        self._clear()
        # Load and save the data in the raw directory.
        self._load(filepath)
        self.save(folder="raw")

    # -------------------------------------------------------------------------------------------------------------------- #
    #                                               DATA IO METHODS                                                        #
    # -------------------------------------------------------------------------------------------------------------------- #

    def _clear(self) -> None:
        """Deletes file from data directories all directories except external."""
        if self._clear_data:
            directories = ["raw", "prepped", "transformed", "processed"]
            for folder in directories:
                self._delete(folder)

    def _load(self, filepath: str) -> None:
        """Called only once from the constructor. Loads data from the external directory.

        Note: This method also adds a row for column names and changes behavior 'pv' for page_view to 'click'

        Args:
            filepath (str): The filepath from which to load the data.
        """
        self._df = pd.read_csv(filepath)

        # Add columns
        self._df.columns = ["user", "item", "category", "behavior", "timestamp"]

        # Change 'pv' for page view to click
        self._df.loc[self._df["behavior"] == "pv", "behavior"] = "click"

        # Add datetime object for the corresponding timestamp.
        self._df["datetime"] = pd.to_datetime(self._df["timestamp"], unit="s")

    def _delete(self, folder: str = "raw") -> None:
        """Deletes the file in the designated folder.

        Args:
            folder (str):  'raw', 'prepped', 'transformed', or 'processed'
        """
        directory = self._config.get_directory(folder)
        filepath = os.path.join(directory, self._filename)
        if os.path.exists(filepath):
            os.remove(filepath)

    def load(self, folder: str) -> None:
        """Loads the Taobao data from the requested folder

        Args:
            folder (str): The data folder from which to load the data.

        """
        directory = self._config.get_directory(folder)
        filepath = os.path.join(directory, self._filename)
        self._df = pd.read_csv(filepath)

    def save(self, folder: str) -> None:
        """Saves the data to the designated folder.

        Args:
            folder (str):  'raw', 'prepped', 'transformed', or 'processed'
        """
        directory = self._config.get_directory(folder)
        filepath = os.path.join(directory, self._filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self._df.to_csv(filepath)

    # -------------------------------------------------------------------------------------------------------------------- #
    #                                             DATA EXPLORATION                                                         #
    # -------------------------------------------------------------------------------------------------------------------- #
    def head(self, n: int = 5) -> None:
        """Wraps pandas head method for DatFrames.

        Args:
            n (int): The number of lines to print

        """
        print(self._df.head(n))

    def info(self) -> None:
        """Wraps Pandas info method for DataFrames."""
        print(self._df.info())

    def sample(
        self, n: int = 5, behavior: str = None, random_state=55, na_ok: bool = True
    ) -> pd.DataFrame:
        """Returns n observations randomly sampled from the dataset.

        Args:
            n (int): The number of rows to sample from the dataset.
            behavior (str): One of 'click', 'fav', 'cart', 'buy'.
            na_ok (bool): True if NaT, NaN, etc... are acceptable, False if they are not.
            random_state (int): Seed for pseudorandom generator

        Returns:
            pd.DataFrame containing the samples.
        """
        if behavior is not None:
            try:
                return self._df.loc[~pd.isna(self._df[behavior])].sample(
                    n, random_state=random_state
                )
            except:
                logger.error("Invalid Behavior")
        else:
            return self._df.sample(n, random_state=random_state)

    def get(
        self,
        idx=None,
        user=None,
        item=None,
    ) -> None:
        """Returns observation(s) based on idx, user, or item. Parameters may be ints or lists of ints.
        All ints are converted to single element lists.

        Args:
            idx (int,list): The row index or indices to print. Ignored if user and/or item are/is provided
            user (int, list): The user or users for which rows shall be printed.
            item (int, list): The item or items for which rows shall be printed.
        """

        # Convert scalars to lists.
        idx = self._to_list(idx, ignore=None)
        user = self._to_list(user, ignore=None)
        item = self._to_list(item, ignore=None)

        if user is not None:
            if item is not None:
                df = self._df.loc[
                    (self._df["user"].isin(user)) & (self._df["item"].isin(item))
                ]
            else:
                df = self._df.loc[self._df["user"].isin(user)]
        elif item is not None:
            df = self._df.loc[self._df["item"].isin(item)]
        elif idx is not None:
            df = self._df.loc[idx]
        return df

    def print(
        self,
        idx=None,
        user=None,
        item=None,
    ) -> None:
        """Prints observation(s) based on idx, user, or item. Parameters may be ints or lists of ints.
        All ints are converted to single element lists.

        Args:
            idx (int,list): The row index or indices to print. Ignored if user and/or item are/is provided
            user (int, list): The user or users for which rows shall be printed.
            item (int, list): The item or items for which rows shall be printed.
        """
        print(self.get(idx, user, item))

    # -------------------------------------------------------------------------------------------------------------------- #
    #                                            DATA ANALYSIS I                                                           #
    # -------------------------------------------------------------------------------------------------------------------- #

    def missing(self) -> pd.DataFrame:
        """Produces the missing data report."""
        self._analysis.compute_missing(self._df)
        self._summarize_missing()
        return self._analysis.missing["missing_data"]

    def outliers(self) -> pd.DataFrame:
        """Produces the outliers data report."""
        self._analysis.compute_outliers(self._df)
        self._summarize_outliers()
        return self._analysis.outliers["outliers_data"]

    def orphans(self) -> pd.DataFrame:
        """Produces the orphans data report."""
        self._analysis.compute_orphans(self._df)
        self._summarize_orphans()
        return self._analysis.orphans["orphans_data"]

    def late_clicks(self) -> pd.DataFrame:
        """Produces the late_clicks data report."""
        self._analysis.compute_late_clicks(self._df)
        self._summarize_late_clicks()
        return self._analysis.late_clicks["late_clicks_data"]

    def rates(self) -> pd.DataFrame:
        """Produces conversion rate and related rate statistics."""
        self._analysis.compute_rates(self._df)
        self._summarize_rates()

    # -------------------------------------------------------------------------------------------------------------------- #
    #                                             DATA PREPARATION I                                                       #
    # -------------------------------------------------------------------------------------------------------------------- #
    def dropna(self) -> None:
        """Wraps the pandas dropna method."""
        self._df = self._df.dropna()
        self._handle_update()

    def impute_timestamp(self, idx: int) -> None:
        """Impute timestamp at idx.

        Args
            idx (int): The index of the row in which imputation is happening
        """

        dt = datetime.strptime(Taobao.start, Taobao.date_format)

        self._df.at[idx, "timestamp"] = datetime.timestamp(dt)
        self._df.at[idx, "datetime"] = dt
        self._handle_update()

    # -------------------------------------------------------------------------------------------------------------------- #
    #                                           DATA VISUALIZATION                                                         #
    # -------------------------------------------------------------------------------------------------------------------- #

    def hist(
        self, behavior: str = None, bins: int = 43, sample: float = 1, random_state=55
    ) -> None:
        """Plots a histogram of behaviors over time.

        Args:
            behavior (str): The behavior to plot. If None, all behaviors are plotted as groups
            bins (int): The number of bins. Default is 43, approximately every 4 hours.

        """

        if behavior is not None:
            fig, ax = plt.subplots(figsize=(16, 8))
            sns.histplot(
                data=self._df[self._df["behavior"] == behavior].sample(
                    frac=sample, random_state=random_state
                ),
                x="datetime",
                bins=bins,
                kde=True,
                ax=ax,
            )
            title = "Histogram of {}s from November 25 through December 3, 2017".format(
                behavior
            )
        else:
            behaviors = self._df["behavior"].unique()
            n = len(behaviors)
            cols = n // 2
            rows = n - cols
            width = cols * 8
            height = rows * 4
            fig, axs = plt.subplots(rows, cols, figsize=(width, height), sharex=True)
            behaviors = self._df["behavior"].unique()
            i, j = [0, 0]
            for k, behavior in enumerate(behaviors):
                i = k // 2
                j = k % 2
                sns.histplot(
                    data=self._df[self._df["behavior"] == behavior].sample(
                        frac=sample, random_state=random_state
                    ),
                    x="datetime",
                    bins=bins,
                    ax=axs[i, j],
                )
                axs[i, j].set_title(behavior.capitalize())
                axs[i, j].tick_params(labelrotation=45)
            title = "Histogram of Behaviors from November 25 through December 3, 2017"

        fig.suptitle(title)
        fig.tight_layout()
        plt.show()

    # -------------------------------------------------------------------------------------------------------------------- #
    #                                             DATA TRANSFORMATION                                                      #
    # -------------------------------------------------------------------------------------------------------------------- #
    def transform(self) -> None:
        """Controls the transformation process in which the data is converted to wide format and features are added.

        Concretely, this method:
            1. Adds an interaction variable to represent the number of interactions in a journey from click to conversion.
            2. Converts the data to a wide format in which each row represents a customer journey, containing datetime objects for specific interactions.
            3. Adds a target variable that indicates whether a conversion occurred or not.
            4. Adds a feedback delay variable to capture the time between conversion and the initial click.
        """
        self.dropna()

        self._create_interactions_feature()

        self._wide()

        self._create_target()

        self._create_delayed_feedback()

        self._handle_update(stage="transformed")

    def _create_interactions_feature(self) -> None:
        "Creates variable that reflects the number of interactions for a user, item, category triple."
        df = self._df[["user", "item", "category", "behavior"]]
        self._df["interactions"] = df.groupby(
            by=["user", "item", "category"]
        ).transform("count")

    def _wide(self) -> None:
        """Pivots the table and groups by user, item, category, triplet, taking first value of columns."""
        df = self._df[
            ["user", "item", "category", "behavior", "interactions", "datetime"]
        ]
        df = pd.pivot_table(
            df,
            index=["user", "item", "category", "interactions"],
            columns=["behavior"],
            values="datetime",
        )

        self._df = (
            df.groupby(by=["user", "item", "category", "interactions"])
            .last()
            .reset_index()
            .rename_axis("index", axis=1)
        )

    def _create_target(self) -> None:
        """Creates a conversion indicator variable."""
        self._df["conversion"] = 0
        self._df.loc[~pd.isna(self._df["buy"]), "conversion"] = 1

    def _create_delayed_feedback(self) -> None:
        """Creates a delayed feedback variable."""
        self._df["delay"] = self._df["buy"] - self._df["click"]

    # -------------------------------------------------------------------------------------------------------------------- #
    #                                             DATA PREPARATION II                                                      #
    # -------------------------------------------------------------------------------------------------------------------- #
    def impute_missing_clicks(self) -> None:
        """Updates missing click data with earliest click time."""

        dt = datetime.strptime(Taobao.start, Taobao.date_format)
        self._df.loc[pd.isna(self._df["click"]), "click"] = dt
        self._handle_update()

    def replace_late_clicks(self) -> None:
        """Replaces late clicks for conversions with start date."""
        dt = datetime.strptime(Taobao.start, Taobao.date_format)
        self._df.loc[
            (~pd.isna(self._df["buy"]))
            & (
                (self._df["click"] > self._df["buy"])
                or (self._df["click"] > self._df["fav"])
                or (self._df["click"] > self._df["cart"])
            ),
            "click",
        ] = dt
        self._handle_update()

    # -------------------------------------------------------------------------------------------------------------------- #
    #                                               DATA ANALYSIS                                                          #
    # -------------------------------------------------------------------------------------------------------------------- #

    def summary(
        self, df: pd.DataFrame = None, title: str = "Taobao Dataset Summary"
    ) -> None:
        """Summarizes the internal dataset or the dataset passed as parameter

        Args:
            df (pd.DataFrame): Optional dataframe to be summarized. If None
                The internal dataframe is summarized.
            title (str): The title for the dataset being summarized

        ."""
        self._printer.print_title(title)
        self._analysis.compute_all(self._df)
        self.info()
        self._summarize_user_item_visits()
        self._summarize_interactions()
        self._summarize_rates()
        self._summarize_timestamps()
        self._summarize_missing()
        self._summarize_outliers()
        self._summarize_orphans()
        self._summarize_late_clicks()

    def _summarize_user_item_visits(self) -> None:
        d = self._analysis.visits
        self._printer.print_dictionary(d, title="User Item Visit Analysis")

    def _summarize_interactions(self) -> None:
        d = self._analysis.interactions
        self._printer.print_dictionary(d, title="Interaction Analysis")

    def _summarize_rates(self) -> None:
        d = self._analysis.rates
        self._printer.print_dictionary(d, title="Rates")

    def _summarize_timestamps(self) -> None:
        d = self._analysis.time_stats
        self._printer.print_dictionary(d, title="Timestamp Analysis")

    def _summarize_missing(self) -> None:
        d = self._analysis.missing
        self._printer.print_dictionary(d["missing"], title="Missing Data Analysis")
        self._printer.print_dataframe(d["missing_data"].head(20), "Missing Data")

    def _summarize_outliers(self) -> None:
        d = self._analysis.outliers
        self._printer.print_title("Outlier Analysis")
        print(
            "There are {} outliers, representing {}% of the data.".format(
                str(d["outliers"]), str(round(d["outliers_pct"], 2))
            )
        )
        self._printer.print_dataframe(d["outliers_data"].head(20), "Outliers")

    def _summarize_orphans(self) -> None:
        d = self._analysis.orphans
        self._printer.print_title("Orphan Analysis")
        print(
            "There are {} orphans, representing {}% of the data.".format(
                str(d["orphans"]), str(round(d["orphans_pct"], 2))
            )
        )
        self._printer.print_dataframe(d["orphans_data"].head(20), "Outliers")

    def _summarize_late_clicks(self) -> None:
        d = self._analysis.late_clicks
        self._printer.print_title("Orphan Analysis")
        print(
            "There are {} late_clicks, representing {}% of the data.".format(
                str(d["late_clicks"]), str(round(d["late_clicks_pct"], 2))
            )
        )
        self._printer.print_dataframe(d["late_clicks_data"].head(20), "Outliers")

    # -------------------------------------------------------------------------------------------------------------------- #
    #                                               HOUSEKEEPING                                                           #
    # -------------------------------------------------------------------------------------------------------------------- #
    def _handle_update(self, stage: str = None) -> None:
        """Performs post update housekeeping like clearing caches"""
        self._analysis.clear_cache()
        if stage is not None:
            self._analysis.current_stage = stage


# ======================================================================================================================== #
#                                             TAOBAO STATS                                                                 #
# ======================================================================================================================== #
class TaobaoAnalysis:
    """Class the computes, manages the dataset statistics, and associated data"""

    def __init__(self) -> None:

        # Users, items, categories and visits
        self.visits = {"users": None, "items": None, "categories": None, "visits": None}

        # Behaviors and interactions
        self.interactions = {
            "interactions": None,
            "clicks": None,
            "favorites": None,
            "carts": None,
            "conversions": None,
        }

        # Rates
        self.rates = {"conversion": None, "cart": None, "abandoned_cart": None}

        # Descriptive Statistics on timestamps and datetime objects
        self.time_stats = None

        # Data cleaning statistics and data
        self.missing = {"missing": None, "missing_pct": None, "missing_data": None}

        self.outliers = {"outliers": None, "outliers_pct": None, "outliers_data": None}

        self.orphans = {"orphans": None, "orphans_pct": None, "orphans_data": None}

        self.late_clicks = {
            "late_clicks": None,
            "late_clicks_pct": None,
            "late_clicks_data": None,
        }

        self._cache_available = {
            "basic": False,
            "visit": False,
            "interactions": False,
            "rates": False,
            "time": False,
            "missing": False,
            "outliers": False,
            "orphans": False,
            "late_clicks": False,
        }

        self._current_stage = "prepped"

    @property
    def current_stage(self) -> str:
        return self._current_stage

    @current_stage.setter
    def current_stage(self, stage: str) -> None:
        self._current_stage = stage

    def clear_cache(self) -> None:
        """Clears the cache. Called when data has changed."""
        for k in self._cache_available.keys():
            self._cache_available[k] = False

    def basic_stats(self, df: pd.DataFrame) -> None:
        """Computes basic shape statistics.

        Args:
            df (pd.DataFrame): Data
        """
        if not self._cache_available["basic"]:
            self.rows = df.shape[0]
            self.columns = df.shape[1]
            self._cache_available["basic"] = True

    def compute_user_item_visits(self, df: pd.DataFrame) -> None:
        """Computes cardinality of users and items

        Args:
            df (pd.DataFrame): Data
        """
        if not self._cache_available["visit"]:
            self.visits["users"] = len(df["user"].dropna().unique())
            self.visits["items"] = len(df["item"].dropna().unique())
            self.visits["categories"] = len(df["category"].dropna().unique())
            # Visits are not unique user, item, category triplets.
            self.visits["visits"] = len(df[["user", "item", "category"]].dropna())
            self._cache_available["visit"] = True

    def compute_interactions(self, df: pd.DataFrame) -> None:
        """Computes behavior statistics.

        Args:
            df (pd.DataFrame): Data
        """
        if not self._cache_available["interactions"]:
            if self._current_stage == "transformed":
                self._compute_interactions_wide(df)
            else:
                self._compute_interactions_long(df)
            self._cache_available["interactions"] = True

    def _compute_interactions_wide(self, df: pd.DataFrame) -> None:
        """Computes behavior statistics for wide dataset."""
        self.interactions["clicks"] = len(df["click"].dropna())
        self.interactions["favorites"] = len(df["fav"].dropna())
        self.interactions["carts"] = len(df["cart"].dropna())
        self.interactions["conversions"] = len(df["buy"].dropna())
        self.interactions["interactions"] = (
            self.interactions["clicks"]
            + self.interactions["favorites"]
            + self.interactions["carts"]
            + self.interactions["conversions"]
        )

    def _compute_interactions_long(self, df: pd.DataFrame) -> None:
        """Computes behavior statistics for wide dataset."""
        self.interactions["clicks"] = len(df[df["behavior"] == "click"].dropna())
        self.interactions["favorites"] = len(df[df["behavior"] == "fav"].dropna())
        self.interactions["carts"] = len(df[df["behavior"] == "cart"].dropna())
        self.interactions["conversions"] = len(df[df["behavior"] == "buy"].dropna())
        self.interactions["interactions"] = (
            self.interactions["clicks"]
            + self.interactions["favorites"]
            + self.interactions["carts"]
            + self.interactions["conversions"]
        )

    def compute_rates(self, df: pd.DataFrame) -> None:
        """Computes conversion, cart, and abandoned cart rates.

        Args:
            df (pd.DataFrame): Data
        """
        if not self._cache_available["rates"]:
            self.compute_interactions(df)
            self.rates["conversion"] = (
                self.interactions["conversions"] / self.interactions["clicks"] * 100
            )
            self.rates["cart"] = (
                self.interactions["carts"] / self.interactions["clicks"] * 100
            )
            self.rates["abandoned_cart"] = (
                (self.interactions["conversions"] - self.interactions["carts"])
                / self.interactions["clicks"]
                * 100
            )
            self._cache_available["rates"] = True

    def compute_time_stats(self, df: pd.DataFrame) -> None:
        """Computes timestamp and datetime object stats

        Args:
            df (pd.DataFrame): Data
        """
        if not self._cache_available["time"]:
            if self._current_stage == "transformed":
                self._compute_time_stats_wide(df)
            else:
                self._compute_time_stats_long(df)
            self._cache_available["time"] = True

    def _compute_time_stats_wide(self, df: pd.DataFrame) -> None:
        """Computes time stats for wide dataset."""
        self.time_stats = df[["click", "fav", "cart", "buy"]].describe().T

    def _compute_time_stats_long(self, df: pd.DataFrame) -> None:
        """Computes time stats for wide dataset."""
        self.time_stats = df[["timestamp", "datetime"]].describe().T

    def compute_missing(self, df: pd.DataFrame) -> None:
        """Computes missing items.

        Args:
            df (pd.DataFrame): Data
        """
        if not self._cache_available["missing"]:
            is_na = df.isna()
            row_has_na = is_na.any(axis=1)
            self.missing["missing_data"] = df[row_has_na]
            s = is_na.sum()
            self.missing["missing"] = s.to_dict()
            self.missing["missing_pct"] = {}
            for k, v in self.missing["missing"].items():
                self.missing["missing_pct"][k] = v / len(df) * 100
            self._cache_available["missing"] = True

    def compute_outliers(self, df: pd.DataFrame, distance: int = 3) -> None:
        """Computes outlier timestamps

        Args:
            df (pd.DataFrame): Data
            distance (int): The number of mean absolute deviations from the median
                that would identify an outlier.
        """
        if not self._cache_available["outliers"]:
            n = len(df)
            mad = df["timestamp"].mad()
            zish = abs(df["timestamp"] - df["timestamp"].median()) / (mad / np.sqrt(n))
            outliers = zish > distance * mad
            self.outliers["outliers_data"] = df[outliers]
            self.outliers["outliers"] = len(df[outliers])
            self.outliers["outliers_pct"] = self.outliers["outliers"] / n * 100
            self._cache_available["outliers"] = True

    def compute_orphans(self, df: pd.DataFrame) -> None:
        """Computes behaviors without associated clicks

        Args:
            df (pd.DataFrame): Data
        """
        if not self._cache_available["orphans"]:
            df1 = df.loc[(~pd.isnull(df["buy"])) & (pd.isnull(df["click"]))]
            df2 = df.loc[(~pd.isnull(df["fav"])) & (pd.isnull(df["click"]))]
            df3 = df.loc[(~pd.isnull(df["cart"])) & (pd.isnull(df["click"]))]
            df4 = (
                pd.concat([df1, df2, df3])
                .drop_duplicates()
                .sort_values(by=["user", "item", "category"])
            )
            self.orphans["orphans"] = len(df4)
            self.orphans["orphans_pct"] = len(df4) / len(df) * 100
            self.orphans["orphans_data"] = df4
            self._cache_available["orphans"] = True

    def compute_late_clicks(self, df: pd.DataFrame) -> None:
        """Computes behaviors in which the click follows a behavior.

        Args:
            df (pd.DataFrame): Data
        """
        if not self._cache_available["late_clicks"]:
            df1 = df.loc[df["click"] > df["buy"]]
            df2 = df.loc[df["click"] > df["fav"]]
            df3 = df.loc[df["click"] > df["cart"]]
            df4 = (
                pd.concat([df1, df2, df3])
                .drop_duplicates()
                .sort_values(by=["user", "item", "category"])
            )
            self.late_clicks["late_clicks"] = len(df4)
            self.late_clicks["late_clicks_pct"] = len(df4) / len(df) * 100
            self.late_clicks["late_clicks_data"] = df4
            self._cache_available["late_clicks"] = True

    def compute_all(self, df: pd.DataFrame) -> None:
        self.basic_stats(df)
        self.compute_user_item_visits(df)
        self.compute_interactions(df)
        self.compute_rates(df)
        self.compute_time_stats(df)
        self.compute_missing(df)
        self.compute_outliers(df)
        self.compute_orphans(df)
        self.compute_late_clicks(df)
