#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project  : Deep Learning for Conversion Rate Prediction (CVR)                                    #
# Version  : 0.1.0                                                                                 #
# File     : \dataset.py                                                                           #
# Language : Python 3.7.12                                                                         #
# ------------------------------------------------------------------------------------------------ #
# Author   : John James                                                                            #
# Email    : john.james.ai.studio@gmail.com                                                        #
# URL      : https://github.com/john-james-ai/cvr                                                  #
# ------------------------------------------------------------------------------------------------ #
# Created  : Thursday, January 13th 2022, 2:22:59 am                                               #
# Modified : Thursday, February 10th 2022, 8:53:04 pm                                              #
# Modifier : John James (john.james.ai.studio@gmail.com)                                           #
# ------------------------------------------------------------------------------------------------ #
# License  : BSD 3-clause "New" or "Revised" License                                               #
# Copyright: (c) 2022 Bryant St. Labs                                                              #
# ================================================================================================ #
"""Dataset Module"""
import pandas as pd
from pandas.api.types import is_numeric_dtype

from cvr.core.asset import AssetPassport, AssetRepo, Asset
from cvr.utils.printing import Printer


# ============================================================================ #
#                                 DATASET                                      #
# ============================================================================ #


class Dataset(Asset):
    def __init__(self, passport: AssetPassport, data: pd.DataFrame) -> None:
        super(Dataset, self).__init__(passport=passport)
        self._df = data

        # Cache for computations
        # Dataset summaries and info
        self._summary = None
        self._info = None
        # Column-wise computations
        self._rank_frequency_table = {}
        self._descriptive_statistics = {}
        self._printer = Printer()

    # ------------------------------------------------------------------------ #
    #                             PROPERTIES                                   #
    # ------------------------------------------------------------------------ #
    @property
    def size(self) -> float:
        return self._df.memory_usage(deep=True).sum() / (1024 * 1024)

    @property
    def shape(self) -> tuple:
        return self._df.shape

    @property
    def data(self) -> pd.DataFrame:
        return self._df

    @property
    def info(self) -> pd.DataFrame:
        return self._infoize()

    @property
    def summary(self) -> pd.DataFrame:
        return self._summarize()

    # ------------------------------------------------------------------------ #
    #                             DATA ACCESS                                  #
    # ------------------------------------------------------------------------ #
    def head(self, n: int = 5) -> pd.DataFrame:
        """Prints and returns the top n rows from a dataset.

        Args:
            n (int): Number of observations to print/return
        """
        df = self._df.head(n)
        subtitle = "First {} Rows".format(str(n))
        self._printer.print_title(self.passport.name, subtitle)
        self._printer.print_dataframe(df)

    def tail(self, n: int = 5) -> pd.DataFrame:
        """Prints and returns the last n rows from a dataset.

        Args:
            n (int): Number of observations to print/return
        """
        df = self._df.tail(n)
        subtitle = "Last {} Rows".format(str(n))
        self._printer.print_title(self.passport.name, subtitle)
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
        self._printer.print_title(self.passport.name, subtitle)
        if as_dict is True:
            d = df.to_dict(orient="index")
            for index, data in d.items():
                subtitle = "Index = {}".format(index)
                self._printer.print_dictionary(data, subtitle)
                self._printer.print_blank_line()
        else:
            self._printer.print_dataframe(df)

    # ------------------------------------------------------------------------ #
    #                AGGREGATION AND SUMMARIZATION                             #
    # ------------------------------------------------------------------------ #
    def _infoize(self) -> pd.DataFrame:
        """Prepares dataset information similar to pandas info method."""
        if self._info is not None:
            pass
        else:
            df1 = self._df.dtypes.to_frame()
            df2 = self._df.count(axis=0, numeric_only=False).to_frame()
            df3 = self._df.isna().sum().to_frame()
            df4 = df3[0] / len(self._df) * 100
            df5 = self._df.nunique().to_frame()
            df6 = df5[0] / len(self._df) * 100
            df7 = self._df.memory_usage(deep=True).to_frame()
            df8 = pd.concat([df1, df2, df3, df4, df5, df6, df7], axis=1, join="inner")
            df8.columns = [
                "Data Type",
                "Count",
                "Missing",
                "% Missing",
                "Unique",
                "% Unique",
                "Memory Usage",
            ]
            self._info = df8
        return self._info

    # ------------------------------------------------------------------------ #
    def _summarize(self) -> dict:
        """Renders dataset level statistics."""
        if not self._summary:
            d = {}
            d["Rows"] = self._df.shape[0]
            d["Columns"] = self._df.shape[1]
            d["Cells"] = self._df.shape[0] * self._df.shape[1]
            d["Size in Memory (Mb)"] = round(
                self._df.memory_usage(deep=True).sum() / (1024 * 1024), 2
            )
            d["Non-Null Cells"] = self._df.notna().sum().sum()
            d["Missing Cells"] = self._df.isna().sum().sum()
            d["Sparsity"] = round(d["Missing Cells"] / d["Cells"] * 100, 2)
            d["Duplicate Rows"] = self._df.duplicated(keep="first").sum()
            d["Duplicate Rows %"] = round(d["Duplicate Rows"] / d["Rows"] * 100, 2)
            datatypes = self._datatypes()
            d.update(datatypes)

    def _datatypes(self) -> dict:
        """Returns a dictionary of data type counts."""
        d = self._df.dtypes.astype(str).value_counts().to_frame().to_dict()[0]
        d2 = {}
        for k, v in d.items():
            k = k + " datatypes"
            d2[k] = v
        return d2

    # ------------------------------------------------------------------------ #
    def describe(self, column: str) -> pd.DataFrame:
        """Descriptive statistics for a column in a dataframe

        Args:
            column (str): Name of a column in the dataset.
        """
        if is_numeric_dtype(self._df[column]):
            return self.describe_numeric(column)
        else:
            return self.describe_categorical(column)

    # ------------------------------------------------------------------------ #
    def describe_numeric(self, column: str) -> pd.DataFrame:
        """Descriptive statistics for a numeric column in a dataframe

        Args:
            column (str): Name of a column in the dataset.
        """
        if not self._descriptive_statistics.get(column, None):
            stats = self._df[column].describe().to_frame().T
            stats["skew"] = self._df[column].skew(axis=0, skipna=True)
            stats["kurtosis"] = self._df[column].kurtosis(axis=0, skipna=True)
            stats["missing"] = self._df[column].isna().sum()
            stats["missingness"] = self._df[column].isna().sum() / len(self._df[column]) * 100
            stats["unique"] = self._df[column].nunique()
            stats["uniqueness"] = self._df[column].nunique() / len(self._df[column]) * 100
            self._descriptive_statistics[column] = stats
        return self._descriptive_statistics[column]

    # ------------------------------------------------------------------------ #
    def describe_categorical(self, column: str) -> pd.DataFrame:
        """Descriptive statistics for a non-numeric column in a dataframe

        Args:
            column (str): Name of a column in the dataset.
        """
        if not self._descriptive_statistics.get(column, None):
            stats = self._df[column].describe().to_frame().T
            stats["missing"] = self._df[column].isna().sum()
            stats["missingness"] = self._df[column].isna().sum() / len(self._df[column]) * 100
            stats["unique"] = self._df[column].nunique()
            stats["uniqueness"] = self._df[column].nunique() / len(self._df[column]) * 100
            self._descriptive_statistics[column] = stats
        return self._descriptive_statistics[column]

    # ------------------------------------------------------------------------ #
    def rank_frequencies(self, column: str, n: int = None) -> dict:
        """Returns frequencies for a categorical variable ordered by rank

        Args:
            column (str): Column name of categorical or object data in the
                dataset.
        Returns:
            numpy array
        """
        if self._rank_frequency_table.get(column, None) is None:
            self._rank_frequency_table[column] = self._df[column].value_counts().to_numpy()
        return self._rank_frequency_table[column]

    # ------------------------------------------------------------------------ #
    def cum_rank_frequencies(self, column: str, n: int) -> dict:
        """Returns cumulative frequencies ordered by rank

        Args:
            column (str): Column name of categorical or object data in the
                dataset.
        Returns:
            numpy array
        """
        if self._rank_frequency_table.get(column, None) is None:
            self._rank_frequency_table[column] = self._df[column].value_counts().to_numpy()
        cumfreq = self._rank_frequency_table[column].cumsum()
        return cumfreq


# ============================================================================ #
#                            DATASET REPO                                      #
# ============================================================================ #
class DatasetRepo(AssetRepo):
    """Dataset Repository

    Args:
        directory (str): Home directory for the repository
        registry (str): Filepath to the registry file
    """

    def __init__(self, directory: str, registry: str) -> None:
        super(DatasetRepo, self).__init__(directory, registry)

    def create(self, passport, AssetPassport, data: pd.DataFrame) -> Dataset:
        """Creates a Dataset object.

        Args:
            passport (AssetPassport): Identification object
        """

        aid = self.aid_gen()
        passport.aid = aid

        dataset = Dataset(
            passport=passport,
            data=data,
        )

        dataset = self.set_version(dataset)

        return dataset
