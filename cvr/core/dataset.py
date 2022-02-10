#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ============================================================================ #
# Project  : Deep Learning for Conversion Rate Prediction (CVR)                #
# Version  : 0.1.0                                                             #
# File     : \dataset.py                                                       #
# Language : Python 3.7.12                                                     #
# ---------------------------------------------------------------------------- #
# Author   : John James                                                        #
# Email    : john.james.ai.studio@gmail.com                                    #
# URL      : https://github.com/john-james-ai/cvr                              #
# ---------------------------------------------------------------------------- #
# Created  : Thursday, January 13th 2022, 2:22:59 am                           #
# Modified : Wednesday, February 9th 2022, 11:48:53 pm                         #
# Modifier : John James (john.james.ai.studio@gmail.com)                       #
# ---------------------------------------------------------------------------- #
# License  : BSD 3-clause "New" or "Revised" License                           #
# Copyright: (c) 2022 Bryant St. Labs                                          #
# ============================================================================ #
"""Dataset Module"""
import pandas as pd
import os
from collections import OrderedDict
import math
from datetime import datetime
import requests
import tarfile
import inspect
import shutil
from tqdm import tqdm
import tempfile
from pandas.api.types import is_numeric_dtype

from cvr.core.task import Task
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

    def sample(
        self, n: int = 5, as_dict: bool = True, random_state: int = None
    ) -> pd.DataFrame:
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
        if not self._info:
            df1 = self._df.dtypes.to_frame()
            df2 = self._df.count(axis=0, numeric_only=False).to_frame()
            df3 = self._df.isna().sum().to_frame()
            df4 = df3[0] / df2[0] * 100
            df5 = self._df.nunique().to_frame()
            df6 = df5[0] / df2[0] * 100
            df7 = self._df.memory_usage(deep=True).to_frame()
            df8 = pd.concat(
                [df1, df2, df3, df4, df5, df6, df7], axis=1, join="inner"
            )
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
            d["Duplicate Rows %"] = round(
                d["Duplicate Rows"] / d["Rows"] * 100, 2
            )
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
            stats["missingness"] = (
                self._df[column].isna().sum() / len(self._df[column]) * 100
            )
            stats["unique"] = self._df[column].nunique()
            stats["uniqueness"] = (
                self._df[column].nunique() / len(self._df[column]) * 100
            )
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
            stats["missingness"] = (
                self._df[column].isna().sum() / len(self._df[column]) * 100
            )
            stats["unique"] = self._df[column].nunique()
            stats["uniqueness"] = (
                self._df[column].nunique() / len(self._df[column]) * 100
            )
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
            self._rank_frequency_table[column] = (
                self._df[column].value_counts().to_numpy()
            )
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
            self._rank_frequency_table[column] = (
                self._df[column].value_counts().to_numpy()
            )
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


# ============================================================================ #
#                            DATASET FACTORY                                   #
# ============================================================================ #
class DatasetFactory(Task):
    """Creates Dataset objects."""

    def __init__(self, passport: AssetPassport) -> None:
        super(DatasetFactory, self).__init__(passport=passport)

    def run(self, data: pd.DataFrame) -> Dataset:
        self._logger = self._config.logger

        self._logger.info(
            "\tStarted {} {}".format(
                self.__class__.__name__, inspect.stack()[0][3]
            )
        )

        aid = self._config.dataset_repo.aid_gen()
        self._passport.aid = aid

        dataset = Dataset(
            passport=self._passport,
            data=data,
        )

        dataset = self._config.dataset_repo.set_version(dataset)

        self._logger.info(
            "\tCompleted {} {}".format(
                self.__class__.__name__, inspect.stack()[0][3]
            )
        )

        return dataset


# ============================================================================ #
#                            DATASET READER                                    #
# ============================================================================ #
class DatasetReader(Task):
    """Reads Dataset objects from the asset repository

    Args:
        dataset_repo (DatasetRepo): Asset repository

    """

    def __init__(self) -> None:
        self._asset_type = "dataset"

    def run(self, passport: AssetPassport) -> Dataset:
        self._logger = self._config.logger
        self._logger.info(
            "\tStarted {} {}".format(
                self.__class__.__name__, inspect.stack()[0][3]
            )
        )
        self._logger.info(
            "\tCompleted {} {}".format(
                self.__class__.__name__, inspect.stack()[0][3]
            )
        )
        return self._config.dataset_repo.get(
            asset_type=passport.asset_type,
            stage=passport.stage,
            name=passport.name,
        )


# ============================================================================ #
#                            DATASET WRITER                                    #
# ============================================================================ #
class DatasetWriter(Task):
    """Writes Dataset objects to the asset repository

    Args:
        dataset_repo (DatasetRepo): Asset repository

    """

    def __init__(self, passport: AssetPassport) -> None:
        self._passport = passport

    def run(self, dataset: Dataset) -> None:
        self._logger = self._config.logger
        self._logger.info(
            "\tStarted {} {}".format(
                self.__class__.__name__, inspect.stack()[0][3]
            )
        )
        self._logger.info(
            "\tCompleted {} {}".format(
                self.__class__.__name__, inspect.stack()[0][3]
            )
        )
        return self._config.dataset_repo.add(dataset)


# ---------------------------------------------------------------------------- #
#                             DOWNLOAD TASK                                    #
# ---------------------------------------------------------------------------- #


class Download(Task):
    """Downloads the data from the source website."""

    def __init__(
        self,
        passport: AssetPassport,
        source: str,
        destination: str,
        chunk_size=20,
        n_groups=20,
    ) -> None:
        super(Download, self).__init__(passport=passport)
        self._source = source
        self._destination = destination
        self._chunk_size = chunk_size
        self._n_groups = n_groups
        self._response = OrderedDict()
        self._chunk_metrics = OrderedDict()
        self._download_end = None
        self._download_start = datetime.now()

    def run(self, data: pd.DataFrame = None) -> None:
        """Downloads the data if it doesn't exist or if force is True

        Args:
            data (dict): Dictionary containing the Datasource parameters.

        """

        self._logger = self._config.logger
        self._destination = os.path.join(
            self._config.directory, self._destination
        )

        if not os.path.exists(self._destination):
            self._run(source=self._source, destination=self._destination)
            self._executed = True
            self._passed = True
            self._result = "Ok"

        else:
            self._executed = False
            self._passed = True
            self._result = "Ok"

    def _run(
        self,
        source: str,
        destination: str,
    ) -> dict:
        """Downloads and extracts the data to the raw directory"""

        # Download step
        with requests.get(source, stream=True) as response:
            self._status_code = response.status_code
            if response.status_code == 200:
                self._download(destination=destination, response=response)
            else:
                raise ConnectionError(response)

    def _download(self, destination: str, response: requests.Response) -> dict:
        """Downloads data

        Args:
            destination (str): The download destination file path
            response (requests.Response): HTTP Response object.
        """

        self._logger.info(
            "\tStarted {} {}".format(
                self.__class__.__name__, inspect.stack()[0][3]
            )
        )

        self._setup(response)

        # Set chunk_size, defaults to 10Mb
        chunk_size = 1024 * 1024 * self._chunk_size

        # Get size of content
        size_in_bytes = int(response.headers.get("content-length", 0))

        # Get the number of chunks
        n_chunks = math.ceil(size_in_bytes / chunk_size)

        # Set number of iterations in each group of chunks for reporting
        # purposes
        group_size = self._group_size(n_chunks)

        # Setup data for progress bar
        if self._config.progress and self._config.verbose:
            progress_bar = tqdm(total=size_in_bytes, unit="iB", unit_scale=True)

        # Create the directory
        os.makedirs(os.path.dirname(destination))

        with open(destination, "wb") as fd:
            i = 0
            downloaded = 0
            for chunk in response.iter_content(chunk_size=chunk_size):
                if self._config.progress and self._config.verbose:
                    progress_bar.update(len(chunk))

                # Download the chunk and capture transmission metrics.
                downloaded = self._download_chunk(
                    chunk=chunk,
                    fd=fd,
                    response=response,
                    group_size=group_size,
                    i=i,
                    downloaded=downloaded,
                )

                i += 1

        if self._config.progress and self._config.verbose:
            progress_bar.close()

        self._teardown(destination=destination, i=i, downloaded=downloaded)

        self._logger.info(
            "\tCompleted {} {}".format(
                self.__class__.__name__, inspect.stack()[0][3]
            )
        )

    def _group_size(self, n_chunks: int, idx: int = 0) -> int:
        """Computes the number of chunk download iterations in a group."""

        group_sizes = [1, 5, 10, 20, 50, 100]
        if n_chunks / group_sizes[idx] <= self._n_groups:
            return group_sizes[idx]
        else:
            idx += 1
            return self._group_size(n_chunks, idx)

    def _setup(self, response: requests.Response) -> None:
        """Grab some metadata from the content header.

        Args:
            logger (logging): Voice, every method must have a voice, right?
            response (requests.Response): HTTP Response object.
        """

        # Grab response header information
        self._response["Status Code"] = response.status_code
        self._response["Content Type"] = response.headers.get(
            "Content-Type", ""
        )
        self._response["Last Modified"] = response.headers.get(
            "Last-Modified", ""
        )
        self._response["Content Length (Mb)"] = round(
            int(response.headers.get("Content-Length", 0)) / (1024 * 1024), 3
        )

    def _download_chunk(
        self,
        chunk,
        fd,
        response: requests.Response,
        group_size: int,
        i: int = 0,
        downloaded: int = 0,
    ) -> dict:
        """Downloads a chunk of data from source site and produces some metrics.

        Args:
            chunk (requests.chunk) Chunk of data downloaded.
            fd (file descriptor): Pointer to the file.
            response (requests.Response): HTTP Response object.
            i (int): Current iteration
            downloaded (int): Bytes downloaded
        """

        # Anal about indexes
        chunk_number = i + 1

        # Write the data
        fd.write(chunk)

        # Capture cumulative downloaded and duration
        downloaded += int(len(chunk))
        duration = datetime.now() - self._download_start

        # Duration may be zero for first few iterations. In case of zero
        # division error, we'll set average speed to 0
        try:
            pct_downloaded = (
                downloaded
                / int(response.headers.get("Content-Length", 0))
                * 100
            )
            average_mbps = int(
                (downloaded / duration.total_seconds()) / (1024 * 1024)
            )
        except ZeroDivisionError:
            pct_downloaded = 0
            average_mbps = 0

        # If progress is disabled and verbose is True, provide progress every
        # 'check_download' iterations
        if not self._config.progress and self._config.verbose:
            if (chunk_number) % group_size == 0:
                self._logger.info(
                    "\tChunk #{}: {} percent downloaded at {} Mbps".format(
                        str(chunk_number),
                        str(round(pct_downloaded, 2)),
                        str(round(average_mbps, 2)),
                    )
                )

        self._chunk_metrics[chunk_number] = {
            "Downloaded": downloaded,
            "Duration": duration,
            "Pct Downloaded": pct_downloaded,
            "Avg Mbps": average_mbps,
        }

        return downloaded

    def _teardown(self, destination: str, i: int, downloaded: int) -> dict:
        """Captures metadata for the response variable.

        Args:
            destination (str): The download destination file path
            i (int): The current number of chunks downloaded
            downloaded (int): Total bytes downloaded.
        """
        self._download_end = datetime.now()
        duration = self._download_end - self._download_start

        Mb = os.path.getsize(destination) / (1024 * 1024)
        self._response["Chunk Size (Mb)"] = self._chunk_size
        self._response["Chunks Downloaded"] = i + 1
        self._response["Downloaded (Mb)"] = round(downloaded / (1024 * 1024), 3)
        self._response["Mbps"] = round(Mb / duration.total_seconds(), 3)

        self._logger.info(
            "Downloaded {} in {} chunks at {} Mbps".format(
                str(round(self._response["Downloaded (Mb)"], 2)),
                str(round(self._response["Chunks Downloaded"], 2)),
                str(round(self._response["Mbps"], 2)),
            )
        )


# ---------------------------------------------------------------------------- #
#                                 EXTRACT                                      #
# ---------------------------------------------------------------------------- #
class Extract(Task):
    """Extracts data from a gzip archive

    Args:
        download (str): Compressed file downloaded from the source
        extract (str): The file to extract from the gzip archive
        destination (str): The filepath to the raw data
    """

    def __init__(
        self,
        passport,
        download: str,
        extract: str,
        destination: str,
    ) -> None:
        super(Extract, self).__init__(passport=passport)
        self._download = download
        self._extract = extract
        self._destination = destination
        self._response = OrderedDict()

    def run(self, data: pd.DataFrame = None) -> pd.DataFrame:

        self._logger = self._config.logger

        self._logger.info(
            "\tStarted {} {}".format(
                self.__class__.__name__, inspect.stack()[0][3]
            )
        )

        if not os.path.exists(self._destination) or self._config.force is True:

            data = tarfile.open(self._download)
            with tempfile.TemporaryDirectory() as tempdirname:
                data.extractall(tempdirname)
                tempfilepath = os.path.join(tempdirname, self._extract)

                self._response["Size Extracted (Mb)"] = round(
                    int(os.path.getsize(tempfilepath)) / (1024 * 1024), 2
                )

                os.makedirs(os.path.dirname(self._destination), exist_ok=True)
                shutil.copyfile(tempfilepath, self._destination)

        self._logger.info(
            "\tCompleted {} {}".format(
                self.__class__.__name__, inspect.stack()[0][3]
            )
        )

        return pd.read_csv(self._destination, sep="\t", low_memory=False)
