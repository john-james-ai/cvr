#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ======================================================================================================================== #
# Project  : Conversion Rate Prediction (CVR)                                                                              #
# Version  : 0.1.0                                                                                                         #
# File     : \etl.py                                                                                                       #
# Language : Python 3.7.12                                                                                                 #
# ------------------------------------------------------------------------------------------------------------------------ #
# Author   : John James                                                                                                    #
# Email    : john.james.ai.studio@gmail.com                                                                                #
# URL      : https://github.com/john-james-ai/cvr                                                                          #
# ------------------------------------------------------------------------------------------------------------------------ #
# Created  : Friday, January 21st 2022, 1:39:53 pm                                                                         #
# Modified : Thursday, January 27th 2022, 5:29:52 am                                                                       #
# Modifier : John James (john.james.ai.studio@gmail.com)                                                                   #
# ------------------------------------------------------------------------------------------------------------------------ #
# License  : BSD 3-clause "New" or "Revised" License                                                                       #
# Copyright: (c) 2022 Bryant St. Labs                                                                                      #
# ======================================================================================================================== #
"""Provides base classes for operators, including pipelines and tasks."""
from abc import ABC, abstractmethod
import os
import math
from collections import OrderedDict
from datetime import datetime, date
import pandas as pd
import numpy as np
import logging
import inspect
import requests
from typing import Union
import tarfile
import shutil
from tqdm import tqdm
import tempfile
import warnings

warnings.filterwarnings("ignore")

from cvr.data import criteo_columns, criteo_dtypes
from cvr.core.task import Task, STATUS_CODES
from cvr.core.workspace import Workspace
from cvr.utils.config import CriteoConfig
from cvr.core.pipeline import DataPipelineConfig
from cvr.core.dataset import Dataset
from cvr.utils.sampling import sample_file

# ------------------------------------------------------------------------------------------------------------------------ #
class Extract(Task):
    """Extracts the data from its source and creates araw Dataset object.

    Args:
        config (CriteoConfig): Configuration object with parameters needed download the source data.

    """

    def __init__(self, datasource_config: dict, chunk_size: int = 20) -> None:
        super(Extract, self).__init__()
        self._datasource_config = datasource_config
        self._chunk_size = chunk_size

        self._chunk_metrics = OrderedDict()

        self._config = None
        self._filepath_download = None
        self._filepath_raw = None
        self._n_groups = 10

    @property
    def chunk_metrics(self) -> dict:
        "Times and download rates for each requests chunk of data."
        return self._chunk_metrics

    def _run(self, data: Dataset = None) -> Dataset:
        """Downloads the data if it doesn't already exist or if self._config.force is True.

        Args:
            data (Dataset): Dataset created by the previous step.

        Returns:
            Dataset object
        """
        self._logger.debug("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        # Format filepaths.
        self._source = self._datasource_config.url
        self._filepath_download = self._datasource_config.destination
        self._filepath_raw = os.path.join("data", self._config.workspace.name, self._datasource_config.filepath_raw)

        # --------------------------------------- DOWNLOAD STEP --------------------------------------- #

        # Download the data unless it already exists or force is True
        if not os.path.exists(self._filepath_download) or self._config.force is True:
            if os.path.exists(self._filepath_download):
                x = input("Data already downloaded. Do you want to download again and overwrite these data? [y\\n]")
                if "y" in x.lower():
                    self._download(source=self._source, destination=self._filepath_download)

        # -------------------------------------- DECOMPRESS STEP -------------------------------------- #

        # Extract to raw data in the workspace, unless it already exists or force is True
        if not os.path.exists(self._filepath_raw) or self._config.force is True:
            self._decompress(source=self._filepath_download, destination=self._filepath_raw)

        # ----------------------------------------- LOAD STEP ----------------------------------------- #

        df = pd.read_csv(
            self._filepath_raw, sep="\t", header=None, names=criteo_columns, dtype=criteo_dtypes, low_memory=False
        )

        # Create Dataset object
        dataset = self._build_dataset(data=df)

        # Update status code
        self._status_code = "200"

        self._logger.debug("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))
        return dataset

    def _download(self, source: str, destination: str) -> dict:
        """Downloads the data from the site"""
        self._logger.debug("\t\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        os.makedirs(os.path.dirname(destination), exist_ok=True)

        # Override  start time since we're measuring internet speed
        self._start = datetime.now()

        with requests.get(source, stream=True) as response:
            self._status_code = response.status_code
            if response.status_code == 200:
                self._process_response(destination=destination, response=response)
                self._logger.info(
                    "\n\tDownload complete! {} Mb downloaded in {} {} Mb chunks.".format(
                        str(self._summary["Downloaded (Mb)"]),
                        str(self._summary["Chunks Downloaded"]),
                        str(self._chunk_size),
                    )
                )
            else:
                raise ConnectionError(response)

        self._logger.debug("\t\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def _process_response(self, destination: str, response: requests.Response) -> dict:
        """Processes an HTTP Response

        Args:
            destination (str): The download destination file path
            response (requests.Response): HTTP Response object.
        """
        self._logger.debug("\t\t\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        self._setup_process_response(response)

        # Set chunk_size, defaults to 10Mb
        chunk_size = 1024 * 1024 * self._chunk_size

        # Get size of content
        size_in_bytes = int(response.headers.get("content-length", 0))

        # Get the number of chunks
        n_chunks = math.ceil(size_in_bytes / chunk_size)

        # Set number of iterations in each group of chunks for reporting purposes
        group_size = self._group_size(n_chunks)

        # Setup data for progress bar
        if self._config.progress and self._config.verbose:
            progress_bar = tqdm(total=size_in_bytes, unit="iB", unit_scale=True)

        with open(destination, "wb") as fd:
            i = 0
            downloaded = 0
            for chunk in response.iter_content(chunk_size=chunk_size):
                if self._config.progress and self._config.verbose:
                    progress_bar.update(len(chunk))

                # Download the chunk and capture transmission metrics.
                downloaded = self._download_chunk(
                    chunk=chunk, fd=fd, response=response, group_size=group_size, i=i, downloaded=downloaded
                )

                i += 1

        if self._config.progress and self._config.verbose:
            progress_bar.close()

        self._teardown_process_response(destination=destination, i=i, downloaded=downloaded)

        self._logger.debug("\t\t\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def _group_size(self, n_chunks: int, idx: int = 0) -> int:
        """Computes the number of chunk download iterations in a group for reporting purposes"""
        group_sizes = [1, 5, 10, 20, 50, 100]
        if n_chunks / group_sizes[idx] <= self._n_groups:
            return group_sizes[idx]
        else:
            idx += 1
            return self._group_size(n_chunks, idx)

    def _setup_process_response(self, response: requests.Response) -> None:
        """Grab some metadata from the content header.

        Args:
            logger (logging): Voice, every method must have a voice, right?
            response (requests.Response): HTTP Response object.
        """

        self._logger.info(
            "\tDownloading {} Mb in {} Mb chunks\n".format(
                str(round(int(response.headers.get("Content-Length", 0)) / (1024 * 1024), 2)), str(self._chunk_size)
            )
        )
        # Grab response header information

        self._summary["Status Code"] = response.status_code
        self._summary["Content Type"] = response.headers.get("Content-Type", "")
        self._summary["Last Modified"] = response.headers.get("Last-Modified", "")
        self._summary["Content Length (Mb)"] = round(int(response.headers.get("Content-Length", 0)) / (1024 * 1024), 3)

    def _download_chunk(
        self, chunk, fd, response: requests.Response, group_size: int, i: int = 0, downloaded: int = 0
    ) -> dict:
        """Downloads a chunk of data from source site and produces some metrics.

        Args:
            chunk (requests.chunk) Chunk of data downloaded from the source site.
            fd (file descriptor): Pointer to the file.
            response (requests.Response): HTTP Response object.
            i (int): Current iteration
            downloaded (int): Bytes downloaded
        """
        self._logger.debug("\t\t\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        # Anal about indexes
        chunk_number = i + 1

        # Write the data
        fd.write(chunk)

        # Capture cumulative downloaded and duration
        downloaded += int(len(chunk))
        duration = datetime.now() - self._start

        # Duration may be zero for first few iterations. In case of zero division error, we'll set
        # average speed to 0
        try:
            pct_downloaded = downloaded / int(response.headers.get("Content-Length", 0)) * 100
            average_mbps = int((downloaded / duration.total_seconds()) / (1024 * 1024))
        except ZeroDivisionError as e:
            average_mbps = 0

        # If progress is disabled and verbose is True, provide progress every 'check_download' iterations
        if not self._config.progress and self._config.verbose:
            if (chunk_number) % group_size == 0:
                self._logger.info(
                    "\tChunk #{}: {} percent downloaded at {} Mbps".format(
                        str(chunk_number), str(round(pct_downloaded, 2)), str(round(average_mbps, 2))
                    )
                )

        self._chunk_metrics[chunk_number] = {
            "Downloaded": downloaded,
            "Duration": duration,
            "Pct Downloaded": pct_downloaded,
            "Avg Mbps": average_mbps,
        }

        self._logger.debug("\t\t\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))
        return downloaded

    def _teardown_process_response(self, destination: str, i: int, downloaded: int) -> dict:
        """Deletes temporary files, and updates the response header with additional information

        Args:
            destination (str): The download destination file path
            i (int): The current number of chunks downloaded
            downloaded (int): Total bytes downloaded.
        """
        self._logger.debug("\t\t\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        duration = datetime.now() - self._start
        Mb = os.path.getsize(destination) / (1024 * 1024)
        self._summary["Chunk Size (Mb)"] = self._chunk_size
        self._summary["Chunks Downloaded"] = i + 1
        self._summary["Downloaded (Mb)"] = round(downloaded / (1024 * 1024), 3)
        self._summary["Mbps"] = round(Mb / duration.total_seconds(), 3)

        self._logger.debug("\t\t\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def _decompress(self, source: str, destination: str) -> None:
        """Decompresses the gzip file from source and extracts data to the destination."""
        self._logger.debug("\t\t\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))
        self._logger.info("\tDecompression initiated.")
        data = tarfile.open(source)
        with tempfile.TemporaryDirectory() as tempdirname:
            data.extractall(tempdirname)
            tempfilepath = os.path.join(tempdirname, self._datasource_config.filepath_extract)

            self._summary["Size Extracted (Mb)"] = round(int(os.path.getsize(tempfilepath)) / (1024 * 1024), 2)

            # If sampling, copy a sample from the temp file, otherwise copy the tempfile in its entirety
            if self._config.dataset_config.sample_size is not None:
                self._decompress_sample(
                    source=tempfilepath,
                    destination=destination,
                    nrows=self._config.dataset_config.sample_size,
                    random_state=self._config.random_state,
                )
            else:
                os.makedirs(os.path.dirname(destination), exist_ok=True)
                shutil.copyfile(tempfilepath, destination)
        self._logger.info("\tDecompression Complete! {} Mb Extracted.".format(str(self._summary["Size Extracted (Mb)"])))
        self._logger.debug("\t\t\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def _decompress_sample(self, source: str, destination: str, nrows: int, random_state: int = None) -> None:
        self._logger.debug("\t\t\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        """Reads a sample from the source file and stores in destination."""
        self._logger.info("\tSampling dataset initiated.")
        sample_file(source=source, destination=destination, nrows=nrows, random_state=random_state)
        self._summary["Sampled Dataset Observations"] = nrows
        self._summary["Sampled Dataset Size (Mb)"] = round(int(os.path.getsize(destination)) / (1024 * 1024), 2)
        self._logger.info("\tSampling Complete! {} Rows Sampled.".format(str(nrows)))

        self._logger.debug("\t\t\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def _process_non_response(self, response: requests.Response) -> dict:
        """In case non 200 response from HTTP server.

        Args:
            response (requests.Response): HTTP Response object.

        """
        self._summary["HTTP Response Status Code"] = response.status_code

    @property
    def summary(self) -> dict:
        self._print_summary_dict(self._summary)
        return self._summary


# ------------------------------------------------------------------------------------------------------------------------ #
class TransformETL(Task):
    """Replaces designated missing values (-1) with NaN

    Args:
        value (str,list): The value or values to replace
    Returns:
        DataFrame with replaced values

    """

    def __init__(self, value: Union[str, list]) -> None:
        super(TransformETL, self).__init__()
        self._value = value
        self._before = None
        self._after = None

    def _run(self, data: Dataset = None) -> Dataset:
        """Transforms source data.

        Args:
            data (Dataset): Dataset created by the previous step.

        Returns:
            Dataset object
        """

        self._logger.debug("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        data.set_task_data(self)

        # Check Missing before the operation
        self._before = self._df.isna().sum()

        # Transform the missing values
        self._df = self._df.replace(self._value, np.nan)
        self._status_code = "200"

        # Check missing after
        self._after = self._df.isna().sum()

        # Create Dataset object
        dataset = self._build_dataset(data=self._df)
        return dataset

        # Summarize
        self._summary = missing

        self._logger.debug("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        return dataset

    @property
    def summary(self) -> dict:
        """Prepares missing values by column and presents totals."""
        rows, columns = self._df.shape
        cells = rows * columns
        # Prepare Missing Values Detail
        missing = self._before.to_frame(name="Before").join(self._after.to_frame(name="After"))

        subtitle = "Missing Values Replacement: Dataset {} / {} Stage".format(self._config.name, self._config.stage)
        self._print_summary_df(missing, subtitle)

        # Prepare missing values summary
        before = self._before.sum()
        after = self._after.sum()
        d = {
            "Missing": {"Before": before, "After": after},
            "Total": {"Before": cells, "After": cells},
            "Pct": {"Before": round(before / cells * 100, 2), "After": round(after / cells * 100, 2)},
        }
        df = pd.DataFrame.from_dict(d, orient="columns")
        self._print_summary_df(df, " ")
        return missing


# ------------------------------------------------------------------------------------------------------------------------ #
class LoadDataset(Task):
    """Loads the dataset into the current workspace."""

    def __init__(self) -> None:
        super(LoadDataset, self).__init__()

    def _run(self, data: Dataset = None) -> Dataset:
        """Add the dataset to the current workspace

        Args:
            data (Dataset): Dataset created by the previous step.

        Returns:
            Dataset object
        """

        self._logger.debug("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        filepath = self._config.workspace.add_dataset(data)
        # Update status code
        self._status_code = "200"
        self._summary = OrderedDict()
        self._summary["AID"] = data.aid
        self._summary["Workspace"] = self._config.workspace.name
        self._summary["Dataset Name"] = data.name
        self._summary["Stage"] = data.stage
        self._summary["filepath"] = filepath

        self._logger.debug("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        return data

    @property
    def summary(self) -> dict:
        """Prints a summary of the task result."""
        self._print_summary_dict(self._summary)
        return self._summary
