#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ======================================================================================================================== #
# Project  : Conversion Rate Prediction (CVR)                                                                              #
# Version  : 0.1.0                                                                                                         #
# File     : \etl.py                                                                                                       #
# Language : Python 3.10.1                                                                                                 #
# ------------------------------------------------------------------------------------------------------------------------ #
# Author   : John James                                                                                                    #
# Email    : john.james.ai.studio@gmail.com                                                                                #
# URL      : https://github.com/john-james-ai/cvr                                                                          #
# ------------------------------------------------------------------------------------------------------------------------ #
# Created  : Friday, January 21st 2022, 1:39:53 pm                                                                         #
# Modified : Sunday, January 23rd 2022, 5:05:31 am                                                                         #
# Modifier : John James (john.james.ai.studio@gmail.com)                                                                   #
# ------------------------------------------------------------------------------------------------------------------------ #
# License  : BSD 3-clause "New" or "Revised" License                                                                       #
# Copyright: (c) 2022 Bryant St. Labs                                                                                      #
# ======================================================================================================================== #
#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ======================================================================================================================== #
# Project  : Conversion Rate Prediction (CVR)                                                                              #
# Version  : 0.1.0                                                                                                         #
# File     : \etl.py                                                                                                       #
# Language : Python 3.10.1                                                                                                 #
# ------------------------------------------------------------------------------------------------------------------------ #
# Author   : John James                                                                                                    #
# Email    : john.james.ai.studio@gmail.com                                                                                #
# URL      : https://github.com/john-james-ai/cvr                                                                          #
# ------------------------------------------------------------------------------------------------------------------------ #
# Created  : Friday, January 21st 2022, 1:39:53 pm                                                                         #
# Modified :                                                                                                               #
# Modifier : John James (john.james.ai.studio@gmail.com)                                                                   #
# ------------------------------------------------------------------------------------------------------------------------ #
# License  : BSD 3-clause "New" or "Revised" License                                                                       #
# Copyright: (c) 2022 Bryant St. Labs                                                                                      #
# ======================================================================================================================== #
"""Provides base classes for operators, including pipelines and tasks."""
from abc import ABC, abstractmethod
import os
from collections import OrderedDict
from datetime import datetime, date
import pandas as pd
import numpy as np
import logging
import requests
from typing import Union
import tarfile
import shutil
import warnings
import tempfile

warnings.filterwarnings("ignore")

from cvr.data import criteo_columns, criteo_dtypes
from cvr.core.task import Task, STATUS_CODES
from cvr.core.workspace import Workspace
from cvr.utils.config import CriteoConfig
from cvr.core.pipeline import PipelineCommand
from cvr.data.datasets import Dataset


# ------------------------------------------------------------------------------------------------------------------------ #
class Extract(Task):
    """Extracts the data from its source and creates a Dataset object.

    Args:
        config (CriteoConfig): Configuration object with parameters needed download the source data.
        chunk_size(int): Chunk size in Mb

    """

    def __init__(self, config: dict, chunk_size: int = 10) -> None:
        super(Extract, self).__init__()
        self._config = config
        self._chunk_size = chunk_size
        self._logger = None

        self._chunk_metrics = OrderedDict()

    @property
    def chunk_metrics(self) -> dict:
        return self._chunk_metrics

    def _run(self, command: PipelineCommand, data: Dataset = None) -> Dataset:
        """Downloads the data if it doesn't already exist or if command.force is True."""

        # Unpack command logger
        self._logger = command.logger

        # Download the data and decompress if not already there. If force is True, download is mandatory
        if not os.path.exists(self._config["destination"]) or command.force is True:
            self._summary = self._download()
            self._decompress()

        # Load raw data
        df = pd.read_csv(
            self._config["filepath_raw"], sep="\t", header=None, names=criteo_columns, dtype=criteo_dtypes, low_memory=False
        )

        # Create Dataset object
        dataset = self._build_dataset(command, df)
        return dataset

    def _download(self) -> dict:
        """Downloads the data from the site"""

        os.makedirs(os.path.dirname(self._config["destination"]), exist_ok=True)

        # Override  start time since we're measuring internet speed
        self._start = datetime.now()

        with requests.get(self._config["source"], stream=True) as response:
            self._status_code = response.status_code
            if response.status_code == 200:
                response_footer = self._process_response(response)
            else:
                raise ConnectionError(response)
        return response_footer

    def _process_response(self, response: requests.Response) -> dict:
        """Processes an HTTP Response

        Args:
            response (requests.Response): HTTP Response object.
        """

        response_header = self._setup_process_response(response)

        # Set chunk_size, defaults to 10Mb
        chunk_size = 1024 * 1024 * self._chunk_size

        with open(self._config["destination"], "wb") as fd:
            i = 0
            downloaded = 0
            for chunk in response.iter_content(chunk_size=chunk_size):

                # Download the chunk and capture transmission metrics.
                downloaded = self._download_chunk(chunk, fd, response, i, downloaded)

                i += 1

        response_footer = self._teardown_process_response(response_header, i, downloaded)
        return response_footer

    def _setup_process_response(self, response: requests.Response) -> None:
        """Grab some metadata from the content header.

        Args:
            logger (logging): Voice, every method must have a voice, right?
            response (requests.Response): HTTP Response object.
        """
        self._logger.info(
            "\tDownloading {} Mb\n".format(str(round(int(response.headers.get("Content-Length", 0)) / (1024 * 1024), 2)))
        )
        # Grab response header information
        d = OrderedDict()
        d["Status Code"] = response.status_code
        d["Content Type"] = response.headers.get("Content-Type", "")
        d["Last Modified"] = response.headers.get("Last-Modified", "")
        d["Content Length (Mb)"] = round(int(response.headers.get("Content-Length", 0)) / (1024 * 1024), 3)
        return d

    def _download_chunk(self, chunk, fd, response: requests.Response, i: int = 0, downloaded: int = 0) -> dict:
        """Downloads a chunk of data from source site and produces some metrics.

        Args:
            chunk (requests.chunk) Chunk of data downloaded from the source site.
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
        duration = datetime.now() - self._start

        # Duration may be zero for first few iterations. In case of zero division error, we'll set
        # average speed to 0
        try:
            pct_downloaded = downloaded / int(response.headers.get("Content-Length", 0)) * 100
            average_mbps = int((downloaded / duration.total_seconds()) / (1024 * 1024))
        except ZeroDivisionError as e:
            average_mbps = 0

        # Every 10th chunk, we'll report progress
        if (chunk_number) % 10 == 0:
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
        return downloaded

    def _teardown_process_response(self, response_header, i: int, downloaded: int) -> dict:
        """Deletes temporary files, and updates the response header with additional information

        Args:
            response_header (dict): HTTP Response object.
            downloaded (int): Total bytes downloaded.
        """
        duration = datetime.now() - self._start
        Mb = os.path.getsize(self._config["filepath_raw"]) / (1024 * 1024)
        response_header["Chunk Size (Mb)"] = self._chunk_size
        response_header["Chunks Downloaded"] = i + 1
        response_header["Downloaded (Mb)"] = round(downloaded / (1024 * 1024), 3)
        response_header["File Size (Mb)"] = round(Mb, 3)
        response_header["Mbps"] = round(Mb / duration.total_seconds(), 3)

        return response_header

    def _decompress(self) -> None:
        data = tarfile.open(self._config["destination"])
        with tempfile.TemporaryDirectory() as tempdirname:
            data.extractall(tempdirname)
            tempfilepath = os.path.join(tempdirname, self._config["filepath_extract"])
            shutil.copyfile(tempfilepath, self._config["filepath_raw"])

    def _process_non_response(self, response: requests.Response) -> dict:
        """In case non 200 response from HTTP server.

        Args:
            response (requests.Response): HTTP Response object.

        """
        response_footer = {}
        response_footer["HTTP Response Status Code"] = response.status_code
        return response_footer

    @property
    def summary(self) -> dict:
        return self._summary


# ------------------------------------------------------------------------------------------------------------------------ #
class TransformMissing(Task):
    """Replaces designated missing values (-1) with NaN

    Args:
        value (str,list): The value or values to replace
    Returns:
        DataFrame with replaced values

    """

    def __init__(self, value: Union[str, list]) -> None:
        super(TransformMissing, self).__init__()
        self._value = value
        self._before = None
        self._after = None

    def _run(self, command: PipelineCommand, data: Dataset) -> Dataset:

        data.set_task_data(self)

        # Check Missing before the operation
        self._before = self._df.isna().sum().sum()

        # Transform the missing values
        self._df = self._df.replace(self._value, np.nan)

        # Check missing after
        self._after = self._df.isna().sum().sum()

        # Create Dataset object
        dataset = self._build_dataset(command, self._df)
        return dataset

        # Summarize
        self._summary = self._response_normal()
        return dataset

    def _response_normal(self) -> dict:
        rows, columns = self._data.shape
        cells = rows * columns
        response_footer = OrderedDict()
        response_footer["Rows"] = rows
        response_footer["Columns"] = columns
        response_footer["Cells"] = cells
        response_footer["Missing Before"] = self._before
        response_footer["Missing % Before"] = round(self._before / cells * 100, 2)
        response_footer["Missing After"] = self._after
        response_footer["Missing % After"] = round(self._after / cells * 100, 2)
        response_footer["% Change"] = round(
            (response_footer["Missing After"] - response_footer["Missing Before"]) / response_footer["Missing Before"] * 100,
            2,
        )

        self._status_code = "200"

        return response_footer

    @property
    def summary(self) -> dict:
        return self._summary


# ------------------------------------------------------------------------------------------------------------------------ #
class LoadDataset:
    """Loads the dataset into the current workspace."""

    def _run(self, command: PipelineCommand, data: Dataset) -> Dataset:
        """Add the dataset to the current workspace

        Args:
            command (PipelineCommand): Parameters for the Pipeline
            data (Dataset): Dataset created by the previous step.
        """
        self._workspace = Workspace(command.workspace)
        filepath = self._workspace.add_dataset(data)
        self._summary = OrderedDict()
        self._summary["Status Code"] = 200
        self._summary["AID"] = data.aid
        self._summary["Workspace"] = data.workspace
        self._summary["Dataset Name"] = data.name
        self._summary["Stage"] = data.stage
        self._summary["name"] = data.name
        self._summary["filepath"] = filepath

    @property
    def summary(self) -> dict:
        return self._summary
