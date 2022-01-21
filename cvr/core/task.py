#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ======================================================================================================================== #
# Project  : Conversion Rate Prediction (CVR)                                                                              #
# Version  : 0.1.0                                                                                                         #
# File     : \task.py                                                                                                      #
# Language : Python 3.10.1                                                                                                 #
# ------------------------------------------------------------------------------------------------------------------------ #
# Author   : John James                                                                                                    #
# Email    : john.james.ai.studio@gmail.com                                                                                #
# URL      : https://github.com/john-james-ai/cvr                                                                          #
# ------------------------------------------------------------------------------------------------------------------------ #
# Created  : Wednesday, January 19th 2022, 5:34:06 pm                                                                      #
# Modified : Friday, January 21st 2022, 12:52:32 am                                                                        #
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

warnings.filterwarnings("ignore")

STATUS_CODES = {
    "102": "Processing",
    "200": "OK",
    "202": "Accepted",
    "215": "Complete - Not Executed: Output Data Already Exists",
    "404": "Bad Request",
    "500": "Internal Server Error. See HTTP Response Code",
}

# ------------------------------------------------------------------------------------------------------------------------ #
class Task(ABC):
    """Defines interface for task classes."""

    def __init__(self, **kwargs) -> None:
        self._name = self.__class__.__name__
        self._start = None
        self._end = None
        self._duration = None
        self._logger = None
        self._data = None
        self._status_code = "202"
        self._status_text = STATUS_CODES[self._status_code]
        self._status = self._status_code + ": " + self._status_text
        self._summary = OrderedDict()

    @property
    def name(self) -> None:
        return self._name

    @property
    def output(self) -> None:
        return self._output

    @property
    def start(self) -> datetime:
        return self._start

    @property
    def end(self) -> datetime:
        return self._end

    @property
    def status(self) -> str:
        self._status_text = STATUS_CODES[self._status_code]
        self._status = self._status_code + ": " + self._status_text
        return self._status

    @property
    def duration(self) -> datetime:
        return self._duration

    def _setup(self, logger: logging) -> None:
        self._start = datetime.now()
        self._status_code = "102"
        logger.info("Started Task: {}".format(self._name))

    def _teardown(self, logger: logging) -> None:
        self._end = datetime.now()
        self._duration = self._end - self._start
        self._summary["Start"] = self._start
        self._summary["End"] = self._end
        self._summary["Duration"] = self._duration
        self._summary["Status"] = self.status
        self._summary["Status Date"] = date.today()
        self._summary["Status Time"] = self._end.strftime("%H:%M:%S")
        logger.info("Ended Task: {}. Status: {}".format(self._name, self.status))

    def run(
        self, logger: logging, data: Union[str, pd.DataFrame, dict] = None, force=False
    ) -> Union[str, pd.DataFrame, dict]:
        """Runs the task through delegation to a private member on the subclass

        Args:
            logger (logging): A logging object
            data (str, pd.DataFrame, dict): Optional. Input data

        """
        self._setup(logger)
        self._output = self._run(logger=logger, data=data, force=force)
        self._teardown(logger)
        return self._output

    def _abort_existence(self) -> None:
        """Status to report when task not executed because output data already exists."""
        self._status_code = "215"

    @property
    @abstractmethod
    def summary(self) -> None:
        pass

    @abstractmethod
    def _run(
        self, logger: logging, data: Union[str, pd.DataFrame, dict] = None, force=False
    ) -> Union[str, pd.DataFrame, dict]:
        pass


# ------------------------------------------------------------------------------------------------------------------------ #
class Download(Task):
    """Downloads a file to a specified file.

    Args:
        source (str): The URL
        destination (str): The filepath to the downloaded file.
        chunk_size(int): Chunk size in Mb

    """

    def __init__(self, source: str, destination: str, chunk_size=10) -> None:
        super(Download, self).__init__()
        self._source = source
        self._destination = destination
        self._chunk_size = chunk_size

        self._chunk_metrics = OrderedDict()

    @property
    def chunk_metrics(self) -> dict:
        return self._chunk_metrics

    def _run(
        self, logger: logging, data: Union[str, pd.DataFrame, dict] = None, force=False
    ) -> Union[str, pd.DataFrame, dict]:
        """Downloads the data if it doesn't already exist or if force is True."""

        if not os.path.exists(self._destination) or force is True:
            self._summary = self._download(logger)
        else:
            self._summary = self._abort_existence()
            self._abort_existence()

    def _download(self, logger: logging) -> dict:
        """Downloads the data from the site.

        Args:
            logger (logging): Voice, every method must have a voice, right?
        """

        os.makedirs(os.path.dirname(self._destination), exist_ok=True)

        # Override  start time since we're measuring internet speed
        self._start = datetime.now()

        with requests.get(self._source, stream=True) as response:
            if response.status_code == 200:
                response_footer = self._process_response(logger, response)
                self._status_code = "200"
            else:
                response_footer = self._process_non_response(logger, response)
                self._status_code = "500"

        return response_footer

    def _process_response(self, logger: logging, response: requests.Response) -> dict:
        """Processes an HTTP Response

        Args:
            logger (logging): Voice, every method must have a voice, right?
            response (requests.Response): HTTP Response object.
        """

        response_header = self._setup_process_response(logger, response)

        # Set chunk_size, defaults to 10Mb
        chunk_size = 1024 * 1024 * self._chunk_size

        with open(self._destination, "wb") as fd:
            i = 0
            downloaded = 0
            for chunk in response.iter_content(chunk_size=chunk_size):

                # Download the chunk and capture transmission metrics.
                downloaded = self._download_chunk(chunk, fd, logger, response, i, downloaded)

                i += 1

        response_footer = self._teardown_process_response(logger, response_header, i, downloaded)
        return response_footer

    def _setup_process_response(self, logger: logging, response: requests.Response) -> None:
        """Grab some metadata from the content header.

        Args:
            logger (logging): Voice, every method must have a voice, right?
            response (requests.Response): HTTP Response object.
        """
        logger.info(
            "\tDownloading {} Mb\n".format(str(round(int(response.headers.get("Content-Length", 0)) / (1024 * 1024), 2)))
        )
        # Grab response header information
        d = OrderedDict()
        d["Status Code"] = response.status_code
        d["Content Type"] = response.headers.get("Content-Type", "")
        d["Last Modified"] = response.headers.get("Last-Modified", "")
        d["Content Length (Mb)"] = round(int(response.headers.get("Content-Length", 0)) / (1024 * 1024), 3)
        return d

    def _download_chunk(
        self, chunk, fd, logger: logging, response: requests.Response, i: int = 0, downloaded: int = 0
    ) -> dict:
        """Downloads a chunk of data from source site and produces some metrics.

        Args:
            chunk (requests.chunk) Chunk of data downloaded from the source site.
            fd (file descriptor): Pointer to the file.
            logger (logging): Voice, every method must have a voice, right?
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
            logger.info(
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

    def _teardown_process_response(self, logger: logging, response_header, i: int, downloaded: int) -> dict:
        """Deletes temporary files, and updates the response header with additional information

        Args:
            logger (logging): Voice, every method must have a voice, right?
            response (requests.Response): HTTP Response object.
            downloaded (int): Total bytes downloaded.
        """
        duration = datetime.now() - self._start
        Mb = os.path.getsize(self._destination) / (1024 * 1024)
        response_header["Chunk Size (Mb)"] = self._chunk_size
        response_header["Chunks Downloaded"] = i + 1
        response_header["Downloaded (Mb)"] = round(downloaded / (1024 * 1024), 3)
        response_header["File Size (Mb)"] = round(Mb, 3)
        response_header["Mbps"] = round(Mb / duration.total_seconds(), 3)

        return response_header

    def _process_non_response(self, logger: logging, response: requests.Response) -> dict:
        """In case non 200 response from HTTP server.

        Args:
            logger (logging): Voice, every method must have a voice, right?
            response (requests.Response): HTTP Response object.

        """
        response_footer = {}
        response_footer["HTTP Response Status Code"] = response.status_code
        return response_footer

    def _abort_existence(self) -> dict:
        """In case of existing data and force is false. Report existing data.

        Args:
            logger (logging): Voice, every method must have a voice, right?

        """
        Mb = os.path.getsize(self._destination) / (1024 * 1024)
        response_footer = OrderedDict()
        response_footer["Destination"] = self._destination
        response_footer["File Size (Mb)"] = round(Mb, 3)
        self._status_code = "215"
        return response_footer

    @property
    def summary(self) -> dict:
        return self._summary


# ------------------------------------------------------------------------------------------------------------------------ #
class Decompress(Task):
    """Decompresses and extracts a gz file to a specified directory.

    Args:
        source (str): The location of the gz file
        destination (str): The directory to which the data should be extracted.

    """

    def __init__(self, source: str, destination: str) -> None:
        super(Decompress, self).__init__()
        self._source = source
        self._destination = destination

    @property
    def summary(self) -> dict:
        self._summary

    def _run(
        self, logger: logging, data: Union[str, pd.DataFrame, dict] = None, force=False
    ) -> Union[str, pd.DataFrame, dict]:
        """Decompresses a file to the designated destination directory"""

        if not os.path.exists(self._destination) or force:
            data = tarfile.open(self._source)
            data.extractall(os.path.dirname(os.path.dirname(self._destination)))
            self._summary = self._response_normal()
        else:
            self._summary = self._abort_existence()

    def _response_normal(self) -> dict:
        response_footer = OrderedDict()
        if os.path.exists(self._source):
            response_footer["Source"] = self._source
            response_footer["Compressed Size"] = os.path.getsize(self._source)
        if os.path.exists(self._destination):
            response_footer["Destination"] = self._destination
            response_footer["Expanded Size"] = os.path.getsize(self._destination)

        self._status_code = "200"

        return response_footer

    def _abort_existence(self) -> dict:
        """In case of existing data and force is false. Report existing data.

        Args:
            logger (logging): Voice, every method must have a voice, right?

        """
        response_footer = self._response_normal()
        self._status_code = "215"
        return response_footer

    @property
    def summary(self) -> dict:
        return self._summary


# ------------------------------------------------------------------------------------------------------------------------ #
class Copy(Task):
    """Copies a file from source to destination.

    Args:
        source (str): Source filepath
        destination (str): Destination filepath

    """

    def __init__(self, source: str, destination: str) -> None:
        super(Copy, self).__init__()
        self._source = source
        self._destination = destination

    def _run(
        self, logger: logging, data: Union[str, pd.DataFrame, dict] = None, force=False
    ) -> Union[str, pd.DataFrame, dict]:
        """Copies a file from source to destination."""

        if not os.path.exists(self._destination) or force:
            os.makedirs(os.path.dirname(self._destination), exist_ok=True)
            shutil.copyfile(self._source, self._destination)
            self._summary = self._response_normal()
        else:
            self._summary = self._abort_existence()
        return {}

    def _response_normal(self) -> dict:
        response_footer = {}
        if os.path.exists(self._source):
            response_footer["Source"] = self._source
        if os.path.exists(self._destination):
            response_footer["Destination"] = self._destination

        self._status_code = "200"

        return response_footer

    def _abort_existence(self) -> dict:
        """In case of existing data and force is false. Report existing data.

        Args:
            logger (logging): Voice, every method must have a voice, right?

        """
        response_footer = self._response_normal()
        self._status_code = "215"
        return response_footer

    @property
    def summary(self) -> dict:
        return self._summary


# ------------------------------------------------------------------------------------------------------------------------ #
class LoadTSV(Task):
    """Loads a tab-delimited file into a DataFrame.

    Args:
        source (str): The path to the file to be loaded
        columns (list): A list of column names
        dtypes (dict): Mapping of column names to datatypes

    Returns:
        DataFrame

    """

    def __init__(self, source: str, columns: list = None, dtypes: dict = None) -> None:
        super(LoadTSV, self).__init__()
        self._source = source
        self._columns = columns
        self._dtypes = dtypes

    def _run(
        self, logger: logging, data: Union[str, pd.DataFrame, dict] = None, force=False
    ) -> Union[str, pd.DataFrame, dict]:

        self._data = pd.read_csv(
            self._source,
            sep="\t",
            names=self._columns,
            low_memory=False,
            index_col=False,
            dtype=self._dtypes,
        )
        self._summary = self._response_normal()
        return self._data

    def _response_normal(self) -> dict:
        dtypes = self._data.dtypes.astype(str).value_counts().to_frame().to_dict()[0]

        response_footer = OrderedDict()
        response_footer["Rows"] = self._data.shape[0]
        response_footer["Columns"] = self._data.shape[1]
        response_footer["Size"] = self._data.memory_usage(deep=True).sum()
        response_footer["Missing"] = self._data.isna().sum().sum()
        response_footer["Rows w/ Missing"] = self._data.isna().any(axis=1).sum()
        response_footer["Columns w/ Missing"] = self._data.isna().any(axis=0).sum()
        response_footer.update(dtypes)

        self._status_code = "200"

        return response_footer

    @property
    def summary(self) -> dict:
        return self._summary


# ------------------------------------------------------------------------------------------------------------------------ #
class SetNA(Task):
    """Replaces designated values with NaN

    Args:
        value (str,list): The value or values to replace
    Returns:
        DataFrame with replaced values

    """

    def __init__(self, value: Union[str, list]) -> None:
        super(SetNA, self).__init__()
        self._value = value
        self._before = None
        self._after = None

    def _run(
        self, logger: logging, data: Union[str, pd.DataFrame, dict] = None, force=False
    ) -> Union[str, pd.DataFrame, dict]:

        self._data = data
        self._before = self._data.isna().sum().sum()

        self._data = self._data.replace(self._value, np.nan)

        self._after = self._data.isna().sum().sum()
        self._summary = self._response_normal()
        return self._data

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
class LoadPKLDataFrame(Task):
    """Pickles the data to the destination

    Args:
        source (str): The filepath to the pickled file.
    """

    def __init__(self, source: str) -> None:
        super(LoadPKLDataFrame, self).__init__()
        self._source = source

    def _run(
        self, logger: logging, data: Union[str, pd.DataFrame, dict] = None, force=False
    ) -> Union[str, pd.DataFrame, dict]:

        self._data = pd.read_pickle(self._source)

        self._summary = self._response_normal()

        return self._data

    def _response_normal(self) -> dict:
        dtypes = self._data.dtypes.astype(str).value_counts().to_frame().to_dict()[0]

        response_footer = OrderedDict()
        response_footer["Rows"] = self._data.shape[0]
        response_footer["Columns"] = self._data.shape[1]
        response_footer["Size"] = self._data.memory_used(deep=True)
        response_footer["Missing"] = self._data.isna().sum().sum()
        response_footer["Rows w/ Missing"] = self._data.isna().any(axis=1).sum()
        response_footer["Columns w/ Missing"] = self._data.isna().any(axis=0).sum()
        print(dtypes)
        response_footer.update(dtypes)

        self._status_code = "200"

        return response_footer

    @property
    def summary(self) -> dict:
        return self._summary


# ------------------------------------------------------------------------------------------------------------------------ #
class SavePKLDataFrame(Task):
    """Pickles the data to the destination

    Args:
        destination (str): The filepath to which the data should be pickled.
    """

    def __init__(self, destination: str) -> None:
        super(SavePKLDataFrame, self).__init__()
        self._destination = destination

    def _run(
        self, logger: logging, data: Union[str, pd.DataFrame, dict] = None, force=False
    ) -> Union[str, pd.DataFrame, dict]:

        if not os.path.exists(self._destination):
            os.makedirs(os.path.dirname(self._destination), exist_ok=True)
        else:
            os.remove(self._destination)
        data.to_pickle(path=self._destination)
        self._summary = self._response_normal()

    def _response_normal(self) -> dict:
        response_footer = OrderedDict()
        response_footer["Destination"] = self._destination
        response_footer["File Size (Mb)"] = round(int(os.path.getsize(self._destination)) / (1024 * 1024), 2)

        self._status_code = "200"

        return response_footer

    @property
    def summary(self) -> dict:
        return self._summary
