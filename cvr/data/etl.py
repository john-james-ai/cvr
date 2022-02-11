#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project  : Deep Learning for Conversion Rate Prediction (CVR)                                    #
# Version  : 0.1.0                                                                                 #
# File     : \etl.py                                                                               #
# Language : Python 3.7.12                                                                         #
# ------------------------------------------------------------------------------------------------ #
# Author   : John James                                                                            #
# Email    : john.james.ai.studio@gmail.com                                                        #
# URL      : https://github.com/john-james-ai/cvr                                                  #
# ------------------------------------------------------------------------------------------------ #
# Created  : Friday, January 21st 2022, 1:39:53 pm                                                 #
# Modified : Thursday, February 10th 2022, 8:35:00 pm                                              #
# Modifier : John James (john.james.ai.studio@gmail.com)                                           #
# ------------------------------------------------------------------------------------------------ #
# License  : BSD 3-clause "New" or "Revised" License                                               #
# Copyright: (c) 2022 Bryant St. Labs                                                              #
# ================================================================================================ #

"""IO Related Tasks"""
import os
import math
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import inspect
import requests
import tarfile
import shutil
from tqdm import tqdm
import tempfile
from dateutil.parser import parse as parsedate

from cvr.core.asset import AssetPassport
from cvr.core.task import Task, TaskResponse, TaskResult
from cvr.utils.printing import Printer
from cvr.utils.sampling import sample_file

# ---------------------------------------------------------------------------- #
MEGABYTE = 1024 * 1024
GIGABYTE = 1024 * MEGABYTE
# ============================================================================ #
#                              EXTRACT TASK                                    #
# ============================================================================ #

# ---------------------------------------------------------------------------- #
#                         EXTRACT TASK RESPONSE                                #
# ---------------------------------------------------------------------------- #


@dataclass
class ExtractResponse(TaskResponse):
    status_code: str = field(default=100)
    content_type: str = field(default=None)
    last_modified: str = field(default=None)
    content_length: str = field(default=0)
    chunk_size: int = field(default=0)
    chunks_downloaded: int = field(default=0)
    download_start: datetime = field(default=None)
    download_end: datetime = field(default=None)
    download_duration: timedelta = field(default=0)
    downloaded_mb: int = field(default=0)
    downloaded_mbps: int = field(default=0)
    filesize_compressed: int = field(default=0)
    filesize: int = field(default=0)
    filepath: str = field(default=None)

    def __post_init__(self) -> None:
        self._printer = Printer()

    def parse_response(self, response: requests.Response) -> None:
        """Extracts data from the requests.Response object.

        Args:
            response (requests.Response): Server responnse
        """
        self.status_code = response.status_code
        self.content_type = response.headers.get("Content-Type", "")
        self.last_modified = (
            response.headers.get("Last-Modified", "")
            if response.headers.get("Last-Modified", "") == ""
            else parsedate(response.headers.get("Last-Modified", ""))
        )
        self.content_length = round(int(response.headers.get("Content-Length", 0)) / MEGABYTE, 3)

    def todict(self) -> None:
        d = {
            "Status Code": self.status_code,
            "Content Type": self.content_type,
            "Last Modified": self.last_modified,
            "Content Length (Mb)": self.content_length,
            "Download Start": self.download_start,
            "Download End": self.download_end,
            "Download Duration": self.download_duration,
            "Downloaded (Mb)": self.downloaded_mb,
            "Downloaded (Mbps)": self.downloaded_mbps,
            "Chunks Downloaded": self.chunks_downloaded,
            "Chunk Size (Mb)": self.chunk_size,
            "Filesize (Compressed Mb)": self.filesize_compressed,
            "Filesize (Mb)": self.filesize,
            "Filepath": self.filepath,
        }
        return d

    def print(self) -> None:
        d = self.todict()
        self._printer.print_title("Extract Task Response")
        self._printer.print_dictionary(d)


# ---------------------------------------------------------------------------- #
#                              EXTRACT TASK                                    #
# ---------------------------------------------------------------------------- #


class Extract(Task):
    """Downloads the data from the source website."""

    def __init__(
        self,
        passport: AssetPassport,
        url: str,
        download_filepath: str,
        extract_filepath: str,
        destination: str,
        sample_size: int = None,
        chunk_size=20,
        n_groups=20,
    ) -> None:
        super(Extract, self).__init__(passport=passport)
        self._url = url
        self._download_filepath = download_filepath
        self._extract_filepath = extract_filepath  # The compressed archive file
        self._destination = destination  # The raw data filepath
        self._sample_size = sample_size

        self._chunk_size = chunk_size
        self._n_groups = n_groups

    @property
    def passport(self) -> AssetPassport:
        return self._passport

    @property
    def response(self) -> TaskResponse:
        return self._response

    @property
    def result(self) -> TaskResult:
        return self._result

    def _setup(self) -> None:
        self._result = TaskResult()
        self._response = ExtractResponse()
        self._response.begin()

        # Change filepaths to point to the current studio
        self._destination = os.path.join(self._config.directory, self._destination)
        self._download_filepath = os.path.join(self._config.directory, self._download_filepath)

    def run(self, data: pd.DataFrame = None) -> None:
        """Downloads the data if it doesn't exist or if force is True

        Args:
            data (pd.DataFrame) Input data

        """
        self.setup()
        self._logger.debug("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        self._process_download()
        self._process_decompression()
        self.teardown()

        self._logger.debug(
            "\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3])
        )

    def _process_download(self) -> None:
        """Orchestrates the download process"""
        self._logger.debug("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        # Make the server connection in context
        with requests.get(self._url, stream=True) as response:
            self._response.status_code = response.status_code
            if response.status_code == 200:
                self._process_response(response=response)
            else:
                raise ConnectionError(response)

        self._logger.debug(
            "\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3])
        )

    def _process_response(self, response: requests.Response) -> None:
        """Orchestrates the extract process.

        Checks the response variable and evaluates the last modified date. The download proceeds if
        there is no local version of the data or the local data is out of date vis-a-vis the last
        modified date from the server.

        Args:
            response (requests.Response) Requests server response
        """
        self._logger.debug("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        self._response.parse_response(response)

        # Determine if download should proceed based upon the availability and currency of local
        # data vis-a-vis the server version and execute accordingly
        if self._proceed_download():
            self._setup_download(response)
            self._download(response)
            self._teardown_download()
        else:
            self._skip_download()

        self._logger.debug(
            "\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3])
        )

    def _proceed_download(self) -> bool:
        """Returns False if local data are up-to-date and force is not selected."""
        return not os.path.exists(self._download_filepath)

    def _skip_download(self) -> None:
        self._result.executed = False
        self._result.comment = "Download skipped. Local data are current."

    def _setup_download(self, response: requests.Response) -> None:
        """Prepares member variables used while downloading."""
        # Set chunk_size, defaults to 10Mb
        self._chunk_size_bytes = MEGABYTE * self._chunk_size

        # Get size of content
        size_in_bytes = int(response.headers.get("content-length", 0))

        # Get the number of chunks
        n_chunks = math.ceil(size_in_bytes / self._chunk_size_bytes)

        # Set number of iterations in each group of chunks for reporting
        # purposes
        self._chunk_group_size = self._group_size(n_chunks)

        # Setup data for progress bar
        if self._config.progress and self._config.verbose:
            self._progress_bar = tqdm(total=size_in_bytes, unit="iB", unit_scale=True)

        # Create the directory
        os.makedirs(os.path.dirname(self._download_filepath))

        # Start the download clock on the response variable
        self._response.download_start = datetime.now()

    def _download(self, response: requests.Response) -> int:
        """Performs the download operation."""
        self._logger.debug("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        with open(self._download_filepath, "wb") as fd:
            self._chunks_downloaded = 0
            self._bytes_downloaded = 0
            for chunk in response.iter_content(chunk_size=self._chunk_size_bytes):
                if self._config.progress and self._config.verbose:
                    self._progress_bar.update(len(chunk))

                # Download the chunk and capture transmission metrics.
                self._download_chunk(chunk=chunk, fd=fd, response=response)

                self._chunks_downloaded += 1

        self._logger.debug(
            "\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3])
        )

    def _teardown_download(self) -> None:
        """Wraps up download processing."""

        # Shut down progress bar if using it.
        if self._config.progress and self._config.verbose:
            self._progress_bar.close()

        # Update the response variable with metrics
        self._response.download_end = datetime.now()
        self._response.download_duration = (
            self._response.download_end - self._response.download_start
        )

        self._response.chunk_size = self._chunk_size
        self._response.chunks_downloaded = self._chunks_downloaded + 1

        self._response.downloaded_mb = round(self._bytes_downloaded / MEGABYTE, 2)
        self._response.downloaded_mbps = round(
            self._response.downloaded_mb / self._response.download_duration.total_seconds(), 3
        )

        self._response.filesize_compressed = round(
            os.path.getsize(self._download_filepath) / MEGABYTE, 2
        )

        self._result.executed = True
        self._result.passed = self._passed_download()

        # Log summary results

        self._logger.debug(
            "Downloaded {} in {} chunks at {} Mbps".format(
                str(
                    round(
                        self._response.downloaded_mb,
                        2,
                    )
                ),
                str(round(self._response.chunks_downloaded, 2)),
                str(round(self._response.downloaded_mbps, 2)),
            )
        )

    def _passed_download(self) -> bool:
        """Returns True if download was successful or content exists.

        If the file exists and is at least 300 Mb compressed, take a lap."""
        return (
            os.path.exists(self._download_filepath)
            and os.path.getsize(self._download_filepath) > 300
        )

    def _group_size(self, n_chunks: int, idx: int = 0) -> int:
        """Computes the number of chunk download iterations in a group."""

        group_sizes = [1, 5, 10, 20, 50, 100]
        if n_chunks / group_sizes[idx] <= self._n_groups:
            return group_sizes[idx]
        else:
            idx += 1
            return self._group_size(n_chunks, idx)

    def _download_chunk(self, chunk, fd, response: requests.Response) -> None:
        """Downloads a chunk of data from source site and produces some metrics.

        Args:
            chunk (requests.chunk) Chunk of data downloaded.
            fd (file descriptor): Pointer to the file.
            response (requests.Response): HTTP Response object.
            i (int): Current iteration
            downloaded (int): Bytes downloaded
        """

        # Anal about indexes
        chunk_number = self._chunks_downloaded + 1

        # Write the data
        fd.write(chunk)

        # Capture cumulative downloaded and duration
        self._bytes_downloaded += int(len(chunk))
        duration = datetime.now() - self._response.download_start

        # Duration may be zero for first few iterations. In case of zero
        # division error, we'll set average speed to 0
        try:
            pct_downloaded = (
                self._bytes_downloaded / int(response.headers.get("Content-Length", 0)) * 100
            )
            average_mbps = int((self._bytes_downloaded / duration.total_seconds()) / MEGABYTE)
        except ZeroDivisionError:
            pct_downloaded = 0
            average_mbps = 0

        # If progress is disabled and verbose is True, provide progress every
        # 'check_download' iterations
        if not self._config.progress and self._config.verbose:
            if (chunk_number) % self._chunk_group_size == 0:
                self._logger.debug(
                    "\tChunk #{}: {} percent downloaded at {} Mbps".format(
                        str(chunk_number),
                        str(round(pct_downloaded, 2)),
                        str(round(average_mbps, 2)),
                    )
                )

    def _process_decompression(self) -> None:
        """Extracts data from gzip archive if necessary"""
        self._logger.debug("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        if self._proceed_decompression():
            self._decompress()
        else:
            self._skip_decompression()

        self._logger.debug(
            "\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3])
        )

    def _proceed_decompression(self) -> bool:
        """Returns True if data were downloaded or no local copy exists."""
        return not os.path.exists(self._destination) or self._config.force or self._result.executed

    def _skip_decompression(self) -> None:
        """Gracefully skips decompression"""
        self._response.filesize = round(os.path.getsize(self._destination) / MEGABYTE, 2)

    def _decompress(self) -> None:
        """Decompress the designated file and copy to raw directory."""
        data = tarfile.open(self._download_filepath)
        with tempfile.TemporaryDirectory() as tempdirname:
            data.extractall(tempdirname)
            tempfilepath = os.path.join(tempdirname, self._extract_filepath)

            os.makedirs(os.path.dirname(self._destination), exist_ok=True)

            # Sampling Functionality Option
            if self._sample_size is None:
                shutil.copyfile(tempfilepath, self._destination)
            else:
                sample_file(
                    tempfilepath,
                    self._destination,
                    nrows=self._sample_size,
                    sep="\t",
                    random_state=601,
                )

    def _passed_decompression(self) -> bool:
        """Returns True if file exists and is at least 6 Gb in size."""
        return os.path.exists(self._destination) and os.path.getsize(self._destination) > 1 * (
            GIGABYTE
        )

    def _teardown(self) -> None:
        """Updates Task Summary and Response."""
        self._response.stop()
        self._response.filepath = self._destination
        self._response.filesize = round(os.path.getsize(self._destination) / MEGABYTE, 2)
        self._result.passed = self._passed_download() and self._passed_decompression()
        self._result.complete = True
        self._result.completed = datetime.now()

    def passed(self) -> bool:
        return self._passed_download() and self._passed_decompression()


# ---------------------------------------------------------------------------- #
#                                 EXTRACT                                      #
# ---------------------------------------------------------------------------- #
class Transform(Task):
    """Transforms data via data type preprocessing and column naming.

    This task adds column names to the data, converts string data types
    to pandas categories, and replaces missing value indicator with
    NaN encoding.

    Args:
        source (str): The filepath to the raw data
        colnames (list): List of column names
        dtypes (dict): Mapping of column names to data types
        missing_values (list): List of missing indicator values to replace.
        sep (str): Column separator for the data
    """

    def __init__(
        self,
        passport: AssetPassport,
        source: str,
        colnames: list,
        dtypes: dict,
        sep: str,
        missing_values: list,
    ) -> None:
        super(Transform, self).__init__(passport=passport)
        self._source = source
        self._colnames = colnames
        self._dtypes = dtypes
        self._sep = sep
        self._missing_values = missing_values

    @property
    def passport(self) -> AssetPassport:
        return self._passport

    @property
    def response(self) -> TaskResponse:
        return self._response

    @property
    def result(self) -> TaskResult:
        return self._result

    def run(self, data: pd.DataFrame = None) -> pd.DataFrame:
        """Main entry point for the Transformation Task"""

        self.setup()

        self._logger.debug("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        data = pd.read_csv(
            self._source,
            header=None,
            names=self._colnames,
            dtype=self._dtypes,
            sep="\t",
            low_memory=False,
            encoding_errors="ignore",
        )
        data = data.replace(to_replace=self._missing_values, value=np.nan)
        self.teardown()

        self._logger.debug(
            "\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3])
        )
        return data

    def _setup(self) -> None:
        self._result = TaskResult()
        self._response = TaskResponse()
        self._response.begin()

        # Change filepaths to point to the current studio
        self._source = os.path.join(self._config.directory, self._source)

    def _teardown(self) -> None:
        self._response.stop()
        self._result.executed = True
        self._result.passed = True
        self._result.complete = True
        self._result.completed = datetime.now()

    def passed(self) -> bool:
        return True
