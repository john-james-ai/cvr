#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ======================================================================================================================== #
# Project  : Conversion Rate Prediction (CVR)                                                                              #
# Version  : 0.1.0                                                                                                         #
# File     : \source.py                                                                                                    #
# Language : Python 3.10.1                                                                                                 #
# ------------------------------------------------------------------------------------------------------------------------ #
# Author   : John James                                                                                                    #
# Email    : john.james.ai.studio@gmail.com                                                                                #
# URL      : https://github.com/john-james-ai/cvr                                                                          #
# ------------------------------------------------------------------------------------------------------------------------ #
# Created  : Monday, December 27th 2021, 5:54:01 pm                                                                        #
# Modified : Thursday, January 20th 2022, 1:36:32 pm                                                                       #
# Modifier : John James (john.james.ai.studio@gmail.com)                                                                   #
# ------------------------------------------------------------------------------------------------------------------------ #
# License  : BSD 3-clause "New" or "Revised" License                                                                       #
# Copyright: (c) 2022 Bryant St. Labs                                                                                      #
# ======================================================================================================================== #
"""Defines Tasks for the extraction, transformation and loading of source data."""
from abc import ABC, abstractmethod
import os
import logging
import pandas as pd
import numpy as np
import requests
import tarfile
import shutil
from datetime import datetime
from typing import Union
import warnings

warnings.filterwarnings("ignore")
from cvr.utils.config import CriteoConfig
from cvr.core.pipeline import PiplineCommand, PipelineBuilder, Pipeline
from cvr.data import criteo_dtypes, criteo_columns
from cvr.utils.config import DataSourceConfig, ProjectConfig
from cvr.core.task import Task

# ------------------------------------------------------------------------------------------------------------------------ #
class CriteoETLCommand:
    """Command object for the ExtractWebGZ"""

    def __init__(self, config: CriteoConfig) -> None:
        criteo = config.get_criteo()
        self._name = criteo["name"]
        self._url = criteo["url"]
        self._filepath_compressed = criteo["filepath_compressed"]
        self._filepath_decompressed = criteo["filepath_decompressed"]
        self._filepath_raw = criteo["filepath_raw"]
        self._filepath_staged = criteo["filepath_staged"]
        self._sep = criteo["sep"]
        self._missing = criteo["missing"]
        self._force = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def url(self) -> str:
        return self._url

    @property
    def filename_compressed(self) -> str:
        return self._filename_compressed

    @property
    def filename_decompressed(self) -> str:
        return self._filename_decompressed

    @property
    def filename_raw(self) -> str:
        return self._filename_raw

    @property
    def filename_staged(self) -> str:
        return self._filename_staged

    @property
    def sep(self) -> str:
        return self._sep

    @property
    def missing(self) -> str:
        return self._missing

    @property
    def force(self) -> bool:
        return self._force

    @force.setter
    def force(self, force: bool) -> None:
        self._force = force


# ------------------------------------------------------------------------------------------------------------------------ #
class CriteoExtract(Task):
    """Defines the Criteo extract transform load pipeline."""

    def __init__(self, command: CriteoETLCommand) -> None:
        self._command = command

    def _run(self, logger: logging) -> None:
        self._logger = logger
        self._logger.info("\t{} started.".format(__class__.__name__))

        self.download()
        self.decompress()
        self.save()

        self._end = datetime.now()
        self._extract_duration = self._end - self._start
        self._logger.info("\t{} Completed. Duration: {}".format(__class__.__name__, self._extract_duration))
        return self

    def download(self) -> None:
        start = datetime.now()
        self._logger.info("\t\tDownload process began at {}.".format(start))

        self._download()

        end = datetime.now()
        self._download_duration = end - start
        self._logger.info("\t\tDownload completed at {}. Duration: {}".format(end, self._download_duration))
        return self

    def _download(self) -> None:
        """Downloads Criteo data to a local directory."""

        if not os.path.exists(self._command.filepath_compressed):
            os.makedirs(os.path.dirname(self._command.filepath_compressed), exist_ok=True)

            response = requests.get(self._command.url, stream=True)
            if response.status_code == 200:
                with open(self._command.filepath_compressed, "wb") as f:
                    f.write(response.raw.read())

    def decompress(self) -> None:
        start = datetime.now()
        self._logger.info("\t\tDecompression started at {}.".format(start))

        self._decompress()

        end = datetime.now()
        self._decompress_duration = end - start
        self._logger.info("\t\tDecompression completed at {}. Duration: {}".format(end, self._decompress_duration))

    def _decompress(self) -> None:

        if not os.path.exists(self._command.filepath_decompressed) or self._force:
            data = tarfile.open(self._command.filepath_compressed)
            data.extractall(os.path.dirname(os.path.dirname(self._command.filepath_decompressed)))

    def save(self) -> None:
        start = datetime.now()
        self._logger.info("\t\tSave started at {}.".format(start))

        self._save()

        end = datetime.now()
        self._save_duration = end - start
        self._logger.info("\t\tSave completed at {}. Duration: {}".format(end, self._save_duration))

    def _save(self) -> None:
        """Copies the data to the raw data directory."""

        if not os.path.exists(self._command.filepath_raw) or self._force:
            os.makedirs(os.path.dirname(self._command.filepath_raw), exist_ok=True)
            shutil.copyfile(self._command.filepath_decompressed, self._command.filepath_raw)

    def transform(self) -> None:
        start = datetime.now()
        self._logger.info("\tTransform started at {}.".format(start))

        self._transform()

        end = datetime.now()
        self._transform_duration = end - start
        self._logger.info("\tTransform completed at {}. Duration: {}".format(end, self._transform_duration))

    def load(self) -> None:
        start = datetime.now()
        self._logger.info("\tLoad started at {}.".format(start))

        self._load()

        end = datetime.now()
        self._load_duration = end - start
        self._logger.info("\tLoad completed at {}. Duration: {}".format(end, self._load_duration))


# ------------------------------------------------------------------------------------------------------------------------ #


class CriteoETL(ETL):
    """Acquires Data from a web source. Stores the raw, transforms the data and stages it for processing.

    Args:
        config (DataSourceConfig): The configuration object for the data source.
        source (str): The name for the specific datasource
        force (bool): If true, the dataset will be downloaded and existing files will be overwritten.

    """

    def __init__(
        self,
        datasource_config: DataSourceConfig,
        project_config: ProjectConfig,
        source: str = "criteo",
        force: bool = False,
        verbose=False,
    ) -> None:
        super(CriteoETL, self).__init__(datasource_config, project_config, source, force, verbose)
        self.config()

    def config(self) -> None:
        """Sets configuration parameters."""
        project_config = self._project_config.get_config()
        datasource_config = self._datasource_config.get_config(source=self._source)

        self._name = datasource_config["name"]
        self._url = datasource_config["url"]
        self._filepath_external = os.path.join(project_config["external_data_dir"], datasource_config["filename"])
        self._filepath_decompressed = os.path.join(project_config["external_data_dir"], datasource_config["decompressed"])
        self._filepath_raw = project_config["raw_data_filepath"]
        self._filepath_staged = project_config["staged_data_filepath"]

    def _download(self) -> None:
        """Extracts and downloads Criteo data to a local directory."""

        if not os.path.exists(self._filepath_external):
            os.makedirs(os.path.dirname(self._filepath_external), exist_ok=True)

            response = requests.get(self._url, stream=True)
            if response.status_code == 200:
                with open(self._filepath_external, "wb") as f:
                    f.write(response.raw.read())

    def _decompress(self) -> None:

        if not os.path.exists(self._filepath_decompressed) or self._force:
            data = tarfile.open(self._filepath_external)
            data.extractall(os.path.dirname(self._filepath_external))

    def _save(self) -> None:
        """Copies the data to the raw data directory."""

        if not os.path.exists(self._filepath_raw) or self._force:
            os.makedirs(os.path.dirname(self._filepath_raw), exist_ok=True)
            shutil.copyfile(self._filepath_decompressed, self._filepath_raw)

    def _transform(self) -> None:

        if not os.path.exists(self._filepath_staged) or self._force:
            self._df = pd.read_csv(
                self._filepath_raw,
                sep="\t",
                names=criteo_columns,
                low_memory=False,
                index_col=False,
                dtype=criteo_dtypes,
            )

            self._df = self._df.replace(-1, np.nan)
            self._df = self._df.replace("-1", np.nan)

    def _load(self) -> None:
        if not os.path.exists(self._filepath_staged) or self._force:
            self._df.to_pickle(path=self._filepath_staged)

    @property
    def summary(self) -> None:
        d = {
            "URL": self._url,
            "Filepath": self._filepath_raw,
            "Size MB": os.path.getsize(self._filepath_decompressed),
            "Started": self._start,
            "Completed": self._end,
            "Duration": self._extract_duration,
            "Raw Filepath": self._filepath_raw,
            "Staged Filepath": self._filepath_staged,
        }
        self._printer.print_title(self._name, "Source Summary")
        self._printer.print_dictionary(d)
