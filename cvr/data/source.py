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
# Modified : Tuesday, January 18th 2022, 9:48:47 pm                                                                        #
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
# File     : \source.py                                                                                                    #
# Language : Python 3.10.1                                                                                                 #
# ------------------------------------------------------------------------------------------------------------------------ #
# Author   : John James                                                                                                    #
# Email    : john.james.ai.studio@gmail.com                                                                                #
# URL      : https://github.com/john-james-ai/cvr                                                                          #
# ------------------------------------------------------------------------------------------------------------------------ #
# Created  : Monday, December 27th 2021, 5:54:01 pm                                                                        #
# Modified : Tuesday, January 18th 2022, 9:09:57 pm                                                                        #
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
# File     : \source.py                                                                                                    #
# Language : Python 3.10.1                                                                                                 #
# ------------------------------------------------------------------------------------------------------------------------ #
# Author   : John James                                                                                                    #
# Email    : john.james.ai.studio@gmail.com                                                                                #
# URL      : https://github.com/john-james-ai/cvr                                                                          #
# ------------------------------------------------------------------------------------------------------------------------ #
# Created  : Monday, December 27th 2021, 5:54:01 pm                                                                        #
# Modified : Monday, January 17th 2022, 10:55:08 pm                                                                        #
# Modifier : John James (john.james.ai.studio@gmail.com)                                                                   #
# ------------------------------------------------------------------------------------------------------------------------ #
# License  : BSD 3-clause "New" or "Revised" License                                                                       #
# Copyright: (c) 2022 Bryant St. Labs                                                                                      #
# ======================================================================================================================== #

from abc import ABC, abstractmethod
import os
import logging
import pandas as pd
import numpy as np
import requests
import tarfile
import shutil
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

from cvr.data import raw_dtypes, criteo_columns
from cvr.utils.config import DataSourceConfig, ProjectConfig
from cvr.utils.printing import Printer

# ------------------------------------------------------------------------------------------------------------------------ #
logging.basicConfig(level=logging.ERROR)
# ------------------------------------------------------------------------------------------------------------------------ #
class ETL(ABC):
    """Abstract base class for extract transform load."""

    def __init__(
        self,
        datasource_config: DataSourceConfig,
        project_config: ProjectConfig,
        source: str,
        force: bool = False,
        verbose=False,
    ) -> None:
        # Parameters
        self._datasource_config = datasource_config
        self._project_config = project_config
        self._source = source
        self._force = force
        self._verbose = verbose

        self._filepath_external = None
        self._filepath_decompressed = None
        self._filepath_raw = None
        self._filepath_staged = None

        self._df = None

        self._start = None
        self._end = None

        self._printer = Printer()
        self._config_logging(verbose)

    @abstractmethod
    def config(self) -> None:
        """Sets configuration parameters."""
        pass

    @abstractmethod
    def _download(self) -> None:
        pass

    @abstractmethod
    def _decompress(self) -> None:
        pass

    @abstractmethod
    def _transform(self) -> None:
        pass

    @abstractmethod
    def _load(self) -> None:
        pass

    @abstractmethod
    def _save(self) -> None:
        pass

    def etl(self) -> None:
        """Primary entry point. Method executes entire data pipeline from source to staging."""
        self.extract()
        self.transform()
        self.load()
        self.summary

    def extract(self) -> None:
        self._start = datetime.now()
        self._logger.info("\tExtract process started at {}.".format(self._start))

        self.config()
        self.download()
        self.decompress()
        self.save()

        self._end = datetime.now()
        self._extract_duration = self._end - self._start
        self._logger.info("\tExtract completed at {}. Duration: {}".format(self._end, self._extract_duration))
        return self

    def download(self) -> None:
        start = datetime.now()
        self._logger.info("\t\tDownload process began at {}.".format(start))

        self._download()

        end = datetime.now()
        self._download_duration = end - start
        self._logger.info("\t\tDownload completed at {}. Duration: {}".format(end, self._download_duration))
        return self

    def decompress(self) -> None:
        start = datetime.now()
        self._logger.info("\t\tDecompression started at {}.".format(start))

        self._decompress()

        end = datetime.now()
        self._decompress_duration = end - start
        self._logger.info("\t\tDecompression completed at {}. Duration: {}".format(end, self._decompress_duration))

    def save(self) -> None:
        start = datetime.now()
        self._logger.info("\t\tSave started at {}.".format(start))

        self._save()

        end = datetime.now()
        self._save_duration = end - start
        self._logger.info("\t\tSave completed at {}. Duration: {}".format(end, self._save_duration))

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

    @property
    @abstractmethod
    def summary(self) -> None:
        pass

    def _config_logging(self, verbose: bool = False) -> None:
        # Log Messaging
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.ERROR)
        if verbose:
            self._logger.setLevel(logging.INFO)


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
        self._filepath_decompressed = os.path.join(project_config["external_data_dir"], datasource_config["expanded"])
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
                dtype=raw_dtypes,
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
