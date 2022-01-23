#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ======================================================================================================================== #
# Project  : Explainable Recommendation (XRec)                                                                             #
# Version  : 0.1.0                                                                                                         #
# File     : \config.py                                                                                                    #
# Language : Python 3.8                                                                                                    #
# ------------------------------------------------------------------------------------------------------------------------ #
# Author   : John James                                                                                                    #
# Email    : john.james.ai.studio@gmail.com                                                                                #
# URL      : https://github.com/john-james-ai/xrec                                                                         #
# ------------------------------------------------------------------------------------------------------------------------ #
# Created  : Saturday, December 25th 2021, 11:07:50 am                                                                     #
# Modified : Saturday, January 22nd 2022, 11:45:02 pm                                                                      #
# Modifier : John James (john.james.ai.studio@gmail.com)                                                                   #
# ------------------------------------------------------------------------------------------------------------------------ #
# License  : BSD 3-clause "New" or "Revised" License                                                                       #
# Copyright: (c) 2021 Bryant St. Labs                                                                                      #
# ======================================================================================================================== #
from abc import ABC, abstractmethod
import os
import pandas as pd
import numpy as np
import yaml
import logging
from pprint import pprint

from cvr.utils.printing import Printer

# ------------------------------------------------------------------------------------------------------------------------ #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------------------------------ #
class Config(ABC):
    """Abstract base class for Config classes."""

    def __init__(self) -> None:
        self._printer = Printer()

    @property
    @abstractmethod
    def config_filepath(self) -> str:
        pass

    @abstractmethod
    def print(self) -> None:
        pass

    def load_config(self) -> dict:
        if os.path.exists(self.config_filepath):
            with open(self.config_filepath, "r") as f:
                return yaml.full_load(f)
        else:
            return {}

    def save_config(self, config: dict) -> None:
        os.makedirs(os.path.dirname(self.config_filepath), exist_ok=True)
        with open(self.config_filepath, "w") as f:
            yaml.dump(config, f)


# ======================================================================================================================== #
class WorkspaceConfig(Config):

    filepath = "config\workspace.yaml"

    def __init__(self) -> None:
        super(WorkspaceConfig, self).__init__()
        self._config_filepath = WorkspaceConfig.filepath

    @property
    def config_filepath(self) -> str:
        return WorkspaceConfig.filepath

    def get_workspace(self) -> str:
        config = self.load_config()
        return config["current"]

    def set_workspace(self, workspace) -> str:
        config = self.load_config()
        config["current"] = workspace
        self.save_config(config)

    def print(self) -> None:
        config = self.load_config()
        self._printer.print_dictionary(config, "Workspace Configuration")


# ======================================================================================================================== #
class CriteoConfig(Config):
    """Encapsulates the Criteo Labs data source configuration.

    Args:
        config_filepath (str): A string containing the path to the YAML configuration file.

    """

    filepath = "config\criteo.yaml"

    def __init__(self) -> None:
        super(CriteoConfig, self).__init__()
        self._config_filepath = CriteoConfig.filepath

    @property
    def config_filepath(self) -> str:
        return CriteoConfig.filepath

    def get_config(self) -> dict:
        return self.load_config()

    def print(self) -> None:
        """Prints the configuration to stdout."""
        config = self.get_config()
        self._printer.print_title("Criteo Data Source Configuration")
        self._printer.print_dictionary(config)
