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
# Modified : Tuesday, January 18th 2022, 8:20:45 pm                                                                        #
# Modifier : John James (john.james.ai.studio@gmail.com)                                                                   #
# ------------------------------------------------------------------------------------------------------------------------ #
# License  : BSD 3-clause "New" or "Revised" License                                                                       #
# Copyright: (c) 2021 Bryant St. Labs                                                                                      #
# ======================================================================================================================== #
from abc import ABC, abstractmethod
import os
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
        self._config = None
        self._printer = Printer()

    @abstractmethod
    def get_config(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def load_config(self) -> None:
        pass

    @abstractmethod
    def save_config(self, config: dict) -> None:
        pass

    @abstractmethod
    def print_config(self) -> None:
        pass


class ProjectConfig(Config):
    """Encapsulates project configuration."""

    filepath = "config\project.yaml"

    def get_config(self):
        self.load_config()
        return self._config

    def load_config(self) -> None:
        with open(ProjectConfig.filepath) as f:
            self._config = yaml.full_load(f)

    def save_config(self, config: dict) -> None:
        with open(ProjectConfig.filepath, "w") as f:
            self._config = yaml.dump(config, f)

    def print_config(self) -> None:
        self.load_config()
        self._printer.print_title("Project Configuration")
        for k, v in self._config.items():
            self._printer.print_dictionary(v)


class WorkspaceConfig(Config):
    """Encapsulates project workspaces."""

    filepath = "config\workspace.yaml"

    def get_config(self):
        self.load_config()
        return self._config

    def load_config(self) -> None:
        with open(WorkspaceConfig.filepath) as f:
            self._config = yaml.full_load(f)

    def save_config(self, config: dict) -> None:
        with open(WorkspaceConfig.filepath, "w") as f:
            self._config = yaml.dump(config, f)

    def print_config(self) -> None:
        self.load_config()
        self._printer.print_title("Workspace Configurations")
        for k, v in self._config.items():
            self._printer.print_dictionary(v)


class DataSourceConfig(Config):
    """Manages configurations for DataSource objects."""

    filepath = "config\datasource.yaml"

    def get_config(self, source: str = None):
        self.load_config()
        if source is None:
            return self._config
        else:
            return self._config[source]

    def load_config(self) -> None:
        with open(DataSourceConfig.filepath) as f:
            self._config = yaml.full_load(f)

    def save_config(self, config) -> None:
        with open(DataSourceConfig.filepath, "w") as f:
            self._config = yaml.dump(config, f)

    def print_config(self) -> None:
        self.load_config()
        self._printer.print_title("DataSource Configurations")
        for k, v in self._config.items():
            self._printer.print_dictionary(v)
