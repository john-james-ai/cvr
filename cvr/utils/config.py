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
# Modified : Thursday, January 20th 2022, 2:21:56 pm                                                                       #
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

    def __init__(self, config_filepath: str) -> None:
        self._config_filepath = config_filepath
        self._config = None
        self._printer = Printer()

    @abstractmethod
    def get_config(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def save_config(self, config: dict) -> None:
        pass

    @abstractmethod
    def print_config(self) -> None:
        pass


class ProjectConfig(Config):
    """Encapsulates project configuration."""

    def get_config(self):
        self.load_config()
        return self._config

    def load_config(self) -> None:
        with open(self._config_filepath, "r") as f:
            self._config = yaml.full_load(f)

    def save_config(self, config: dict) -> None:
        with open(self._config_filepath, "w") as f:
            self._config = yaml.dump(config, f)

    def print_config(self) -> None:
        self.load_config()
        self._printer.print_title("Project Configuration")
        for k, v in self._config.items():
            self._printer.print_dictionary(v)


class DatastoreConfig(Config):
    """Stage Configuration Manager."""

    def bump_version(self) -> None:
        config = self.get_config()
        version = int(config["version"])
        new_version = version + 1
        config["version"] += new_version
        save_config(config)
        return version

    def bump_stage(self) -> None:
        config = self.get_config()
        stage = int(config["stage"])
        new_stage = stage + 1
        config["stage"] = new_stage
        save_config(config)
        return stage

    def get_config(self):
        self.load_config()
        return self._config

    def load_config(self) -> None:
        with open(self._config_filepath, "r") as f:
            self._config = yaml.full_load(f)

    def save_config(self, config: dict) -> None:
        with open(self._config_filepath, "w") as f:
            self._config = yaml.dump(config, f)


class WorkspaceConfig(Config):
    """Encapsulates project workspaces."""

    def get_config(self):
        self.load_config()
        return self._config

    def load_config(self) -> None:
        with open(self._config_filepath, "r") as f:
            self._config = yaml.full_load(f)

    def save_config(self, config: dict) -> None:
        with open(self._config_filepath, "w") as f:
            self._config = yaml.dump(config, f)

    def print_config(self) -> None:
        self.load_config()
        self._printer.print_title("Workspace Configurations")
        for k, v in self._config.items():
            self._printer.print_dictionary(v)


class CriteoConfig(Config):
    """Manages configurations for DataSource objects."""

    def __init__(self, config_filepath: str) -> None:
        super(CriteoConfig, self).__init__(config_filepath)
        self._config = self.get_config()

    @property
    def name(self) -> str:
        return self._config["name"]

    @property
    def url(self) -> str:
        return self._config["url"]

    @property
    def destination(self) -> str:
        return self._config["destination"]

    @property
    def filepath_compressed(self) -> str:
        return self._config["filepath_compressed"]

    @property
    def filepath_decompressed(self) -> str:
        return self._config["filepath_decompressed"]

    @property
    def filepath_raw(self) -> str:
        return self._config["filepath_raw"]

    @property
    def filepath_staged(self) -> str:
        return self._config["filepath_staged"]

    @property
    def sep(self) -> str:
        return self._config["sep"]

    @property
    def missing(self) -> str:
        return self._config["missing"]

    def get_config(self) -> None:
        with open(self._config_filepath, "r") as f:
            return yaml.full_load(f)

    def save_config(self, config) -> None:
        with open(self._config_filepath, "w") as f:
            yaml.dump(config, f)

    def print_config(self) -> None:
        config = self.get_config()
        self._printer.print_title("Criteo Configurations")
        for k, v in config.items():
            self._printer.print_dictionary(v)