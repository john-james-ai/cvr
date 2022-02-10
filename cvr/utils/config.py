#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ============================================================================ #
# Project  : Deep Learning for Conversion Rate Prediction (CVR)                #
# Version  : 0.1.0                                                             #
# File     : \config.py                                                        #
# Language : Python 3.7.12                                                     #
# ---------------------------------------------------------------------------- #
# Author   : John James                                                        #
# Email    : john.james.ai.studio@gmail.com                                    #
# URL      : https://github.com/john-james-ai/cvr                              #
# ---------------------------------------------------------------------------- #
# Created  : Saturday, December 25th 2021, 11:07:50 am                         #
# Modified : Wednesday, February 9th 2022, 6:34:03 pm                          #
# Modifier : John James (john.james.ai.studio@gmail.com)                       #
# ---------------------------------------------------------------------------- #
# License  : BSD 3-clause "New" or "Revised" License                           #
# Copyright: (c) 2022 Bryant St. Labs                                          #
# ============================================================================ #

from abc import ABC
import os
import yaml

from cvr.utils.printing import Printer

# ---------------------------------------------------------------------------- #


class Config(ABC):
    """Abstract base class for Config classes."""

    def __init__(self) -> None:
        self._printer = Printer()

    def load_config(self, filepath: str) -> dict:
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                return yaml.full_load(f)
        else:
            return {}

    def save_config(self, config: dict, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            yaml.dump(config, f)


# ---------------------------------------------------------------------------- #


class CriteoConfig(Config):
    """Encapsulates the Criteo Labs data source configuration.

    Args:
        config_filepath (str): A string containing the path
        to the YAML configuration file.

    """

    __filepath = "config\criteo.yaml"

    def __init__(self, config_filepath: str = None) -> None:
        super(CriteoConfig, self).__init__()

        self._config_filepath = (
            config_filepath if config_filepath else CriteoConfig.__filepath
        )

        self._config = self.get_config()

    @property
    def url(self) -> str:
        return self._config["url"]

    @property
    def download_filepath(self) -> str:
        return self._config["download_filepath"]

    @property
    def extract_filepath(self) -> str:
        return self._config["extract_filepath"]

    @property
    def destination(self) -> str:
        return self._config["destination"]

    def get_config(self) -> dict:
        return self.load_config(self._config_filepath)

    def print(self) -> None:
        """Prints the configuration to stdout."""
        config = self.get_config()
        self._printer.print_title("Criteo Data Source Configuration")
        self._printer.print_dictionary(config)


# ---------------------------------------------------------------------------- #


class AIDConfig(Config):
    """Asset ID generator and configuration

    Simple class that returns a sequentially incremented asset id

    Args:
        aid_filepath (str): An optional path to the AID file.

    """

    __filepath = "config\\aid.yaml"

    def __init__(self, aid_filepath: str = None) -> None:
        super(AIDConfig, self).__init__()

        self._config_filepath = (
            aid_filepath if aid_filepath else AIDConfig.__filepath
        )
        self._config = self.get_config()

    def reset(self) -> None:
        """Sets the aid number to zero."""
        config = self.get_config()
        config["aid"] = 0
        self.save_config(config, self._config_filepath)

    def gen(self) -> int:
        """Autoincrements an external asset id number."""

        config = self.get_config()
        try:
            aid = int(config["aid"])
        except KeyError:
            aid = 0
        new_aid = aid + 1
        config["aid"] = new_aid
        self.save_config(config, self._config_filepath)
        return str(aid).zfill(4)

    def get_config(self) -> dict:
        return self.load_config(self._config_filepath)
