#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ============================================================================ #
# Project  : Deep Learning for Conversion Rate Prediction (CVR)                #
# Version  : 0.1.0                                                             #
# File     : \atelier.py                                                       #
# Language : Python 3.7.12                                                     #
# ---------------------------------------------------------------------------- #
# Author   : John James                                                        #
# Email    : john.james.ai.studio@gmail.com                                    #
# URL      : https://github.com/john-james-ai/cvr                              #
# ---------------------------------------------------------------------------- #
# Created  : Friday, February 4th 2022, 6:54:57 pm                             #
# Modified : Wednesday, February 9th 2022, 7:55:11 pm                          #
# Modifier : John James (john.james.ai.studio@gmail.com)                       #
# ---------------------------------------------------------------------------- #
# License  : BSD 3-clause "New" or "Revised" License                           #
# Copyright: (c) 2022 Bryant St. Labs                                          #
# ============================================================================ #
import os

from cvr.core.asset import Asset
from cvr.core.dataset import DatasetRepo
from cvr.utils.io import Pickler
from cvr.utils.logger import LoggerFactory
from cvr.utils.format import titlelize
from cvr.utils.config import AIDConfig
from cvr.utils.printing import Printer

# ---------------------------------------------------------------------------- #
BASEDIR = "ateliers"
# ============================================================================ #
#                                STUDIO                                        #
# ============================================================================ #


class Atelier:
    """Defines a workspace that encapsulates data, and operations.

    Args:
        name (str): The name of the atelier
        description (str): Concise, yet descriptive atelier for the atelier
        logging_level (str): Level of logging for the atelier. Valid
            values are 'info' (default) and 'debug'.
    """

    def __init__(
        self,
        name: str,
        description: str = None,
        logging_level: str = "info",
    ) -> None:

        self._name = name
        self._description = description or titlelize("{} Atelier".format(name))
        self._directory = os.path.join(BASEDIR, name)
        self._asset_directory = os.path.join(self._directory, "assets")
        self._registry = os.path.join(self._directory, "registry.pkl")
        self._aid_config_filepath = os.path.join(
            self._directory, "aid_config.yaml"
        )

        # Logging Configuration
        self._logging_level = logging_level
        self._logger_factory = LoggerFactory(
            name=name, directory=self._directory, logging_level=logging_level
        )
        self._logger = self._logger_factory.get_logger()

        # Persistence
        self._dataset_repo = DatasetRepo(
            directory=self._directory, registry=self._registry
        )

        # asset id config
        self._aid_config = AIDConfig(self._aid_config_filepath)
        self._aid_config.reset()

        # Printer
        self._printer = Printer()
        self._active = True

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def assets_directory(self) -> str:
        return self._asset_directory

    @property
    def active(self) -> bool:
        return self._active

    @active.setter
    def active(self, active) -> None:
        self._active = active

    @property
    def logger(self) -> bool:
        return self._logger

    @property
    def assets(self) -> DatasetRepo:
        return self._dataset_repo

    @property
    def next_aid(self) -> str:
        return self._aid_config.gen()

    def get_asset(
        self, asset_type: str, name: str, stage: str, version: int = None
    ) -> Asset:
        return self._dataset_repo.get(
            asset_type=asset_type, stage=stage, name=name, version=version
        )

    def get_asset_by_aid(self, aid: str) -> Asset:
        return self._dataset_repo.get_by_aid(aid=aid)

    def add_asset(self, asset) -> None:
        self._dataset_repo.add(asset)

    def print_assets(self) -> None:
        assets = self._dataset_repo.get_assets()
        self._printer.print_title(self._description, "Asset Inventory")
        self._printer.print_dataframe(assets)


# ============================================================================ #
#                              STUDIO BUILDER                                  #
# ============================================================================ #
class AtelierFabrique:
    def __init__(self) -> None:

        self._atelier_repository = os.path.join(BASEDIR, "ateliers.pkl")

        self._io = Pickler()

    def create(
        self, name: str, description: str, logging_level: str = "info"
    ) -> Atelier:
        """Creates and returns an Atelier object.

        Args:
            name (str): Atelier name
            description (str): Brief, creative, non-pedestrian atelier
            logging_level (str): The designated logging level for the atelier.
                Valid values are 'info', and 'debug'
        """
        registry = self._io.load(self._atelier_repository)
        if registry is not None:
            if name in registry.keys():
                msg = "Atelier {} already exists.".format(name)
                raise FileExistsError(msg)
        return Atelier(
            name=name,
            description=description,
            logging_level=logging_level,
        )

    def get_atelier(self, name: str) -> Atelier:
        """Obtains the atelier by name

        Args:
            name (str): The name of the atelier
        """
        ateliers = self._io.load(self._atelier_repository) or {}
        try:
            return ateliers[name]
        except KeyError:
            msg = "Atelier {} not found.".format(name)
            self._logger.error(msg)

    def add_atelier(self, atelier: Atelier) -> None:
        """Adds a Atelier to the repository.

        Args:
            atelier (Atelier): Atelier object
        """
        ateliers = self._io.load(self._atelier_repository) or {}
        if ateliers.name in ateliers.keys():
            msg = "Atelier {} already exists.".format(atelier.name)
            raise FileExistsError(msg)
        else:
            ateliers[atelier.name] = atelier
            self._io.save(ateliers, self._atelier_repository)

    def update_atelier(self, atelier: Atelier) -> None:
        """Saves an existing atelier.

        Args:
            atelier (Atelier): Atelier object
        """

        # Obtain current atelier inventory
        ateliers = self._io.load(self._atelier_repository) or {}
        ateliers[atelier.name] = atelier
        self._io.save(ateliers, self._atelier_repository)

    def delete_atelier(self, name: str) -> None:
        """Mark a atelier as not active.

        Args:
            name (str): The name of the atelier to delete.
        """
        atelier = self.get_atelier(name)
        atelier.active = False
        self.update_atelier(atelier)

    def exists(self, name) -> bool:
        """Determines whether a named atelier exists.

        Args:
            name (str): The name of the atelier

        """
        ateliers = self._io.load(self._atelier_repository) or {}
        return name in ateliers.keys()
