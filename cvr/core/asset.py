#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project  : Deep Learning for Conversion Rate Prediction (CVR)                                    #
# Version  : 0.1.0                                                                                 #
# File     : \asset.py                                                                             #
# Language : Python 3.7.12                                                                         #
# ------------------------------------------------------------------------------------------------ #
# Author   : John James                                                                            #
# Email    : john.james.ai.studio@gmail.com                                                        #
# URL      : https://github.com/john-james-ai/cvr                                                  #
# ------------------------------------------------------------------------------------------------ #
# Created  : Saturday, January 22nd 2022, 5:11:23 pm                                               #
# Modified : Thursday, February 10th 2022, 7:48:22 pm                                              #
# Modifier : John James (john.james.ai.studio@gmail.com)                                           #
# ------------------------------------------------------------------------------------------------ #
# License  : BSD 3-clause "New" or "Revised" License                                               #
# Copyright: (c) 2022 Bryant St. Labs                                                              #
# ================================================================================================ #

"""Base class for lab assets that get persisted within labs."""
from abc import ABC, abstractmethod
import os
import pandas as pd
from datetime import datetime
from dataclasses import dataclass, field
from collections import OrderedDict

from cvr.utils.io import Pickler
from cvr.utils.format import titlelize
from cvr.utils.config import AIDConfig
from cvr.utils.printing import Printer


# ============================================================================ #
#                             DATASET PASSPORT                                 #
# ============================================================================ #


@dataclass
class AssetPassport:
    aid: str
    asset_type: str
    name: str
    description: str
    stage: str
    version: int = field(default="0000")
    filepath: str = field(default=None)
    creator: str = field(default="cvr")
    created: datetime = field(default=datetime.now())
    stamps: OrderedDict() = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._printer = Printer()

    def asdict(self) -> dict:
        d = {
            "aid": self.aid,
            "asset_type": self.asset_type,
            "name": self.name,
            "description": self.description,
            "stage": self.stage,
            "version": self.version,
            "filepath": self.filepath,
            "creator": self.creator,
            "created": self.created,
            "stamps": self.stamps,
        }
        return d

    def stamp(self, asset) -> None:
        stamp = {
            "asset": asset.__class__.__name__.lower(),
            "datetime": datetime.now(),
        }
        self.stamps[asset.aid] = stamp

    def print(self) -> None:
        d = {
            "aid": self.aid,
            "asset_type": self.asset_type,
            "name": self.name,
            "description": self.description,
            "stage": self.stage,
            "version": self.version,
            "filepath": self.filepath,
        }
        title = self.name + " " + self.asset_type
        self._printer.print_title(titlelize(title), titlelize(self.description))
        self._printer.print_dictionary(d)
        if len(self.stamps) > 0:
            for stamp in self.stamps.items():
                self._printer.print_dictionary(stamp)


# ============================================================================ #
#                                  ASSET                                       #
# ============================================================================ #


class Asset(ABC):
    """Abstract base class for lab assets.

    Args:
        passport (AssetPassport): passport
    """

    def __init__(self, passport: AssetPassport) -> None:
        self._passport = passport

    @property
    def passport(self) -> str:
        return self._passport

    @property
    def aid(self) -> str:
        return self._passport.aid

    @property
    def name(self) -> str:
        return self._passport.name

    @property
    def description(self) -> str:
        return self._passport.description

    @property
    def stage(self) -> str:
        return self._passport.stage

    @property
    def version(self) -> str:
        return self._passport.version

    @property
    def filepath(self) -> str:
        return self._passport.filepath


# ============================================================================ #
#                               ASSET BUILDER                                  #
# ============================================================================ #
class AssetBuilder(ABC):
    """Abstract base class for asset builders"""

    @property
    @abstractmethod
    def set_passport(self, passport) -> None:
        pass

    @abstractmethod
    def set_config(self, config) -> None:
        pass

    @abstractmethod
    def add_task(self, task) -> None:
        pass


# ============================================================================ #
#                             ASSET REPOSITORY                                 #
# ============================================================================ #
class AssetRepo(ABC):
    """Base class for asset repositories and persistence.

    Args:
        directory (str): Directory containing the asset repository.
        registry (str): Filepath of the registry file.
    """

    def __init__(self, directory: str, registry: str) -> None:
        self._directory = directory
        self._registry = registry
        self._aid_config_filepath = os.path.join(directory, "aid_config.yaml")

        #  IO dependencies
        self._io = Pickler()
        # AID Generator
        self._aid_config = AIDConfig(self._aid_config_filepath)

    # ------------------------------------------------------------------------ #
    def aid_gen(self) -> str:
        return self._aid_config.gen()

    # ------------------------------------------------------------------------ #
    def add(self, asset: Asset) -> None:
        """Commits an asset to persistence.

        Args:
            asset (Asset): Object to be persisted

        """
        asset.passport.filepath = self._create_filepath(asset=asset)
        self._register_asset(asset)
        self._io.save(asset=asset, filepath=asset.filepath)

    # ------------------------------------------------------------------------ #
    def get(self, asset_type: str, stage: str, name: str, version: int = None) -> Asset:
        """Retrieves an asset by asset_type, stage, name, and optional version

        Args:
            asset_type (str): The class of the asset in lower case
            stage (str): The stage in the development pipeline
            name (str): The name of the asset
            version (int): The version of the asset
        Returns:
            asset (Asset): Asset being requested.
        """
        if version:
            registry = self._search_registry_by_version(asset_type, name, stage, version)
        else:
            registry = self._search_registry_by_asset(asset_type, name, stage)

        try:
            filepath = registry["filepath"].values[0]
            return self._io.load(filepath)
        except IndexError:
            return None

    # ------------------------------------------------------------------------ #
    def get_by_aid(self, aid: str) -> Asset:
        """Retrieves an asset by aid.

        Args:
            aid (str): asset id
        Returns:
            asset (Asset): Asset being requested.
        """
        registry = self._io.load(self._registry)

        item = registry.loc[registry["aid"] == aid]
        try:
            filepath = item["filepath"].values[0]
            return self._io.load(filepath)
        except IndexError:
            return None

    # ------------------------------------------------------------------------ #
    def get_assets(self, asset_type: str = None) -> pd.DataFrame:
        """Returns the registry, optionally filtered by asset_type

        Args:
            asset_type (str): asset type

        Returns:
            assets (pd.DataFrame):
        """
        assets = self._io.load(self._registry)
        if asset_type:
            return assets.loc[assets["asset_type"] == asset_type]
        else:
            return assets

    # ------------------------------------------------------------------------ #
    def set_version(self, asset: Asset) -> Asset:
        """Sets the version number on the asset and returns it.

        Args:
            asset (Asset): Asset

        Returns:
            asset (Asset): Asset with version property set

        """
        matching_assets = self._search_registry_by_asset(
            asset_type=asset.passport.asset_type,
            name=asset.passport.name,
            stage=asset.passport.stage,
        )
        if matching_assets is not None:
            asset.passport.version = len(matching_assets)
        else:
            asset.passport.version = 0

        return asset

    # ------------------------------------------------------------------------ #
    def delete(self, aid: str, ignore_errors: bool = True) -> None:
        """Deletes an asset, parameterized by the asset id.

        Args:
            aid (str): asset id
        Returns:
            asset (Asset): Asset being requested.
        """
        registry = self._io.load(self._registry)

        try:
            item = registry.loc[registry["aid"] == aid]
            filepath = item["filepath"].values[0]
            self._io.remove(filepath=filepath)
            registry = registry.loc[registry["aid"] != aid]
            self._io.save(registry, self._registry)

        except AttributeError:
            return None

    # ------------------------------------------------------------------------ #
    def exists(self, aid: str) -> bool:
        """Returns true if the asset version exists.

        Args:
            aid (str): asset id

        Returns:
            bool True if asset exists, False otherwise.
        """
        registry = self._io.load(self._registry)
        item = registry.loc[registry["aid"] == aid]
        return len(item) > 00

    # ------------------------------------------------------------------------ #
    @abstractmethod
    def create(self, name: str, description: str, stage: str, creator: str, **kwargs) -> Asset:
        pass

    # ------------------------------------------------------------------------ #
    def _create_filepath(self, asset: Asset, fileext=".pkl") -> str:
        """Forms the filepath for an asset."""
        filename = (
            asset.passport.stage
            + "_"
            + asset.passport.asset_type
            + "_"
            + asset.passport.name
            + "_v"
            + str(asset.passport.version).zfill(3)
            + fileext
        )
        return os.path.join(self._directory, filename)

    # ------------------------------------------------------------------------ #
    def _register_asset(self, asset: Asset) -> None:
        """Posts the asset to the registry."""
        registry = self._io.load(self._registry)
        registry = registry if registry is not None else pd.DataFrame()

        item = {
            "aid": asset.passport.aid,
            "stage": asset.passport.stage,
            "asset_type": asset.passport.asset_type,
            "created": asset.passport.created,
            "name": asset.passport.name,
            "version": asset.passport.version,
            "creator": asset.passport.creator,
            "filepath": asset.passport.filepath,
        }
        item = pd.DataFrame(data=item, index=[0])

        registry = pd.concat([registry, item], axis=0)
        self._io.save(registry, self._registry)

    # ----------------------------------------------------------------------- #
    def _search_registry_by_version(
        self, asset_type: str, name: str, stage: str, version: int
    ) -> pd.DataFrame:
        """Return one-row dataframe containing version registration."""

        registry = self._io.load(self._registry)
        try:
            return registry.loc[
                (registry["asset_type"] == asset_type)
                & (registry["stage"] == stage)
                & (registry["name"] == name)
                & (registry["version"] == version)
            ]

        except AttributeError:
            return None

    # ------------------------------------------------------------------------ #
    def _search_registry_by_asset(self, asset_type: str, name: str, stage: str) -> pd.DataFrame:
        """Returns latest version of asset."""

        registry = self._io.load(self._registry)
        try:
            assets = registry.loc[
                (registry["asset_type"] == asset_type)
                & (registry["stage"] == stage)
                & (registry["name"] == name)
            ]
            asset = assets.loc[assets["version"] == assets["version"].max()]
            return asset

        except AttributeError:
            return None
