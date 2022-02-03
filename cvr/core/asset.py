#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ============================================================================ #
# Project  : Deep Learning for Conversion Rate Prediction (CVR)                #
# Version  : 0.1.0                                                             #
# File     : \asset.py                                                         #
# Language : Python 3.7.12                                                     #
# ---------------------------------------------------------------------------- #
# Author   : John James                                                        #
# Email    : john.james.ai.studio@gmail.com                                    #
# URL      : https://github.com/john-james-ai/cvr                              #
# ---------------------------------------------------------------------------- #
# Created  : Saturday, January 22nd 2022, 5:11:23 pm                           #
# Modified : Thursday, February 3rd 2022, 1:48:25 pm                           #
# Modifier : John James (john.james.ai.studio@gmail.com)                       #
# ---------------------------------------------------------------------------- #
# License  : BSD 3-clause "New" or "Revised" License                           #
# Copyright: (c) 2022 Bryant St. Labs                                          #
# ============================================================================ #
"""Base class for workspace assets that get persisted within workspaces."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
import os
import shutil
import pandas as pd
import logging

from cvr.utils.io import Pickler
from cvr.utils.format import titlelize, DT_FORMAT_FILENAMES
from cvr.utils.printing import Printer


# ============================================================================ #
#                                  ASSET                                       #
# ============================================================================ #
class Asset(ABC):
    """Abstract base class for workspace assets.

    Args:
        aid (str): Asset id
        asset_type (str): Class of asset
        name (str): Name of asset
        stage (str): Development lifecycle stage
        creator (str): Task requesting the asset.
        version (int): The version of the asset.
        description (str): Brief, distinguishing, non-pedestrian label

    """

    def __init__(
        self,
        asset_type: str,
        aid: str,
        name: str,
        stage: str,
        creator: str,
        version: int,
        description: str = None,
    ) -> None:
        self._asset_type = asset_type.lower()
        self._aid = aid.lower()
        self._name = name.lower()
        self._stage = stage.lower()
        self._creator = creator.lower()
        self._version = version
        self._description = self._format_description(description)
        self._created = datetime.now()

    @property
    def aid(self) -> str:
        return self._aid

    @property
    def asset_type(self) -> str:
        return self._asset_type

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def stage(self) -> str:
        return self._stage

    @property
    def creator(self) -> str:
        return self._creator

    @property
    def version(self) -> str:
        return self._version

    @property
    def filepath(self) -> str:
        return self._filepath

    @filepath.setter
    def filepath(self, filepath) -> None:
        self._filepath = filepath

    @property
    def created(self) -> datetime:
        return self._created

    def _format_description(self, description) -> str:
        if description is None:
            description = "{} {}. Stage: {} created by {}".format(
                self._asset_type, self._name, self._stage, self._creator
            )
            description = titlelize(description)
        return description


# ============================================================================ #
#                             ASSET FACTORY                                    #
# ============================================================================ #
class AssetFactory(ABC):
    """Abstract base class that defines the interface for Factory classes."""

    def __init__(self, workspace_directory: str, logger: logging) -> None:
        self._versions_filepath = os.path.join(workspace_directory, "versions.pk")
        self._logger = logger
        self._io = Pickler()

    # ------------------------------------------------------------------------ #
    @abstractmethod
    def create(name: str, description: str, stage: str, **kwargs) -> Asset:
        pass

    # ------------------------------------------------------------------------ #
    def _get_version(self, asset_type: str, name: str, stage: str) -> str:
        """Obtains a version, updates the controller and returns the number

        Args:
            asset_type (str): The class of the asset
            name (str): Name of the asset
            stage (str): Development lifecycle stage

        Returns:
            version (int)
        """
        versions = self._io.load(self._versions_filepath)

        version = self._search_registry_by_asset(
            asset_type=asset_type, name=name, stage=stage, versions=versions
        )

        # If found update existing asset with version number found
        if version is not None:
            version += 1
            versions.loc[
                (versions["asset_type"] == asset_type)
                & (versions["name"] == name)
                & (versions["stage"] == stage),
                "version",
            ] = version

        # Otherwise, initialize version to 1 and add to registry
        elif version is None:
            version = 1
            d = {
                "asset_type": asset_type,
                "name": name,
                "stage": stage,
                "version": version,
            }
            df = pd.DataFrame(data=d, index=[0])
            versions = pd.concat([versions, df], axis=0)

        else:
            msg = "Version {} is not valid for {} - {} {}".format(
                version, asset_type, name, stage
            )
            raise ValueError(msg)

        self._io.save(versions, self._versions_filepath)

        self._logger.debug(
            "Obtained version {} for {} {} Stage: {}".format(
                str(version), asset_type, name, stage
            )
        )

        return version

    # ------------------------------------------------------------------------ #
    def _search_registry_by_asset(
        self, asset_type: str, name: str, stage: str, versions: pd.DataFrame
    ) -> int:
        """Returns the version number for an asset."""

        # Initialize as None and set only if found

        version = None

        try:
            asset = versions.loc[
                (versions["asset_type"] == asset_type)
                & (versions["name"] == name)
                & (versions["stage"] == stage)
            ]
            if len(asset) > 0:  # Found
                version = asset["version"].values[0]
        except AttributeError as e:  # Not Found
            pass
        except Exception as e:  # Error
            raise Exception(e)

        return version

    # ------------------------------------------------------------------------ #
    def _create_aid(
        self, asset_type: str, name: str, stage: str, creator: str, version: int
    ) -> str:
        """Creates and returns an asset ID

        Asset IDs are unique for every asset and are used to form filenames, perform metadata lookups.

        Args:
            asset_type (str): The class of asset
            name (str): Name of asset
            stage (str): The development lifecycle stage in which the asset is
                created
            creator (str): The pipeline or task creating the asset.
            version (int): Version of the asset

        Returns
            aid (str)

        """
        aid = (
            stage
            + "_"
            + asset_type
            + "_"
            + datetime.now().strftime(DT_FORMAT_FILENAMES)
            + "_"
            + name
            + "_v"
            + str(version).zfill(3)
            + "_"
            + creator
        )

        self._logger.debug(
            "Created AID: {} for {} {} Stage: {}".format(aid, asset_type, name, stage)
        )

        return aid


# ============================================================================ #
#                             ASSET REPOSITORY                                 #
# ============================================================================ #
class AssetRepo(ABC):
    """Abstract base class for asset repositories and persistence.

    Args:
        name (str): Name for the repository
    """

    def __init__(self, workspace_directory: str, logger: logging) -> None:
        self._logger = logger
        self._assets_directory = os.path.join(workspace_directory, "assets")
        self._registry_filepath = os.path.join(workspace_directory, "registry.pkl")

        #  IO dependencies
        self._io = Pickler()

    # ------------------------------------------------------------------------ #
    def add(self, asset: Asset) -> None:
        """Commits an asset to persistence.

        Args:
            asset (Asset): Object to be persisted

        """
        asset.filepath = self._create_filepath(asset=asset)
        self._register_asset(asset)
        self._io.save(asset=asset, filepath=asset.filepath)

        self._logger.debug(
            "Added {} {} Stage: {} Version: {} to {}".format(
                asset.asset_type,
                asset.name,
                asset.stage,
                str(asset.version),
                "AssetRepo",
            )
        )

    # ------------------------------------------------------------------------ #
    def get(self, asset_type: str, name: str, stage: str, version: int) -> Asset:
        """Retrieves an asset by asset type, stage, name and version

        Args:
            asset_type (str): Class of asset
            name (str): Name of object to be retrieved
            stage (str): Project stage
            version (int): Version number for object.
        Returns:
            asset (Asset): Asset being requested.
        """

        registration = self._search_registry_by_version(
            asset_type=asset_type, name=name, stage=stage, version=version
        )

        try:
            filepath = registration["filepath"].values[0]
            return self._io.load(filepath)
        except AttributeError as e:
            return None

    # ------------------------------------------------------------------------ #
    def get_assets(self, asset_type: str = None) -> pd.DataFrame:
        """Returns the asset registry."""
        assets = self._io.load(self._registry_filepath)
        if asset_type:
            return assets.loc[assets["asset_type"] == asset_type]
        else:
            return assets

    # ------------------------------------------------------------------------ #
    def delete_asset(
        self,
        asset_type: str,
        stage: str,
        name: str,
        version: int = None,
        ignore_errors: bool = True,
    ) -> None:
        """Deletes an asset and all versions from repository.

        Args:
            asset_type (str): Class of asset
            name (str): Name of object to be retrieved
            stage (str): Project stage
            version (int) Version of the asset. If None, all versions
            for the asset are deleted.

        """
        registry = self._io.load(self._registry_filepath)

        if version is not None:
            assets_to_delete = self._search_registry_by_version(
                asset_type=asset_type, name=name, stage=stage, version=version
            )
        else:
            assets_to_delete = self._search_registry_by_asset(
                asset_type=asset_type, name=name, stage=stage
            )

        if len(assets_to_delete) > 0:
            # Convert DataFrame to dictionary
            assets_to_delete = assets_to_delete.to_dict("index")

            # Iterate over each row now dictionary
            for idx, asset in assets_to_delete.items():
                x = input(
                    "Confirm deletion of {} {} Stage: {} Version: {}".format(
                        asset["asset_type"],
                        asset["name"],
                        asset["stage"],
                        str(asset["version"]),
                    )
                )

                # Delete the asset and the entry in the registry
                if "y" in x:
                    self._io.remove(asset["filepath"])
                    registry = registry.loc[registry["aid"] != asset["aid"]]

                self._io.save(registry, self._registry_filepath)

                self._logger.debug(
                    "Deleted {} {} Stage: {} Version: {} from {}".format(
                        asset["asset_type"],
                        asset["name"],
                        asset["stage"],
                        str(asset["version"]),
                        "AssetRepo",
                    )
                )
        else:
            if version is None:
                logger.error(
                    "{} {} of Stage {} not found.".format(asset_type, name, stage)
                )
            else:
                logger.error(
                    "{} {} of Stage {} Version: {} not found.".format(
                        asset_type, name, stage, version
                    )
                )
            if not ignore_errors:
                raise FileNotFoundError()

    # ------------------------------------------------------------------------ #
    def asset_exists(
        self, asset_type: str, name: str, stage: str, version: int
    ) -> bool:
        """Returns true if the asset version exists.

        Args:
            asset_type (str): Type of asset
            stage (str): Develop lifecycle stage
            name (str): Asset name
            version (int): Version of asset

        Returns:
            bool True if asset exists, False otherwise.
        """
        asset = self._search_registry_by_version(
            asset_type=asset_type, name=name, stage=stage, version=version
        )

        return len(asset) > 0

    # ------------------------------------------------------------------------ #
    def _create_filepath(self, asset: Asset) -> str:
        """Forms the filepath for an asset."""
        filename = asset.aid + ".pkl"
        return os.path.join(self._assets_directory, filename)

    # ------------------------------------------------------------------------ #
    def _get_filepath(
        self, asset_type: str, name: str, stage: str, version: int
    ) -> str:
        return self._find_registration(
            asset_type=asset_type, name=name, stage=stage, version=version
        )

    # ------------------------------------------------------------------------ #
    def _register_asset(self, asset: Asset) -> None:
        """Posts the asset to the registry."""
        registry = self._io.load(self._registry_filepath)
        registry = registry if registry is not None else pd.DataFrame()

        item = {
            "aid": asset.aid,
            "stage": asset.stage,
            "asset_type": asset.asset_type,
            "created": asset.created,
            "name": asset.name,
            "version": asset.version,
            "creator": asset.creator,
            "filepath": asset.filepath,
        }
        item = pd.DataFrame(data=item, index=[0])

        registry = pd.concat([registry, item], axis=0)
        self._io.save(registry, self._registry_filepath)

        self._logger.debug(
            "Registered asset {} {} Stage: {} Version: {}".format(
                asset.asset_type, asset.name, asset.stage, str(asset.version)
            )
        )

    # ----------------------------------------------------------------------- #
    def _search_registry_by_version(
        self, asset_type: str, name: str, stage: str, version: int
    ) -> pd.DataFrame:
        """Return one-row dataframe containing version registration."""

        registry = self._io.load(self._registry_filepath)
        try:
            return registry.loc[
                (registry["asset_type"] == asset_type)
                & (registry["stage"] == stage)
                & (registry["name"] == name)
                & (registry["version"] == version)
            ]

        except AttributeError as e:
            msg = "The {} asset {} version {} of the {} stage was not found.".format(
                asset_type, name, version, stage
            )
            raise FileNotFoundError(msg)

    # ------------------------------------------------------------------------ #
    def _search_registry_by_asset(
        self, asset_type: str, name: str, stage: str
    ) -> pd.DataFrame:
        """Return dataframe with registrations for an asset."""

        registry = self._io.load(self._registry_filepath)
        try:
            return registry.loc[
                (registry["asset_type"] == asset_type)
                & (registry["stage"] == stage)
                & (registry["name"] == name)
            ]

        except AttributeError as e:
            msg = "The {} asset {} of the {} stage was not found.".format(
                asset_type, name, stage
            )
            raise FileNotFoundError(msg)
