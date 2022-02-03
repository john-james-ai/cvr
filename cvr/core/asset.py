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
# Modified : Thursday, February 3rd 2022, 4:58:47 am                           #
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

# ---------------------------------------------------------------------------- #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# ---------------------------------------------------------------------------- #

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
        aid: str,
        name: str,
        stage: str,
        creator: str,
        version: int,
        filepath: str,
        description: str = None,
    ) -> None:
        self._aid = aid.lower()
        self._name = name.lower()
        self._stage = stage.lower()
        self._creator = creator.lower()
        self._version = version
        self._description = self._format_description(description)
        self._created = datetime.now()

        self._asset_type = self.__class__.__name__

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

    def __init__(self, directory: str) -> None:
        self._versions_filepath = os.path.join(directory, "versions.pk")
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

        version = self._get_version_number(
            asset_type=asset_type, name=name, stage=stage, versions=versions
        )

        versions = self._update_versions(
            asset_type=asset_type,
            name=name,
            stage=stage,
            version=version,
            versions=versions,
        )

        self._io.save(versions, self._versions_filepath)

        return version

    # ------------------------------------------------------------------------ #
    def _get_version_number(
        self, asset_type: str, name: str, stage: str, versions: pd.DataFrame
    ) -> int:
        """Returns the version number for an asset."""
        if versions is None:
            version = 1
        else:
            try:
                version = versions.loc[
                    (versions["asset_type"] == asset_type)
                    & (versions["name"] == name)
                    & (versions["stage"] == stage)
                ]["version"]
                version += 1
            except AttributeError as e:
                version = 1
            except Exception as e:
                raise Exception(e)
        return version

    # ------------------------------------------------------------------------ #
    def _update_versions(
        self,
        asset_type: str,
        name: str,
        stage: str,
        version: int,
        versions: pd.DataFrame,
    ) -> pd.DataFrame:
        """Returns the versions registry with the new asset version updated."""

        d = {"asset_type": asset_type, "name": name, "stage": stage, "version": version}

        version_entry = pd.DataFrame(data=d, index=[0])

        if versions is not None:
            # Screen the current version record out if it exists.
            versions = versions.loc[
                (versions["asset_type"] != asset_type)
                & (versions["name"] != name)
                & (versions["stage"] != stage)
            ]
        else:
            versions = pd.DataFrame()

        versions = pd.concat([versions, version_entry], axis=0)
        return versions

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
            + str(version)
            + "_"
            + creator
        )
        return aid


# ============================================================================ #
#                             ASSET REGISTRY                                   #
# ============================================================================ #
class AssetRegistry:
    """Provides asset registry and versioning functionality."""

    def __init__(self, directory: str) -> None:
        self._io = Pickler()
        self._asset_registry = os.path.join(directory, "asset_registry.pkl")
        self._version_registry = os.path.join(directory, "version_registry.pkl")

    def register_asset(self, asset: Asset) -> None:
        # Filter the asset registries, excluding the current asset
        asset_registry = self._filter_asset_registry(
            asset_type=asset.asset_type, name=asset.name, stage=asset.stage
        )

        asset_item = {
            "asset_type": asset.asset_type,
            "name": asset.name,
            "stage": asset.stage,
            "version": asset.version,
            "updated": datetime.now(),
        }
        asset_item = pd.DataFrame(data=asset_item, index=[0])
        asset_registry = pd.concat([asset_registry, asset_item], axis=0)
        self._io.save(asset_registry, self._asset_registry)

    def register_version(self, asset: Asset) -> None:
        versions = self._io.load(self._version_registry)
        versions = versions if versions is not None else pd.DataFrame()

        version_item = {
            "asset_type": asset.asset_type,
            "name": asset.name,
            "stage": asset.stage,
            "creator": asset.creator,
            "created": asset.created,
            "version": asset.version,
            "filepath": asset.filepath,
        }
        version_item = pd.DataFrame(data=version_item, index=[0])

        versions = pd.concat([versions, version_item], axis=0)
        self._io.save(versions, self._version_registry)

    def get_assets(self) -> pd.DataFrame:
        """Returns all assets and versions in the registry."""
        return self._io.load(self._version_registry)

    def get_version(
        self,
        asset_type: str,
        name: str,
        stage: str,
        version: int = None,
        ignore_errors: bool = True,
    ) -> pd.DataFrame:
        """Returns the version registry for an asset version.

        Returns the version registry for the designated version number. If version is None, the latest version is returned.

        Args:
            asset_type (str): The class of the object
            name (str): The name of the object
            stage (str): The stage of the object
            version (int): Optional. If None, latest version registry returned.
        """
        if version is None:
            version = self.get_current_version(
                asset_type=asset_type, name=name, stage=stage
            )

        registry = self._io.load(self._version_registry)

        try:
            return registry.loc[
                (registry["asset_type"] == asset_type)
                & (registry["name"] == name)
                & (registry["stage"] == stage)
                & (registry["version"] == version)
            ]
        except Exception as e:
            msg = "No regisration for {} {} of stage {} version {}".format(
                asset_type, name, stage, str(version)
            )
            logger.error(msg)
            return None

    def get_current_version(
        self,
        asset_type: str,
        name: str,
        stage: str,
        ignore_errors: bool = True,
    ) -> int:
        registry = self._io.load(self._asset_registry)

        try:
            return registry.loc[
                (registry["asset_type"] == asset_type)
                & (registry["name"] == name)
                & (registry["stage"] == stage)
            ]["version"]
        except Exception:
            return 1

    def get_filepath(
        self,
        asset_type: str,
        name: str,
        stage: str,
        version: int = None,
        ignore_errors: bool = True,
    ) -> str:
        """Retrieves the filepath for the designated asset and version

        Args:
            asset_type (str): Class of object for the asset
            name (str): Single word lower-case name for the asset
            stage (str): The stage in the develop lifecycle
            version (int): Version of asset.
            ignore_errors (bool): If True, exceptions are captured
        """
        try:
            return versions.loc[
                (versions["asset_type"] == asset_type)
                & (versions["name"] == name)
                & (versions["stage"] == stage)
                & (versions["version"] == version)
            ]["filepath"]
        except Exception as e:
            logger.error(
                "Asset {} of type {} and stage {} version {} not found in version registry.  Panic!".format(
                    name, asset_type, stage, str(version)
                )
            )
            if ignore_errors:
                return False
            else:
                raise Exception(e)

    def delete_asset(
        self, asset_type: str, name: str, stage: str, ignore_errors: bool = True
    ) -> None:
        """Deletes the asset registration for the designated asset.

        Args:
            asset_type (str): The class of the object
            name (str): The name of the object
            stage (str): The stage of the object
        """
        # Delete the asset registration
        assets = self._filter_asset_registry(
            asset_type=asset_type, name=name, stage=stage
        )
        self._io.save(assets, self._asset_registry)

    def delete_version(
        self,
        asset_type: str,
        name: str,
        stage: str,
        version: int = None,
        ignore_errors: bool = True,
    ) -> None:
        """Removes a version from the registry

        Args:
            asset_type (str): The class of the object
            name (str): The name of the object
            stage (str): The stage of the object
            version (int): Version number. If None, all versions for the asset will be deleted.
        """
        versions = self._filter_version_registry(
            asset_type=asset_type, name=name, stage=stage, version=version
        )
        self._io.save(versions, self._version_registry)

    def asset_exists(
        self, asset_type: str, name: str, stage: str, ignore_errors: bool = True
    ) -> bool:
        assets = self._io.load(self._asset_registry)
        try:
            asset = assets.loc[
                (assets["asset_type"] == asset_type)
                & (assets["name"] == name)
                & (assets["stage"] == stage)
            ]
            return len(asset) > 0
        except AttributeError as e:
            if ignore_errors:
                return False
            else:
                raise Exception(e)

    def version_exists(
        self,
        asset_type: str,
        name: str,
        stage: str,
        version: int = None,
        ignore_errors: bool = True,
    ) -> bool:
        """Evaluates existence of a version or versions

        Evaluates to False if a specific version does not exist. If no version number is provided, this method evaluates to False iff no versions
        exist for the asset.

        Args:
            asset_type (str): The class of the object
            name (str): The name of the object
            stage (str): The stage of the object
            version (int): Version number. If None, return False if no versions were found for the asset. True otherwise.
        """
        versions = self._io.load(self._version_registry)
        if version:
            try:
                registry = versions.loc[
                    (versions["asset_type"] == asset_type)
                    & (versions["name"] == name)
                    & (versions["stage"] == stage)
                    & (versions["version"] == version)
                ]
                return len(version) > 0
            except Exception as e:
                if ignore_errors:
                    return False
                else:
                    raise Exception(e)
        else:
            try:
                registry = versions.loc[
                    (versions["asset_type"] == asset_type)
                    & (versions["name"] == name)
                    & (versions["stage"] == stage)
                ]
                return len(version) > 0
            except Exception as e:
                if ignore_errors:
                    return False
                else:
                    raise Exception(e)

    def _filter_asset_registry(
        self, asset_type: str, name: str, stage: str, ignore_errors: bool = True
    ) -> pd.DataFrame:
        asset_registry = self._io.load(self._asset_registry)
        if asset_registry:
            return asset_registry.loc[
                (asset_registry["asset_type"] != asset_type)
                & (asset_registry["name"] != name)
                & (asset_registry["stage"] != stage)
            ]
        else:
            return pd.DataFrame()

    def _filter_version_registry(
        self,
        asset_type: str,
        name: str,
        stage: str,
        version: int = None,
        ignore_errors: bool = True,
    ) -> pd.DataFrame:

        version_registry = self._io.load(self._version_registry)

        if version_registry is not None:
            if version:

                return version_registry.loc[
                    (version_registry["asset_type"] != asset_type)
                    & (version_registry["name"] != name)
                    & (version_registry["stage"] != stage)
                    & (version_registry["version"] != version)
                ]
            else:
                return version_registry.loc[
                    (version_registry["asset_type"] != asset_type)
                    & (version_registry["name"] != name)
                    & (version_registry["stage"] != stage)
                ]
        else:
            return pd.DataFrame()


# ============================================================================ #
#                             ASSET REPOSITORY                                 #
# ============================================================================ #
class AssetRepo(ABC):
    """Abstract base class for asset repositories and persistence.

    Args:
        name (str): Name for the repository
    """

    def __init__(self, workspace_directory: str) -> None:
        self._assets_directory = os.path.join(workspace_directory, "assets")
        self._registry_filepath = os.path.join(workspace_directory, "registry.pkl")

        #  IO dependencies
        self._io = Pickler()

    def add(self, asset: Asset) -> None:
        """Commits an asset to persistence.

        Args:
            asset (Asset): Object to be persisted

        """
        self._io.save(asset=asset, filepath=asset.filepath)

        # Update registries
        self._registry.register_asset(asset)
        self._registry.register_version(asset)

    def get(self, asset_type: str, name: str, stage: str, version: int = None) -> Asset:
        """Retrieves an asset by asset type, stage, name and version

        Will retrieve the designated version. If version is None, the latest
        version is returned.

        Args:
            asset_type (str): Class of asset
            name (str): Name of object to be retrieved
            stage (str): Project stage
            version (int): Version number for object.
        Returns:
            asset (Asset): Asset being requested.
        """
        if version is None:
            version = self._registry.get_current_version(
                asset_type=asset_type, name=name, stage=stage
            )

        filepath = self._registry.get_filepath(
            asset_type=asset_type, name=name, stage=stage
        )

        return self._io.load(filepath)

    def get_assets(self) -> pd.DataFrame:
        """Returns the asset registry."""
        return self._registry.get_assets()

    def delete_asset(
        self, asset_type, stage: str, name: str, ignore_errors=True
    ) -> None:
        """Deletes an asset and all versions from repository.

        Args:
            asset_type (str): Class of asset
            name (str): Name of object to be retrieved
            stage (str): Project stage
            version (int): Version number for object.

        """
        if self._registry.asset_exists(asset_type=asset_type, name=name, stage=stage):

            # Obtain the current version number from the asset registry
            version = self._registry.get_current_version(
                asset_type=asset_type, name=name, stage=stage
            )
            # Iteratively delete each version, ignoring errors in case the version has been previously deleted.
            for version in np.arange(1, version + 1):
                self.delete_version(
                    asset_type=asset_type,
                    name=name,
                    stage=stage,
                    version=version,
                    ignore_errors=True,
                )

        if self._registry.asset_exists(asset_type=asset_type, stage=stage, name=name):
            self._registry.delete_asset(asset_type=asset_type, stage=stage, name=name)

    def delete_version(
        self, asset_type, stage: str, name: str, version: int = None, ignore_errors=True
    ) -> None:
        """Deletes an asset from the repository.

        Args:
            asset_type (str): Type of asset
            stage (str): Develop lifecycle stage
            name (str): Asset name
            version (int): Asset version. If None, most recent version will be deleted.

        """
        # Get current version if not provided
        if version is None:
            version = self._registry.get_current_version(
                asset_type=asset_type, name=name, stage=stage, ignore_errors=True
            )

        filepath = self._registry.get_filepath(
            asset_type=asset_type, name=name, stage=stage, ignore_errors=True
        )

        if filepath:
            self._io.remove(filepath)

        self._registry.delete_version(asset_type, stage, name, version)

        # If this was the remaining version, delete the asset from registry as well.
        if not self._registry.version_exists(
            asset_type=asset_type, name=name, stage=stage, ignore_errors=True
        ):
            self._registry.delete_asset(
                asset_type=asset_type, name=name, stage=stage, ignore_errors=True
            )

    def asset_exists(self, asset_type: str, name: str, stage: str) -> bool:
        """Returns true if the asset exists.

        Args:
            asset_type (str): Type of asset
            stage (str): Develop lifecycle stage
            name (str): Asset name

        Returns:
            bool True if asset exists, False otherwise.
        """

        return self._registry.asset_exists(
            asset_type=asset_type, stage=stage, name=name
        )

    def version_exists(
        self, asset_type: str, name: str, stage: str, version: int = None
    ) -> bool:
        """Returns true if the asset version exists.

        Args:
            asset_type (str): Type of asset
            stage (str): Develop lifecycle stage
            name (str): Asset name
            version (int): Optional. If None, Returns True if any version exists.

        Returns:
            bool True if asset exists, False otherwise.
        """
        return self._registry.version_exists(asset_type, stage, name, version)

    def _get_version_number(self, asset_type: str, name: str, stage: str) -> int:
        """Returns the version number for an asset

        Version numbers start at 1 and are incremented when an asset of the same type, stage, and name are created.

        Args:
            asset_type (str): The class of asset
            name (str): Name of asset
            stage (str): The development lifecycle stage in which the asset is
                created

        Returns
            version (int)
        """
        return self._registry.get_current_version(
            asset_type=asset_type, name=name, stage=stage
        )
