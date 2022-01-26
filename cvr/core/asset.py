#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ======================================================================================================================== #
# Project  : Conversion Rate Prediction (CVR)                                                                              #
# Version  : 0.1.0                                                                                                         #
# File     : \asset.py                                                                                                     #
# Language : Python 3.10.1                                                                                                 #
# ------------------------------------------------------------------------------------------------------------------------ #
# Author   : John James                                                                                                    #
# Email    : john.james.ai.studio@gmail.com                                                                                #
# URL      : https://github.com/john-james-ai/cvr                                                                          #
# ------------------------------------------------------------------------------------------------------------------------ #
# Created  : Saturday, January 22nd 2022, 5:11:23 pm                                                                       #
# Modified : Wednesday, January 26th 2022, 3:28:18 am                                                                      #
# Modifier : John James (john.james.ai.studio@gmail.com)                                                                   #
# ------------------------------------------------------------------------------------------------------------------------ #
# License  : BSD 3-clause "New" or "Revised" License                                                                       #
# Copyright: (c) 2022 Bryant St. Labs                                                                                      #
# ======================================================================================================================== #
"""Base class for workspace assets that get persisted within workspaces."""
from abc import ABC, abstractmethod
import os
import shutil
import pandas as pd

from cvr.utils.io import Pickler
from cvr.utils.format import titlelize
from cvr.utils.printing import Printer

# ------------------------------------------------------------------------------------------------------------------------ #
class Asset(ABC):
    """Abstract base class for workspace assets.

    Args:
        name (str): The name of the asset
        stage (str): The stage in which the asset was created
    """

    def __init__(self, name: str, stage: str, version: int = None) -> None:
        self._name = name
        self._stage = stage
        self._version = 0 if version is None else version
        self._aid = stage + "_" + name
        self._filepath = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def stage(self) -> str:
        return self._stage

    @property
    def aid(self) -> str:
        return self._aid

    @property
    def filepath(self) -> str:
        return self._filepath

    @filepath.setter
    def filepath(self, filepath) -> None:
        self._filepath = filepath


# ======================================================================================================================== #
#                                               ASSET MANAGER                                                              #
# ======================================================================================================================== #
class AssetManager(ABC):
    """Defines the interface inventory management of workspace assets such as Datasets, Models and Pipelines"""

    def __init__(self, workspace_name: str, workspace_directory: str) -> None:
        self._workspace_name = workspace_name
        self._workspace_directory = workspace_directory

        self._printer = Printer()
        self._io = Pickler()

    @property
    def workspace_name(self) -> str:
        return self._workspace_name

    @property
    @abstractmethod
    def asset_type(self) -> str:
        pass

    @property
    @abstractmethod
    def inventory_filepath(self) -> str:
        pass

    @property
    @abstractmethod
    def asset_directory(self) -> str:
        pass

    @property
    def count(self) -> int:
        inventory = self._io.load(self.inventory_filepath)
        if inventory is None:
            count = 0
        else:
            count = inventory.shape[0]
        return count

    def add(self, asset: Asset) -> None:
        asset = self._set_filepath(asset)
        self._add_asset(asset)
        self._add_inventory(asset)
        return asset.filepath

    def get(self, stage: str, name: str) -> Asset:
        filepath = self._get_filepath(stage, name)
        if filepath is None:
            print("No filepath found for {} {}".format(stage, name))
        else:
            return self._io.load(filepath)

    def delete(self, stage: str, name: str) -> None:
        self._remove_asset(stage, name)
        self._remove_inventory(stage, name)

    def exists(self, name: str, stage: str) -> bool:
        filepath = self._get_filepath(stage, name)
        if filepath is None:
            return False
        else:
            return self._io.exists(filepath)

    def _add_asset(self, asset: Asset) -> None:
        asset = self._set_filepath(asset)
        self._io.save(asset, asset.filepath)

    def _add_inventory(self, asset: Asset) -> None:
        inventory = self._io.load(self.inventory_filepath)
        new_item = {
            "aid": asset.aid,
            "workspace": self._workspace_name,
            "asset_type": asset.__class__.__name__,
            "stage": asset.stage,
            "name": asset.name,
            "filepath": asset.filepath,
        }
        new_item = pd.DataFrame(new_item, index=[0])
        if inventory is None:
            inventory = new_item
        else:
            inventory = pd.concat([inventory, new_item], axis=0)
        if os.path.exists(self.inventory_filepath):
            os.remove(self.inventory_filepath)
        self._io.save(inventory, self.inventory_filepath)

    def _remove_asset(self, stage: str, name: str) -> None:
        filepath = self._get_filepath(stage, name)
        if filepath is None:
            print("No filepath found for {} {}".format(stage, name))
        else:
            self._io.remove(filepath)

    def _remove_inventory(self, stage: str, name: str) -> None:
        inventory = self._io.load(self.inventory_filepath)
        if inventory is not None:
            # Get AID
            aid = inventory[(inventory["stage"] == stage) & (inventory["name"] == name)]["aid"].values
            if len(aid) != 0:
                inventory = inventory[(inventory["aid"] != aid[0])]

                if os.path.exists(self.inventory_filepath):
                    os.remove(self.inventory_filepath)
                self._io.save(inventory, self.inventory_filepath)
            else:
                print("No filepath found for {} {}".format(stage, name))

    def _get_filepath(self, stage: str, name: str) -> list:
        filepath = None
        inventory = self._io.load(self.inventory_filepath)
        if inventory is not None:
            filepath = inventory[
                (inventory["workspace"] == self._workspace_name)
                & (inventory["stage"] == stage)
                & (inventory["name"] == name)
            ]["filepath"].values

            if len(filepath) == 0:
                return None
            elif isinstance(filepath, str):
                return filepath.lower()
            else:
                return filepath[0].lower()

    def _set_filepath(self, asset: Asset) -> str:
        filename = asset.__class__.__name__.lower() + "_" + asset.stage + "_" + asset.name + ".pkl"
        asset.filepath = os.path.join(self.asset_directory, asset.stage, filename)
        return asset

    def print(self) -> None:
        inventory = self._io.load(self.inventory_filepath)
        self._printer.print_title("Workspace Asset Inventory", titlelize(self._workspace_name))
        self._printer.print_dataframe(inventory)


# ------------------------------------------------------------------------------------------------------------------------ #
class DatasetManager(AssetManager):
    """Manages persistance of Datasets:

    Args:
        name (str): The name of the workspace
        workspace_directory (str): The base workspace_directory of the workspace
    """

    _asset_type = "datasets"

    def __init__(self, workspace_name: str, workspace_directory: str) -> None:
        super(DatasetManager, self).__init__(workspace_name, workspace_directory)
        self._inventory_filepath = os.path.join(self._workspace_directory, self._asset_type, "inventory.pkl")
        self._asset_directory = os.path.join(self._workspace_directory, self._asset_type)

    @property
    def asset_type(self) -> str:
        return self._asset_type

    @property
    def inventory_filepath(self) -> str:
        return self._inventory_filepath

    @property
    def asset_directory(self) -> str:
        return self._asset_directory
