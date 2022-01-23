#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ======================================================================================================================== #
# Project  : Conversion Rate Prediction (CVR)                                                                              #
# Version  : 0.1.0                                                                                                         #
# File     : \workspace.py                                                                                                 #
# Language : Python 3.10.1                                                                                                 #
# ------------------------------------------------------------------------------------------------------------------------ #
# Author   : John James                                                                                                    #
# Email    : john.james.ai.studio@gmail.com                                                                                #
# URL      : https://github.com/john-james-ai/cvr                                                                          #
# ------------------------------------------------------------------------------------------------------------------------ #
# Created  : Sunday, January 16th 2022, 4:42:38 am                                                                         #
# Modified : Sunday, January 23rd 2022, 4:04:50 am                                                                         #
# Modifier : John James (john.james.ai.studio@gmail.com)                                                                   #
# ------------------------------------------------------------------------------------------------------------------------ #
# License  : BSD 3-clause "New" or "Revised" License                                                                       #
# Copyright: (c) 2022 Bryant St. Labs                                                                                      #
# ======================================================================================================================== #
from abc import ABC, abstractmethod
import os
from datetime import datetime
import pandas as pd
import logging
import pickle

from cvr.utils.config import WorkspaceConfig
from cvr.utils.printing import Printer
from cvr.data.datasets import Dataset
from cvr.core.asset import Asset

# ------------------------------------------------------------------------------------------------------------------------ #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------------------------------ #


class Workspace:
    """Defines a workspace encapsulating datasets and the pipeline operations."""

    def __init__(self, name: str, description: str = None, current: bool = True) -> None:
        self._name = name
        self._description = description
        self._directory = os.path.join("workspaces", name)
        self._asset_manager = AssetManager(self)
        self._config = WorkspaceConfig()
        self._config.set_workspace(name)

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def directory(self) -> str:
        return self._directory

    def make_current(self) -> None:
        self._config.set_workspace(self._name)

    def add_dataset(self, dataset: Dataset) -> None:
        """Adds a dataset object to the inventory and persistent storage."""
        return self._asset_manager.add(dataset)

    def get_dataset(self, stage: str, name: str) -> None:
        return self._asset_manager.get(stage, name)

    def delete_dataset(self, stage: str, name: str) -> None:
        self._asset_manager.delete(stage, name)

    def print(self) -> None:
        self._asset_manager.print()


# ======================================================================================================================== #
#                                                   ASSET                                                                  #
# ======================================================================================================================== #
class AssetManager(ABC):
    """Defines the interface inventory management of workspace assets such as Datasets, Models and Pipelines"""

    def __init__(self, workspace) -> None:
        self._workspace = workspace
        inventory_filename = "inventory.pkl"
        self._inventory_filepath = os.path.join(workspace.directory, inventory_filename)
        self._printer = Printer()

    def add(self, asset: Asset) -> None:
        asset = self._set_filepath(asset)
        self._add_asset(asset)
        self._add_inventory(asset)
        return asset.filepath

    def get(self, stage: str, name: str) -> Dataset:
        filepath = self._get_filepath(stage, name)
        if filepath is None:
            print("No filepath found for {} {}".format(stage, name))
        else:
            return self.load(filepath)

    def delete(self, stage: str, name: str) -> None:
        self._remove_asset(stage, name)
        self._remove_inventory(stage, name)

    def _add_asset(self, asset: Asset) -> None:
        asset = self._set_filepath(asset)
        self.save(asset, asset.filepath)

    def _add_inventory(self, asset: Asset) -> None:
        inventory = self.load(self._inventory_filepath)
        new_item = {
            "aid": asset.aid,
            "workspace": self._workspace.name,
            "classname": asset.__class__.__name__,
            "stage": asset.stage,
            "name": asset.name,
            "filepath": asset.filepath,
        }
        new_item = pd.DataFrame(new_item, index=[0])
        if inventory is None:
            inventory = new_item
        else:
            inventory = pd.concat([inventory, new_item], axis=0)
        if os.path.exists(self._inventory_filepath):
            os.remove(self._inventory_filepath)
        self.save(inventory, self._inventory_filepath)

    def _remove_asset(self, stage: str, name: str) -> None:
        filepath = self._get_filepath(stage, name)
        if filepath is None:
            print("No filepath found for {} {}".format(stage, name))
        else:
            self.remove(filepath)

    def _remove_inventory(self, stage: str, name: str) -> None:
        inventory = self.load(self._inventory_filepath)
        if inventory is not None:
            # Get AID
            aid = inventory[(inventory["stage"] == stage) & (inventory["name"] == name)]["aid"].values
            if len(aid) != 0:
                inventory = inventory[(inventory["aid"] != aid[0])]

                if os.path.exists(self._inventory_filepath):
                    os.remove(self._inventory_filepath)
                self.save(inventory, self._inventory_filepath)
            else:
                print("No filepath found for {} {}".format(stage, name))

    def _get_filepath(self, stage: str, name: str) -> list:
        filepath = None
        inventory = self.load(self._inventory_filepath)
        if inventory is not None:
            filepath = inventory[
                (inventory["workspace"] == self._workspace.name)
                & (inventory["stage"] == stage)
                & (inventory["name"] == name)
            ]["filepath"].values

            if len(filepath) == 0:
                return None
            elif isinstance(filepath, str):
                return filepath
            else:
                return filepath[0]

    def _set_filepath(self, asset: Asset) -> str:
        filename = self._workspace.name + "_" + asset.__class__.__name__ + "_" + asset.stage + "_" + asset.name + ".pkl"
        asset.filepath = os.path.join(self._workspace.directory, asset.__class__.__name__, asset.stage, filename)
        return asset

    def load(self, filepath) -> Dataset:
        """Unpickles a asset.

        Args:
            filepath (str): Location of asset.
        """

        try:
            picklefile = open(filepath, "rb")
            asset = pickle.load(picklefile)
            picklefile.close()
            return asset
        except FileNotFoundError as e:
            return None

    def remove(self, filepath) -> Dataset:
        """Removes an asset.

        Args:
            filepath (str): Location of asset.
        """
        try:
            os.remove(filepath)
        except FileNotFoundError as e:
            print(e)

    def save(self, asset, filepath) -> None:
        """Pickles a asset

        Args:
            asset (Asset): Payload
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        picklefile = open(filepath, "wb")
        pickle.dump(asset, picklefile)
        picklefile.close()

    def print(self) -> None:
        inventory = self.load(self._inventory_filepath)
        self._printer.print_title("Workspace Inventory", self._workspace.name)
        self._printer.print_dataframe(inventory)
