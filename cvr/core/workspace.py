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
# Modified : Saturday, January 22nd 2022, 7:26:28 pm                                                                       #
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

from cvr.utils.config import WorkspaceConfig, StageConfig
from cvr.utils.printing import Printer
from cvr.data.datasets import Dataset, DatasetBuilder
from cvr.data.datastore import Datastore

# ------------------------------------------------------------------------------------------------------------------------ #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------------------------------ #


class Workspace:
    """Defines a workspace encapsulating datasets and the pipeline operations."""

    def __init__(self, name: str, description: str = None) -> None:
        self._name = name
        self._description = description
        self._directory = os.path.join("workspaces", name)
        self._asset_manager = AssetManager(self)

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def directory(self) -> str:
        return self._directory

    def add_dataset(self, dataset: Dataset) -> None:
        """Adds a dataset object to the inventory and persistent storage."""
        self._asset_manager.add(dataset)

    def get_dataset(self, dataset_id: str) -> None:
        """Retrieve a dataset by id."""
        config = self._workspace_config.get_config()
        if dataset_id in config["datasets"].keys():
            filepath = config["datasets"][dataset_id]["filepath"]
            return self._load_dataset(filepath)

    @property
    def datasets(self) -> None:
        self._print_inventory()


# ======================================================================================================================== #
#                                                   ASSET                                                                  #
# ======================================================================================================================== #
class AssetManager(ABC):
    """Defines the interface inventory management of workspace assets such as Datasets, Models and Pipelines"""

    def __init__(self, workspace) -> None:
        self._workspace = workspace
        inventory_filename = "inventory.pkl"
        self._inventory_filepath = os.path.join(workspace.directory, inventory)

    def add(self, asset: Asset) -> None:
        asset = self._set_filepath(asset)
        self._add_asset(asset)
        self._add_inventory(asset)
        return asset.filepath

    def remove(self, asset_id: str) -> None:
        self._remove_asset(asset_id)
        self._remove_inventory(asset_id)

    def _add_asset(self, asset: Asset) -> None:
        asset = self._set_filepath(asset)
        self._save(asset, asset.filepath)

    def _add_inventory(self, asset: Asset) -> None:
        inventory = self.load(self._inventory_filepath)
        new_item = {
            "id": asset.id,
            "workspace": self._workspace.name,
            "classname": asset.__class__.__name__,
            stage: asset.stage,
            name: asset.name,
            "filepath": asset.filepath,
        }
        new_item = pd.DataFrame.from_dict(new_item)
        if inventory.shape[0] == 0:
            inventory = new_item
        else:
            inventory = pd.concat([inventory, new_item], axis=0)
        self.save(inventory, self._inventory_filepath)

    def _remove_asset(self, asset_id: str) -> None:
        filepath = self._get_filepath(asset_id)
        self.remove(filepath)

    def _remove_inventory(self, asset_id: str) -> None:
        inventory = self.load(self._inventory_filepath)
        inventory = inventory[inventory["asset_id"] != asset_id]
        self.save(inventory, self._inventory_filepath)

    def _get_filepath(self) -> list:
        inventory = self.load_inventory()
        filepath = inventory[inventory["id"] == self._workspace.name]["id"].values
        return filepath

    def set_filepath(self, asset: Asset) -> str:
        filename = self._workspace.name + "_" + asset.__class__.__name__ + "_" + asset.stage + "_" + asset.name + ".pkl"
        asset.filename = os.path.join(self._workspace.directory, asset.__class__.__name__, asset.stage, filename)
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
            print(e)

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
        os.makedirs(os.path.dirname(filepath))
        picklefile = open(filepath, "wb")
        pickle.dump(asset, picklefile)
        picklefile.close()
