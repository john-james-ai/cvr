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
# Modified : Tuesday, January 25th 2022, 4:40:19 pm                                                                        #
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
import shutil

from cvr.utils.config import WorkspaceConfig
from cvr.utils.printing import Printer
from cvr.data.datasets import Dataset
from cvr.core.asset import Asset
from cvr.utils.io import Pickler

# ------------------------------------------------------------------------------------------------------------------------ #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------------------------------ #


class Workspace:
    """Defines a workspace encapsulating datasets and the pipeline operations."""

    def __init__(self, name: str, description: str = None, sample_size: int = None, random_state: int = None) -> None:
        self._name = name
        self._description = description
        self._sample_size = sample_size
        self._random_state = random_state
        self._directory = os.path.join("workspaces", name)
        self._asset_manager = AssetManager(name=name, directory=self._directory)
        self._dataset_count = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def directory(self) -> str:
        return self._directory

    @property
    def sample_size(self) -> str:
        return self._sample_size

    @property
    def random_state(self) -> str:
        return self._random_state

    @property
    def dataset_count(self) -> int:
        return self._dataset_count

    def add_dataset(self, dataset: Dataset) -> None:
        """Adds a dataset object to the inventory and persistent storage."""
        filepath = self._asset_manager.add(dataset)
        self._dataset_count += 1
        return filepath

    def get_dataset(self, stage: str, name: str) -> None:
        return self._asset_manager.get(stage, name)

    def delete_dataset(self, stage: str, name: str) -> None:
        self._asset_manager.delete(stage, name)
        self._dataset_count -= 1

    def print(self) -> None:
        self._asset_manager.print()


# ======================================================================================================================== #
#                                               ASSET MANAGER                                                              #
# ======================================================================================================================== #
class AssetManager(ABC):
    """Defines the interface inventory management of workspace assets such as Datasets, Models and Pipelines"""

    def __init__(self, name: str, directory: str) -> None:
        self._name = name
        self._directory = directory

        inventory_filename = "inventory.pkl"
        self._inventory_filepath = os.path.join(directory, inventory_filename)
        self._printer = Printer()
        self._io = Pickler()

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
            return self._io.load(filepath)

    def delete(self, stage: str, name: str) -> None:
        self._remove_asset(stage, name)
        self._remove_inventory(stage, name)

    def _add_asset(self, asset: Asset) -> None:
        asset = self._set_filepath(asset)
        self._io.save(asset, asset.filepath)

    def _add_inventory(self, asset: Asset) -> None:
        inventory = self._io.load(self._inventory_filepath)
        new_item = {
            "aid": asset.aid,
            "workspace": self._name,
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
        self._io.save(inventory, self._inventory_filepath)

    def _remove_asset(self, stage: str, name: str) -> None:
        filepath = self._get_filepath(stage, name)
        if filepath is None:
            print("No filepath found for {} {}".format(stage, name))
        else:
            self._io.remove(filepath)

    def _remove_inventory(self, stage: str, name: str) -> None:
        inventory = self._io.load(self._inventory_filepath)
        if inventory is not None:
            # Get AID
            aid = inventory[(inventory["stage"] == stage) & (inventory["name"] == name)]["aid"].values
            if len(aid) != 0:
                inventory = inventory[(inventory["aid"] != aid[0])]

                if os.path.exists(self._inventory_filepath):
                    os.remove(self._inventory_filepath)
                self._io.save(inventory, self._inventory_filepath)
            else:
                print("No filepath found for {} {}".format(stage, name))

    def _get_filepath(self, stage: str, name: str) -> list:
        filepath = None
        inventory = self._io.load(self._inventory_filepath)
        if inventory is not None:
            filepath = inventory[
                (inventory["workspace"] == self._name) & (inventory["stage"] == stage) & (inventory["name"] == name)
            ]["filepath"].values

            if len(filepath) == 0:
                return None
            elif isinstance(filepath, str):
                return filepath.lower()
            else:
                return filepath[0].lower()

    def _set_filepath(self, asset: Asset) -> str:
        filename = self._name + "_" + asset.__class__.__name__ + "_" + asset.stage + "_" + asset.name + ".pkl"
        asset.filepath = os.path.join(self._directory, asset.__class__.__name__, asset.stage, filename)
        return asset

    def print(self) -> None:
        inventory = self._io.load(self._inventory_filepath)
        self._printer.print_title("Workspace Asset Inventory", self._name)
        self._printer.print_dataframe(inventory)


# ======================================================================================================================== #
#                                            WORKSPACE MANAGER                                                             #
# ======================================================================================================================== #
class WorkspaceManager:
    """Builds and manages workspace objects."""

    __config = None
    __printer = None
    __directory = None
    __workspaces_filepath = None
    __io = None

    @staticmethod
    def _initialize():
        WorkspaceManager.__config = WorkspaceConfig() if WorkspaceManager.__config is None else WorkspaceManager.__config
        WorkspaceManager.__printer = Printer() if WorkspaceManager.__printer is None else WorkspaceManager.__printer
        WorkspaceManager.__directory = "workspaces" if WorkspaceManager.__directory is None else WorkspaceManager.__directory
        WorkspaceManager.__workspaces_filepath = (
            os.path.join(WorkspaceManager.__directory, "workspaces.pkl")
            if WorkspaceManager.__workspaces_filepath is None
            else WorkspaceManager.__workspaces_filepath
        )
        WorkspaceManager.__io = Pickler() if WorkspaceManager.__io is None else WorkspaceManager.__io

    @staticmethod
    def create_workspace(
        name: str, description: str = None, sample_size: int = None, current: bool = True, random_state: int = None
    ) -> Workspace:
        """Creates and returns a workspace object.

        Args:
            name (str): The name for the workspace
            description (str): Short description of workspace
            current (bool): Whether to set the workspace as current
            random_state (int): Seed for pseudo-random generator
        """
        WorkspaceManager._initialize()

        # Obtain current workspace inventory
        workspaces = WorkspaceManager.__io.load(WorkspaceManager.__workspaces_filepath) or {}
        # Create Workspace instance
        workspace = Workspace(name=name, description=description, sample_size=sample_size, random_state=random_state)
        # Add workspace instance to inventory of workspaces
        workspaces[name] = workspace
        # Save workspace inventory
        WorkspaceManager.__io.save(workspaces, WorkspaceManager.__workspaces_filepath)
        # If workspace is current, set workspace to current in workspace configuration
        if current:
            config = WorkspaceManager.__config.get_config()
            config["current"] = name
            WorkspaceManager.__config.set_config(config)

        return workspace

    @staticmethod
    def get_workspace(name) -> Workspace:
        """Obtains the workspace by name

        Args:
            name (str): The name of the workspace
        """
        WorkspaceManager._initialize()

        # Obtain current workspace inventory
        workspaces = WorkspaceManager.__io.load(WorkspaceManager.__workspaces_filepath) or {}
        try:
            return workspaces[name]
        except KeyError as e:
            raise KeyError("Workspace {} does not exist.".format(name))

    @staticmethod
    def get_current_workspace() -> Workspace:
        WorkspaceManager._initialize()

        # Obtain name of current workspace
        config = WorkspaceManager.__config.get_config()
        current = config.get("current", None)
        if current is None:
            raise Exception("Current workspace has not been set.")

        # Obtain current workspace from inventory by name
        workspaces = WorkspaceManager.__io.load(WorkspaceManager.__workspaces_filepath) or {}
        try:
            return workspaces[current]
        except KeyError as e:
            raise KeyError("Workspace {} does not exist".format(current))

    @staticmethod
    def update_workspace(workspace) -> None:
        """Saves an existing workspace.

        Args:
            workspace (Workspace): Workspace object
        """
        WorkspaceManager._initialize()

        # Obtain current workspace inventory
        workspaces = WorkspaceManager.__io.load(WorkspaceManager.__workspaces_filepath) or {}
        workspaces[workspace.name] = workspace
        WorkspaceManager.__io.save(workspaces, WorkspaceManager.__workspaces_filepath)

    @staticmethod
    def delete_workspace(name: str) -> None:
        """Deletes a Workspace object and all of its data.

        Args:
            name (str): The name of the workspace to delete.
        """
        x = input = "Deleting this workspace will remove all of its data. Are you sure you want to proceed? [y/n]"
        if "y" in x:
            WorkspaceManager._initialize()

            # Get the named workspace
            workspace = WorkspaceManager.get_workspace(name)

            # Delete the directory
            shutil.rmtree(workspace.directory, ignore_errors=True)

            # Obtain workspace inventory
            workspaces = WorkspaceManager.__io.load(WorkspaceManager.__workspaces_filepath) or {}
            try:
                del workspaces[name]
            except KeyError as e:
                raise RuntimeWarning("Workspace {} does not workspace inventory.".format(name))
