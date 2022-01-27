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
# Modified : Wednesday, January 26th 2022, 11:57:40 pm                                                                     #
# Modifier : John James (john.james.ai.studio@gmail.com)                                                                   #
# ------------------------------------------------------------------------------------------------------------------------ #
# License  : BSD 3-clause "New" or "Revised" License                                                                       #
# Copyright: (c) 2022 Bryant St. Labs                                                                                      #
# ======================================================================================================================== #
from abc import ABC, abstractmethod
import os
from datetime import datetime
import pandas as pd
import pickle
import shutil

from cvr.utils.config import WorkspaceConfig
from cvr.utils.logger import LoggerFactory
from cvr.utils.printing import Printer
from cvr.core.dataset import Dataset
from cvr.core.asset import Asset, AssetManager, DatasetManager
from cvr.utils.io import Pickler


class Workspace:
    """Defines a workspace encapsulating datasets and the pipeline operations."""

    def __init__(self, name: str, directory: str, description: str = None, random_state: int = None) -> None:
        self._name = name
        self._description = description
        self._random_state = random_state
        self._directory = os.path.join(directory, name)

        # self._logger = LoggerFactory().get_logger(name=name, directory=directory, verbose=True)

        self._dataset_manager = DatasetManager(workspace_name=name, workspace_directory=self._directory)
        # self._model_manager = ModelManager(name=name, directory=self._directory)
        # self._experiment_manager = ExperimentManager(name=name, directory=self._directory)
        # self._pipeline_manager = PipelineManager(name=name, directory=self._directory)
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
    def random_state(self) -> str:
        return self._random_state

    @property
    def dataset_count(self) -> int:
        return self._dataset_manager.count

    def add_dataset(self, dataset: Dataset) -> None:
        """Adds a dataset object to the inventory and persistent storage."""
        filepath = self._dataset_manager.add(dataset)
        self._dataset_count += 1
        return filepath

    def get_dataset(self, stage: str, name: str) -> None:
        return self._dataset_manager.get(stage, name)

    def delete_dataset(self, stage: str, name: str) -> None:
        self._dataset_manager.delete(stage, name)
        self._dataset_count -= 1

    def dataset_exists(self, name: str, stage: str) -> bool:
        return self._dataset_manager.exists(name=name, stage=stage)

    def print(self) -> None:
        self._dataset_manager.print()


# ======================================================================================================================== #
#                                            WORKSPACE MANAGER                                                             #
# ======================================================================================================================== #
class WorkspaceManager(ABC):
    """Quasi Singleton: Builds and manages workspace objects."""

    def __init__(self) -> None:
        self._workspaces_dir = "workspaces"
        self._workspaces_filepath = os.path.join(self._workspaces_dir, "workspaces.pkl")
        self._io = Pickler()
        self._printer = Printer()
        self._config = WorkspaceConfig()

    def count(self):
        """Returns the count of workspaces being managed."""
        # Obtain current workspace inventory
        workspaces = self._io.load(self._workspaces_filepath) or {}
        return len(workspaces)

    def reset(self):
        """Resets the workspace manager"""

        # Obtain current workspace inventory
        workspaces = self._io.load(self._workspaces_filepath) or {}
        # Loop through workspaces and get approval to delete each workspace.
        for name in workspaces.keys():
            x = input("Do you want to delete workspace {}? [y/n]".format(name))
            if "y" in x:
                self.delete_workspace(name)
        # Set the number of workspaces
        workspaces = self._io.load(self._workspaces_filepath) or {}

    def create_workspace(
        self, name: str, description: str = None, current: bool = True, random_state: int = None
    ) -> Workspace:
        """Creates and returns a workspace object.

        Args:
            name (str): The name for the workspace
            description (str): Short description of workspace
            current (bool): Whether to set the workspace as current
            random_state (int): Seed for pseudo-random generator
        """

        # Obtain current workspace inventory
        workspaces = self._io.load(self._workspaces_filepath) or {}
        # Confirm the workspace of that name doesn't already exist
        if name in workspaces.keys():
            raise FileExistsError("A workspace named {} already exists.".format(name))
        # Create Workspace instance
        workspace = Workspace(name=name, description=description, directory=self._workspaces_dir, random_state=random_state)
        # Add workspace instance to inventory of workspaces
        workspaces[name] = workspace
        # Save workspace inventory
        self._io.save(workspaces, self._workspaces_filepath)
        # If workspace is current, set workspace to current in workspace configuration
        if current:
            config = self._config.get_config()
            config["current"] = name
            self._config.set_config(config)

        return workspace

    def get_workspace(self, name: str) -> Workspace:
        """Obtains the workspace by name

        Args:
            name (str): The name of the workspace
        """
        # Obtain current workspace inventory
        workspaces = self._io.load(self._workspaces_filepath) or {}
        try:
            return workspaces[name]
        except KeyError as e:
            raise KeyError("Workspace {} does not exist.".format(name))

    def get_current_workspace(self) -> Workspace:
        """Gets the current workspace from the workspace config"""

        # Obtain name of current workspace
        config = self._config.get_config()
        current = config.get("current", None)
        if current is None:
            raise Exception("Current workspace has not been set.")

        # Obtain current workspace from inventory by name
        workspaces = self._io.load(self._workspaces_filepath) or {}
        try:
            return workspaces[current]
        except KeyError as e:
            raise KeyError("Workspace {} does not exist".format(current))

    def set_current_workspace(self, name: str) -> None:
        """Sets the current workspace in the workspace config file

        Args:
            name (str): Name of the workspace to be made current
        """

        # Confirm workspace exists
        if not self.exists(name):
            raise FileNotFoundError("Workspace {} does not exist.".format(name))

        # Update workspace config
        config = self._config.get_config()
        config["current"] = name
        self._config.set_config(config)

    def get_current_workspace_name(self) -> Workspace:
        """Returns the current workspace object."""
        workspace = self.get_current_workspace()
        return workspace.name

    def update_workspace(self, workspace: Workspace) -> None:
        """Saves an existing workspace.

        Args:
            workspace (Workspace): Workspace object
        """

        # Obtain current workspace inventory
        workspaces = self._io.load(self._workspaces_filepath) or {}
        workspaces[workspace.name] = workspace
        self._io.save(workspaces, self._workspaces_filepath)

    def delete_workspace(self, name: str) -> None:
        """Deletes a Workspace object and all of its data.

        Args:
            name (str): The name of the workspace to delete.
        """
        x = input = "Deleting this workspace will remove all of its data. Are you sure you want to proceed? [y/n]"
        if "y" in x:

            # Get the named workspace
            workspace = self.get_workspace(name)

            # Delete the directory
            shutil.rmtree(workspace.directory, ignore_errors=True)

            # Obtain workspace inventory, delete the workspace from inventory and resave the inventory
            workspaces = self._io.load(self._workspaces_filepath) or {}
            if name in workspaces.keys():
                del workspaces[name]
                self._io.save(workspaces, self._workspaces_filepath)

    def exists(self, name) -> bool:
        """Determines whether a named workspace exists.

        Args:
            name (str): The name of the workspace

        """
        workspaces = self._io.load(self._workspaces_filepath) or {}
        return name in workspaces.keys()


# ======================================================================================================================== #
#                                               WORKSPACE BUILDER                                                          #
# ======================================================================================================================== #
class WorkspaceBuilder(ABC):
    """Abstract interface for workspace builders."""

    @property
    @abstractmethod
    def workspace(self) -> None:
        pass

    @abstractmethod
    def set_workspace_name(self, name: str) -> None:
        pass

    @abstractmethod
    def set_workspace_description(self, name: str) -> None:
        pass

    @abstractmethod
    def set_dataset_config(self, name: str) -> None:
        pass

    @abstractmethod
    def set_dataset_name(self, name: str) -> None:
        pass

    @abstractmethod
    def set_dataset_name(self, name: str) -> None:
        pass

    @abstractmethod
    def set_dataset_size(self, name: str) -> None:
        pass

    @abstractmethod
    def set_istest(self) -> None:
        pass

    @abstractmethod
    def set_random_state(self, model: AssetManager) -> None:
        pass
