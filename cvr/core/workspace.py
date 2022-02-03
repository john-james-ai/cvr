#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ============================================================================ #
# Project  : Deep Learning for Conversion Rate Prediction (CVR)                #
# Version  : 0.1.0                                                             #
# File     : \workspace.py                                                     #
# Language : Python 3.7.12                                                     #
# ---------------------------------------------------------------------------- #
# Author   : John James                                                        #
# Email    : john.james.ai.studio@gmail.com                                    #
# URL      : https://github.com/john-james-ai/cvr                              #
# ---------------------------------------------------------------------------- #
# Created  : Sunday, January 16th 2022, 4:42:38 am                             #
# Modified : Wednesday, February 2nd 2022, 3:14:38 pm                          #
# Modifier : John James (john.james.ai.studio@gmail.com)                       #
# ---------------------------------------------------------------------------- #
# License  : BSD 3-clause "New" or "Revised" License                           #
# Copyright: (c) 2022 Bryant St. Labs                                          #
# ============================================================================ #
import os
from datetime import datetime
import pandas as pd
import pickle

from cvr.core.dataset import Dataset, DatasetFactory
from cvr.core.asset import Asset, AssetFactory, AssetRepo
from cvr.utils.io import Pickler
from cvr.utils.logger import LoggerFactory

# ============================================================================ #
#                               WORKSPACE                                      #
# ============================================================================ #


class Workspace:
    """Defines a workspace encapsulating datasets and operations."""

    def __init__(self, name: str, description: str = None) -> None:

        self._name = name
        self._description = description
        self._directory = os.path.join("workspaces", name)
        self._active = True

        # Compositions
        self._logger = LoggerFactory().get_logger(
            name=name, directory=self._directory, verbose=True
        )
        # Factories
        self._dataset_factory = DatasetFactory()
        # Persistence
        self._asset_repo = AssetRepo(workspace_directory=self._directory)

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def active(self) -> bool:
        return self._active

    @property
    def logger(self) -> bool:
        return self._logger

    @active.setter
    def active(self, active) -> None:
        self._active = active

    def add_dataset(self, dataset: Dataset) -> None:
        """Persists new dataset object."""
        self._asset_repo.add(dataset)
        self._logger.info(
            "Added {}, version {} to the {} workspace".format(
                dataset.description, str(dataset.version), self._name
            )
        )

    def get_dataset(self, stage: str, name: str, version: int) -> Asset:
        return self._asset_repo.get(
            asset_type="dataset", name=name, stage=stage, version=version
        )

    def remove_dataset(self, stage: str, name: str, version: int) -> None:
        self._asset_repo.remove(
            asset_type="dataset", name=name, stage=stage, version=version
        )
        self._logger.info(
            "Removed Dataset {} version {} of the {} stage from the {} workspace".format(
                name, str(version), stage, self._name
            )
        )

    def dataset_exists(self, name: str, stage: str) -> bool:
        return self._asset_repo.exists(asset_type="dataset", name=name, stage=stage)

    def list_datasets(self) -> pd.DataFrame:
        return self._asset_repo.list_datasets()


# ============================================================================ #
#                          WORKSPACE  FACTORY                                  #
# ============================================================================ #
class WorkspaceFactory(AssetFactory):
    """Factory Class for Workspace objects."""

    def create(self, name: str, description: str) -> Workspace:
        """Creates and returns a workspace object.

        Args:
            name (str): The name for the workspace
            description (str): Short description of workspace
        """
        return Workspace(name=name, description=description)


# ============================================================================ #
#                          WORKSPACE CONTAINER                                 #
# ============================================================================ #
class WorkspaceContainer:
    def __init__(self) -> None:
        self._workspace_factory = WorkspaceFactory()
        self._workspace_repository = os.path.join("workspaces", "repository.pkl")
        self._io = Pickler()

    def create(self, name: str, description: str) -> Workspace:
        """Creates and returns a Workspace object.

        Args:
            name (str): Workspace name
            description (str): Brief, creative, non-pedestrian label
        """
        return self._workspace_factory.create(name, description)

    def get_workspace(self, name: str) -> Workspace:
        """Obtains the workspace by name

        Args:
            name (str): The name of the workspace
        """
        workspaces = self._io.load(self._workspace_repository) or {}
        try:
            return workspaces[name]
        except KeyError as e:
            msg = "Workspace {} not found.".format(name)
            logger.error(msg)

    def add_workspace(self, workspace: Workspace) -> None:
        """Adds a Workspace to the repository.

        Args:
            workspace (Workspace): Workspace object
        """
        workspaces = self._io.load(self._workspace_repository) or {}
        if workspaces.name in workspaces.keys():
            msg = "Workspace {} already exists.".format(workspace.name)
            logger.error(msg)
        else:
            workspaces[workspace.name] = workspace
            self._io.save(workspaces, self._workspace_repository)

    def update_workspace(self, workspace: Workspace) -> None:
        """Saves an existing workspace.

        Args:
            workspace (Workspace): Workspace object
        """

        # Obtain current workspace inventory
        workspaces = self._io.load(self._workspace_repository) or {}
        try:
            workspaces[workspace.name] = workspace
            self._io.save(workspaces, self._workspace_repository)
        except Exception as e:
            logger.error(e)

    def delete_workspace(self, name: str) -> None:
        """Mark a workspace as not active.

        Args:
            name (str): The name of the workspace to delete.
        """
        workspace = self.get_workspace(name)
        workspace.active = False
        self.update_workspace(workspace)

    def exists(self, name) -> bool:
        """Determines whether a named workspace exists.

        Args:
            name (str): The name of the workspace

        """
        workspaces = self._io.load(self._workspace_repository) or {}
        return name in workspaces.keys()
