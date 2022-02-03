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
# Modified : Thursday, February 3rd 2022, 1:58:38 pm                           #
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

    def __init__(
        self, name: str, workspace_directory: str, description: str = None
    ) -> None:

        self._name = name
        self._description = description
        self._workspace_directory = os.path.join(workspace_directory, name)

        self._active = True

        # Compositions
        self._logger = LoggerFactory().get_logger(
            name=name,
            directory=self._workspace_directory,
            logging_level="debug",
            verbose=True,
        )
        # Factories
        self._dataset_factory = DatasetFactory(
            workspace_directory=self._workspace_directory, logger=self._logger
        )
        # Persistence
        self._asset_repo = AssetRepo(
            workspace_directory=self._workspace_directory, logger=self._logger
        )

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

    def get_dataset(self, stage: str, name: str, version: int) -> Asset:
        return self._asset_repo.get(
            asset_type="dataset", name=name, stage=stage, version=version
        )

    def remove_dataset(self, stage: str, name: str, version: int) -> None:
        self._asset_repo.delete_asset(
            asset_type="dataset", name=name, stage=stage, version=version
        )

    def dataset_exists(self, name: str, stage: str, version: int) -> bool:
        return self._asset_repo.asset_exists(
            asset_type="dataset", name=name, stage=stage, version=version
        )

    def list_datasets(self) -> pd.DataFrame:
        return self._asset_repo.get_assets(asset_type="dataset")


# ============================================================================ #
#                          WORKSPACE ADMIN                                     #
# ============================================================================ #
class WorkspaceAdmin:
    def __init__(self, workspace_directory: str = None) -> None:

        self._workspace_directory = (
            workspace_directory if workspace_directory else "workspaces"
        )
        self._workspace_repository = os.path.join(
            self._workspace_directory, "workspaces.pkl"
        )

        self._io = Pickler()

    def create(
        self, name: str, workspace_directory: str, description: str
    ) -> Workspace:
        """Creates and returns a Workspace object.

        Args:
            name (str): Workspace name
            description (str): Brief, creative, non-pedestrian label
        """
        return Workspace(
            name=name, workspace_directory=workspace_directory, description=description
        )

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
