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
# Modified : Wednesday, January 19th 2022, 5:33:41 pm                                                                      #
# Modifier : John James (john.james.ai.studio@gmail.com)                                                                   #
# ------------------------------------------------------------------------------------------------------------------------ #
# License  : BSD 3-clause "New" or "Revised" License                                                                       #
# Copyright: (c) 2022 Bryant St. Labs                                                                                      #
# ======================================================================================================================== #
import os
from datetime import datetime
import pandas as pd
import logging
import yaml

from cvr.core.repository import Repository
from cvr.core.workspace import Workspace
from cvr.utils.config import WorkspaceConfig, ProjectConfig
from cvr.data.source import DataSourceConfig, CriteoETL
from cvr.utils.printing import Printer
from cvr.data.datasets import Dataset, DatasetBuilder
from cvr.data.datastore import Datastore
from cvr.utils.data import sample

# ------------------------------------------------------------------------------------------------------------------------ #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------------------------------ #


class Project:
    """Encapsulates a workspace.."""

    filepath = "workspaces/inventory.yaml"

    def __init__(self, name) -> None:
        self._name = name
        self._source = "criteo"

        self._workspace_name = None
        self._workspace_description = None
        self._sample_size = None

        self._datasource_config = DataSourceConfig()
        self._project_config = ProjectConfig()

    @property
    def name(self) -> str:
        return self._name

    def source_data(self, force: bool = False, verbose: bool = False) -> None:
        """Extracts, transforms and loads the data into a staged directory."""

        xtr = CriteoETL(
            datasource_config=self._datasource_config,
            project_config=self._project_config,
            source=self._source,
            force=force,
            verbose=verbose,
        )
        xtr.etl()
        return self

    def create_workspace(self, workspace_name: str) -> None:
        self._workspace_name = workspace_name
        return self

    def with_description(self, workspace_description: str) -> None:
        self._workspace_description = workspace_description
        return self

    def add_full_data(self) -> None:
        self._sample_size = 1
        return self

    def add_sample_data(self, sample_size: float = 0.01) -> None:
        self._sample_size = sample_size
        return self

    def build(self) -> None:
        """Builds the workspace."""
        if self._is_valid():
            df = self._get_data()
            dataset = self._build_dataset(df)
            self._workspace = self._build_workspace(dataset)
        return self._workspace

    def _is_valid(self) -> None:
        return self._workspace_name and self._workspace_description

    def _get_data(self) -> None:
        """Obtains the data for a workspace."""
        config = self._project_config.get_config()
        df = pd.read_pickle(config["staged_data_filepath"])

        if self._sample_size is not None and self._sample_size < 1:
            df = sample(data=df, frac=self._sample_size, stratify="sale", random_state=config["random_state"])
        return df

    def _build_dataset(self, df: pd.DataFrame) -> Dataset:
        builder = DatasetBuilder(self._workspace_name)
        dataset = builder.set_data(df).set_name(self._workspace_name).set_stage("staged").set_creator("Oracle").build()
        return dataset

    def _build_workspace(self, dataset: Dataset) -> None:
        """Constructs the workspace"""

        workspace = Workspace(name=self._workspace_name, description=self._workspace_description)
        workspace.initialize()
        workspace.add_dataset(dataset)
        self._add_workspace(workspace)
        return workspace

    @property
    def workspace(self) -> None:
        return self._workspace

    def add_workspace(self, workspace: Workspace) -> None:
        inventory = {} or self._load_inventory()
        d = {"description": workspace.description, "directory": workspace.directory}
        inventory[workspace.name] = d
        self._save_inventory(inventory)

    def remove_workspace(self, name: str) -> None:
        inventory = {} or self._load_inventory()
        if name not in inventory.keys():
            raise KeyError("Workspace {} does not exist.".format(name))
        else:
            del inventory[name]
        self._save_inventory(inventory)

    @property
    def inventory(self) -> None:
        inventory = self._load_inventory()
        df = pd.DataFrame.from_dict(inventory, orient="index")
        self._printer.print_title("Project {} Workspaces".format(self._name))
        self._printer.print_dataframe(df)
        return df
