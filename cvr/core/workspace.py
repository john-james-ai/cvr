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
# Modified : Wednesday, January 19th 2022, 12:15:49 am                                                                     #
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

from cvr.utils.config import WorkspaceConfig
from cvr.utils.printing import Printer
from cvr.data.datasets import Dataset, DatasetBuilder
from cvr.data.datastore import Datastore

# ------------------------------------------------------------------------------------------------------------------------ #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------------------------------ #


class Workspace:
    """Encapsulates a workspace.."""

    columns = ["name", "description", "stage", "path", "created", "saved"]

    def __init__(self, name: str, description: str) -> None:
        self._name = name
        self._description = description
        self._basedir = os.path.join("workspaces", name)
        self._datastore = Datastore()
        self._config = WorkspaceConfig()
        self._printer = Printer()

    @property
    def name(self) -> str:
        return self._name

    def initialize(self) -> None:
        config = self._config.get_config()
        # Create directories and sequence numbers for each stage
        config["workspaces"] = {self._name: {"seq": {}, "datasets": {}}}
        for stage in config["stages"]:
            directory = os.path.join(self._basedir, stage)
            os.makedirs(directory, exist_ok=True)
            config["workspaces"][self._name]["seq"][stage] = 0
            config["workspaces"][self._name]["datasets"][stage] = {}
        self._config.save_config(config)

    def add_dataset(self, dataset: Dataset) -> None:
        """Adds a dataset object to the inventory and persistent storage."""
        self._datastore.save_dataset(dataset)
        self._add_dataset_inventory(dataset)

    def get_dataset(self, dataset_id: str) -> None:
        """Retrieve a dataset by id."""
        return self._datastore.load_dataset(dataset_id)

    def delete_dataset(self, dataset_id: str) -> None:
        """Deletes a dataset from the workspace, storage and the inventory."""
        self._datastore.delete_dataset(filepath)

    def reset_stage(self, stage: str) -> None:
        """Deletes all datasets in a stage and resets the sequence numbers."""
        # Remove the datasets from disk.
        datasets = self._get_datasets(stage)
        for dataset_id in datasets.keys():
            self.delete_dataset(dataset_id)

        # Clear the inventory for the stage
        config = self._config.get_config()
        config["workspaces"][self._name]["datasets"][stage] = {}
        self._config.save_config(config)

    @property
    def datasets(self) -> None:
        config = self._config.get_config()
        datasets = config["workspaces"][self._name]["datasets"]
        df = pd.DataFrame(datasets)
        self._printer.print_title("Workspace {}".format(self._name), "Dataset Inventory")
        self._printer.print_dataframe(df)

    def _add_dataset_inventory(self, dataset: Dataset) -> None:
        config = self._config.get_config()
        ds = {
            "id": dataset.id,
            "name": dataset.name,
            "stage": dataset.stage,
            "creator": dataset.creator,
            "created": dataset.created,
            "saved": datetime.now(),
        }
        dsd = {dataset.id: ds}
        config["workspaces"][self._name]["datasets"][dataset.stage] = dsd
        self._config.save_config(config)

    def _get_dataset_id(self, stage: str, name: str) -> None:
        config = self._config.get_config()
        try:

            seq = config["workspaces"][self._name]["seq"][stage]
            config["workspaces"][self._name]["seq"][stage] += 1
            self._config.save_config(config)
            seq = str(seq).zfill(config["id_digits"])
            key = [seq, self._name, stage, name]
            dataset_id = "_".join(key)
            return dataset_id.lower()
        except KeyError as e:
            logger.error("Stage {} does not exist in this workspace.".format(stage))

    def _get_datasets(self, stage: str = None) -> None:
        """Returns dictionary of dataset specifications."""
        config = self._config.get_config()
        if stage:
            datasets = config["workspaces"][self_name]["datasets"][stage]
        else:
            datasets = config["workspaces"][self_name]["datasets"]
        return datasets
