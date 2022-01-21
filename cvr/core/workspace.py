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
# Modified : Wednesday, January 19th 2022, 9:17:56 pm                                                                      #
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
    """Defines a workspace encapsulating datasets and stages."""

    columns = ["name", "description", "stage", "path", "created", "saved"]

    def __init__(self, name: str, description: str) -> None:
        self._name = name
        self._description = description
        self._directory = os.path.join("workspaces", name)
        self._datasets_filepath = os.path.join(self._directory, "datasets")
        config_file = os.path.join(self._directory, "config.yaml")
        self._workspace_config = WorkspaceConfig(config_file)
        self._pipeline = None

        self._printer = Printer()
        self._initialize()

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def directory(self) -> str:
        return self._directory

    def initialize(self) -> None:
        """Initialize the workspace configuration."""
        # Create the stage sequence tables
        stages = self._project_config.get_config("stages")
        config = {}
        config["stages"] = {}
        for stage in stages:
            config["stages"][stage] = 0
        self._workspace_config.save_config(config)

        # Delete all data and recreate stage directories.
        shutil.rmtree(self._dataset_filepath, ignore_errors=True)

    def build_dataset(self)

    def add_dataset(self, dataset: Dataset) -> None:
        """Adds a dataset object to the inventory and persistent storage."""
        self._add_to_inventory(dataset)
        self._save_dataset(dataset)

    def get_dataset(self, dataset_id: str) -> None:
        """Retrieve a dataset by id."""
        config = self._workspace_config.get_config()
        if dataset_id in config["datasets"].keys():
            filepath = config["datasets"][dataset_id]["filepath"]
            return self._load_dataset(filepath)

    @property
    def datasets(self) -> None:
        self._print_inventory()

    # --------------------------------------------------------------------------------------------------------------------- #
    #                                                      IO                                                               #
    # --------------------------------------------------------------------------------------------------------------------- #

    def _load_dataset(self, filepath) -> Dataset:
        """Unpickles a dataset object.

        Args:
            filepath (str): Location of file.
        """
        picklefile = open(filepath, "rb")
        dataset = pickle.load(picklefile)
        picklefile.close()
        return dataset

    def _save_dataset(self, dataset: Dataset) -> None:
        """Pickles a dataset

        Args:
            dataset (Dataset): Data
        """
        picklefile = open(dataset.filepath, "wb")
        pickle.dump(dataset, picklefile)
        picklefile.close()

    # --------------------------------------------------------------------------------------------------------------------- #
    #                                            INVENTORY MANAGEMENT                                                       #
    # --------------------------------------------------------------------------------------------------------------------- #

    def get_ids(self, stage: str = None) -> None:
        ids = []
        config = self._workspace_config.get_config()
        if stage is not None:
            for id, dataset in config.items():
                if stage == dataset["stage"]:
                    ids.append(dataset["ids"])
        return ids

    def _add_to_inventory(self, dataset: Dataset) -> None:
        config = self._workspace_config.get_config()
        ds = {
            "name": dataset.name,
            "stage": dataset.stage,
            "filepath": dataset.filepath,
            "creator": dataset.creator,
            "created": dataset.created,
            "saved": datetime.now(),
        }
        config["datasets"][dataset.id] = ds
        self._workspace_config.save_config(config)

    def _remove_from_inventory(self, dataset_id: str) -> None:
        config = self._workspace_config.get_config()
        try:
            del config[dataset_id]
        except KeyError as e:
            logger.warning("Dataset with id {} does not exist.".format(dataset_id))
        self._workspace_config.save_config(config)

    def _print_inventory(self) -> None:
        """Returns dictionary of dataset specifications."""
        config = self._config.get_config()
        df = pd.DataFrame.from_dict(config[self._name]["datasets"], orient="index")
        self._printer.print_title("Workspace Inventory", self._name)
        self._printer.print_dataframe(df)