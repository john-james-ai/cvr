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
# Modified : Wednesday, January 19th 2022, 9:21:14 am                                                                      #
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


from cvr.utils.config import DatastoreConfig
from cvr.utils.printing import Printer
from cvr.data.datasets import Dataset

# ------------------------------------------------------------------------------------------------------------------------ #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------------------------------ #


class Datastore:
    """Manages dataset persistence."""

    def __init__(self, workspace_name: str) -> None:
        self._name = name
        self._home = os.path.join("workspaces", workspace_name, "datasets")
        directory_file = os.path.join(self._home, "directory.yaml")
        config_file = os.path.join(self._home, "config.yaml")
        self._config = DatastoreConfig(config_file)
        self._directory = DatastoreConfig(directory_file)
        self._printer = Printer()
        self._initialize()
        self._stage = 0
        self._version = 0

    @property
    def directory(self) -> str:
        return self._directory

    def initialize(self) -> None:
        """Initialize the workspace configuration."""
        # Create data directory
        shutil.rmtree(self._directory, ignore_errors=True)
        # Write the config file
        config = {"stage": 0, "version": 0}
        self._config.save_config(config)

    def add_dataset(self, dataset: Dataset) -> None:
        """Adds a dataset object to the inventory and persistent storage."""
        self._add_to_inventory(dataset)
        self._save_dataset(dataset)

    def get_dataset(self, dataset_id: str) -> None:
        """Retrieve a dataset by id."""
        config = self._directory.get_config()
        if dataset_id in config.keys():
            filepath = config[dataset_id]["filepath"]
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
        config = self._config.get_config()
        ds = {
            "name": dataset.name,
            "stage": dataset.stage,
            "filepath": dataset.filepath,
            "creator": dataset.creator,
            "created": dataset.created,
            "saved": datetime.now(),
        }
        config["datasets"][dataset.id] = ds
        self._config.save_config(config)

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
