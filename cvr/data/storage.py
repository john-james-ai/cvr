#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ======================================================================================================================== #
# Project  : Conversion Rate Prediction (CVR)                                                                              #
# Version  : 0.1.0                                                                                                         #
# File     : \storage.py                                                                                                   #
# Language : Python 3.10.1                                                                                                 #
# ------------------------------------------------------------------------------------------------------------------------ #
# Author   : John James                                                                                                    #
# Email    : john.james.ai.studio@gmail.com                                                                                #
# URL      : https://github.com/john-james-ai/cvr                                                                          #
# ------------------------------------------------------------------------------------------------------------------------ #
# Created  : Monday, December 27th 2021, 3:14:22 am                                                                        #
# Modified : Sunday, January 16th 2022, 12:29:55 am                                                                        #
# Modifier : John James (john.james.ai.studio@gmail.com)                                                                   #
# ------------------------------------------------------------------------------------------------------------------------ #
# License  : BSD 3-clause "New" or "Revised" License                                                                       #
# Copyright: (c) 2022 Bryant St. Labs                                                                                      #
# ======================================================================================================================== #
import os
import pandas as pd
import logging
from datetime import datetime
import json

from cvr.utils.printing import Printer

# ------------------------------------------------------------------------------------------------------------------------ #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------------------------------------------------ #
#                                                 DATASTORE                                                                #
# ------------------------------------------------------------------------------------------------------------------------ #
class DataStore:
    """Dataset repository and inventory management."""

    inventory = "data/datastore.json"
    columns = ["name", "workspace", "stage", "version", "path", "created", "saved"]

    def __init__(self) -> None:
        self._printer = Printer()

    def get(self, workspace: str, stage: str, name: str, version: int) -> Dataset:
        filepath = self._get_filepath(workspace, stage, name, version)

        if os.path.exists(filepath):
            return self._load_dataset(filepath)

    def add(self, dataset: Dataset) -> None:
        """Adds a Dataset Object to the repository.

        Args:
            dataset (Dataset): The Dataset object
        """
        filepath = self._format_filepath(dataset)
        if os.path.exists(filepath):
            raise FileExistsError(filepath)
        else:
            self._save_dataset(dataset, filepath)
            self._add_inventory(dataset, filepath)

    def update(self, dataset: Dataset) -> None:
        """Updates an existing Dataset object.

        Args:
            dataset (Dataset): The Dataset object
        """
        filepath = self._format_filepath(dataset)
        if not os.path.exists(filepath):
            message = "Dataset with filepath = {} does not exist.".format(filepath)
            raise Exception(message)
        else:
            self._save_dataset(dataset, filepath)
            self._add_inventory(dataset, filepath)

    def delete(self, dataset: Dataset) -> None:
        """Removes a dataset from the repository

        Args:
            dataset (Dataset): The Dataset object
        """
        filepath = self._format_filepath(dataset)
        self._delete_dataset(filepath)
        self._delete_inventory(dataset)

    def get_inventory(self) -> pd.DataFrame:
        """Returns inventory as a dataframe"""
        inventory = self._load_inventory()
        df = pd.DataFrame.from_dict(inventory, orient="index", columns=DataStore.columns)
        return df

    def print_inventory(self) -> None:
        df = self.get_inventory()
        title = "Inventory of Dataset Objects"
        self._printer.print_dataframe(df, title)

    def _load_inventory(self) -> dict:
        if os.path.exists(DataStore.inventory):
            with open(DataStore.inventory, "r") as f:
                return json.loads(f)
        else:
            return {}

    def _save_inventory(self, inventory: dict) -> None:
        with open(DataStore.inventory, "w") as f:
            json.dump(inventory, f)

    def _add_inventory(self, dataset: Dataset, filepath: str) -> None:
        inventory = self._load_inventory()
        inventory[dataset.id] = {
            "name": dataset.name,
            "workspace": dataset.workspace,
            "stage": dataset.stage,
            "version": dataset.version,
            "path": filepath,
            "created": dataset.created,
            "saved": datetime.now(),
        }
        self._save_inventory(inventory)

    def _delete_inventory(self, dataset: Dataset) -> None:
        inventory = self._load_inventory()
        inventory.pop(dataset.id, None)
        self._save_inventory(inventory)

    def _get_filepath(self, workspace: str, stage: str, name: str, version: int) -> str:
        inventory = self.get_inventory()
        return inventory.loc[
            (inventory["workspace"] == workspace)
            & (inventory["stage"] == stage)
            & (inventory["name"] == name)
            & (inventory["version"] == version)
        ]["filepath"].values

    def _format_filepath(self, dataset: Dataset) -> str:
        directory = os.path.join("data", dataset.workspace, dataset.stage)
        filename = ""
        filename += dataset.workspace + "_"
        filename += dataset.stage + "_"
        filename += str(dataset.version) + ".pkl"
        return os.path.join(directory, filename)

    def _load_dataset(self, filepath) -> Dataset:
        picklefile = open(filepath, "rb")
        dataset = pickle.load(picklefile)
        picklefile.close()
        return dataset

    def _save_dataset(self, dataset: Dataset, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        picklefile = open(filepath, "wb")
        pickle.dump(dataset, picklefile)
        picklefile.close()

    def _delete_dataset(self, filepath: str) -> None:
        if os.path.exists(filepath):
            os.remove(filepath)
