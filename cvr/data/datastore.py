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
# Modified : Wednesday, January 19th 2022, 12:19:53 am                                                                     #
# Modifier : John James (john.james.ai.studio@gmail.com)                                                                   #
# ------------------------------------------------------------------------------------------------------------------------ #
# License  : BSD 3-clause "New" or "Revised" License                                                                       #
# Copyright: (c) 2022 Bryant St. Labs                                                                                      #
# ======================================================================================================================== #
import os
from datetime import datetime
import pickle
import pandas as pd
import logging

from cvr.data.datasets import Dataset

# ------------------------------------------------------------------------------------------------------------------------ #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------------------------------ #


class Datastore:
    """Stores and retrieves Datasetes within a workspace."""

    def load_dataset(self, dataset_id: str) -> Dataset:
        """Loads a dataset object

        Args:
            dataset_id (str): Unique dataset identifier
        """
        filepath = self._get_filepath(dataset_id)
        return self._load_dataset(filepath)

    def save_dataset(self, dataset: Dataset) -> None:
        """Saves a Dataset object."""
        dataset = self._add_filepath(dataset)
        self._save_dataset(dataset)

    def delete_dataset(self, dataset_id) -> None:
        print("Are you sure you want to delete this dataset? [y/n]")
        answer = input()
        if "y" in answer:
            filepath = self._get_filepath(dataset_id)
            os.remove(filepath)

    def _add_filepath(self, dataset) -> None:
        filename = dataset.id + ".pkl"
        filepath = os.path.join("workspaces", dataset.workspace, dataset.stage, filename)
        dataset.filepath = filepath.lower()
        return dataset

    def _get_filepath(self, dataset_id) -> None:
        filename = dataset_id + ".pkl"
        components = dataset_id.split("_")
        return os.path.join("workspaces", components[1], components[2], filename)

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
