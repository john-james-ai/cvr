#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ======================================================================================================================== #
# Project  : Conversion Rate Prediction (CVR)                                                                              #
# Version  : 0.1.0                                                                                                         #
# File     : \asset.py                                                                                                     #
# Language : Python 3.10.1                                                                                                 #
# ------------------------------------------------------------------------------------------------------------------------ #
# Author   : John James                                                                                                    #
# Email    : john.james.ai.studio@gmail.com                                                                                #
# URL      : https://github.com/john-james-ai/cvr                                                                          #
# ------------------------------------------------------------------------------------------------------------------------ #
# Created  : Saturday, January 22nd 2022, 5:11:23 pm                                                                       #
# Modified : Sunday, January 23rd 2022, 2:37:51 am                                                                         #
# Modifier : John James (john.james.ai.studio@gmail.com)                                                                   #
# ------------------------------------------------------------------------------------------------------------------------ #
# License  : BSD 3-clause "New" or "Revised" License                                                                       #
# Copyright: (c) 2022 Bryant St. Labs                                                                                      #
# ======================================================================================================================== #
"""Base class for workspace assets that get persisted within workspaces."""
from abc import ABC, abstractmethod

# ------------------------------------------------------------------------------------------------------------------------ #
class Asset(ABC):
    """Abstract base class for workspace assets.

    Args:
        name (str): The name of the asset
        stage (str): The stage in which the asset was created
    """

    def __init__(self, name: str, stage: str, version: int = None) -> None:
        self._name = name
        self._stage = stage
        self._version = 0 if version is None else version
        self._aid = stage + "_" + name
        self._filepath = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def stage(self) -> str:
        return self._stage

    @property
    def aid(self) -> str:
        return self._aid

    @property
    def filepath(self) -> str:
        return self._filepath

    @filepath.setter
    def filepath(self, filepath) -> None:
        self._filepath = filepath
