#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ============================================================================ #
# Project  : Deep Learning for Conversion Rate Prediction (CVR)                #
# Version  : 0.1.0                                                             #
# File     : \io.py                                                            #
# Language : Python 3.7.12                                                     #
# ---------------------------------------------------------------------------- #
# Author   : John James                                                        #
# Email    : john.james.ai.studio@gmail.com                                    #
# URL      : https://github.com/john-james-ai/cvr                              #
# ---------------------------------------------------------------------------- #
# Created  : Tuesday, January 25th 2022, 1:35:16 pm                            #
# Modified : Wednesday, February 2nd 2022, 8:21:20 pm                          #
# Modifier : John James (john.james.ai.studio@gmail.com)                       #
# ---------------------------------------------------------------------------- #
# License  : BSD 3-clause "New" or "Revised" License                           #
# Copyright: (c) 2022 Bryant St. Labs                                          #
# ============================================================================ #
"""IO related utilities."""
import os
import pickle

# ---------------------------------------------------------------------------- #
class Pickler:
    """Wraps basic pickle operations."""

    def load(self, filepath) -> dict:
        """Unpickles a asset"""
        try:
            picklefile = open(filepath, "rb")
            asset = pickle.load(picklefile)
            picklefile.close()
            return asset
        except FileNotFoundError as e:
            return None

    def remove(self, filepath) -> None:
        """Removes an asset.

        Args:
            filepath (str): Location of asset.
        """
        try:
            os.remove(filepath)
        except FileNotFoundError as e:
            print(e)

    def save(self, asset, filepath) -> None:
        """Pickles a asset

        Args:
            asset (Asset): Payload
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        picklefile = open(filepath, "wb")
        pickle.dump(asset, picklefile)
        picklefile.close()

    def exists(self, filepath) -> bool:
        """Returns a boolean indicating whether a file exists at the filepath

        Args:
            filepath (str): The path to the file
        Returns:
            bool
        """
        return os.path.exists(filepath)
