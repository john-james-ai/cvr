#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ============================================================================ #
# Project  : Deep Learning for Conversion Rate Prediction (CVR)                #
# Version  : 0.1.0                                                             #
# File     : \logger.py                                                        #
# Language : Python 3.7.12                                                     #
# ---------------------------------------------------------------------------- #
# Author   : John James                                                        #
# Email    : john.james.ai.studio@gmail.com                                    #
# URL      : https://github.com/john-james-ai/cvr                              #
# ---------------------------------------------------------------------------- #
# Created  : Saturday, January 22nd 2022, 7:48:42 pm                           #
# Modified : Friday, February 4th 2022, 9:24:57 pm                             #
# Modifier : John James (john.james.ai.studio@gmail.com)                       #
# ---------------------------------------------------------------------------- #
# License  : BSD 3-clause "New" or "Revised" License                           #
# Copyright: (c) 2022 Bryant St. Labs                                          #
# ============================================================================ #
import os
import logging

# ---------------------------------------------------------------------------- #
FORMATTER_CONSOLE = logging.Formatter("%(message)s")
FORMATTER_FILE = logging.Formatter(
    "%(asctime)s — %(name)s — %(levelname)s — %(message)s"
)
LOGGING_LEVELS = {"info": logging.INFO, "debug": logging.DEBUG}
# ---------------------------------------------------------------------------- #


class LoggerFactory:
    def __init__(
        self, name: str, directory: str, logging_level: str = "info"
    ) -> None:
        self._name = name
        self._directory = directory
        self._logging_level = logging_level
        self._logger = None

    @property
    def logging_level(self) -> str:
        return self._logging_level

    @logging_level.setter
    def logging_level(self, logging_level: str) -> None:
        self._logging_level = logging_level

    def _get_console_handler(self) -> logging.StreamHandler:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(FORMATTER_CONSOLE)
        return console_handler

    def _get_file_handler(self) -> logging.handlers.TimedRotatingFileHandler:
        log_filename = self._name + ".log"
        log_filepath = os.path.join(self._directory, log_filename)
        os.makedirs(os.path.dirname(log_filepath), exist_ok=True)
        file_handler = logging.handlers.TimedRotatingFileHandler(
            filename=log_filepath, when="d", encoding=None, delay=False
        )
        file_handler.setFormatter(FORMATTER_FILE)
        return file_handler

    def get_logger(self) -> logging:
        """Returns a configured logging object."""
        logger = logging.getLogger(self._name)
        logger.setLevel(LOGGING_LEVELS.get(self._logging_level, logging.INFO))
        logger.addHandler(self._get_console_handler())
        logger.addHandler(self._get_file_handler())
        return logger
