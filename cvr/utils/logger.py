#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ======================================================================================================================== #
# Project  : Conversion Rate Prediction (CVR)                                                                              #
# Version  : 0.1.0                                                                                                         #
# File     : \logger.py                                                                                                    #
# Language : Python 3.10.1                                                                                                 #
# ------------------------------------------------------------------------------------------------------------------------ #
# Author   : John James                                                                                                    #
# Email    : john.james.ai.studio@gmail.com                                                                                #
# URL      : https://github.com/john-james-ai/cvr                                                                          #
# ------------------------------------------------------------------------------------------------------------------------ #
# Created  : Saturday, January 22nd 2022, 7:48:42 pm                                                                       #
# Modified : Wednesday, January 26th 2022, 3:03:46 am                                                                      #
# Modifier : John James (john.james.ai.studio@gmail.com)                                                                   #
# ------------------------------------------------------------------------------------------------------------------------ #
# License  : BSD 3-clause "New" or "Revised" License                                                                       #
# Copyright: (c) 2022 Bryant St. Labs                                                                                      #
# ======================================================================================================================== #
import os
import logging

# ------------------------------------------------------------------------------------------------------------------------ #


class LoggerFactory:
    """Custom Logger to for each workspace"""

    def get_logger(self, name: str, directory: str, verbose: bool) -> None:

        #  Create log filename and directory
        logfilename = name + ".log"
        logfilepath = os.path.join(directory, name, logfilename)
        os.makedirs(os.path.dirname(logfilepath), exist_ok=True)

        # Clear existing loggers
        logging.root.handlers = []

        # Create logger
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        # Set formatters
        file_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        stream_format = logging.Formatter("%(message)s")

        # Add file handler
        fh = logging.handlers.TimedRotatingFileHandler(logfilepath, when="d", encoding=None, delay=False)
        fh.setFormatter(file_format)
        logger.addHandler(fh)

        # Add console handler if verbose.
        if verbose:
            ch = logging.StreamHandler()
            ch.setFormatter(stream_format)
            logger.addHandler(ch)

        return logger
