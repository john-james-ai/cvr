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
# Modified : Sunday, January 23rd 2022, 5:30:33 am                                                                         #
# Modifier : John James (john.james.ai.studio@gmail.com)                                                                   #
# ------------------------------------------------------------------------------------------------------------------------ #
# License  : BSD 3-clause "New" or "Revised" License                                                                       #
# Copyright: (c) 2022 Bryant St. Labs                                                                                      #
# ======================================================================================================================== #
import os
import logging

# ------------------------------------------------------------------------------------------------------------------------ #


class LoggerFactory:
    """Custom Logger to File and to Console"""

    def get_logger(self, workspace: str, stage: str, name: str, verbose: bool) -> None:

        #  Create log filename and directory
        logname = stage + "_" + name
        logfilename = logname + ".log"
        logfilepath = os.path.join("workspaces", workspace, "logs", logfilename)
        os.makedirs(os.path.dirname(logfilepath), exist_ok=True)

        # Clear existing loggers
        logging.root.handlers = []

        # Create logger
        logger = logging.getLogger(logname)
        logger.setLevel(logging.DEBUG)

        # Set formatter
        format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        # Add file handler
        fh = logging.handlers.TimedRotatingFileHandler(logfilepath, when="d", encoding=None, delay=False)
        fh.setFormatter(format)
        logger.addHandler(fh)

        # Add console handler if verbose.
        if verbose:
            ch = logging.StreamHandler()
            ch.setFormatter(format)
            logger.addHandler(ch)

        return logger
