#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ======================================================================================================================== #
# Project  : Explainable Recommendation (XRec)                                                                             #
# Version  : 0.1.0                                                                                                         #
# File     : \test_profiler.py                                                                                             #
# Language : Python 3.8                                                                                                    #
# ------------------------------------------------------------------------------------------------------------------------ #
# Author   : John James                                                                                                    #
# Email    : john.james.ai.studio@gmail.com                                                                                #
# URL      : https://github.com/john-james-ai/xrec                                                                         #
# ------------------------------------------------------------------------------------------------------------------------ #
# Created  : Tuesday, December 28th 2021, 1:06:31 am                                                                       #
# Modified : Sunday, January 2nd 2022, 2:00:21 pm                                                                          #
# Modifier : John James (john.james.ai.studio@gmail.com)                                                                   #
# ------------------------------------------------------------------------------------------------------------------------ #
# License  : BSD 3-clause "New" or "Revised" License                                                                       #
# Copyright: (c) 2021 Bryant St. Labs                                                                                      #
# ======================================================================================================================== #
#%%
import os
import pytest
import logging
import numpy as np
import pandas as pd
import shutil
import inspect

from cvr.data.outliers import (
    OutlierZScore,
    OutlierIQR,
    OutlierDetector,
    OutlierAVF,
    OutlierWAVF,
    OutlierSCF,
)

# ------------------------------------------------------------------------------------------------------------------------ #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------------------------------ #


class OutlierTests:
    def __init__(self):
        filepath = "tests\\test_data\staged\criteo.csv"
        self._X = pd.read_csv(filepath, low_memory=False)
        self._z = OutlierZScore()
        self._iqr = OutlierIQR()
        self._o = OutlierDetector()

    def test_zscore(self):
        logger.info(
            "    Started {} {}".format(self.__class__.__name__, inspect.stack()[0][3])
        )
        self._z.fit(self._X)
        labels = self._z.predict(self._X)
        print("OutliersZScore detected {} outliers.".format(labels.sum()))

        assert isinstance(labels, (list, np.ndarray)), logger.error(
            "Failure in {}.".format(inspect.stack()[0][3])
        )

        logger.info(
            "\t\tSuccessfully completed {} {}".format(
                self.__class__.__name__, inspect.stack()[0][3]
            )
        )

    def test_iqr(self):
        logger.info(
            "    Started {} {}".format(self.__class__.__name__, inspect.stack()[0][3])
        )
        self._iqr.fit(self._X)
        labels = self._iqr.predict(self._X)
        print("OutliersIQR detected {} outliers.".format(labels.sum()))

        assert isinstance(labels, (list, np.ndarray)), logger.error(
            "Failure in {}.".format(inspect.stack()[0][3])
        )

        logger.info(
            "\t\tSuccessfully completed {} {}".format(
                self.__class__.__name__, inspect.stack()[0][3]
            )
        )

    def test_avf(self):
        logger.info(
            "    Started {} {}".format(self.__class__.__name__, inspect.stack()[0][3])
        )
        self._avf = OutlierAVF()
        self._avf.fit(self._X)
        labels = self._avf.predict(self._X)
        print("OutlierAVF detected {} outliers.".format(labels.sum()))

        assert isinstance(labels, (list, np.ndarray)), logger.error(
            "Failure in {}.".format(inspect.stack()[0][3])
        )

        logger.info(
            "\t\tSuccessfully completed {} {}".format(
                self.__class__.__name__, inspect.stack()[0][3]
            )
        )

    def test_outliers(self):
        logger.info(
            "    Started {} {}".format(self.__class__.__name__, inspect.stack()[0][3])
        )
        self._o.fit(self._X)
        labels = self._o.predict(self._X)

        print(
            "Summary of outlier analysis \n\n{}.".format(
                self._o.results_["ensemble"]["summary"]
            )
        )

        logger.info(
            "\t\tSuccessfully completed {} {}".format(
                self.__class__.__name__, inspect.stack()[0][3]
            )
        )

    def test_scf(self):
        logger.info(
            "    Started {} {}".format(self.__class__.__name__, inspect.stack()[0][3])
        )
        self._scf = OutlierSCF()
        self._scf.fit(self._X)
        labels = self._scf.predict(self._X)
        print("OutlierSCF detected {} outliers.".format(labels.sum()))

        assert isinstance(labels, (list, np.ndarray)), logger.error(
            "Failure in {}.".format(inspect.stack()[0][3])
        )

        logger.info(
            "\t\tSuccessfully completed {} {}".format(
                self.__class__.__name__, inspect.stack()[0][3]
            )
        )

    def test_wavf(self):
        logger.info(
            "    Started {} {}".format(self.__class__.__name__, inspect.stack()[0][3])
        )
        self._wavf = OutlierWAVF()
        self._wavf.fit(self._X)
        labels = self._wavf.predict(self._X)
        print("OutlierWAVF detected {} outliers.".format(labels.sum()))

        assert isinstance(labels, (list, np.ndarray)), logger.error(
            "Failure in {}.".format(inspect.stack()[0][3])
        )

        logger.info(
            "\t\tSuccessfully completed {} {}".format(
                self.__class__.__name__, inspect.stack()[0][3]
            )
        )

    def test_outliers(self):
        logger.info(
            "    Started {} {}".format(self.__class__.__name__, inspect.stack()[0][3])
        )
        self._o.fit(self._X)
        labels = self._o.predict(self._X)

        print(
            "Summary of outlier analysis \n\n{}.".format(
                self._o.results_["ensemble"]["summary"]
            )
        )

        logger.info(
            "\t\tSuccessfully completed {} {}".format(
                self.__class__.__name__, inspect.stack()[0][3]
            )
        )


if __name__ == "__main__":
    t = OutlierTests()
    # t.test_zscore()
    # t.test_iqr()
    t.test_avf()
    t.test_scf()
    t.test_wavf()
    t.test_outliers()
#%%
