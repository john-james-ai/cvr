#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ======================================================================================================================== #
# Project  : Explainable Recommendation (XRec)                                                                             #
# Version  : 0.1.0                                                                                                         #
# File     : \criteo.py                                                                                                    #
# Language : Python 3.8                                                                                                    #
# ------------------------------------------------------------------------------------------------------------------------ #
# Author   : John James                                                                                                    #
# Email    : john.james.ai.studio@gmail.com                                                                                #
# URL      : https://github.com/john-james-ai/xrec                                                                         #
# ------------------------------------------------------------------------------------------------------------------------ #
# Created  : Sunday, December 26th 2021, 3:56:00 pm                                                                        #
# Modified : Friday, January 14th 2022, 6:46:32 pm                                                                         #
# Modifier : John James (john.james.ai.studio@gmail.com)                                                                   #
# ------------------------------------------------------------------------------------------------------------------------ #
# License  : BSD 3-clause "New" or "Revised" License                                                                       #
# Copyright: (c) 2021 Bryant St. Labs                                                                                      #
# ======================================================================================================================== #
from abc import ABC, abstractmethod
import os
import pandas as pd
import numpy as np
import logging
import math
from scipy import stats
from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from cvr.utils.printing import Printer

# ------------------------------------------------------------------------------------------------------------------------ #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------------------------------ #

DEFAULT_N_JOBS = 18
# ======================================================================================================================== #
#                                              OUTLIER DETECTION                                                           #
# ======================================================================================================================== #
class OutlierDetector:
    """Outlier detection with selected outlier detection algorithms.

    Args:
        criterion (str): Indicates criterion for final determination of an observation, given results
            from various outlier detection algorithms. Values include 'any', 'all', 'vote' for
            majority vote.
        numeric_algorithms(dict): Dictionary of instantiated numeric outlier detection algorithms
        categorical_algorithms(dict): Dictionary of instantiated categorical outlier detection algorithms
        random_state (int): Pseudo random generator seed for Isolation Forest

    Attributes:
        results_: Contains a nested dictionary with three numeric, categorical and combined outlier labels
        summary_:

    Returns:
        Numpy array containing the labels labels

    """

    def __init__(
        self,
        criterion="vote",
        numeric_algorithms: dict = None,
        categorical_algorithms: dict = None,
        random_state=None,
    ) -> None:
        self._criterion = criterion
        self._random_state = random_state
        self.results_ = {}
        self._n = None

        # Numeric Outlier Detection Algorithms
        self._numeric_algorithms = (
            numeric_algorithms
            if numeric_algorithms
            else {
                "Z-Score": OutlierZScore(),
                "IQR": OutlierIQR(),
                "Robust Covariance": OutlierEllipticEnvelope(random_state=random_state),
                "Isolation Forest": OutlierIsolationForest(random_state=random_state),
                "Local Outlier Factor": OutlierLocalOutlierFactor(),
            }
        )
        # Categorical Outlier Detection Algorithms
        self._categorical_algorithms = (
            categorical_algorithms
            if categorical_algorithms
            else {
                "Attribute Value Frequency": OutlierAVF(),
                "Square of Complement Frequency": OutlierSCF(),
                "Weighted Attribute Value Frequency": OutlierWAVF(),
            }
        )
        # Algorithms for numeric and categorical (object) data outlier detection
        self._detectors = {
            "number": self._numeric_algorithms,
            "object": self._categorical_algorithms,
        }

    def fit(self, X, y=None):
        """Fits several outlier detection algorithms.

        Args:
            X (pd.DataFrame): Input
        """
        self._n = len(X)
        labels_ensemble = {}
        for datatype, algorithms in self._detectors.items():
            labels_datatype = {}
            X_datatype = X.select_dtypes(include=datatype)
            for name, algorithm in algorithms.items():
                name_datatype = name + " (" + datatype + ")"
                print(
                    "Currently fitting outlier detector {}.".format(name_datatype),
                    end=" ",
                )
                algorithm.fit(X_datatype)
                labels = algorithm.predict(X_datatype)

                o = labels.sum()
                p = round(o / self._n * 100, 2)
                print("Detected {} outliers, {}% of the data.".format(str(o), str(p)))

                labels_datatype[name] = labels
                labels_ensemble[name_datatype] = labels
            self.results_[datatype] = self._compute_results(labels_datatype, datatype)

        # Combine results for numeric and categorical outlier labels
        self.results_["ensemble"] = self._compute_results(labels_ensemble, "combined")

    def predict(self, X) -> pd.DataFrame:
        o = self.results_["ensemble"]["labels"].sum()
        p = round(o / self._n * 100, 2)
        print(
            "\nThe ensemble detected {} outliers constituting {}% of the data using the {} criterion.".format(
                str(o), str(p), str(self._criterion)
            )
        )
        return self.results_["ensemble"]["labels"].to_frame().reset_index()

    def _compute_results(self, labels: dict, datatype: str) -> dict:
        """Aggregates results for several outlier detection algorithms."""
        d = {}
        # Store labels by algorithm
        d["labels_by_algorithm"] = pd.DataFrame.from_dict(labels, orient="columns")
        # Store aggregated labels based upon the criteria
        d["labels_any"] = d["labels_by_algorithm"].any(axis=1)
        d["labels_all"] = d["labels_by_algorithm"].all(axis=1)
        d["labels_vote"] = d["labels_by_algorithm"].mean(axis=1) > 0.5

        # Store the labels according to the selected criterion
        if self._criterion == "any":
            d["labels"] = d["labels_any"]
        elif self._criterion == "all":
            d["labels"] = d["labels_all"]
        else:
            d["labels"] = d["labels_vote"]

        # Update labels by algorithms to include the labels aggregated by the three criteria
        all_desc = self._get_label_description(datatype, " (All)")
        any_desc = self._get_label_description(datatype, " (Any)")
        vote_desc = self._get_label_description(datatype, " (Majority Vote)")
        ensemble_desc = self._get_label_description(datatype, "")

        d["labels_by_algorithm"][all_desc] = d["labels_all"]
        d["labels_by_algorithm"][any_desc] = d["labels_any"]
        d["labels_by_algorithm"][vote_desc] = d["labels_vote"]
        d["labels_by_algorithm"][ensemble_desc] = d["labels"]

        # Aggregate the total counts for all algorithms for selected and criteria
        d["summary"] = d["labels_by_algorithm"].sum()
        return d

    def _get_label_description(self, datatype: str, criterion: str) -> str:
        if datatype == "number":
            return "Numeric Ensemble" + criterion
        elif datatype == "object":
            return "Categorical Ensemble" + criterion
        else:
            return "Combined Ensemble" + criterion


# ------------------------------------------------------------------------------------------------------------------------ #
#                                          OUTLIER ANALYSIS Z-SCORE                                                        #
# ------------------------------------------------------------------------------------------------------------------------ #
class OutlierZScore:
    def __init__(self, threshold: int = 3) -> None:
        self._threshold = threshold
        self._labels = None

    def fit(self, X, y=None) -> None:
        """Computes the zscores for a 2 dimensional array.

        Args:
            X (pd.DataFrame): Input
        """
        # Convert dataframe to numpy array.
        X = X.select_dtypes(include="number").values
        z = stats.zscore(X)
        labels = np.where(np.abs(z) > self._threshold, 1, 0)
        self._labels = np.any(labels, axis=1)

    def predict(self, X):
        """Returns the prediction

        Args:
            X (np.array): Input
        """
        return self._labels


# ------------------------------------------------------------------------------------------------------------------------ #
#                                            OUTLIER ANALYSIS IQR                                                          #
# ------------------------------------------------------------------------------------------------------------------------ #
class OutlierIQR:
    def __init__(self, threshold: float = 1.5) -> None:
        self._threshold = threshold
        self._labels = None

    def fit(self, X, y=None) -> None:
        """Computes the zscores for a 2 dimensional array.

        X (pd.DataFrame): Input
        """
        # Convert dataframe to numpy array.
        X = X.select_dtypes(include="number").values
        q1, q3 = np.percentile(a=X, q=[25, 75], axis=0)
        iqr = q3 - q1
        lower = q1 - (iqr * self._threshold)
        upper = q3 + (iqr * self._threshold)

        labels = np.where(np.greater(X, upper) | np.less(X, lower), 1, 0)
        self._labels = np.any(labels, axis=1)

    def predict(self, X) -> np.array:
        return self._labels


# ======================================================================================================================== #
#                                   SKLEARN OUTLIER DETECTOR WRAPPERS                                                      #
# ======================================================================================================================== #
class OutliersSKLearn(ABC):
    """Abstract base class for sklearn outlier detectors wrappers.

    The SKLearn classifiers cannot handle NaNs. Hence, NaNs were replaced as follows:
        - Numeric variables replace NaNs with the mean.
        - Categorical variables replace NaNs with -1
    """

    def __init__(
        self,
        contamination: float = None,
        n_jobs: int = DEFAULT_N_JOBS,
        random_state: int = None,
        **kwargs
    ) -> None:
        self._contamination = contamination
        self._n_jobs = n_jobs
        self._random_state = random_state
        self._clf = self.get_clf()

    @abstractmethod
    def get_clf(self) -> None:
        pass

    def fit(self, X: pd.DataFrame, y: np.ndarray = None) -> None:
        X = X.select_dtypes(include="number")
        X = self._impute(X).values
        self._clf.fit(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = X.select_dtypes(include="number")
        X = self._impute(X).values
        labels = self._clf.predict(X)
        return np.where(labels == -1, 1, 0)

    def _impute(self, X) -> pd.DataFrame:
        """Imputes missing numerics with their means and missing categoricals with '-1'"""
        imputer = {
            "sale": 0,
            "sales_amount": X["sales_amount"].mean(),
            "conversion_time_delay": X["conversion_time_delay"].mean(),
            "click_ts": X["click_ts"].mean(),
            "n_clicks_1week": X["n_clicks_1week"].mean(),
            "product_price": X["product_price"].mean(),
            "product_age_group": "-1",
            "device_type": "-1",
            "audience_id": "-1",
            "product_gender": "-1",
            "product_brand": "-1",
            "product_category_1": "-1",
            "product_category_2": "-1",
            "product_category_3": "-1",
            "product_category_4": "-1",
            "product_category_5": "-1",
            "product_category_6": "-1",
            "product_category_7": "-1",
            "product_country": "-1",
            "product_id": "-1",
            "product_title": "-1",
            "partner_id": "-1",
            "user_id": "-1",
        }
        X.fillna(value=imputer, inplace=True)
        return X


# ------------------------------------------------------------------------------------------------------------------------ #
#                                    OUTLIER ANALYSIS ELLIPTIC ENVELOPE                                                    #
# ------------------------------------------------------------------------------------------------------------------------ #
class OutlierEllipticEnvelope(OutliersSKLearn):
    """Wrapper for sklearn's Elliptic Envelope class which accepts dataframes as input.

    Args:
        support_fraction (float): The proportion of points to be included in the support of the raw MCD estimate. If None, the minimum value of support_fraction will be used within the algorithm: [n_sample + n_features + 1] / 2. Range is (0, 1). Default is None.
        contamination (float): The amount of contamination of the data set, i.e. the proportion of outliers in the data set. Range is (0, 0.5]. Default is 0.1
        random_state (int): Pseudo random generator seed. Default is None.
    """

    def __init__(
        self,
        support_fraction: float = 0.6,
        contamination: float = 0.1,
        random_state: int = None,
    ) -> None:
        self._support_fraction = support_fraction
        super(OutlierEllipticEnvelope, self).__init__(
            contamination=contamination, random_state=random_state
        )

    def get_clf(self):
        return EllipticEnvelope(
            support_fraction=self._support_fraction,
            contamination=self._contamination,
            random_state=self._random_state,
        )


# ------------------------------------------------------------------------------------------------------------------------ #
#                                    OUTLIER ANALYSIS ISOLATION FOREST                                                     #
# ------------------------------------------------------------------------------------------------------------------------ #
class OutlierIsolationForest(OutliersSKLearn):
    """Wrapper for sklearn's Isolation Forest class which accepts dataframes as input.

    Args:
        contamination (float): The amount of contamination of the data set, i.e. the proportion of outliers in
            the data set. Range is (0, 0.5]. Default is 0.1
        n_jobs (int). The number of jobs to run in parallel.
        random_state (int): Pseudo random generator seed. Default is None.
    """

    def __init__(
        self,
        contamination="auto",
        n_jobs: int = DEFAULT_N_JOBS,
        random_state: int = None,
    ) -> None:
        super(OutlierIsolationForest, self).__init__(
            contamination=contamination, n_jobs=n_jobs, random_state=random_state
        )

    def get_clf(self):
        return IsolationForest(
            contamination=self._contamination,
            n_jobs=self._n_jobs,
            random_state=self._random_state,
        )


# ------------------------------------------------------------------------------------------------------------------------ #
#                                    OUTLIER ANALYSIS ISOLATION FOREST                                                     #
# ------------------------------------------------------------------------------------------------------------------------ #
class OutlierLocalOutlierFactor(OutliersSKLearn):
    """Wrapper for sklearn's Local Outlier Factor class which accepts dataframes as input.

    Args:
        contamination (float): The amount of contamination of the data set, i.e. the proportion of outliers in
            the data set. Range is (0, 0.5]. Default is 0.1
        n_jobs (int). The number of jobs to run in parallel.
        random_state (int): Pseudo random generator seed. Default is None.
    """

    def __init__(self, contamination="auto", n_jobs: int = DEFAULT_N_JOBS) -> None:
        super(OutlierLocalOutlierFactor, self).__init__(
            contamination=contamination, n_jobs=n_jobs
        )

    def get_clf(self):
        return LocalOutlierFactor(
            contamination=self._contamination, n_jobs=self._n_jobs
        )

    def predict(self, X: pd.DataFrame) -> None:
        X = X.select_dtypes(include="number")
        X = self._impute(X).values
        labels = self._clf.fit_predict(X)
        return np.where(labels == -1, 1, 0)


# ======================================================================================================================== #
#                                        OUTLIER CATEGORICAL ANALYSIS                                                      #
# ======================================================================================================================== #
# ------------------------------------------------------------------------------------------------------------------------ #
#                                 OUTLIER ANALYSIS ATTRIBUTE VALUE FREQUENCY                                               #
# ------------------------------------------------------------------------------------------------------------------------ #
class OutlierAVF:
    """Detects outliers using the Attribute Value Frequency method.

    Args:
        threshold (float): The threshold used to determine the lowest M AVF scores. Assuming frequencies are normally
            distributed.
    """

    def __init__(self, threshold: float = 0.1) -> None:
        self._threshold = threshold
        self._labels = None

    def fit(self, X, y=None) -> None:
        """Fits the model

        X (pd.DataFrame): Input
        """
        X = X.select_dtypes(include="object")
        df = pd.DataFrame()

        # Iterative over columns and create dataframe that contains the frequencies of the values.
        for col in X.columns:
            # Create a one column dataframe
            df1 = X[col].to_frame()
            # Compute value counts and convert series to frame
            df2 = df1.value_counts().to_frame().reset_index()
            df2.columns = ["value", "count"]
            # Merge the two dataframes and extract the column with the frequencies and add to new dataframe
            merged = pd.merge(df1, df2, how="left", left_on=col, right_on="value")
            df[col] = merged["count"]

        # We need to determine a threhold in terms of the observations with the M lowest AVF scores.
        # Taking the assumption that frequences are normally distributed, we can select the
        # observations with avf scores below a number of standard deviations below the mean avf.
        avf = df.mean(axis=1)
        n = len(df)
        k = math.ceil(n * self._threshold)
        threshold = avf.sort_values().head(k).max()
        self._labels = avf < threshold

    def predict(self, X) -> np.array:
        # Convert the dataframe to a numpy array to comport with the other estimators.
        return self._labels.values


# ------------------------------------------------------------------------------------------------------------------------ #
#                             OUTLIER ANALYSIS WEIGHTED ATTRIBUTE VALUE FREQUENCY                                          #
# ------------------------------------------------------------------------------------------------------------------------ #
class OutlierWAVF:
    """Detects outliers using the Weighted Attribute Value Frequency method.

    Args:
        threshold (float): The threshold used to determine the lowest M WAVF scores. Assuming frequencies are normally
            distributed.
    """

    def __init__(self, threshold: float = 0.1) -> None:
        self._threshold = threshold
        self._labels = None

    def fit(self, X, y=None) -> None:
        """Fits the model

        X (pd.DataFrame): Input
        """
        X = X.select_dtypes(include="object")
        weights = self._compute_weights(X)
        df = pd.DataFrame()

        # Iterative over columns and create dataframe that contains the frequencies of the values.
        for col in X.columns:
            # Create a one column dataframe
            df1 = X[col].to_frame()
            # Compute value counts and convert series to frame
            df2 = df1.value_counts().to_frame().reset_index()
            df2.columns = ["value", "count"]
            # Merge the two dataframes and extract the column with the frequencies and add to new dataframe
            merged = pd.merge(df1, df2, how="left", left_on=col, right_on="value")
            df[col] = merged["count"] * weights[col]

        # We need to determine a threhold in terms of the observations with the M lowest AVF scores.
        # Taking the assumption that frequences are normally distributed, we can select the
        # observations with avf scores below a number of standard deviations below the mean avf.
        avf = df.mean(axis=1)
        n = len(df)
        k = math.ceil(n * self._threshold)
        threshold = avf.sort_values().head(k).max()
        self._labels = avf < threshold

    def predict(self, X) -> np.array:
        # Convert the dataframe to a numpy array to comport with the other estimators.
        return self._labels.values

    def _compute_weights(self, X: pd.DataFrame) -> dict:
        """Computes the weights as the range of frequencies for each variable."""
        weights = {}
        for col in X.columns:
            counts = X[col].value_counts()
            weights[col] = max(counts) - min(counts)
        return weights


# ------------------------------------------------------------------------------------------------------------------------ #
#                              OUTLIER ANALYSIS SQUARE OF THE COMPLEMENT FREQUENCY                                         #
# ------------------------------------------------------------------------------------------------------------------------ #
class OutlierSCF:
    """Detects outliers using the Square of the Complement Frequency (SCF).

    Args:
        threshold (float): The threshold used to determine the lowest M SCF scores. Assuming frequencies are normally
            distributed.
    """

    def __init__(self, threshold: float = 3) -> None:
        self._threshold = threshold
        self._labels = None

    def fit(self, X, y=None) -> None:
        """Fits the model

        X (pd.DataFrame): Input
        """
        X = X.select_dtypes(include="object")
        df = pd.DataFrame()
        n = X.shape[0]

        # Iterative over columns and create dataframe that contains the frequencies of the values.
        for col in X.columns:
            # Create a one column dataframe
            df1 = X[col].to_frame()
            # Get the number of categories in col
            c = X[col].nunique()
            # Compute the marginal relative frequency  (value counts / number of observations)
            p = df1.value_counts().to_frame() / n
            # Compute the square complement frequency
            df2 = (1 - p) ** 2 / c
            # Merge the two dataframes and extract the column with the frequencies and add to new dataframe
            df[col] = pd.merge(df1, df2, on=[col], how="left")[0]

        # Take the sum across columns
        scf = df.sum(axis=1)
        # Designate the scores above threshold standard deviations of the man as outliers
        upper_bound = scf.mean() + scf.std() * self._threshold
        self._labels = scf > upper_bound

    def predict(self, X) -> np.array:
        # Convert the dataframe to a numpy array to comport with the other estimators.
        return self._labels.values
