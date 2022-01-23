#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ======================================================================================================================== #
# Project  : Explainable Recommendation (XRec)                                                                             #
# Version  : 0.1.0                                                                                                         #
# File     : \__init__.py                                                                                                  #
# Language : Python 3.8                                                                                                    #
# ------------------------------------------------------------------------------------------------------------------------ #
# Author   : John James                                                                                                    #
# Email    : john.james.ai.studio@gmail.com                                                                                #
# URL      : https://github.com/john-james-ai/xrec                                                                         #
# ------------------------------------------------------------------------------------------------------------------------ #
# Created  : Tuesday, December 21st 2021, 7:45:33 pm                                                                       #
# Modified : Sunday, January 23rd 2022, 12:02:31 am                                                                        #
# Modifier : John James (john.james.ai.studio@gmail.com)                                                                   #
# ------------------------------------------------------------------------------------------------------------------------ #
# License  : BSD 3-clause "New" or "Revised" License                                                                       #
# Copyright: (c) 2021 Bryant St. Labs                                                                                      #
# ======================================================================================================================== #
import numpy as np

criteo_columns = [
    "sale",
    "sales_amount",
    "conversion_time_delay",
    "click_ts",
    "n_clicks_1week",
    "product_price",
    "product_age_group",
    "device_type",
    "audience_id",
    "product_gender",
    "product_brand",
    "product_category_1",
    "product_category_2",
    "product_category_3",
    "product_category_4",
    "product_category_5",
    "product_category_6",
    "product_category_7",
    "product_country",
    "product_id",
    "product_title",
    "partner_id",
    "user_id",
]

feature_columns = [
    "click_ts",
    "n_clicks_1week",
    "product_price",
    "product_age_group",
    "device_type",
    "audience_id",
    "product_gender",
    "product_brand",
    "product_category_1",
    "product_category_2",
    "product_category_3",
    "product_category_4",
    "product_category_5",
    "product_category_6",
    "product_category_7",
    "product_country",
    "product_id",
    "product_title",
    "partner_id",
    "user_id",
]

numeric_columns = [
    "sales_amount",
    "conversion_time_delay",
    "click_ts",
    "sale_ts",
    "n_clicks_1week",
    "product_price",
]
categorical_columns = [
    "sale",
    "product_age_group",
    "device_type",
    "audience_id",
    "product_gender",
    "product_brand",
    "product_category_1",
    "product_category_2",
    "product_category_3",
    "product_category_4",
    "product_category_5",
    "product_category_6",
    "product_category_7",
    "product_country",
    "product_id",
    "product_title",
    "partner_id",
    "user_id",
]
target_columns = ["sale", "sales_amount", "conversion_time_delay"]

numeric_descriptive_stats = [
    "column",
    "datatype",
    "size",
    "count",
    "mean",
    "min",
    "25%",
    "50%",
    "75%",
    "max",
]

categorical_descriptive_stats = [
    "column",
    "datatype",
    "size",
    "count",
    "unique",
    "uniqueness",
    "top",
    "freq",
    "freq_%",
]


criteo_dtypes = {
    "sale": "category",
    "sales_amount": np.float64,
    "conversion_time_delay": np.int64,
    "click_ts": np.int64,
    "n_clicks_1week": np.int64,
    "product_price": np.float64,
    "product_age_group": "category",
    "device_type": "category",
    "audience_id": "category",
    "product_gender": "category",
    "product_brand": "category",
    "product_category_1": "category",
    "product_category_2": "category",
    "product_category_3": "category",
    "product_category_4": "category",
    "product_category_5": "category",
    "product_category_6": "category",
    "product_category_7": "category",
    "product_country": "category",
    "product_id": "category",
    "product_title": str,
    "partner_id": "category",
    "user_id": "category",
}
