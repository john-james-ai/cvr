#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ======================================================================================================================== #
# Project  : Explainable Recommendation (XRec)                                                                             #
# Version  : 0.1.0                                                                                                         #
# File     : \formatting.py                                                                                                #
# Language : Python 3.8                                                                                                    #
# ------------------------------------------------------------------------------------------------------------------------ #
# Author   : John James                                                                                                    #
# Email    : john.james.ai.studio@gmail.com                                                                                #
# URL      : https://github.com/john-james-ai/xrec                                                                         #
# ------------------------------------------------------------------------------------------------------------------------ #
# Created  : Wednesday, December 29th 2021, 11:58:51 pm                                                                    #
# Modified : Friday, December 31st 2021, 4:01:13 am                                                                        #
# Modifier : John James (john.james.ai.studio@gmail.com)                                                                   #
# ------------------------------------------------------------------------------------------------------------------------ #
# License  : BSD 3-clause "New" or "Revised" License                                                                       #
# Copyright: (c) 2021 Bryant St. Labs                                                                                      #
# ======================================================================================================================== #
def proper(x):
    x = x.replace("_", " ")
    x = x.capitalize()
    x = x.replace("Nan", "NaN")
    return x


def titlelize(x):
    x = x.replace("_", " ")
    x = x.title()
    x = x.replace("Nan", "NaN")
    return x


def titlelize_df(df):
    new_cols = []
    cols = df.columns
    for col in cols:
        new_cols.append(titlelize(col))
    df.columns = new_cols
    return df


def s_to_dict(df):
    s = df.to_dict("list")
    for k, v in s.items():
        if isinstance(v, str):
            s[k] = v[0].split()[0]
        else:
            s[k] = v[0]
    return s
