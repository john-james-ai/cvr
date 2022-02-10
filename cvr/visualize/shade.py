#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ============================================================================ #
# Project  : Deep Learning for Conversion Rate Prediction (CVR)                #
# Version  : 0.1.0                                                             #
# File     : \shade.py                                                         #
# Language : Python 3.7.12                                                     #
# ---------------------------------------------------------------------------- #
# Author   : John James                                                        #
# Email    : john.james.ai.studio@gmail.com                                    #
# URL      : https://github.com/john-james-ai/cvr                              #
# ---------------------------------------------------------------------------- #
# Created  : Thursday, February 3rd 2022, 6:07:19 pm                           #
# Modified : Friday, February 4th 2022, 2:42:08 pm                             #
# Modifier : John James (john.james.ai.studio@gmail.com)                       #
# ---------------------------------------------------------------------------- #
# License  : BSD 3-clause "New" or "Revised" License                           #
# Copyright: (c) 2022 Bryant St. Labs                                          #
# ============================================================================ #
#%%
import holoviews as hv
import numpy as np
import panel as pn
import pandas as pd
from holoviews.operation.datashader import (
    dynspread,
    datashade,
    rasterize,
    shade,
)
from holoviews.operation import decimate
import datashader as ds

pn.extension()

N = 1000000
x = np.random.random(N)
y = np.random.random(N)

dset = hv.Dataset(pd.DataFrame({"x": x, "y": y, "z1": x * y, "z2": -x * y}))

pts1 = hv.Points(dset, kdims=["x", "y"], vdims=["z1"])
pts2 = hv.Points(dset, kdims=["x", "y"], vdims=["z2"])

agg1 = ds.mean("z1")
agg2 = ds.mean("z2")

opts = dict(height=800, width=800)

plot1 = datashade(pts1, aggregator=agg1).options(**opts) * decimate(pts1)
plot2 = datashade(pts2, aggregator=agg2).options(**opts) * decimate(pts2)

plots = [("z1", pn.panel(plot1)), ("z2", pn.panel(plot2))]

pn.Tabs(*plots)
