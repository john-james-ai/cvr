#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ======================================================================================================================== #
# Project  : Conversion Rate Prediction (CVR)                                                                              #
# Version  : 0.1.0                                                                                                         #
# File     : \system.py                                                                                                    #
# Language : Python 3.7.12                                                                                                 #
# ------------------------------------------------------------------------------------------------------------------------ #
# Author   : John James                                                                                                    #
# Email    : john.james.ai.studio@gmail.com                                                                                #
# URL      : https://github.com/john-james-ai/cvr                                                                          #
# ------------------------------------------------------------------------------------------------------------------------ #
# Created  : Sunday, January 23rd 2022, 8:39:55 pm                                                                         #
# Modified : Sunday, January 23rd 2022, 10:45:37 pm                                                                        #
# Modifier : John James (john.james.ai.studio@gmail.com)                                                                   #
# ------------------------------------------------------------------------------------------------------------------------ #
# License  : BSD 3-clause "New" or "Revised" License                                                                       #
# Copyright: (c) 2022 Bryant St. Labs                                                                                      #
# ======================================================================================================================== #
#%%
from collections import OrderedDict
import psutil
import platform
import datetime as datetime
import speedtest
import subprocess
import cpuinfo

from cvr.utils.printing import Printer

# ------------------------------------------------------------------------------------------------------------------------ #
class System:
    """Provides system information"""

    def __init__(self) -> None:
        self._printer = Printer()

    def summary(self) -> None:
        s = platform.uname()
        d = OrderedDict()
        d["System"] = s.system
        d["Release"] = s.release
        d["Version"] = s.version
        self._printer.print_dictionary(d, title="Operating System")

    def cpu(self) -> None:
        d = OrderedDict()
        cinfo = cpuinfo.get_cpu_info()

        try:
            d["Model"] = (
                subprocess.check_output("wmic computersystem get model", stderr=open(os.devnull, "w"), shell=True)
                .decode("utf-8")
                .partition("Model")[2]
                .strip(" \r\n")
            )
        except Exception as error:
            d["Model"] = None

        try:
            d["Manufacturer"] = (
                str(
                    subprocess.check_output("wmic computersystem get manufacturer", stderr=open(os.devnull, "w"), shell=True)
                )
                .split()[1]
                .strip("\\r\\n")
            )
        except Exception as error:
            d["Manufacturer"] = None

        d["Processor"] = cinfo["brand_raw"]

        # CPU Cores
        d["Physical CPU Cores"] = psutil.cpu_count(logical=False)
        d["Total CPU Cores"] = psutil.cpu_count(logical=True)

        # Frequencies
        cpufreq = psutil.cpu_freq()
        d["Max Frequency"] = str(round(cpufreq.max, 2)) + " Mhz"
        d["Min Frequency"] = str(round(cpufreq.min, 2)) + " Mhz"
        d["Current Frequency"] = str(round(cpufreq.current, 2)) + " Mhz"
        self._printer.print_dictionary(d, title="CPU Information")

    def memory(self) -> None:
        d = OrderedDict()
        svmem = psutil.virtual_memory()
        d["Total"] = self._get_size(svmem.total)
        d["Available"] = self._get_size(svmem.available)
        d["Used"] = self._get_size(svmem.used)
        d["Percentage"] = str(svmem.percent) + "%"
        self._printer.print_dictionary(d, title="Memory Information")

    def disk(self) -> None:
        d = OrderedDict()
        partitions = psutil.disk_partitions()
        for i, partition in enumerate(partitions):
            try:
                partition_usage = psutil.disk_usage(partition.mountpoint)
            except PermissionError:
                continue
            d[f"Partition {i} Total Size"] = self._get_size(partition_usage.total)
            # d[f"Partition {i} Used"] = self._get_size(partition_usage.used)
            # d[f"Partition {i} Free"] = self._get_size(partition_usage.free)
            # d[f"Partition {i} Percentage"] = str(self._get_size(partition_usage.percent)) + "%"
        self._printer.print_dictionary(d, title="Disk Information")

    def network(self) -> None:
        d = OrderedDict()
        s = speedtest.Speedtest()
        d["Download Speed"] = self._get_size(s.download())
        d["Upload Speed"] = self._get_size(s.upload())
        self._printer.print_dictionary(d, title="Network Information")

    def _get_size(self, bytes, suffix="B"):
        """
        Scale bytes to its proper format
        e.g:
            1253656 => '1.20MB'
            1253656678 => '1.17GB'
        """
        factor = 1024
        for unit in ["", "K", "M", "G", "T", "P"]:
            if bytes < factor:
                n = "{} {}{}".format(str(round(bytes, 2)), unit, suffix)
                return n
            bytes /= factor


if __name__ == "__main__":
    t = System()
    t.summary()
    t.cpu()
    t.memory()
    t.disk()
#%%
