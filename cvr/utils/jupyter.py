#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ======================================================================================================================== #
# Project  : Conversion Rate Prediction (CVR)                                                                              #
# Version  : 0.1.0                                                                                                         #
# File     : \jupyter.py                                                                                                   #
# Language : Python 3.7.12                                                                                                 #
# ------------------------------------------------------------------------------------------------------------------------ #
# Author   : John James                                                                                                    #
# Email    : john.james.ai.studio@gmail.com                                                                                #
# URL      : https://github.com/john-james-ai/cvr                                                                          #
# ------------------------------------------------------------------------------------------------------------------------ #
# Created  : Tuesday, January 25th 2022, 8:19:18 pm                                                                        #
# Modified : Tuesday, January 25th 2022, 8:31:33 pm                                                                        #
# Modifier : John James (john.james.ai.studio@gmail.com)                                                                   #
# ------------------------------------------------------------------------------------------------------------------------ #
# License  : BSD 3-clause "New" or "Revised" License                                                                       #
# Copyright: (c) 2022 Bryant St. Labs                                                                                      #
# ======================================================================================================================== #
#%%
import nbformat as nbf
from glob import glob

# ------------------------------------------------------------------------------------------------------------------------ #
def prepare_notebooks():
    # Collect a list of all notebooks in the content folder
    notebooks = glob("book/*.ipynb", recursive=True)

    # Text to look for in adding tags
    text_search_dict = {
        "# IMPORTS": "remove-output",  # Remove the whole cell
        "# GLUE": "hide-input",  # Remove only the input
        "# HIDE CODE": "hide-input",  # Hide the input w/ a button to show
    }

    # Search through each notebook and look for the text, add a tag if necessary
    for ipath in notebooks:
        ntbk = nbf.read(ipath, nbf.NO_CONVERT)

        for cell in ntbk.cells:
            cell_tags = cell.get("metadata", {}).get("tags", [])
            for key, val in text_search_dict.items():
                if key in cell["source"]:
                    if val not in cell_tags:
                        cell_tags.append(val)
            if len(cell_tags) > 0:
                cell["metadata"]["tags"] = cell_tags

        nbf.write(ntbk, ipath)


if __name__ == "__main__":
    prepare_notebooks()

#%%
