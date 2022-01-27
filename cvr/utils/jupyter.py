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
# Modified : Thursday, January 27th 2022, 2:49:35 am                                                                       #
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

    # Userful tags
    # Two types of tags, hide and remove.
    #   Hide provides a button to reveal the cell contents
    #   Remove prevents the content from appearing in the HTML at all.
    # Hide Tags:
    #   "hide-input": Hides the cell but displays the output
    #   "hide-output": Hides the output from a cell, but provides a button to show
    #   "hide-cell": Hides both input and output
    # Remove Tags:
    #    "remove-input": Removes cell from HTML, but shows ouput. No botton available
    #    "remove-output": Removes cell output from HTML. No botton
    #    "remove-cell": Removes entire cell, input and output. No botton.
    #
    # remove-cell: remove entire cell
    #

    # Text to look for in adding tags
    text_search_dict = {
        "# IMPORTS": "remove-output",  # Removes the 'module not found' error from output
        "# GLUE": "remove-cell",  # Removes the cell (input/output) which declares glue variables
        "# HIDE CODE": "hide-input",  # Hide the input w/ a button to show
        "# REMOVE": "remove-cell",  # Removes cells so marked
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
