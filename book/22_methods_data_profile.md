---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: 'Python 3.7.12 64-bit (''cvr'': conda)'
  language: python
  name: python3
---

# Data Profile
This first examination of the data seeks to characterize data quality in its (near) raw form. Here, we will discover the scope and breadth of data preprocessing that will be considered before advancing to the exploratory analysis effort. The remainder of this section is organized as follows:

    1. Descriptive statistics, missing values and cardinality.    
    2. Distribution analysis of continuous variables.    
    3. Frequency analysis of categorical variables.         
    4. Summary and recommendations

If you recall, in the last section, we loaded the Dataset object, vesuvio, into the staging area of the workspace of the same name.  Let's instantiate vesuvio (singleton) and obtain the Datasaet.

```{code-cell} ipython3
from cvr.core.workspace import Workspace
```

```{code-cell} ipython3
workspace = Workspace('vesuvio')
dataset = workspace.get_dataset(stage='staging', name='vesuvio')
dataset.summary
```
