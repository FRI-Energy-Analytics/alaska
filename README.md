# AlasKA: The las file aliaser

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4047049.svg)](https://doi.org/10.5281/zenodo.4047049) [![Actions Status](https://github.com/FRI-Energy-Analytics/alaska/workflows/Python%20package/badge.svg)](https://github.com/FRI-Energy-Analytics/alaska/actions) [![codecov](https://codecov.io/gh/FRI-Energy-Analytics/alaska/branch/master/graph/badge.svg)](https://codecov.io/gh/FRI-Energy-Analytics/alaska)

AlasKA is a Python package that reads mnemonics from LAS files and outputs an aliased dictionary of mnemonics and its aliases, as well as a list of mnemonics that cannot be found. It uses three different methods to find aliases to mnemonics: locates exact matches of a mnemonic in an alias dictionary, identifies keywords in mnemonics' description then returns alias from the keyword extractor, and predicts alias using all attributes of the curves.

#### Install

To install the package first clone the repository to your local machine. Then `cd` into the repository and create a virtual environment with `conda env create -f environment.yml`. Next activate the environment with `conda activate alaska`. From inside the `alaska` virtual environment and in the `alaska` top-level directory run `pip install .` and it will install the latest version of `alaska` and all the dependencies except for PyTorch. To install PyTorch go to [https://pytorch.org/](https://pytorch.org/) and select the version of PyTorch that is compatible with your machine. PyTorch is only required if you want to use the pointer generator model to alias mnemonics (~10% of log mnemonics need the pointer generator model).

#### Sample Usage

```python
from alaska import Alias
from welly import Project

path = "testcase.las"
a = Alias()
parsed, not_found = a.parse(path)
```

In this case, `parsed` is the aliased dictionary that contains mnemonics and its aliases, and `not_found` is the list of mnemonics that the aliaser did not find. Users can manually alias mnemonics in the `not_found` list and add them to the dictionary of aliased mnemonics

Parameters of the Alias class can be changed, and the defaults are the following

```python
a = Alias(dictionary=True, keyword_extractor=True, model=False, prob_cutoff=.5)
```

Users can choose which parser to use/not to use by setting the parsers to True/False. The `prob_cutoff` is the confidence the user wants the predictions made by the pointer generator model parser to have.

Then, the aliased mnemonics can be input into `welly` as demonstrated below.

```python
from welly import Project
p = Project.from_las(path)
data = p.df(keys=list(parsed.keys()), alias=parsed)
print(data)
```
