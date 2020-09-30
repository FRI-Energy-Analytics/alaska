# AlasKA: The las file aliaser

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4047049.svg)](https://doi.org/10.5281/zenodo.4047049) [![Build Status](https://travis-ci.org/FRI-Energy-Analytics/alaska.svg?branch=master)](https://travis-ci.org/FRI-Energy-Analytics/alaska) [![codecov](https://codecov.io/gh/FRI-Energy-Analytics/alaska/branch/master/graph/badge.svg)](https://codecov.io/gh/FRI-Energy-Analytics/alaska)

AlasKA is a Python package that reads mnemonics from LAS files and outputs an aliased dictionary of mnemonics and its aliases, as well as a list of mnemonics that cannot be found. It uses three different methods to find aliases to mnemonics: locates exact matches of a mnemonic in an alias dictionary, identifies keywords in mnemonics' description then returns alias from the keyword extractor, and predicts alias using all attributes of the curves.

#### Sample Usage

```python
from alaska import Alias
from welly import Project
import lasio

path = "testcase.las"
a = Alias()
parsed, not_found = a.parse(path)
```

In this case, parsed is the aliased dictionary that contains mnemonics and its aliases, and not_found is the list of mnemonics that the aliaser did not find. Users can manually alias mnemonics in the not_found list and add them to the dictionary of aliased mnemonics

Parameters of the Alias class can be changed, and the defaults are the following

```python
a = Alias(dictionary=True, keyword_extractor=True, model=True, prob_cutoff=.5)
```

Users can choose which parser to use/not to use by setting the parsers to True/False. The prob_cutoff is the confidence the user wants the predictions made by model parser to have.

Then, the aliased mnemonics can be inputted into welly as demonstrated below.

```python
from welly import Project
p = Project.from_las(path)
data = p.df(keys=list(parsed.keys()), alias=parsed)
print(data)
```
