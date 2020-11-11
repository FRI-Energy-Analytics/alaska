"""
This tutorial is meant to be a quick and easy example of 
aliasing mnemonics using AlasKA. For more advanced usage,
please refer to more_examples.py
"""
from aliaser import Alias
from welly import Project
import lasio

path = "data/testcase1.las"  # any lAS file of your choice
# initialize aliaser
a = Alias()

# the parse function returns two dictionaries, one for parsed
# and another one for mnemonics not aliased using the aliaser.
# in this case, not_found is empty as we didn't specify any
# parameters for Alias(), so the aliaser finds a closest match
# for all mnemonics in the LAS file.
parsed, not_found = a.parse(path)

# visualize parsed mnemonics in welly as dataframe
p = Project.from_las(path)
data = p.df(keys=list(parsed.keys()), alias=parsed)
print(data)
