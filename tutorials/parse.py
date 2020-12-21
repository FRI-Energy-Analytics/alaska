"""
Parsing mnemonics of a LAS file
===============================

Often, a mnemonic aliasing process looks like the following:

#. List all the mnemonics in the file
#. Group synonymous mnemonics under a single label
#. Make dictionaries with mnemonics and labels
#. Feed dictionaries into welly

The class alaska.Aliaser takes input mnemonics from a LAS file
or a directory of LAS files, and aliases the mnemonics so that
synonymous mnemonics are grouped under the same label.

See example below.
"""
import os

from welly import Project

from alaska import Alias, get_data_path

path = str(get_data_path("testcase1.las"))


# initialize aliaser
a = Alias()
# the parameters can also be customized as such:
# a = Alias(dictionary=True, keyword_extractor=True,
#                          model=True, prob_cutoff=.5)

# the parse function returns two dictionaries, one for parsed
# and another one for mnemonics not found using the aliaser
parsed, not_found = a.parse(path)

# print aliased mnemonics to screen
print("*" * 10, "Aliased dictionary", "*" * 10)
for i in parsed:
    print("{}: {}".format(i, parsed[i]))
print("Not parsed with Aliaser:", not_found)

# feed parsed dictionary into welly, and leave the not aliased
# ones alone
p = Project.from_las(path)
data = p.df(keys=list(parsed.keys()), alias=parsed)
print(data)

# print the heatmap of the aliased mnemonics to visualize results
a.heatmap()
