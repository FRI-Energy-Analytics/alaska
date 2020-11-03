"""
Example script demonstrating AlasKA parsing
"""
import os
from pathlib import Path

import lasio
import pandas as pd

from alaska import Alias, get_data_path

path = Path(os.path.join(get_data_path(), "testcase3.las"))

a = Alias()
parsed, not_found = a.parse(path)
las = lasio.read(path)
l, l2, l3 = [], [], []
for key, value in parsed.items():
    l.append(key)
    l2.append(str(las.curves[key].descr))
    l3.append(value)
output_df = pd.DataFrame({"mnemonic": l, "description": l2, "parsed_mnemonic": l3})
l, l2, l3, lst = [], [], [], []
for i in not_found:
    l.append(i)
    l2.append(str(las.curves[i].descr))
    l3.append(str(las.curves[i].unit).lower())
output_df2 = pd.DataFrame({"mnemonics": l, "description": l2, "units": l3})
print(output_df)
print(output_df.loc[output_df["mnemonic"].isin(output_df2.mnemonics)])
