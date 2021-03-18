# Copyright (c) 2021 The AlasKA Developers.
# Distributed under the terms of the MIT License.
# SPDX-License_Identifier: MIT
"""
Utility function to reliably get AlasKA data files from the package data
location whether it is a regular package install (pip install alaska) or a
developer/editable install from the alaska repo (pip install -e .)
also see: https://importlib-resources.readthedocs.io/en/latest/
"""
from alaska import data

# Maintenance note:
#    When Python 3.6 support ends, this try/except can be
#   changed to simply `import importlib.resources as pkg_resources`
try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources


def get_data_path(datafile):
    """
    :param datafile: the file name of a file in the alaska.data module
    :return: PosixPath() object containing the full path to the datafile

    Build and return a path object for the input datafile

    Example: get_data_path("my_wonderful.csv")
    """
    with pkg_resources.path(data, datafile) as mypath:
        return mypath
