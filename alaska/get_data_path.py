"""
Utility function to reliably get AlasKA data files from the package data
location whether it is a regular package install (pip install alaska) or a
developer/editable install from the alaska repo (pip install -e .)
also see: https://importlib-resources.readthedocs.io/en/latest/
"""
# Maintenance note:
#    When Python 3.6 support ends, this try/except can be
#   changed to simply `import importlib.resources as pkg_resources`
try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources


def get_data_path():
    with pkg_resources.path("alaska", "data") as mypath:
        return mypath