from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
from .keyword_tree import Alias, search, make_tree, search_child
from .predict_from_model import make_prediction
from .get_data_path import get_data_path
