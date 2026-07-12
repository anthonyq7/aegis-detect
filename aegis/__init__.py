from importlib.metadata import PackageNotFoundError, version

from aegis.predictor import Predictor

try:
    __version__ = version("aegis-detect")
except PackageNotFoundError:
    __version__ = "unknown"
