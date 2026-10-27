from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("aegis-detect")
except PackageNotFoundError:
    __version__ = "unknown"


def __getattr__(name):
    # Lazy import so `aegis.normalize` can be used without loading torch/transformers.
    if name == "Predictor":
        from aegis.predictor import Predictor

        return Predictor
    raise AttributeError(f"module 'aegis' has no attribute {name!r}")
