"""MLTrail — a local, versioned registry (vault) for trained ML models."""
from .config import load_config
from .predict import predict
from .registry import Registry

__all__ = ["Registry", "load_config", "predict"]
__version__ = "0.1.0"
