"""MLTrail — a local, versioned registry (vault) for trained ML models."""
from .config import default_config, load_config
from .predict import predict
from .registry import Registry

__all__ = ["Registry", "load_config", "default_config", "predict"]
__version__ = "0.1.0"
