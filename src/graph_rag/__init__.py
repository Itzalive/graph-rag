def hello() -> str:
    return "Hello from graph-rag!"
from .config import Settings, load_settings

__all__ = ["Settings", "load_settings"]
