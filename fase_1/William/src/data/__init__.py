"""Data module - Carregamento e preprocessamento de dados."""

from .loader import TelcoDataLoader
from .preprocessing import TelcoDataPreprocessor

__all__ = [
    "TelcoDataLoader",
    "TelcoDataPreprocessor",
]
