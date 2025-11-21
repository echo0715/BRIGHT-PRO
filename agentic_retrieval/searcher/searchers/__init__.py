"""
Searchers package for different search implementations.
"""

from enum import Enum
from .base import BaseSearcher
from .bm25_searcher import BM25Searcher

from .custom_searcher import CustomSearcher
from .diver_searcher import DiverSearcher
from .bge_searcher import BGESearcher
from .inst_searcher import InstSearcher
from .qwen_searcher import QwenSearcher
from .reasonir_searcher import ReasonIRSearcher
from .grit_searcher import GritSearcher

class SearcherType(Enum):
    """Enum for managing available searcher types and their CLI mappings."""
    BM25 = ("bm25", BM25Searcher)
    REASONIR_EMBED = ("reasonir", ReasonIRSearcher)
    DIVER = ("diver", DiverSearcher)
    BGE = ("bge", BGESearcher)
    INSTRUCTOR = ("inst-l", InstSearcher)
    INSTRUCTOR_XL = ("inst-xl", InstSearcher)
    QWEN = ("qwen", QwenSearcher)
    GRIT = ("grit", GritSearcher)
    CUSTOM = ("custom", CustomSearcher) # Your custom searcher class, yet to be implemented
    
    def __init__(self, cli_name, searcher_class):
        self.cli_name = cli_name
        self.searcher_class = searcher_class
    
    @classmethod
    def get_choices(cls):
        """Get list of CLI choices for argument parser."""
        return [searcher_type.cli_name for searcher_type in cls]
    
    @classmethod
    def get_searcher_class(cls, cli_name):
        """Get searcher class by CLI name."""
        for searcher_type in cls:
            if searcher_type.cli_name == cli_name:
                return searcher_type.searcher_class
        raise ValueError(f"Unknown searcher type: {cli_name}")


__all__ = [
    "BaseSearcher",
    "SearcherType"
]