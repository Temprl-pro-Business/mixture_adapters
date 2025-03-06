"""
Components for semantic-based routing of queries to appropriate adapters.
"""

from .route import AdapterRoute
from .router import SemanticRouter

__all__ = ["AdapterRoute", "SemanticRouter"] 