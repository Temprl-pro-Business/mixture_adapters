"""
Module for handling adapter routes and their associated utterances.
"""

from dataclasses import dataclass
from typing import List

@dataclass
class AdapterRoute:
    """
    Represents a route for an adapter with its associated utterances for semantic routing.
    
    Attributes:
        adapter_name (str): The name of the adapter this route corresponds to
        training_utterances (List[str]): List of example utterances used for semantic routing
    """
    adapter_name: str
    training_utterances: List[str]

    def __post_init__(self):
        """Validate the route attributes after initialization."""
        if not isinstance(self.adapter_name, str):
            raise ValueError("adapter_name must be a string")
        if not isinstance(self.training_utterances, list):
            raise ValueError("training_utterances must be a list") 