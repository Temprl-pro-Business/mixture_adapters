"""
Module for managing the base language model and tokenizer.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Optional

class ModelManager:
    """
    Manages the base language model and tokenizer initialization and operations.
    """
    
    def __init__(
        self,
        model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        load_in_8bit: bool = False,
        load_in_4bit: bool = True
    ):
        """
        Initialize the model manager.
        
        Args:
            model_name (str): Name of the HuggingFace model to load
            device (str): Device to load the model on ('cuda' or 'cpu')
            load_in_8bit (bool): Whether to load model in 8-bit precision
            load_in_4bit (bool): Whether to load model in 4-bit precision
        """
        self.model_name = model_name
        self.device = device
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        
        self._initialize_components()
        
    def _initialize_components(self) -> None:
        """Initialize the tokenizer and model components."""
        print(f"Loading tokenizer from {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        print("Tokenizer loaded successfully")
        
        print(f"Loading model from {self.model_name}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device,
            load_in_8bit=self.load_in_8bit,
            load_in_4bit=self.load_in_4bit
        )
        print(f"Model loaded successfully and moved to {self.device} device")
        
    def get_model(self) -> AutoModelForCausalLM:
        """Get the loaded model."""
        return self.model
        
    def get_tokenizer(self) -> AutoTokenizer:
        """Get the loaded tokenizer."""
        return self.tokenizer
        
    def move_to_device(self, device: Optional[str] = None) -> None:
        """
        Move the model to a specific device.
        
        Args:
            device (Optional[str]): Device to move the model to. If None, uses the default device
        """
        target_device = device or self.device
        self.model.to(target_device)
        print(f"Model moved to {target_device} device") 