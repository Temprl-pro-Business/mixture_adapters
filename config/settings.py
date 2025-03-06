"""
Configuration settings for the mixture of adapters system.
"""

from typing import Dict

class Settings:
    """Global settings for the mixture of adapters system."""
    
    # Model settings
    BASE_MODEL_NAME: str = "unsloth/Qwen2.5-1.5B-Instruct-unsloth-bnb-4bit"
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Generation settings
    MAX_NEW_TOKENS: int = 512
    TEMPERATURE: float = 0.7
    DO_SAMPLE: bool = True
    
    # Routing settings
    SIMILARITY_THRESHOLD: float = 0.7
    
    # Model loading settings
    LOAD_IN_8BIT: bool = False
    LOAD_IN_4BIT: bool = True
    
    # PEFT adapter configurations
    # Map adapter names to their HuggingFace Hub paths
    PEFT_ADAPTERS: Dict[str, str] = {
        # Example:
        # "go_adapter": "your-username/go-programming-adapter",
        # "python_adapter": "your-username/python-programming-adapter"
    }
    
    @classmethod
    def get_model_settings(cls) -> Dict:
        """Get model-related settings."""
        return {
            "base_model_name": cls.BASE_MODEL_NAME,
            "load_in_8bit": cls.LOAD_IN_8BIT,
            "load_in_4bit": cls.LOAD_IN_4BIT
        }
        
    @classmethod
    def get_generation_settings(cls) -> Dict:
        """Get text generation settings."""
        return {
            "max_new_tokens": cls.MAX_NEW_TOKENS,
            "temperature": cls.TEMPERATURE,
            "do_sample": cls.DO_SAMPLE
        }
        
    @classmethod
    def get_routing_settings(cls) -> Dict:
        """Get semantic routing settings."""
        return {
            "embedding_model_name": cls.EMBEDDING_MODEL_NAME,
            "similarity_threshold": cls.SIMILARITY_THRESHOLD
        } 